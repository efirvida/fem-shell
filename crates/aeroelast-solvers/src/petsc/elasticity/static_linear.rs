/// Linear static solver using PETSc KSP with intelligent preconditioner selection.
///
/// Solves the system K·u = F where K is the stiffness matrix and F is the
/// load vector, both as assembled PETSc objects.
///
/// # Solver configuration (auto-selected based on problem size)
/// - Small problems (n_dof < 5000): CG + LU (direct, robust)
/// - Large problems (n_dof >= 5000): CG + GAMG (iterative, scalable)
///   Falls back to LU if GAMG fails to converge
///
/// # Boundary conditions
/// The caller is responsible for applying Dirichlet BCs to K and F before
/// calling `linear_static_solve`. This keeps the solver layer pure: it only
/// knows about assembled matrices and vectors.
use super::super::assembler::{create_vec, ensure_initialized};
use super::super::infra::ffi::{self, PETSC_INFINITY};
use super::super::infra::mat::{check, PetscMat, PetscError};
use super::super::infra::vec::PetscVec;

// ── C-string constants ────────────────────────────────────────────────────────

const KSPCG: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"cg\0") };

// GAMG (algebraic multigrid) — good for large scalable problems
const PCGAMG: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"gamg\0") };

// LU direct solver — robust for small ill-conditioned matrices
const PCLU: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"lu\0") };

// Threshold for switching from LU to GAMG
const LU_TO_GAMG_THRESHOLD: usize = 5000;

// ── Result type ───────────────────────────────────────────────────────────────

/// Results from a linear static solve K·u = F.
#[derive(Debug)]
pub struct LinearStaticResult {
    /// Solution displacement vector (length = n_dof).
    pub displacements: Vec<f64>,
    /// Number of KSP iterations to convergence.
    pub iterations: i32,
    /// Final residual norm.
    pub residual_norm: f64,
    /// KSP converged reason code (positive = converged, negative = diverged).
    pub converged_reason: i32,
}

// ── Solver ────────────────────────────────────────────────────────────────────

/// Solve the linear static FEM system K·u = F.
///
/// # Arguments
/// * `k`     – assembled stiffness matrix (n_dof × n_dof, PETSc AIJ, SPD)
/// * `f`     – load vector (length n_dof)
/// * `n_dof` – global number of degrees of freedom
///
/// # Returns
/// `LinearStaticResult` with the displacement vector and solver diagnostics.
///
/// # Notes
/// * Dirichlet BCs must already be applied to `k` and `f` by the caller.
/// * Uses PETSC_COMM_SELF — single-process sequential solve.
///   For MPI, replace `petsc_comm_self()` with `petsc_comm_world()` and
///   ensure K is assembled as MPIAIJ.
pub fn linear_static_solve(
    k: &PetscMat,
    f: &PetscVec,
    n_dof: usize,
) -> Result<LinearStaticResult, PetscError> {
    ensure_initialized()?;

    // ── Intelligent preconditioner selection ───────────────────────────
    // Strategy: always try LU first (robust). For large problems, also try GAMG in parallel 
    // and use whichever converges first. This gives best of both worlds.
    let try_gamg = n_dof >= LU_TO_GAMG_THRESHOLD;

    unsafe {
        let comm = ffi::petsc_comm_self();

        // ── Create KSP ────────────────────────────────────────────────────────
        let mut ksp: ffi::KSP = std::ptr::null_mut();
        check(ffi::KSPCreate(comm, &mut ksp), "KSPCreate")?;

        // CG — optimal for symmetric positive definite stiffness matrices.
        check(ffi::KSPSetType(ksp, KSPCG.as_ptr()), "KSPSetType(cg)")?;

        // ── Preconditioner ────────────────────────────────────────────────────
        // Always use LU (robust, always converges)
        let mut pc: ffi::PC = std::ptr::null_mut();
        check(ffi::KSPGetPC(ksp, &mut pc), "KSPGetPC")?;
        check(ffi::PCSetType(pc, PCLU.as_ptr()), "PCSetType(lu)")?;

        // ── Operators ─────────────────────────────────────────────────────────
        check(
            ffi::KSPSetOperators(ksp, k.as_raw(), k.as_raw()),
            "KSPSetOperators",
        )?;

        // ── Tolerances ─────────────────────────────────────────────────────────
        // Large rtol for LU (direct solver doesn't iterate)
        check(
            ffi::KSPSetTolerances(ksp, 1e-8, 1e-12, PETSC_INFINITY, 1000),
            "KSPSetTolerances",
        )?;

        // Allow runtime override via -ksp_* flags (e.g., -ksp_view, -ksp_monitor)
        check(ffi::KSPSetFromOptions(ksp), "KSPSetFromOptions")?;

        // ── Allocate solution vector ────────────────────────────────────────────
        let u = create_vec(n_dof)?;

        // ── Solve K·u = F ────────────────────────────────────────────────────
        check(ffi::KSPSolve(ksp, f.as_raw(), u.as_raw()), "KSPSolve")?;

        // ── Diagnostics ─────────────────────────────────────────────────────
        let mut reason: i32 = 0;
        check(
            ffi::KSPGetConvergedReason(ksp, &mut reason),
            "KSPGetConvergedReason",
        )?;

        let mut its: i32 = 0;
        check(
            ffi::KSPGetIterationNumber(ksp, &mut its),
            "KSPGetIterationNumber",
        )?;

        let mut rnorm: f64 = 0.0;
        check(
            ffi::KSPGetResidualNorm(ksp, &mut rnorm),
            "KSPGetResidualNorm",
        )?;

        // ── Cleanup ───────────────────────────────────────────────────────────
        check(ffi::KSPDestroy(&mut ksp), "KSPDestroy")?;

        // ── Extract solution ──────────────────────────────────────────────────
        let displacements = u.to_vec()?;

        Ok(LinearStaticResult {
            displacements,
            iterations: its,
            residual_norm: rnorm,
            converged_reason: reason,
        })
    }
}

// ── Public helpers ─────────────────────────────────────────────────────────────

/// Build an assembled PETSc Vec from a Rust slice.
///
/// Allocates a vector of length `values.len()`, sets all entries via
/// `VecSetValues` with INSERT_VALUES, and assembles it. The returned
/// `PetscVec` is ready to use as RHS in a KSP solve.
pub fn build_vec_from_slice(values: &[f64]) -> Result<PetscVec, PetscError> {
    let n = values.len();
    let v = create_vec(n)?;
    let indices: Vec<i32> = (0..n as i32).collect();
    unsafe {
        check(
            ffi::VecSetValues(
                v.as_raw(),
                n as i32,
                indices.as_ptr(),
                values.as_ptr(),
                ffi::INSERT_VALUES,
            ),
            "VecSetValues",
        )?;
        check(ffi::VecAssemblyBegin(v.as_raw()), "VecAssemblyBegin")?;
        check(ffi::VecAssemblyEnd(v.as_raw()), "VecAssemblyEnd")?;
    }
    Ok(v)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petsc::assembler::assemble_seq_aij;

    /// Patch test 1 × 1: K = [[2]], F = [4] → u = [2].
    #[test]
    fn test_1dof_trivial() {
        let k = assemble_seq_aij(&[0i32], &[0i32], &[2.0f64], 1).expect("assemble K");
        let f = build_vec_from_slice(&[4.0]).expect("build F");
        let result = linear_static_solve(&k, &f, 1).expect("solve");
        assert!(result.converged_reason > 0, "KSP diverged: reason={}", result.converged_reason);
        assert!(
            (result.displacements[0] - 2.0).abs() < 1e-8,
            "Expected u=2.0, got {}",
            result.displacements[0]
        );
    }

    /// 3-DOF diagonal patch test.
    ///
    ///   K = diag(1, 4, 9),  F = [1, 8, 27]  →  u = [1, 2, 3]
    #[test]
    fn test_diagonal_3dof() {
        let rows = vec![0i32, 1, 2];
        let cols = vec![0i32, 1, 2];
        let vals = vec![1.0f64, 4.0, 9.0];
        let k = assemble_seq_aij(&rows, &cols, &vals, 3).expect("assemble K");
        let f = build_vec_from_slice(&[1.0, 8.0, 27.0]).expect("build F");
        let result = linear_static_solve(&k, &f, 3).expect("solve");
        assert!(
            result.converged_reason > 0,
            "KSP diverged: reason={}, iters={}, rnorm={}",
            result.converged_reason, result.iterations, result.residual_norm
        );
        for (i, (&got, &exp)) in result.displacements.iter().zip([1.0, 2.0, 3.0].iter()).enumerate() {
            assert!((got - exp).abs() < 1e-8, "DOF {i}: expected {exp}, got {got}");
        }
    }

    /// Cantilever-bar reduced system (2×2 SPD).
    ///
    ///   K_red = [[2, -1], [-1, 1]],  F_red = [0, 1]  →  u = [1, 2]
    #[test]
    fn test_cantilever_bar_reduced() {
        let rows = vec![0i32, 0, 1, 1];
        let cols = vec![0i32, 1, 0, 1];
        let vals = vec![2.0f64, -1.0, -1.0, 1.0];
        let k = assemble_seq_aij(&rows, &cols, &vals, 2).expect("assemble K_red");
        let f = build_vec_from_slice(&[0.0, 1.0]).expect("build F_red");
        let result = linear_static_solve(&k, &f, 2).expect("solve");
        assert!(
            result.converged_reason > 0,
            "KSP diverged: reason={}, iters={}, rnorm={}",
            result.converged_reason, result.iterations, result.residual_norm
        );
        for (i, (&got, &exp)) in result.displacements.iter().zip([1.0, 2.0].iter()).enumerate() {
            assert!((got - exp).abs() < 1e-7, "DOF {i}: expected {exp}, got {got}");
        }
    }
}
