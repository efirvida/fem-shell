/// Nonlinear static solver using PETSc SNES.
///
/// Solves the residual equation R(u) = F_int(u) - F_ext = 0 where:
/// - `F_int(u)` is the internal force vector assembled with Green-Lagrange
///   strains (geometric nonlinearity, `nonlinear=true` in `assemble_fint`)
/// - `F_ext` is the constant external load vector
///
/// # Solver configuration
/// The default SNES type is `NEWTONLS` (Newton-Raphson with line search).
/// At runtime this can be switched to arc-length via `-snes_type newtonal`
/// (e.g. from the simulation YAML under `solver_options`).
///
/// The inner KSP uses CG + ICC (same as the linear static solver) — ICC is
/// the correct preconditioner for the symmetric-positive-definite tangent
/// stiffness K_T that arises in structural FEM.
///
/// # Boundary conditions
/// Dirichlet DOFs are enforced in both the residual and the Jacobian:
/// - Residual: R[i] = 0  for i ∈ dirichlet_dofs
/// - Jacobian: K_T[i,i] = 1, all off-diagonal entries zero (identity row)
///
/// This "one-on-diagonal / zero-off-diagonal" approach keeps K_T symmetric
/// and is the standard approach in PETSc FEM codes.
use std::ffi::c_void;

use aeroelast_core::assembly::MeshAssembler;

use super::super::assembler::{assemble_seq_aij, create_vec, ensure_initialized};
use super::super::infra::ffi::{self, PETSC_DEFAULT, PETSC_INFINITY};
use super::super::infra::mat::{check, PetscError, PetscMat};
use super::static_linear::build_vec_from_slice;

// ── C-string constants ────────────────────────────────────────────────────────

const KSPCG: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"cg\0") };
const PCICC: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"icc\0") };

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the nonlinear static solver.
///
/// All fields have defaults suitable for structural FEM. Override from YAML
/// by passing `solver_options: "-snes_type newtonal -snes_monitor"` at
/// runtime via PETSc option flags.
#[derive(Debug, Clone)]
pub struct NonlinearConfig {
    /// Absolute residual tolerance (‖R(u)‖ < atol → converged).
    pub atol: f64,
    /// Relative residual tolerance (‖R(u_k)‖ / ‖R(u_0)‖ < rtol → converged).
    pub rtol: f64,
    /// Step-length tolerance (‖δu‖ / ‖u‖ < stol → converged).
    pub stol: f64,
    /// Maximum Newton iterations.
    pub max_it: i32,
    /// Maximum residual evaluations (−1 = unlimited).
    pub max_funcs: i32,
}

impl Default for NonlinearConfig {
    fn default() -> Self {
        Self {
            atol: 1e-10,
            rtol: 1e-8,
            stol: 1e-8,
            max_it: 50,
            max_funcs: -1,
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────────

/// Results from a nonlinear static solve R(u) = 0.
#[derive(Debug)]
pub struct NonlinearStaticResult {
    /// Solution displacement vector (length = n_dof).
    pub displacements: Vec<f64>,
    /// Number of SNES (Newton) iterations to convergence.
    pub iterations: i32,
    /// Final residual norm ‖R(u)‖.
    pub residual_norm: f64,
    /// SNES converged reason code (positive = converged, negative = diverged).
    pub converged_reason: i32,
}

// ── Application context ───────────────────────────────────────────────────────

/// Data passed to the PETSc callbacks via `SNESSetApplicationContext`.
///
/// We store raw pointers to the Rust data — the data is borrowed for the
/// entire duration of `nonlinear_static_solve`, so the pointers are valid.
struct SnesCtx<'a> {
    assembler: &'a MeshAssembler,
    f_ext: &'a [f64],
    dirichlet_dofs: &'a [usize],
}

// ── Callbacks ─────────────────────────────────────────────────────────────────

/// PETSc residual callback: `R(u) = F_int(u) - F_ext`.
///
/// PETSc passes `x` (current iterate) and writes into `f` (residual).
/// Dirichlet DOFs are zeroed so SNES treats them as already satisfied.
///
/// # Safety
/// Called exclusively by PETSc during `SNESSolve`. The `ctx` pointer is
/// the `SnesCtx` we installed via `SNESSetApplicationContext`.
unsafe extern "C" fn form_residual(
    _snes: ffi::SNES,
    x: ffi::Vec,
    f: ffi::Vec,
    ctx: *mut c_void,
) -> ffi::PetscErrorCode {
    // ctx is passed directly by PETSc — it's the pointer we gave SNESSetFunction
    let ctx = &*(ctx as *const SnesCtx);

    // ── Extract current u from PETSc Vec (read-only) ─────────────────────
    let n = ctx.f_ext.len();
    let mut x_ptr: *const f64 = std::ptr::null();
    let ierr = ffi::VecGetArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }
    let u_slice = std::slice::from_raw_parts(x_ptr, n);

    // ── F_int(u) — Green-Lagrange internal forces ─────────────────────────
    let f_int = ctx.assembler.assemble_fint(u_slice, true);

    let ierr = ffi::VecRestoreArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }

    // ── R = F_int - F_ext ─────────────────────────────────────────────────
    let mut f_ptr: *mut f64 = std::ptr::null_mut();
    let ierr = ffi::VecGetArray(f, &mut f_ptr);
    if ierr != 0 {
        return ierr;
    }
    let r_slice = std::slice::from_raw_parts_mut(f_ptr, n);
    for i in 0..n {
        r_slice[i] = f_int[i] - ctx.f_ext[i];
    }

    // ── Enforce Dirichlet BCs: R[i] = 0 ─────────────────────────────────
    for &dof in ctx.dirichlet_dofs {
        r_slice[dof] = 0.0;
    }

    let ierr = ffi::VecRestoreArray(f, &mut f_ptr);
    if ierr != 0 {
        return ierr;
    }

    0 // PETSC_SUCCESS
}

/// PETSc Jacobian callback: assembles the tangent stiffness `K_T(u)`.
///
/// PETSc passes `x` (current iterate) and expects `jac` / `jpre` to be
/// filled with K_T. Dirichlet rows: diagonal = 1, off-diagonal = 0.
///
/// # Safety
/// Called exclusively by PETSc during `SNESSolve`.
unsafe extern "C" fn form_jacobian(
    _snes: ffi::SNES,
    x: ffi::Vec,
    jac: ffi::Mat,
    _jpre: ffi::Mat,
    ctx: *mut c_void,
) -> ffi::PetscErrorCode {
    // ctx is passed directly by PETSc
    let ctx = &*(ctx as *const SnesCtx);

    // ── Extract current u (read-only — x is locked by PETSc) ─────────────
    let n = ctx.f_ext.len();
    let mut x_ptr: *const f64 = std::ptr::null();
    let ierr = ffi::VecGetArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }
    let u_slice = std::slice::from_raw_parts(x_ptr, n);

    // ── K_T(u) as COO triplets ────────────────────────────────────────────
    let (rows, cols, vals) = ctx.assembler.assemble_kt(u_slice);

    let ierr = ffi::VecRestoreArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }

    // ── Convert i64 → i32 (PETSc uses 32-bit indices here) ───────────────
    let rows_i32: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
    let cols_i32: Vec<i32> = cols.iter().map(|&c| c as i32).collect();

    // ── Fill jac via COO API ──────────────────────────────────────────────
    let nnz = rows_i32.len() as i32;
    let ierr = ffi::MatSetPreallocationCOO(jac, nnz, rows_i32.as_ptr(), cols_i32.as_ptr());
    if ierr != 0 {
        return ierr;
    }
    let ierr = ffi::MatSetValuesCOO(jac, vals.as_ptr(), ffi::INSERT_VALUES);
    if ierr != 0 {
        return ierr;
    }

    let ierr = ffi::MatAssemblyBegin(jac, ffi::MAT_FINAL_ASSEMBLY);
    if ierr != 0 {
        return ierr;
    }
    let ierr = ffi::MatAssemblyEnd(jac, ffi::MAT_FINAL_ASSEMBLY);
    if ierr != 0 {
        return ierr;
    }

    // ── Enforce Dirichlet BCs: identity rows ─────────────────────────────
    // Set diagonal = 1, zero off-diagonal via MatZeroRowsColumns.
    // We use the simpler MatSetValue approach: zero the row/col, then set 1
    // on diagonal. This keeps K_T symmetric.
    let dirichlet_i32: Vec<i32> = ctx.dirichlet_dofs.iter().map(|&d| d as i32).collect();
    if !dirichlet_i32.is_empty() {
        let ierr = ffi::MatZeroRowsColumns(
            jac,
            dirichlet_i32.len() as i32,
            dirichlet_i32.as_ptr(),
            1.0, // diagonal value
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if ierr != 0 {
            return ierr;
        }
    }

    0 // PETSC_SUCCESS
}

// ── Solver ────────────────────────────────────────────────────────────────────

/// Solve the nonlinear static FEM system R(u) = F_int(u) - F_ext = 0.
///
/// # Arguments
/// * `assembler`      – assembled mesh with materials (provides K_T and F_int)
/// * `f_ext`          – external load vector (constant, length = n_dof)
/// * `dirichlet_dofs` – indices of constrained DOFs (zero displacement)
/// * `config`         – solver tolerances and iteration limits
///
/// # Returns
/// `NonlinearStaticResult` with the displacement vector and solver diagnostics.
///
/// # Notes
/// * Default SNES type is `NEWTONLS`. Override at runtime with
///   `-snes_type newtonal` for arc-length continuation.
/// * Uses PETSC_COMM_SELF — single-process sequential solve.
pub fn nonlinear_static_solve(
    assembler: &MeshAssembler,
    f_ext: &[f64],
    dirichlet_dofs: &[usize],
    config: &NonlinearConfig,
) -> Result<NonlinearStaticResult, PetscError> {
    ensure_initialized()?;

    let n_dof = f_ext.len();

    // ── Application context (lives on the stack, borrowed for the solve) ──
    let ctx = SnesCtx {
        assembler,
        f_ext,
        dirichlet_dofs,
    };
    // We need a raw pointer to ctx. ctx is alive for the entire unsafe block.
    let ctx_ptr = &ctx as *const SnesCtx as *mut c_void;

    unsafe {
        let comm = ffi::petsc_comm_self();

        // ── Initial Jacobian matrix (structural zeros from K_T at u=0) ────
        let u_zero = vec![0.0f64; n_dof];
        let (rows0, cols0, vals0) = assembler.assemble_kt(&u_zero);
        let rows0_i32: Vec<i32> = rows0.iter().map(|&r| r as i32).collect();
        let cols0_i32: Vec<i32> = cols0.iter().map(|&c| c as i32).collect();
        let jac = assemble_seq_aij(&rows0_i32, &cols0_i32, &vals0, n_dof)?;

        // ── Residual scratch Vec ──────────────────────────────────────────
        let r_vec = create_vec(n_dof)?;

        // ── Solution Vec (initial guess = 0) ─────────────────────────────
        let x_vec = create_vec(n_dof)?;

        // ── Create SNES ───────────────────────────────────────────────────
        let mut snes: ffi::SNES = std::ptr::null_mut();
        check(ffi::SNESCreate(comm, &mut snes), "SNESCreate")?;

        // Store our context so callbacks can recover it
        check(
            ffi::SNESSetApplicationContext(snes, ctx_ptr),
            "SNESSetApplicationContext",
        )?;

        // ── Register callbacks ────────────────────────────────────────────
        check(
            ffi::SNESSetFunction(snes, r_vec.as_raw(), form_residual, ctx_ptr),
            "SNESSetFunction",
        )?;
        check(
            ffi::SNESSetJacobian(
                snes,
                jac.as_raw(),
                jac.as_raw(),
                form_jacobian,
                ctx_ptr,
            ),
            "SNESSetJacobian",
        )?;

        // ── Tolerances ────────────────────────────────────────────────────
        check(
            ffi::SNESSetTolerances(
                snes,
                config.atol,
                config.rtol,
                config.stol,
                config.max_it,
                config.max_funcs,
            ),
            "SNESSetTolerances",
        )?;

        // ── Inner KSP: CG + ICC (same as linear static solver) ───────────
        let mut ksp: ffi::KSP = std::ptr::null_mut();
        check(ffi::SNESGetKSP(snes, &mut ksp), "SNESGetKSP")?;
        check(ffi::KSPSetType(ksp, KSPCG.as_ptr()), "KSPSetType(cg)")?;
        let mut pc: ffi::PC = std::ptr::null_mut();
        check(ffi::KSPGetPC(ksp, &mut pc), "KSPGetPC")?;
        check(ffi::PCSetType(pc, PCICC.as_ptr()), "PCSetType(icc)")?;
        check(
            ffi::KSPSetTolerances(ksp, 1e-8, 1e-12, PETSC_INFINITY, 1000),
            "KSPSetTolerances",
        )?;

        // ── Allow runtime overrides ───────────────────────────────────────
        // e.g. -snes_type newtonal  → arc-length
        //      -snes_monitor        → print residual each iteration
        check(ffi::SNESSetFromOptions(snes), "SNESSetFromOptions")?;

        // ── Solve R(u) = 0 ────────────────────────────────────────────────
        // b = null (no RHS shift — R already includes -F_ext in the callback)
        check(
            ffi::SNESSolve(snes, std::ptr::null_mut(), x_vec.as_raw()),
            "SNESSolve",
        )?;

        // ── Diagnostics ───────────────────────────────────────────────────
        let mut reason: i32 = 0;
        check(
            ffi::SNESGetConvergedReason(snes, &mut reason),
            "SNESGetConvergedReason",
        )?;

        let mut its: i32 = 0;
        check(
            ffi::SNESGetIterationNumber(snes, &mut its),
            "SNESGetIterationNumber",
        )?;

        let mut fnorm: f64 = 0.0;
        check(
            ffi::SNESGetFunctionNorm(snes, &mut fnorm),
            "SNESGetFunctionNorm",
        )?;

        // ── Cleanup ───────────────────────────────────────────────────────
        check(ffi::SNESDestroy(&mut snes), "SNESDestroy")?;

        // ── Extract solution ──────────────────────────────────────────────
        let displacements = x_vec.to_vec()?;

        Ok(NonlinearStaticResult {
            displacements,
            iterations: its,
            residual_norm: fnorm,
            converged_reason: reason,
        })
    }
}

// ── Public COO-based entry point (for PyO3 binding) ───────────────────────────

/// Convenience wrapper that accepts COO triplets directly (bypasses
/// `MeshAssembler`) for callers that already have the tangent stiffness.
///
/// This is the version exposed to Python via PyO3.
/// For the full assembler-based path, use `nonlinear_static_solve` directly.
pub fn nonlinear_static_solve_coo(
    assembler: &MeshAssembler,
    f_ext_slice: &[f64],
    dirichlet_dofs: &[usize],
    config: NonlinearConfig,
) -> Result<NonlinearStaticResult, PetscError> {
    nonlinear_static_solve(assembler, f_ext_slice, dirichlet_dofs, &config)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use aeroelast_core::assembly::{ElemType, MaterialSpec, MeshAssembler, MeshTopology};

    /// Helper: single MITC3 triangle mesh (patch test).
    ///
    /// Three nodes at (0,0,0), (1,0,0), (0,1,0) — one MITC3 element.
    /// 3 nodes × 6 DOFs = 18 DOFs total.
    fn make_single_triangle_assembler() -> MeshAssembler {
        let node_coords = vec![
            0.0, 0.0, 0.0, // node 0
            1.0, 0.0, 0.0, // node 1
            0.0, 1.0, 0.0, // node 2
        ];
        let connectivity = vec![vec![0usize, 1, 2]];
        let elem_types = vec![ElemType::Mitc3];

        let topology = MeshTopology::new(node_coords, connectivity, elem_types);
        let materials = vec![MaterialSpec::Isotropic {
            e: 200e9,
            nu: 0.3,
            rho: 7800.0,
            thickness: 0.01,
            shear_correction: 5.0 / 6.0,
            drilling_scale: 1.0,
        }];

        MeshAssembler::new(topology, materials)
    }

    /// Patch test: all DOFs clamped, zero load → zero displacement,
    /// SNES converges in 1 step (linear problem with trivial solution u=0).
    #[test]
    fn test_all_clamped_zero_load() {
        let assembler = make_single_triangle_assembler();
        let n_dof = assembler.dofs_count;
        let f_ext = vec![0.0f64; n_dof];
        let dirichlet_dofs: Vec<usize> = (0..n_dof).collect(); // all clamped

        let config = NonlinearConfig::default();
        let result = nonlinear_static_solve(&assembler, &f_ext, &dirichlet_dofs, &config)
            .expect("nonlinear solve failed");

        assert!(
            result.converged_reason > 0,
            "SNES diverged: reason={}, iters={}, fnorm={}",
            result.converged_reason,
            result.iterations,
            result.residual_norm
        );
        for (i, &u) in result.displacements.iter().enumerate() {
            assert!(
                u.abs() < 1e-10,
                "DOF {i}: expected 0.0, got {u} (all clamped, zero load)"
            );
        }
    }
}
