/// Modal solver using SLEPc EPS (Eigenvalue Problem Solver).
///
/// Solves the generalized Hermitian eigenvalue problem: K·x = λ·M·x
/// where K is stiffness and M is mass matrix (both assembled PETSc Mats).
use super::assembler::create_vec;
use super::ffi::{EPS_GHEP, EPS_TARGET_MAGNITUDE, PETSC_DEFAULT};
use super::mat::{check, PetscError, PetscMat};

// C-string constants for KSP/PC/ST types
const STSINVERT: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"sinvert\0") };
const KSPPREONLY: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"preonly\0") };
const PCLU: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"lu\0") };
const MATSOLVERPETSC: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"petsc\0") };

/// Results from a modal analysis.
#[derive(Debug)]
pub struct ModalResult {
    /// Eigenvalues λ = ω² (rad²/s²). Natural frequencies: ω = √λ, f = ω/(2π).
    pub eigenvalues: Vec<f64>,
    /// Mode shapes: eigenvectors[i] is the i-th mode shape (length = n_dof).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of converged modes (may be < n_modes requested).
    pub n_converged: usize,
}

/// Solve the generalized eigenvalue problem K·x = λ·M·x.
///
/// # Arguments
/// * `k`       – stiffness matrix (assembled PETSc Mat, square n_dof × n_dof)
/// * `m`       – mass matrix (assembled PETSc Mat, same size)
/// * `n_modes` – number of smallest eigenvalues to compute
///
/// # Returns
/// `ModalResult` with converged eigenvalues and eigenvectors.
///
/// # Notes
/// * Uses KRYLOVSCHUR solver (SLEPc default, efficient for large sparse problems).
/// * Problem type: EPS_GHEP (generalized hermitian — physically correct for FEM).
/// * SLEPc must be available and linked (via build.rs / pkg-config SLEPc).
pub fn modal_solve(
    k: &PetscMat,
    m: &PetscMat,
    n_modes: usize,
) -> Result<ModalResult, PetscError> {
    crate::petsc::assembler::ensure_initialized()?;

    unsafe {
        let comm = super::ffi::petsc_comm_self();
        let mut eps: super::ffi::EPS = std::ptr::null_mut();

        check(super::ffi::EPSCreate(comm, &mut eps), "EPSCreate")?;

        // Determine matrix size for dimension clamping
        let mut k_rows: i32 = 0;
        let mut k_cols: i32 = 0;
        petsc_mat_get_size(k.as_raw(), &mut k_rows, &mut k_cols)?;

        // Set K (operator A) and M (operator B) for K·x = λ·M·x
        check(
            super::ffi::EPSSetOperators(eps, k.as_raw(), m.as_raw()),
            "EPSSetOperators",
        )?;

        // Generalized Hermitian Eigenvalue Problem
        check(
            super::ffi::EPSSetProblemType(eps, EPS_GHEP),
            "EPSSetProblemType",
        )?;

        // Find smallest eigenvalues (closest to target 0)
        check(
            super::ffi::EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE),
            "EPSSetWhichEigenpairs",
        )?;
        check(super::ffi::EPSSetTarget(eps, 0.0), "EPSSetTarget")?;

        // Configure ST: SINVERT with shift=0, KSP=preonly, PC=LU
        // SINVERT enables shift-and-invert: finds eigenvalues near target=0 efficiently.
        // PC=LU: direct factorization of (K - σ·M).
        let mut st: super::ffi::ST = std::ptr::null_mut();
        check(super::ffi::EPSGetST(eps, &mut st), "EPSGetST")?;
        check(
            super::ffi::STSetType(st, STSINVERT.as_ptr()),
            "STSetType(sinvert)",
        )?;
        check(super::ffi::STSetShift(st, 0.0), "STSetShift")?;
        let mut ksp: super::ffi::KSP = std::ptr::null_mut();
        check(super::ffi::STGetKSP(st, &mut ksp), "STGetKSP")?;
        check(
            super::ffi::KSPSetType(ksp, KSPPREONLY.as_ptr()),
            "KSPSetType(preonly)",
        )?;
        let mut pc: super::ffi::PC = std::ptr::null_mut();
        check(super::ffi::KSPGetPC(ksp, &mut pc), "KSPGetPC")?;
        check(super::ffi::PCSetType(pc, PCLU.as_ptr()), "PCSetType(lu)")?;
        check(
            super::ffi::PCFactorSetMatSolverType(pc, MATSOLVERPETSC.as_ptr()),
            "PCFactorSetMatSolverType(petsc)",
        )?;

        // Request extra modes for better Krylov-Schur convergence (matches Python impl)
        let eff_n_modes = (n_modes + 5).min(k_rows as usize) as i32;
        check(
            super::ffi::EPSSetDimensions(eps, eff_n_modes, PETSC_DEFAULT, PETSC_DEFAULT),
            "EPSSetDimensions",
        )?;

        // Set convergence tolerances (matches Python: tol=1e-10, max_it=1000)
        check(
            super::ffi::EPSSetTolerances(eps, 1e-10, 1000),
            "EPSSetTolerances",
        )?;

        // Allow runtime override via -eps_* options
        check(super::ffi::EPSSetFromOptions(eps), "EPSSetFromOptions")?;

        // Setup and solve
        check(super::ffi::EPSSetUp(eps), "EPSSetUp")?;
        check(super::ffi::EPSSolve(eps), "EPSSolve")?;

        // Query converged count
        let mut nconv: i32 = 0;
        check(
            super::ffi::EPSGetConverged(eps, &mut nconv),
            "EPSGetConverged",
        )?;

        let n_converged = (nconv as usize).min(n_modes);

        let mut eigenvalues = Vec::with_capacity(n_converged);
        let mut eigenvectors = Vec::with_capacity(n_converged);

        // Determine n_dof from K (already queried above as k_rows)
        let n_dof = k_rows as usize;

        // Create scratch vectors for eigenvector extraction.
        // For GHEP with real arithmetic, eigenvalues are real and vi is not used.
        let vr = create_vec(n_dof)?;
        let vi = create_vec(n_dof)?;

        for i in 0..n_converged as i32 {
            let mut kr: f64 = 0.0;
            let mut ki: f64 = 0.0;

            check(
                super::ffi::EPSGetEigenpair(eps, i, &mut kr, &mut ki, vr.as_raw(), vi.as_raw()),
                "EPSGetEigenpair",
            )?;

            eigenvalues.push(kr);
            eigenvectors.push(vr.to_vec()?);
        }

        // Cleanup EPS
        check(super::ffi::EPSDestroy(&mut eps), "EPSDestroy")?;

        Ok(ModalResult {
            eigenvalues,
            eigenvectors,
            n_converged,
        })
    }
}

/// Query global row/col size of a PETSc Mat via MatGetSize.
unsafe fn petsc_mat_get_size(
    mat: super::ffi::Mat,
    m: *mut i32,
    n: *mut i32,
) -> Result<(), PetscError> {
    check(super::ffi::MatGetSize(mat, m, n), "MatGetSize")
}
