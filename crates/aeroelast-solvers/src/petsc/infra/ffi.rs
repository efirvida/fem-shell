/// Raw FFI declarations for PETSc 3.24.x and SLEPc.
///
/// PETSc is built with OpenMPI 4.1.4 — MPI_Comm is `*mut c_void` (opaque pointer).
/// PETSc integer size: 32-bit (PETSC_USE_64BIT_INDICES is NOT defined in petscconf.h).
/// PETSc scalar: f64 (real, non-complex build).
#[allow(non_camel_case_types, dead_code, non_snake_case)]
use std::ffi::c_void;

// ── PETSc type aliases ────────────────────────────────────────────────────────

/// PETSc integer (32-bit — this installation does NOT use 64-bit indices).
pub type PetscInt = i32;

/// PETSc scalar (real, double precision).
pub type PetscScalar = f64;

/// PETSc real.
pub type PetscReal = f64;

/// PETSc error code (0 = success).
pub type PetscErrorCode = i32;

/// Opaque MPI communicator (OpenMPI: pointer to struct).
#[allow(non_camel_case_types)]
pub type MPI_Comm = *mut c_void;

/// Opaque PETSc matrix handle.
pub type Mat = *mut c_void;

/// Opaque PETSc vector handle.
pub type Vec = *mut c_void;

/// Opaque SLEPc EPS handle.
pub type EPS = *mut c_void;

// ── PETSc constants ───────────────────────────────────────────────────────────

/// MAT_FINAL_ASSEMBLY = 0 (PETSc MatAssemblyType: completes the matrix assembly)
pub const MAT_FINAL_ASSEMBLY: i32 = 0;

/// INSERT_VALUES = 1 — overwrite existing entries.
pub const INSERT_VALUES: i32 = 1;

/// ADD_VALUES = 2 — accumulate into existing entries (required for FEM assembly).
pub const ADD_VALUES: i32 = 2;

/// EPS_GHEP = 2 (generalized hermitian eigenvalue problem).
/// SLEPc EPSProblemType enum: HEP=1, GHEP=2, NHEP=3, GNHEP=4.
pub const EPS_GHEP: i32 = 2;

/// EPS_TARGET_MAGNITUDE = 7 — find eigenvalues closest to target in magnitude.
/// SLEPc EPSWhich enum: LARGEST_MAGNITUDE=1, SMALLEST_MAGNITUDE=2, ..., TARGET_MAGNITUDE=7.
pub const EPS_TARGET_MAGNITUDE: i32 = 7;

/// PETSC_INFINITY — used for dtol to disable divergence tolerance.
pub const PETSC_INFINITY: PetscReal = 1.0e300;

/// PETSC_DEFAULT = -2
pub const PETSC_DEFAULT: PetscInt = -2;

/// PETSC_DECIDE = -1
pub const PETSC_DECIDE: PetscInt = -1;

/// MAT_SYMMETRIC = 1 — flag matrix as symmetric (enables Cholesky path in PC).
pub const MAT_SYMMETRIC: i32 = 1;

/// PETSC_TRUE = 1
pub const PETSC_TRUE: i32 = 1;

// ── MPI communicator accessors ────────────────────────────────────────────────
// In OpenMPI, MPI_COMM_WORLD and MPI_COMM_SELF are extern globals, not integer
// constants. We expose them through extern statics.

extern "C" {
    /// OpenMPI global: MPI_COMM_SELF
    pub static mut ompi_mpi_comm_self: c_void;
    /// OpenMPI global: MPI_COMM_WORLD
    pub static mut ompi_mpi_comm_world: c_void;
}

/// Returns the MPI_COMM_SELF communicator.
///
/// # Safety
/// Must be called after MPI/PETSc has been initialized.
#[inline]
pub unsafe fn petsc_comm_self() -> MPI_Comm {
    &raw mut ompi_mpi_comm_self
}

/// Returns the MPI_COMM_WORLD communicator.
///
/// # Safety
/// Must be called after MPI/PETSc has been initialized.
#[inline]
pub unsafe fn petsc_comm_world() -> MPI_Comm {
    &raw mut ompi_mpi_comm_world
}

// ── PETSc extern declarations ─────────────────────────────────────────────────

#[link(name = "petsc")]
extern "C" {
    // Init
    pub fn PetscInitializeNoArguments() -> PetscErrorCode;
    pub fn PetscInitialized(flag: *mut i32) -> PetscErrorCode;

    // Mat lifecycle
    pub fn MatCreate(comm: MPI_Comm, mat: *mut Mat) -> PetscErrorCode;
    pub fn MatSetType(mat: Mat, type_: *const i8) -> PetscErrorCode;
    pub fn MatSetSizes(
        mat: Mat,
        local_m: PetscInt,
        local_n: PetscInt,
        global_m: PetscInt,
        global_n: PetscInt,
    ) -> PetscErrorCode;

    // Preallocation
    pub fn MatSeqAIJSetPreallocation(
        mat: Mat,
        nz: PetscInt,
        nnz: *const PetscInt,
    ) -> PetscErrorCode;
    pub fn MatMPIAIJSetPreallocation(
        mat: Mat,
        d_nz: PetscInt,
        d_nnz: *const PetscInt,
        o_nz: PetscInt,
        o_nnz: *const PetscInt,
    ) -> PetscErrorCode;

    // COO-based batch insert (PETSc 3.20+)
    pub fn MatSetPreallocationCOO(
        mat: Mat,
        ncoo: PetscInt,
        rows: *const PetscInt,
        cols: *const PetscInt,
    ) -> PetscErrorCode;
    pub fn MatSetValuesCOO(mat: Mat, vals: *const PetscScalar, mode: i32) -> PetscErrorCode;
    pub fn MatSetValues(
        mat: Mat,
        m: PetscInt,
        idxm: *const PetscInt,
        n: PetscInt,
        idxn: *const PetscInt,
        v: *const PetscScalar,
        addv: i32,
    ) -> PetscErrorCode;

    // Assembly
    pub fn MatAssemblyBegin(mat: Mat, type_: i32) -> PetscErrorCode;
    pub fn MatAssemblyEnd(mat: Mat, type_: i32) -> PetscErrorCode;
    pub fn MatZeroEntries(mat: Mat) -> PetscErrorCode;

    // Options
    /// MatSetOption(mat, opt, flag) — set matrix property flags.
    /// opt=MAT_SYMMETRIC=18, flag=PETSC_TRUE=1.
    pub fn MatSetOption(mat: Mat, opt: i32, flag: i32) -> PetscErrorCode;

    // Destroy
    pub fn MatDestroy(mat: *mut Mat) -> PetscErrorCode;

    // Vec lifecycle
    pub fn VecCreate(comm: MPI_Comm, vec: *mut Vec) -> PetscErrorCode;
    pub fn VecSetSizes(vec: Vec, local_n: PetscInt, global_n: PetscInt) -> PetscErrorCode;
    pub fn VecSetFromOptions(vec: Vec) -> PetscErrorCode;
    pub fn VecGetArray(vec: Vec, array: *mut *mut PetscScalar) -> PetscErrorCode;
    pub fn VecRestoreArray(vec: Vec, array: *mut *mut PetscScalar) -> PetscErrorCode;
    pub fn VecDestroy(vec: *mut Vec) -> PetscErrorCode;
}

// ── SLEPc extern declarations ─────────────────────────────────────────────────

/// Opaque SLEPc ST (Spectral Transform) handle.
pub type ST = *mut c_void;

#[link(name = "slepc")]
extern "C" {
    pub fn SlepcInitialized(flag: *mut i32) -> PetscErrorCode;
    pub fn SlepcInitializeNoArguments() -> PetscErrorCode;

    pub fn EPSCreate(comm: MPI_Comm, eps: *mut EPS) -> PetscErrorCode;
    pub fn EPSSetOperators(eps: EPS, A: Mat, B: Mat) -> PetscErrorCode;
    pub fn EPSSetProblemType(eps: EPS, problem_type: i32) -> PetscErrorCode;
    pub fn EPSSetDimensions(
        eps: EPS,
        nev: PetscInt,
        ncv: PetscInt,
        mpd: PetscInt,
    ) -> PetscErrorCode;
    pub fn EPSSetWhichEigenpairs(eps: EPS, which: i32) -> PetscErrorCode;
    pub fn EPSSetTarget(eps: EPS, target: PetscReal) -> PetscErrorCode;
    pub fn EPSGetST(eps: EPS, st: *mut ST) -> PetscErrorCode;
    pub fn STSetType(st: ST, type_: *const i8) -> PetscErrorCode;
    pub fn STSetShift(st: ST, shift: PetscReal) -> PetscErrorCode;
    pub fn STGetKSP(st: ST, ksp: *mut KSP) -> PetscErrorCode;
    pub fn EPSSetFromOptions(eps: EPS) -> PetscErrorCode;
    pub fn EPSSetUp(eps: EPS) -> PetscErrorCode;
    pub fn EPSSolve(eps: EPS) -> PetscErrorCode;
    pub fn EPSGetConverged(eps: EPS, nconv: *mut PetscInt) -> PetscErrorCode;
    pub fn EPSGetEigenpair(
        eps: EPS,
        i: PetscInt,
        kr: *mut PetscScalar,
        ki: *mut PetscScalar,
        Vr: Vec,
        Vi: Vec,
    ) -> PetscErrorCode;
    pub fn EPSDestroy(eps: *mut EPS) -> PetscErrorCode;
    pub fn EPSSetTolerances(eps: EPS, tol: PetscReal, max_it: PetscInt) -> PetscErrorCode;
}

// ── PETSc KSP/PC ─────────────────────────────────────────────────────────────

/// Opaque KSP handle.
pub type KSP = *mut c_void;

/// Opaque PC handle.
pub type PC = *mut c_void;

/// Opaque PETSc SNES line search handle.
pub type SNESLineSearch = *mut c_void;

#[link(name = "petsc")]
extern "C" {
    pub fn KSPCreate(comm: MPI_Comm, ksp: *mut KSP) -> PetscErrorCode;
    pub fn KSPSetType(ksp: KSP, type_: *const i8) -> PetscErrorCode;
    pub fn KSPSetOperators(ksp: KSP, Amat: Mat, Pmat: Mat) -> PetscErrorCode;
    pub fn KSPSetTolerances(
        ksp: KSP,
        rtol: PetscReal,
        atol: PetscReal,
        dtol: PetscReal,
        max_it: PetscInt,
    ) -> PetscErrorCode;
    pub fn KSPSetFromOptions(ksp: KSP) -> PetscErrorCode;
    pub fn KSPSolve(ksp: KSP, b: Vec, x: Vec) -> PetscErrorCode;
    pub fn KSPGetConvergedReason(ksp: KSP, reason: *mut i32) -> PetscErrorCode;
    pub fn KSPGetIterationNumber(ksp: KSP, its: *mut PetscInt) -> PetscErrorCode;
    pub fn KSPGetResidualNorm(ksp: KSP, rnorm: *mut PetscReal) -> PetscErrorCode;
    pub fn KSPDestroy(ksp: *mut KSP) -> PetscErrorCode;
    pub fn KSPGetPC(ksp: KSP, pc: *mut PC) -> PetscErrorCode;
    pub fn KSPGMRESSetRestart(ksp: KSP, restart: PetscInt) -> PetscErrorCode;
    pub fn PCSetType(pc: PC, type_: *const i8) -> PetscErrorCode;
    pub fn PCFactorSetMatSolverType(pc: PC, stype: *const i8) -> PetscErrorCode;
    pub fn PCFactorSetLevels(pc: PC, levels: PetscInt) -> PetscErrorCode;
    /// Shift type for incomplete factorization.
    /// MatFactorShiftType: NONE=0, NONZERO=1, POSITIVE_DEFINITE=2, INBLOCKS=3.
    pub fn PCFactorSetShiftType(pc: PC, ftype: i32) -> PetscErrorCode;
    pub fn MatGetSize(mat: Mat, m: *mut PetscInt, n: *mut PetscInt) -> PetscErrorCode;
    pub fn MatMult(mat: Mat, x: Vec, y: Vec) -> PetscErrorCode;
    pub fn VecDuplicate(v: Vec, newv: *mut Vec) -> PetscErrorCode;
    pub fn VecSet(x: Vec, alpha: PetscScalar) -> PetscErrorCode;
    pub fn VecAXPY(y: Vec, alpha: PetscScalar, x: Vec) -> PetscErrorCode;
    pub fn VecSetValues(
        x: Vec,
        ni: PetscInt,
        ix: *const PetscInt,
        y: *const PetscScalar,
        iora: i32,
    ) -> PetscErrorCode;
    pub fn VecAssemblyBegin(vec: Vec) -> PetscErrorCode;
    pub fn VecAssemblyEnd(vec: Vec) -> PetscErrorCode;
    /// Read-only array access — use in SNES residual/Jacobian callbacks.
    pub fn VecGetArrayRead(vec: Vec, array: *mut *const PetscScalar) -> PetscErrorCode;
    pub fn VecRestoreArrayRead(vec: Vec, array: *mut *const PetscScalar) -> PetscErrorCode;
}

// ── PETSc SNES ────────────────────────────────────────────────────────────────

/// Opaque SNES (nonlinear solver) handle.
pub type SNES = *mut c_void;

/// Signature of the residual callback `FormResidual(snes, x, f, ctx)`.
pub type SNESFunction = unsafe extern "C" fn(
    snes: SNES,
    x: Vec,
    f: Vec,
    ctx: *mut c_void,
) -> PetscErrorCode;

/// Signature of the Jacobian callback `FormJacobian(snes, x, J, P, ctx)`.
pub type SNESJacobianFunction = unsafe extern "C" fn(
    snes: SNES,
    x: Vec,
    jac: Mat,
    jpre: Mat,
    ctx: *mut c_void,
) -> PetscErrorCode;

#[link(name = "petsc")]
extern "C" {
    pub fn SNESCreate(comm: MPI_Comm, snes: *mut SNES) -> PetscErrorCode;
    pub fn SNESSetFunction(
        snes: SNES,
        r: Vec,
        f: SNESFunction,
        ctx: *mut c_void,
    ) -> PetscErrorCode;
    pub fn SNESSetJacobian(
        snes: SNES,
        Amat: Mat,
        Pmat: Mat,
        j: SNESJacobianFunction,
        ctx: *mut c_void,
    ) -> PetscErrorCode;
    pub fn SNESSetTolerances(
        snes: SNES,
        atol: PetscReal,
        rtol: PetscReal,
        stol: PetscReal,
        maxit: PetscInt,
        maxf: PetscInt,
    ) -> PetscErrorCode;
    pub fn SNESSetFromOptions(snes: SNES) -> PetscErrorCode;
    pub fn SNESSolve(snes: SNES, b: Vec, x: Vec) -> PetscErrorCode;
    pub fn SNESGetConvergedReason(snes: SNES, reason: *mut i32) -> PetscErrorCode;
    pub fn SNESGetIterationNumber(snes: SNES, its: *mut PetscInt) -> PetscErrorCode;
    pub fn SNESGetFunctionNorm(snes: SNES, fnorm: *mut PetscReal) -> PetscErrorCode;
    pub fn SNESGetLinearSolveIterations(snes: SNES, lits: *mut PetscInt) -> PetscErrorCode;
    pub fn SNESGetNumberFunctionEvals(snes: SNES, nfuncs: *mut PetscInt) -> PetscErrorCode;
    pub fn SNESSetApplicationContext(snes: SNES, usrP: *mut c_void) -> PetscErrorCode;
    pub fn SNESGetApplicationContext(snes: SNES, usrP: *mut *mut c_void) -> PetscErrorCode;
    pub fn SNESDestroy(snes: *mut SNES) -> PetscErrorCode;
    pub fn SNESGetKSP(snes: SNES, ksp: *mut KSP) -> PetscErrorCode;
    pub fn SNESGetLineSearch(snes: SNES, linesearch: *mut SNESLineSearch) -> PetscErrorCode;
    pub fn SNESLineSearchSetType(
        linesearch: SNESLineSearch,
        type_: *const i8,
    ) -> PetscErrorCode;
    pub fn SNESLineSearchSetDamping(
        linesearch: SNESLineSearch,
        damping: PetscReal,
    ) -> PetscErrorCode;
}

#[link(name = "petsc")]
extern "C" {
    /// Zero rows AND columns of `mat` corresponding to `rows[]`, setting the
    /// diagonal to `diag`. Optionally shifts RHS `b` and solution `x` (pass
    /// null to skip).
    pub fn MatZeroRowsColumns(
        mat: Mat,
        numRows: PetscInt,
        rows: *const PetscInt,
        diag: PetscScalar,
        x: Vec,
        b: Vec,
    ) -> PetscErrorCode;
}
