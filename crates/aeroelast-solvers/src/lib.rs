/// `aeroelast-solvers`: PETSc/SLEPc-backed assembler and solvers.
///
/// # Module structure
///
/// ```text
/// aeroelast_solvers
/// └── petsc
///     ├── infra/              — PETSc/SLEPc infrastructure (no solver logic)
///     │   ├── ffi             — Raw C bindings for PETSc/SLEPc
///     │   ├── mat             — RAII PetscMat wrapper
///     │   └── vec             — RAII PetscVec wrapper
///     ├── assembler           — COO triplets → PETSc Mat (stiffness / mass)
///     └── elasticity/         — Structural elasticity solvers
///         ├── static_linear   — KSP CG+ICC:   K·u = F
///         ├── static_nonlinear— SNES Newton:  R(u) = 0 (geometric NL)
///         ├── dynamic_newmark — Newmark-β:    Mü + Ku = F(t)
///         └── modal           — SLEPc EPS:    K·x = λM·x
/// ```
///
/// Additional domains (e.g. `aerodynamics/`, `aeroelasticity/`) will be added
/// at the same level as `elasticity/` when needed.
///
/// Additional solver backends (e.g. `lapack`, `eigen`) will be added at the
/// same level as `petsc/` if needed in the future.
///
/// # Quick start
///
/// ```rust,no_run
/// use aeroelast_solvers::petsc::assembler::assemble_from_coo;
/// use aeroelast_solvers::petsc::elasticity::modal::modal_solve;
/// ```
pub mod petsc {
    pub mod infra;
    pub mod assembler;
    pub mod elasticity;
    #[cfg(feature = "fsi")]
    pub mod fsi;
}

// Re-export the most commonly used types at crate root for ergonomics
pub use petsc::assembler::{assemble_from_coo, assemble_seq_aij, create_vec, ensure_initialized};
pub use petsc::infra::mat::{PetscError, PetscMat};
pub use petsc::infra::vec::PetscVec;
pub use petsc::elasticity::static_linear::{linear_static_solve, LinearStaticResult};
pub use petsc::elasticity::dynamic_newmark::{
    newmark_beta_solve, DynamicResult, NewmarkCheckpoint, NewmarkStepper, StepResult,
};
pub use petsc::elasticity::modal::{modal_solve, ModalResult};
pub use petsc::elasticity::static_nonlinear::{
    nonlinear_static_solve, NonlinearConfig, NonlinearStaticResult,
};
