/// `aeroelast-solvers`: PETSc/SLEPc-backed assembler and solvers.
///
/// # Module structure
///
/// ```text
/// aeroelast_solvers
/// └── petsc
///     ├── infra/         — PETSc/SLEPc infrastructure (no solver logic)
///     │   ├── ffi        — Raw C bindings for PETSc/SLEPc
///     │   ├── mat        — RAII PetscMat wrapper
///     │   └── vec        — RAII PetscVec wrapper
///     ├── assembler      — COO triplets → PETSc Mat (stiffness / mass)
///     ├── linear         — Linear static solver via KSP CG + HYPRE BoomerAMG
///     └── modal          — Modal analysis via SLEPc EPS (K·x = λ·M·x)
/// ```
///
/// Naming convention: each solver module is named after the physics it solves.
/// Future additions: `linear.rs` (static), `nonlinear.rs` (geometric/material NL).
///
/// Additional solver backends (e.g. `lapack`, `eigen`) will be added at the
/// same level as `petsc` if needed in the future.
///
/// # Quick start
///
/// ```rust,no_run
/// use aeroelast_solvers::petsc::assembler::assemble_from_coo;
/// use aeroelast_solvers::petsc::modal::modal_solve;
/// ```
pub mod petsc {
    pub mod infra;
    pub mod assembler;
    pub mod linear;
    pub mod modal;
}

// Re-export the most commonly used types at crate root for ergonomics
pub use petsc::assembler::{assemble_from_coo, assemble_seq_aij, create_vec, ensure_initialized};
pub use petsc::infra::mat::{PetscError, PetscMat};
pub use petsc::infra::vec::PetscVec;
pub use petsc::linear::{linear_static_solve, LinearStaticResult};
pub use petsc::modal::{modal_solve, ModalResult};
