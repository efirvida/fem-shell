/// Structural elasticity solvers.
///
/// Each module implements one solver for a class of structural FEM problems:
///
/// | Module              | Method              | Problem                          |
/// |---------------------|---------------------|----------------------------------|
/// | `static_linear`     | KSP (CG + ICC)      | K·u = F  (linear static)         |
/// | `static_nonlinear`  | SNES (Newton-LS)    | R(u) = 0 (geometric/material NL) |
/// | `dynamic_newmark`   | Newmark-β           | Mü + Ku = F(t) (linear dynamic)  |
/// | `modal`             | SLEPc EPS           | K·x = λM·x (modal analysis)      |
///
/// All solvers are single-process (PETSC_COMM_SELF) and accept assembled
/// COO triplets or `MeshAssembler` references as input.
pub mod static_linear;
pub mod static_nonlinear;
pub mod dynamic_newmark;
pub mod modal;