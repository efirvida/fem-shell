/// FSI (Fluid-Structure Interaction) coupling module.
///
/// Provides the building blocks for preCICE-based FSI coupling:
/// - `checkpoint` — state snapshot for implicit coupling iteration rollback
/// - `force_utils` — force ramping and capping before passing to the stepper
/// - `setup`       — BC reduction, mass lumping, Rayleigh C and interface DOF remapping
/// - `linear_elastic` — full `LinearElasticFsiSolver` coupling loop
/// - `stress_stiffened` — `StressStiffenedFsiSolver` with per-step K_G update

pub mod checkpoint;
pub mod force_utils;
pub mod linear_elastic;
pub mod setup;
pub mod stress_stiffened;

pub use linear_elastic::{FsiConfig, FsiError, FsiInitialState, FsiResult, LinearElasticFsiSolver};
pub use stress_stiffened::StressStiffenedFsiSolver;
