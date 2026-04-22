/// FSI (Fluid-Structure Interaction) coupling module.
///
/// Provides the building blocks for preCICE-based FSI coupling:
/// - `checkpoint` — state snapshot for implicit coupling iteration rollback
/// - `force_utils` — force ramping and capping before passing to the stepper
/// - `linear_elastic` — full `LinearElasticFsiSolver` coupling loop

pub mod checkpoint;
pub mod force_utils;
pub mod linear_elastic;

pub use linear_elastic::{FsiConfig, FsiError, FsiResult, LinearElasticFsiSolver};
