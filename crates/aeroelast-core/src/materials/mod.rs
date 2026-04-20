pub mod composite;
pub mod isotropic;
pub mod orthotropic;
pub mod laminate;

use nalgebra::{Matrix2, Matrix3};

/// Constitutive matrices for a shell element.
///
/// All matrices are in local element coordinates.
#[derive(Clone)]
pub struct ShellConstitutive {
    /// Membrane stiffness (3×3): force-strain [N·h / (1-ν²)] form
    pub cm: Matrix3<f64>,
    /// Bending stiffness (3×3): moment-curvature [N·h³/12 / (1-ν²)] form
    pub cb: Matrix3<f64>,
    /// Transverse shear stiffness (2×2): force-shear strain [k·G·h] form
    pub cs: Matrix2<f64>,
    /// Raw membrane (stress-strain, no thickness): σ = C_raw · ε
    pub cm_raw: Matrix3<f64>,
}

/// Material trait for shell elements.
///
/// Implementors provide constitutive matrices and physical properties.
pub trait Material: Send + Sync {
    /// Compute all constitutive matrices for a given thickness and shear correction factor.
    fn constitutive(&self, thickness: f64, shear_correction: f64) -> ShellConstitutive;

    /// Density (kg/m³)
    fn density(&self) -> f64;
}
