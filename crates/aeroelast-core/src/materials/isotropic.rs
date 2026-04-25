use nalgebra::{Matrix2, Matrix3};

use super::{Material, ShellConstitutive};

/// Isotropic linear elastic material.
#[derive(Debug, Clone, Copy)]
pub struct IsotropicMaterial {
    pub e: f64,   // Young's modulus
    pub nu: f64,  // Poisson's ratio
    pub rho: f64, // Density
}

impl IsotropicMaterial {
    pub fn new(e: f64, nu: f64, rho: f64) -> Self {
        Self { e, nu, rho }
    }
}

impl Material for IsotropicMaterial {
    fn constitutive(&self, thickness: f64, shear_correction: f64) -> ShellConstitutive {
        let e = self.e;
        let nu = self.nu;
        let h = thickness;
        let factor_m = e * h / (1.0 - nu * nu);
        let factor_b = e * h * h * h / (12.0 * (1.0 - nu * nu));
        let g = e / (2.0 * (1.0 + nu));
        let factor_raw = e / (1.0 - nu * nu);
        let half_1_minus_nu = (1.0 - nu) / 2.0;

        let cm = Matrix3::new(
            factor_m,
            factor_m * nu,
            0.0,
            factor_m * nu,
            factor_m,
            0.0,
            0.0,
            0.0,
            factor_m * half_1_minus_nu,
        );

        let cb = Matrix3::new(
            factor_b,
            factor_b * nu,
            0.0,
            factor_b * nu,
            factor_b,
            0.0,
            0.0,
            0.0,
            factor_b * half_1_minus_nu,
        );

        let cs_val = shear_correction * g * h;
        let cs = Matrix2::new(cs_val, 0.0, 0.0, cs_val);

        let cm_raw = Matrix3::new(
            factor_raw,
            factor_raw * nu,
            0.0,
            factor_raw * nu,
            factor_raw,
            0.0,
            0.0,
            0.0,
            factor_raw * half_1_minus_nu,
        );

        ShellConstitutive {
            cm,
            cb_coupling: Matrix3::zeros(), // Zero for isotropic (no membrane-bending coupling)
            cb,
            cs,
            cm_raw,
        }
    }

    fn density(&self) -> f64 {
        self.rho
    }
}
