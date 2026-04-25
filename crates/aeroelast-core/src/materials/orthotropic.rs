use nalgebra::{Matrix2, Matrix3};

use super::{Material, ShellConstitutive};

/// Orthotropic linear elastic material (plane-stress shell formulation).
///
/// Properties are given in the principal material coordinate system (1-2-3):
/// - 1 = fiber direction
/// - 2 = transverse in-plane direction
/// - 3 = through-thickness direction
#[derive(Debug, Clone, Copy)]
pub struct OrthotropicMaterial {
    pub e1: f64,   // Young's modulus in fiber direction [Pa]
    pub e2: f64,   // Young's modulus transverse to fibers [Pa]
    pub e3: f64,   // Young's modulus through thickness [Pa]
    pub g12: f64,  // In-plane shear modulus [Pa]
    pub g23: f64,  // Transverse shear modulus (2-3 plane) [Pa]
    pub g13: f64,  // Transverse shear modulus (1-3 plane) [Pa]
    pub nu12: f64, // Poisson's ratio ν12
    pub nu23: f64, // Poisson's ratio ν23
    pub nu31: f64, // Poisson's ratio ν31
    pub rho: f64,  // Density [kg/m³]
}

impl OrthotropicMaterial {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        e1: f64, e2: f64, e3: f64,
        g12: f64, g23: f64, g13: f64,
        nu12: f64, nu23: f64, nu31: f64,
        rho: f64,
    ) -> Self {
        Self { e1, e2, e3, g12, g23, g13, nu12, nu23, nu31, rho }
    }

    /// Compute the reduced stiffness matrix Q in principal material axes (plane stress).
    ///
    /// Plane-stress assumption (σ3 = 0) gives:
    /// Q11 = E1 / (1 - ν12·ν21)
    /// Q22 = E2 / (1 - ν12·ν21)
    /// Q12 = ν12·E2 / (1 - ν12·ν21)
    /// Q66 = G12
    pub fn compute_q(&self) -> Matrix3<f64> {
        let nu21 = self.nu12 * self.e2 / self.e1;
        let denom = 1.0 - self.nu12 * nu21;

        let q11 = self.e1 / denom;
        let q22 = self.e2 / denom;
        let q12 = self.nu12 * self.e2 / denom;
        let q66 = self.g12;

        Matrix3::new(
            q11, q12, 0.0,
            q12, q22, 0.0,
            0.0, 0.0, q66,
        )
    }

    /// Compute the transformed reduced stiffness matrix Qbar for angle θ [degrees].
    ///
    /// Uses the standard CLT transformation:
    /// Qbar_11 = Q11·c⁴ + 2(Q12 + 2Q66)·s²c² + Q22·s⁴
    /// etc. (Jones, 1999 — eq. 2.78)
    pub fn compute_qbar(&self, theta_deg: f64) -> Matrix3<f64> {
        let q = self.compute_q();
        let q11 = q[(0, 0)];
        let q12 = q[(0, 1)];
        let q22 = q[(1, 1)];
        let q66 = q[(2, 2)];

        let theta = theta_deg.to_radians();
        let c = theta.cos();
        let s = theta.sin();
        let c2 = c * c;
        let s2 = s * s;
        let c4 = c2 * c2;
        let s4 = s2 * s2;

        let qbar11 = q11 * c4 + 2.0 * (q12 + 2.0 * q66) * s2 * c2 + q22 * s4;
        let qbar22 = q11 * s4 + 2.0 * (q12 + 2.0 * q66) * s2 * c2 + q22 * c4;
        let qbar12 = (q11 + q22 - 4.0 * q66) * s2 * c2 + q12 * (s4 + c4);
        let qbar16 = (q11 - q12 - 2.0 * q66) * s * c.powi(3)
            + (q12 - q22 + 2.0 * q66) * s.powi(3) * c;
        let qbar26 = (q11 - q12 - 2.0 * q66) * s.powi(3) * c
            + (q12 - q22 + 2.0 * q66) * s * c.powi(3);
        let qbar66 =
            (q11 + q22 - 2.0 * q12 - 2.0 * q66) * s2 * c2 + q66 * (s4 + c4);

        Matrix3::new(
            qbar11, qbar12, qbar16,
            qbar12, qbar22, qbar26,
            qbar16, qbar26, qbar66,
        )
    }

    /// Compute the transformed transverse shear stiffness Cbar for angle θ [degrees].
    ///
    /// C44 = G23·cos²θ + G13·sin²θ
    /// C55 = G13·cos²θ + G23·sin²θ
    /// C45 = (G13 - G23)·sinθ·cosθ
    ///
    /// Returns [[C55, C45], [C45, C44]]
    pub fn compute_cbar_shear(&self, theta_deg: f64) -> Matrix2<f64> {
        let theta = theta_deg.to_radians();
        let c = theta.cos();
        let s = theta.sin();
        let c2 = c * c;
        let s2 = s * s;

        let c44 = self.g23 * c2 + self.g13 * s2;
        let c55 = self.g13 * c2 + self.g23 * s2;
        let c45 = (self.g13 - self.g23) * s * c;

        Matrix2::new(c55, c45, c45, c44)
    }
}

impl Material for OrthotropicMaterial {
    /// For a single-ply orthotropic shell with given angle=0 (principal coords).
    ///
    /// This is used when the element is a single homogeneous orthotropic layer.
    /// The shear correction is applied as a scalar to both shear terms.
    fn constitutive(&self, thickness: f64, shear_correction: f64) -> ShellConstitutive {
        let h = thickness;
        let h3_12 = h * h * h / 12.0;

        let qbar = self.compute_qbar(0.0);
        let cbar_s = self.compute_cbar_shear(0.0);

        let cm = qbar * h;
        let cb = qbar * h3_12;
        let cs = cbar_s * (shear_correction * h);
        let cm_raw = if h.abs() > 1e-30 { qbar } else { Matrix3::zeros() };

        ShellConstitutive { 
            cm, 
            cb_coupling: Matrix3::zeros(), // Zero for single ply (no membrane-bending coupling)
            cb, 
            cs, 
            cm_raw 
        }
    }

    fn density(&self) -> f64 {
        self.rho
    }
}
