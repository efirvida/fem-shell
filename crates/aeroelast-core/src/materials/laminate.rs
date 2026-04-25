use nalgebra::{Matrix2, Matrix3};

use super::orthotropic::OrthotropicMaterial;
use super::ShellConstitutive;

/// Single ply/lamina in a composite laminate.
#[derive(Debug, Clone)]
pub struct Ply {
    pub material: OrthotropicMaterial,
    pub thickness: f64, // [m]
    pub angle: f64,     // fiber orientation [degrees]
    /// Distance from laminate mid-plane to bottom of ply (set by Laminate)
    pub z_bottom: f64,
    /// Distance from laminate mid-plane to top of ply (set by Laminate)
    pub z_top: f64,
}

impl Ply {
    pub fn new(material: OrthotropicMaterial, thickness: f64, angle: f64) -> Self {
        Self { material, thickness, angle, z_bottom: 0.0, z_top: 0.0 }
    }
}

/// Multi-layer laminate composite with ABD stiffness matrices (CLT).
///
/// Implements Classical Lamination Theory:
///
/// ```text
/// [N]   [A  B] [ε⁰]
/// [M] = [B  D] [κ ]
/// ```
///
/// # References
/// - Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed.
/// - Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells.
#[derive(Debug, Clone)]
pub struct Laminate {
    pub plies: Vec<Ply>,
    pub shear_correction_factor: f64,
    /// 3×3 extensional stiffness matrix [N/m]
    pub a: Matrix3<f64>,
    /// 3×3 coupling stiffness matrix [N]
    pub b: Matrix3<f64>,
    /// 3×3 bending stiffness matrix [N·m]
    pub d: Matrix3<f64>,
    /// 2×2 transverse shear stiffness matrix [N/m]
    pub cs: Matrix2<f64>,
    /// Total laminate thickness [m]
    pub total_thickness: f64,
}

impl Laminate {
    /// Build a laminate from a list of plies (bottom → top).
    ///
    /// Computes layer positions and ABD + Cs matrices immediately.
    pub fn new(plies: Vec<Ply>, shear_correction_factor: f64) -> Result<Self, String> {
        if plies.is_empty() {
            return Err("Laminate must have at least one ply".to_string());
        }

        let mut lam = Self {
            plies,
            shear_correction_factor,
            a: Matrix3::zeros(),
            b: Matrix3::zeros(),
            d: Matrix3::zeros(),
            cs: Matrix2::zeros(),
            total_thickness: 0.0,
        };

        lam.total_thickness = lam.plies.iter().map(|p| p.thickness).sum();
        lam.compute_layer_positions();
        lam.compute_abd_matrices();
        lam.compute_shear_stiffness();

        Ok(lam)
    }

    fn compute_layer_positions(&mut self) {
        let mut z = -self.total_thickness / 2.0;
        for ply in &mut self.plies {
            ply.z_bottom = z;
            z += ply.thickness;
            ply.z_top = z;
        }
    }

    /// Compute ABD stiffness matrices by integrating through thickness.
    ///
    /// ```text
    /// A_ij = Σ Qbar_ij^(k) · (z_{k+1} - z_k)
    /// B_ij = (1/2) · Σ Qbar_ij^(k) · (z_{k+1}² - z_k²)
    /// D_ij = (1/3) · Σ Qbar_ij^(k) · (z_{k+1}³ - z_k³)
    /// ```
    fn compute_abd_matrices(&mut self) {
        self.a = Matrix3::zeros();
        self.b = Matrix3::zeros();
        self.d = Matrix3::zeros();

        // Collect data first to avoid borrow issues
        let ply_data: Vec<(Matrix3<f64>, f64, f64)> = self.plies.iter()
            .map(|p| (p.material.compute_qbar(p.angle), p.z_bottom, p.z_top))
            .collect();

        for (qbar, z_bot, z_top) in &ply_data {
            self.a += qbar * (z_top - z_bot);
            self.b += qbar * 0.5 * (z_top * z_top - z_bot * z_bot);
            self.d += qbar * (1.0 / 3.0) * (z_top.powi(3) - z_bot.powi(3));
        }
    }

    /// Compute transverse shear stiffness using energy equivalence.
    ///
    /// For multi-ply laminates, uses equilibrium-based shear stress distribution
    /// through the thickness. For single-ply, falls back to scalar correction factor.
    fn compute_shear_stiffness(&mut self) {
        let mut a44 = 0.0_f64;
        let mut a45 = 0.0_f64;
        let mut a55 = 0.0_f64;

        let ply_data: Vec<(Matrix3<f64>, Matrix2<f64>, f64, f64, f64)> = self.plies.iter()
            .map(|p| {
                let qbar = p.material.compute_qbar(p.angle);
                let cbar_s = p.material.compute_cbar_shear(p.angle);
                (qbar, cbar_s, p.thickness, p.z_bottom, p.z_top)
            })
            .collect();

        for (_, cbar_s, t, _, _) in &ply_data {
            a55 += cbar_s[(0, 0)] * t;
            a45 += cbar_s[(0, 1)] * t;
            a44 += cbar_s[(1, 1)] * t;
        }

        // Single ply: scalar correction
        if self.plies.len() == 1 {
            let k = self.shear_correction_factor;
            self.cs = Matrix2::new(k * a55, k * a45, k * a45, k * a44);
            return;
        }

        let d11 = self.d[(0, 0)];
        let d22 = self.d[(1, 1)];

        let mut flex_55 = 0.0_f64;
        let mut flex_44 = 0.0_f64;
        let mut cum_55 = 0.0_f64;
        let mut cum_44 = 0.0_f64;

        for (qbar, cbar_s, t, z_bot, z_top) in &ply_data {
            let q11 = qbar[(0, 0)];
            let q22 = qbar[(1, 1)];
            let c55_k = cbar_s[(0, 0)];
            let c44_k = cbar_s[(1, 1)];

            // Phi(z) = A + B*z^2 within this ply
            let a_55 = cum_55 - q11 / 2.0 * z_bot * z_bot;
            let b_55 = q11 / 2.0;
            let a_44 = cum_44 - q22 / 2.0 * z_bot * z_bot;
            let b_44 = q22 / 2.0;

            // integral (A + B*z^2)^2 dz from z_bot to z_top
            let dz3 = (z_top.powi(3) - z_bot.powi(3)) / 3.0;
            let dz5 = (z_top.powi(5) - z_bot.powi(5)) / 5.0;

            let i_55 = a_55 * a_55 * t + 2.0 * a_55 * b_55 * dz3 + b_55 * b_55 * dz5;
            let i_44 = a_44 * a_44 * t + 2.0 * a_44 * b_44 * dz3 + b_44 * b_44 * dz5;

            if c55_k > 0.0 { flex_55 += i_55 / c55_k; }
            if c44_k > 0.0 { flex_44 += i_44 / c44_k; }

            cum_55 += q11 * (z_top * z_top - z_bot * z_bot) / 2.0;
            cum_44 += q22 * (z_top * z_top - z_bot * z_bot) / 2.0;
        }

        let k = self.shear_correction_factor;

        let cs_55 = if flex_55 > 0.0 && d11 > 0.0 {
            d11 * d11 / flex_55
        } else {
            k * a55
        };

        let cs_44 = if flex_44 > 0.0 && d22 > 0.0 {
            d22 * d22 / flex_44
        } else {
            k * a44
        };

        let cs_45 = if a55 > 0.0 && a44 > 0.0 {
            let k55 = cs_55 / a55;
            let k44 = cs_44 / a44;
            (k55 * k44).sqrt() * a45
        } else {
            k * a45
        };

        self.cs = Matrix2::new(cs_55, cs_45, cs_45, cs_44);
    }

    /// Build a `ShellConstitutive` from this laminate's ABD matrices.
    ///
    /// This is the object the element routines consume.
    pub fn to_shell_constitutive(&self) -> ShellConstitutive {
        self.to_shell_constitutive_with_offset(0.0)
    }

    /// Build a `ShellConstitutive` with reference surface offset.
    ///
    /// When the reference surface is offset from the laminate midsurface by `z_offset`:
    /// - A_offset = A (unchanged)
    /// - B_offset = B - z₀·A
    /// - D_offset = D - 2·z₀·B + z₀²·A
    ///
    /// This affects how membrane strain ε_ref relates to strain at material midplane.
    /// Positive z_offset moves reference surface toward the top (positive z).
    ///
    /// # Arguments
    /// * `z_offset` - Distance from laminate midsurface to reference surface [m]
    pub fn to_shell_constitutive_with_offset(&self, z_offset: f64) -> ShellConstitutive {
        let h = self.total_thickness;
        let inv_h = if h.abs() > 1e-30 { 1.0 / h } else { 0.0 };
        let cm_raw = self.a * inv_h;

        // Apply offset transformation to ABD matrices
        // B_offset = B - z₀·A
        // D_offset = D - 2·z₀·B + z₀²·A
        let b_offset = self.b - self.a * z_offset;
        let d_offset = self.d - 2.0 * z_offset * &self.b + z_offset * z_offset * &self.a;

        ShellConstitutive {
            cm: self.a,
            cb_coupling: b_offset, // B matrix with offset applied
            cb: d_offset,           // D matrix with offset applied
            cs: self.cs,
            cm_raw,
        }
    }

    /// Check if laminate is symmetric (B ≈ 0).
    pub fn is_symmetric(&self) -> bool {
        let tol = 1e-10 * self.a.amax();
        self.b.amax() < tol
    }

    /// Check if laminate is balanced (A16 = A26 ≈ 0).
    pub fn is_balanced(&self) -> bool {
        let tol = 1e-10 * self.a.amax();
        self.a[(0, 2)].abs() < tol && self.a[(1, 2)].abs() < tol
    }

    /// Get the full 6×6 ABD matrix [[A, B], [B, D]].
    pub fn abd_matrix_flat(&self) -> [f64; 36] {
        let mut out = [0.0f64; 36];
        // A block (rows 0-2, cols 0-2)
        for i in 0..3 {
            for j in 0..3 {
                out[i * 6 + j] = self.a[(i, j)];
                out[i * 6 + (j + 3)] = self.b[(i, j)];
                out[(i + 3) * 6 + j] = self.b[(i, j)];
                out[(i + 3) * 6 + (j + 3)] = self.d[(i, j)];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 3-ply symmetric cross-ply laminate for testing
    /// Layout: [0°/90°/0°] around midsurface
    fn make_test_laminate() -> Laminate {
        let mat = OrthotropicMaterial::new(
            1.0e11, 1.0e10, 1.0e10,  // E1, E2, E3
            5.0e9, 5.0e9, 5.0e9,    // G12, G23, G13
            0.3, 0.3, 0.3,          // nu12, nu23, nu31
            1600.0,                 // rho
        );
        let ply1 = Ply::new(mat, 0.002, 0.0);   // bottom
        let ply2 = Ply::new(mat, 0.002, 90.0);  // middle
        let ply3 = Ply::new(mat, 0.002, 0.0);   // top
        Laminate::new(vec![ply1, ply2, ply3], 5.0 / 6.0).unwrap()
    }

    /// Create a 3-ply asymmetric laminate for coupling tests
    fn make_asymmetric_laminate() -> Laminate {
        let mat = OrthotropicMaterial::new(
            1.0e11, 1.0e10, 1.0e10,
            5.0e9, 5.0e9, 5.0e9,
            0.3, 0.3, 0.3,
            1600.0,
        );
        let ply1 = Ply::new(mat, 0.002, 0.0);
        let ply2 = Ply::new(mat, 0.003, 45.0);
        let ply3 = Ply::new(mat, 0.001, 0.0);
        Laminate::new(vec![ply1, ply2, ply3], 5.0 / 6.0).unwrap()
    }

    #[test]
    fn test_laminate_symmetric() {
        let lam = make_test_laminate();
        assert!(lam.is_symmetric(), "2-ply with ±angle should be symmetric");
    }

    #[test]
    fn test_laminate_to_shell_constitutive() {
        let lam = make_test_laminate();
        let shell = lam.to_shell_constitutive();
        // A matrix should be non-zero
        assert!(shell.cm.norm() > 1e6, "A matrix should be non-zero");
        // For symmetric laminate, B should be ~0
        assert!(shell.cb_coupling.norm() < 1e-6, "B matrix should be ~0 for symmetric laminate");
        // D matrix should be non-zero
        assert!(shell.cb.norm() > 1e-3, "D matrix should be non-zero");
    }

    #[test]
    fn test_z_offset_zero() {
        // z_offset = 0 should give same result as to_shell_constitutive()
        let lam = make_test_laminate();
        let shell_default = lam.to_shell_constitutive();
        let shell_offset = lam.to_shell_constitutive_with_offset(0.0);
        let diff_a = &shell_default.cm - &shell_offset.cm;
        let diff_b = &shell_default.cb_coupling - &shell_offset.cb_coupling;
        let diff_d = &shell_default.cb - &shell_offset.cb;
        assert!(diff_a.norm() < 1e-12, "A should be unchanged with z_offset=0");
        assert!(diff_b.norm() < 1e-12, "B should be unchanged with z_offset=0");
        assert!(diff_d.norm() < 1e-12, "D should be unchanged with z_offset=0");
    }

    #[test]
    fn test_z_offset_transforms_b_d() {
        // With z_offset, B and D should transform, A should stay same
        let lam = make_test_laminate();
        let z_off = 0.001; // 1mm offset
        let shell = lam.to_shell_constitutive_with_offset(z_off);

        // A unchanged
        assert!((&shell.cm - &lam.a).norm() < 1e-12, "A should be unchanged");

        // B_offset = B - z₀·A (for symmetric B≈0, B_offset ≈ -z₀·A)
        // For this symmetric laminate, B should not be exactly zero after offset
        // because we're applying the transformation
        let expected_b_offset = &lam.b - &lam.a * z_off;
        let b_diff = &shell.cb_coupling - &expected_b_offset;
        assert!(b_diff.norm() < 1e-6, "B offset transformation incorrect");

        // D_offset = D - 2·z₀·B + z₀²·A (for symmetric B≈0, D_offset ≈ D + z₀²·A)
        let expected_d_offset = &lam.d - 2.0 * z_off * &lam.b + z_off * z_off * &lam.a;
        let d_diff = &shell.cb - &expected_d_offset;
        assert!(d_diff.norm() < 1e-6, "D offset transformation incorrect");
    }

    #[test]
    fn test_asymmetric_laminate_has_nonzero_b() {
        let lam = make_asymmetric_laminate();
        assert!(!lam.is_symmetric(), "asymmetric laminate should have B ≠ 0");
        let shell = lam.to_shell_constitutive();
        assert!(shell.cb_coupling.norm() > 1e-6, "B matrix should be non-zero for asymmetric laminate");
    }
}
