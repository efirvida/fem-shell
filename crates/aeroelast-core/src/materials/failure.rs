//! Failure criteria for fiber-reinforced composite laminae.
//!
//! Implements:
//! - Tsai-Wu criterion (quadratic interaction)
//! - Hashin criterion (mode-specific)
//! - Maximum stress criterion
//!
//! # References
//! - Tsai, S.W. and Wu, E.M. (1971). J. Composite Materials, 5, 58–80.
//! - Hashin, Z. (1980). J. Applied Mechanics, 47, 329–334.

/// Strength properties for a composite lamina.
///
/// All values are in Pa (positive magnitudes — the sign convention is
/// handled internally by each criterion).
#[derive(Debug, Clone)]
pub struct StrengthProperties {
    /// Tensile strength in fiber direction [Pa]
    pub xt: f64,
    /// Compressive strength in fiber direction [Pa]
    pub xc: f64,
    /// Tensile strength transverse to fibers [Pa]
    pub yt: f64,
    /// Compressive strength transverse to fibers [Pa]
    pub yc: f64,
    /// In-plane shear strength [Pa]
    pub s12: f64,
    /// Transverse shear strength [Pa] (defaults to s12/2 if not provided)
    pub s23: f64,
}

impl StrengthProperties {
    pub fn new(xt: f64, xc: f64, yt: f64, yc: f64, s12: f64, s23: Option<f64>) -> Self {
        Self { xt, xc, yt, yc, s12, s23: s23.unwrap_or(s12 / 2.0) }
    }
}

/// Composite failure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureMode {
    NoFailure,
    FiberTension,
    FiberCompression,
    MatrixTension,
    MatrixCompression,
    Shear,
    Combined,
}

/// Result of a failure criterion evaluation.
#[derive(Debug, Clone)]
pub struct FailureResult {
    /// True if failure index >= 1.0
    pub failed: bool,
    /// Failure index value (>= 1.0 means failure)
    pub failure_index: f64,
    /// Failure mode
    pub mode: FailureMode,
    /// Reserve factor = 1 / sqrt(FI)
    pub reserve_factor: f64,
    /// Margin of safety = reserve_factor - 1
    pub margin_of_safety: f64,
}

impl FailureResult {
    pub fn new(failure_index: f64, mode: FailureMode) -> Self {
        let (reserve_factor, margin_of_safety) = if failure_index > 0.0 {
            let rf = 1.0 / failure_index.sqrt();
            (rf, rf - 1.0)
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
        Self {
            failed: failure_index >= 1.0,
            failure_index,
            mode,
            reserve_factor,
            margin_of_safety,
        }
    }
}

/// Compute the Tsai-Wu failure index for a composite lamina.
///
/// Criterion: F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2F12·σ1·σ2 = 1
///
/// # Arguments
/// * `sigma` - `[σ1, σ2, τ12]` stress in ply principal coordinates [Pa]
/// * `strength` - material strength properties
/// * `f12_coefficient` - interaction coefficient (default: -0.5)
pub fn tsai_wu(
    sigma: [f64; 3],
    strength: &StrengthProperties,
    f12_coefficient: f64,
) -> FailureResult {
    let [s1, s2, t12] = sigma;
    let (xt, xc, yt, yc, s) = (strength.xt, strength.xc, strength.yt, strength.yc, strength.s12);

    let f1 = 1.0 / xt - 1.0 / xc;
    let f2 = 1.0 / yt - 1.0 / yc;
    let f11 = 1.0 / (xt * xc);
    let f22 = 1.0 / (yt * yc);
    let f66 = 1.0 / (s * s);
    let f12 = f12_coefficient / (xt * xc * yt * yc).sqrt();

    let fi = f1 * s1 + f2 * s2 + f11 * s1 * s1 + f22 * s2 * s2 + f66 * t12 * t12
        + 2.0 * f12 * s1 * s2;

    let mode = if fi >= 1.0 { FailureMode::Combined } else { FailureMode::NoFailure };
    FailureResult::new(fi, mode)
}

/// Compute Hashin failure indices for all failure modes.
///
/// Returns one `FailureResult` per active mode:
/// - Fiber tension OR fiber compression (depending on sign of σ1)
/// - Matrix tension OR matrix compression (depending on sign of σ2)
///
/// # Arguments
/// * `sigma` - `[σ1, σ2, τ12]` stress in ply principal coordinates [Pa]
/// * `strength` - material strength properties
pub fn hashin(sigma: [f64; 3], strength: &StrengthProperties) -> HashinResults {
    let [s1, s2, t12] = sigma;
    let (xt, xc, yt, yc, s, st) =
        (strength.xt, strength.xc, strength.yt, strength.yc, strength.s12, strength.s23);

    let fiber = if s1 >= 0.0 {
        let fi = (s1 / xt).powi(2) + (t12 / s).powi(2);
        FailureResult::new(fi, FailureMode::FiberTension)
    } else {
        let fi = (s1.abs() / xc).powi(2);
        FailureResult::new(fi, FailureMode::FiberCompression)
    };

    let matrix = if s2 >= 0.0 {
        let fi = (s2 / yt).powi(2) + (t12 / s).powi(2);
        FailureResult::new(fi, FailureMode::MatrixTension)
    } else {
        let fi = (s2 / (2.0 * st)).powi(2)
            + ((yc / (2.0 * st)).powi(2) - 1.0) * (s2 / yc)
            + (t12 / s).powi(2);
        FailureResult::new(fi, FailureMode::MatrixCompression)
    };

    HashinResults { fiber, matrix }
}

/// All Hashin mode results.
#[derive(Debug, Clone)]
pub struct HashinResults {
    pub fiber: FailureResult,
    pub matrix: FailureResult,
}

impl HashinResults {
    /// Returns the most critical result (highest failure index).
    pub fn critical(&self) -> &FailureResult {
        if self.fiber.failure_index >= self.matrix.failure_index {
            &self.fiber
        } else {
            &self.matrix
        }
    }
}

/// Compute the maximum stress failure criterion.
///
/// FI = max(σ1/Xt, |σ1|/Xc, σ2/Yt, |σ2|/Yc, |τ12|/S) depending on sign.
///
/// # Arguments
/// * `sigma` - `[σ1, σ2, τ12]` stress in ply principal coordinates [Pa]
/// * `strength` - material strength properties
pub fn max_stress(sigma: [f64; 3], strength: &StrengthProperties) -> FailureResult {
    let [s1, s2, t12] = sigma;
    let (xt, xc, yt, yc, s) = (strength.xt, strength.xc, strength.yt, strength.yc, strength.s12);

    let (r1, mode1) = if s1 >= 0.0 {
        (s1 / xt, FailureMode::FiberTension)
    } else {
        (s1.abs() / xc, FailureMode::FiberCompression)
    };

    let (r2, mode2) = if s2 >= 0.0 {
        (s2 / yt, FailureMode::MatrixTension)
    } else {
        (s2.abs() / yc, FailureMode::MatrixCompression)
    };

    let r12 = t12.abs() / s;

    let (fi, mode) = [(r1, mode1), (r2, mode2), (r12, FailureMode::Shear)]
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    FailureResult::new(fi, mode)
}

/// Compute the stress transformation matrix from laminate (x,y) → ply (1,2) coordinates.
///
/// T such that σ_12 = T · σ_xy, where σ = [σ_xx, σ_yy, τ_xy].
pub fn stress_transformation(theta_deg: f64) -> [[f64; 3]; 3] {
    let theta = theta_deg.to_radians();
    let (c, s) = (theta.cos(), theta.sin());
    let (c2, s2, sc) = (c * c, s * s, s * c);

    [[c2, s2, 2.0 * sc], [s2, c2, -2.0 * sc], [-sc, sc, c2 - s2]]
}

/// Compute the strain transformation matrix from laminate (x,y) → ply (1,2) coordinates.
///
/// T_eps such that ε_12 = T_eps · ε_xy, where ε = [ε_xx, ε_yy, γ_xy].
pub fn strain_transformation(theta_deg: f64) -> [[f64; 3]; 3] {
    let theta = theta_deg.to_radians();
    let (c, s) = (theta.cos(), theta.sin());
    let (c2, s2, sc) = (c * c, s * s, s * c);

    [[c2, s2, sc], [s2, c2, -sc], [-2.0 * sc, 2.0 * sc, c2 - s2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glass_epoxy() -> StrengthProperties {
        // Typical E-glass/epoxy [Pa]
        StrengthProperties::new(1500e6, 1500e6, 40e6, 246e6, 68e6, None)
    }

    #[test]
    fn tsai_wu_no_failure() {
        let s = glass_epoxy();
        let r = tsai_wu([500e6, 20e6, 30e6], &s, -0.5);
        assert!(r.failure_index < 1.0);
        assert!(!r.failed);
        assert!(r.reserve_factor > 1.0);
    }

    #[test]
    fn tsai_wu_failure() {
        let s = glass_epoxy();
        let r = tsai_wu([1600e6, 0.0, 0.0], &s, -0.5);
        assert!(r.failed);
        assert!(r.failure_index >= 1.0);
    }

    #[test]
    fn hashin_fiber_tension() {
        let s = glass_epoxy();
        let h = hashin([800e6, 5e6, 10e6], &s);
        assert_eq!(h.fiber.mode, FailureMode::FiberTension);
    }

    #[test]
    fn hashin_matrix_compression() {
        let s = glass_epoxy();
        let h = hashin([0.0, -200e6, 30e6], &s);
        assert_eq!(h.matrix.mode, FailureMode::MatrixCompression);
    }

    #[test]
    fn max_stress_shear_critical() {
        let s = glass_epoxy();
        let r = max_stress([10e6, 5e6, 70e6], &s);
        assert_eq!(r.mode, FailureMode::Shear);
        assert!(r.failed);
    }

    #[test]
    fn stress_transformation_identity_at_zero() {
        let t = stress_transformation(0.0);
        // At 0°, T should be identity
        assert!((t[0][0] - 1.0).abs() < 1e-12);
        assert!((t[1][1] - 1.0).abs() < 1e-12);
        assert!((t[2][2] - 1.0).abs() < 1e-12);
        assert!(t[0][1].abs() < 1e-12);
    }

    #[test]
    fn reserve_factor_consistency() {
        let s = glass_epoxy();
        let r = tsai_wu([500e6, 20e6, 30e6], &s, -0.5);
        let expected_rf = 1.0 / r.failure_index.sqrt();
        assert!((r.reserve_factor - expected_rf).abs() < 1e-12);
        assert!((r.margin_of_safety - (expected_rf - 1.0)).abs() < 1e-12);
    }
}
