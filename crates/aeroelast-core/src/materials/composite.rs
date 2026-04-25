use nalgebra::{Matrix2, Matrix3};

use super::ShellConstitutive;

/// Construct a `ShellConstitutive` from pre-computed ABD matrices (CLT).
///
/// For composite laminates, the constitutive matrices are the ABD matrices
/// computed from Classical Lamination Theory:
///
/// - `cm` = A matrix (3×3 extensional stiffness, [N/m])
/// - `cb_coupling` = B matrix (3×3 membrane-bending coupling, [N])
/// - `cb` = D matrix (3×3 bending stiffness, [N·m])
/// - `cs` = Cs matrix (2×2 transverse shear stiffness, [N/m])
/// - `cm_raw` = A / h (average membrane stress-strain, for nonlinear)
///
/// # Arguments
///
/// * `a_flat` - A matrix row-major [a11,a12,a13, a21,a22,a23, a31,a32,a33]
/// * `b_flat` - B matrix row-major (coupling) - set to zeros if symmetric laminate
/// * `d_flat` - D matrix row-major
/// * `cs_flat` - Cs matrix row-major [cs11,cs12, cs21,cs22]
/// * `thickness` - Total laminate thickness (for cm_raw = A/h)
pub fn composite_constitutive(
    a_flat: &[f64; 9],
    b_flat: &[f64; 9],
    d_flat: &[f64; 9],
    cs_flat: &[f64; 4],
    thickness: f64,
) -> ShellConstitutive {
    let cm = Matrix3::new(
        a_flat[0], a_flat[1], a_flat[2],
        a_flat[3], a_flat[4], a_flat[5],
        a_flat[6], a_flat[7], a_flat[8],
    );

    let cb_coupling = Matrix3::new(
        b_flat[0], b_flat[1], b_flat[2],
        b_flat[3], b_flat[4], b_flat[5],
        b_flat[6], b_flat[7], b_flat[8],
    );

    let cb = Matrix3::new(
        d_flat[0], d_flat[1], d_flat[2],
        d_flat[3], d_flat[4], d_flat[5],
        d_flat[6], d_flat[7], d_flat[8],
    );

    let cs = Matrix2::new(
        cs_flat[0], cs_flat[1],
        cs_flat[2], cs_flat[3],
    );

    // cm_raw = A / h  (average membrane stress-strain for nonlinear)
    let inv_h = if thickness.abs() > 1e-30 { 1.0 / thickness } else { 0.0 };
    let cm_raw = cm * inv_h;

    ShellConstitutive { cm, cb_coupling, cb, cs, cm_raw }
}
