// MITC4+ Shell Element Kernel
//
// High-performance implementation of the MITC4+ quadrilateral shell element
// (Ko, Lee & Bathe, 2017).
//
// Key differences from MITC3+:
//   - 4 nodes × 6 DOFs = 24 DOFs per element
//   - Bilinear shape functions (variable Jacobian)
//   - 2×2 Gauss integration (4 points)
//   - 5 membrane tying points with MITC4+ blending coefficients
//   - Bubble enrichment: N_b = (1−ξ²)(1−η²) with 2 rotation DOFs condensed out
//   - Rotation-based MITC4 transverse shear interpolation

use nalgebra::{Matrix2, Matrix3, SMatrix, SVector, Vector2, Vector3};

use crate::materials::ShellConstitutive;

// ============================================================================
// Type aliases for fixed-size element matrices
// ============================================================================

/// 24×24 element stiffness/mass matrix (4 nodes × 6 DOFs)
pub type Mat24 = SMatrix<f64, 24, 24>;
/// 26×26 extended matrix (24 nodal + 2 bubble DOFs)
pub type Mat26 = SMatrix<f64, 26, 26>;
/// 24-element force/displacement vector
pub type Vec24 = SVector<f64, 24>;
/// 26-element extended vector
pub type Vec26 = SVector<f64, 26>;

// ============================================================================
// Gauss quadrature (2×2 Gauss-Legendre on quad)
// ============================================================================
// GP value and the 4-point flat rule match crate::quadrature::gauss_quad_2x2().
// Local consts used here for zero-overhead access in hot path.

const N_GAUSS: usize = 4;
const GP: f64 = 0.577_350_269_189_625_8; // 1/√3

const GAUSS_XI:  [f64; N_GAUSS] = [-GP,  GP,  GP, -GP];
const GAUSS_ETA: [f64; N_GAUSS] = [-GP, -GP,  GP,  GP];
const GAUSS_W:   [f64; N_GAUSS] = [1.0, 1.0, 1.0, 1.0];

// ============================================================================
// Precomputed element data
// ============================================================================

/// All data that is constant for a given element geometry.
pub struct Mitc4Precomputed {
    /// Local coordinates of 4 nodes (4×2)
    pub local_coords: [[f64; 2]; 4],
    /// Transformation matrix: local<->global (3×3 rotation part)
    pub t3: Matrix3<f64>,
    /// Constitutive matrices
    pub constitutive: ShellConstitutive,
    /// Drilling stiffness factor: E·h²·0.15
    pub k_drill: f64,
    /// Thickness
    pub thickness: f64,
    /// MITC4+ membrane blending coefficients
    pub a_a: f64,
    pub a_b: f64,
    pub a_c: f64,
    pub a_d: f64,
    pub a_e: f64,
    /// Characteristic vectors (Ko et al. 2017)
    pub x_r: Vector3<f64>,
    pub x_s: Vector3<f64>,
    pub x_d: Vector3<f64>,
    pub n_vec: Vector3<f64>,
    pub m_r: Vector3<f64>,
    pub m_s: Vector3<f64>,
    /// Initial 3D node coordinates (for J3D / covariant membrane)
    pub initial_coords_3d: [[f64; 3]; 4],
    /// Precomputed Jacobian data at each Gauss point
    pub gp_jacobians: [GpJacobian; N_GAUSS],
    /// Precomputed bubble cache at each Gauss point
    pub gp_bubble: [GpBubble; N_GAUSS],
    /// Precomputed covariant membrane B-rows at 5 tying points (each 24-element row)
    pub b_rr: [Vec24; 5],
    pub b_ss: [Vec24; 5],
    pub b_rs: [Vec24; 5],

}

/// Jacobian data at a single Gauss point (3D covariant formulation).
#[derive(Clone, Copy)]
pub struct GpJacobian {
    /// Projection of 3D tangents onto local frame: j_loc[α][β] = g_α · e_β
    pub j_loc: Matrix2<f64>,
    /// Inverse of j_loc
    pub j_inv: Matrix2<f64>,
    /// Surface area element: |g_r × g_s|
    pub sqrt_g: f64,
    /// Shape function derivatives in local orthonormal frame [2×4]
    /// dh[0,i] = dNi/de1,  dh[1,i] = dNi/de2
    pub dh: SMatrix<f64, 2, 4>,
}

/// Bubble data at a single Gauss point.
#[derive(Clone, Copy)]
pub struct GpBubble {
    pub nb: f64,
    pub dnb_dxi: f64,
    pub dnb_deta: f64,
}

// ============================================================================
// Shape functions
// ============================================================================

/// Bilinear shape functions at (xi, eta)
#[inline(always)]
fn shape_functions(xi: f64, eta: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]
}

/// Shape function derivatives at (xi, eta)
/// Returns (dN_dxi[4], dN_deta[4])
#[inline(always)]
fn shape_function_derivatives(xi: f64, eta: f64) -> ([f64; 4], [f64; 4]) {
    let dn_dxi = [
        -0.25 * (1.0 - eta),
         0.25 * (1.0 - eta),
         0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta),
    ];
    let dn_deta = [
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
         0.25 * (1.0 + xi),
         0.25 * (1.0 - xi),
    ];
    (dn_dxi, dn_deta)
}

// ============================================================================
// Bubble function
// ============================================================================

#[inline(always)]
fn bubble_function(xi: f64, eta: f64) -> f64 {
    (1.0 - xi * xi) * (1.0 - eta * eta)
}

#[inline(always)]
fn bubble_derivatives(xi: f64, eta: f64) -> (f64, f64) {
    let dnb_dxi  = -2.0 * xi * (1.0 - eta * eta);
    let dnb_deta = -2.0 * eta * (1.0 - xi * xi);
    (dnb_dxi, dnb_deta)
}

// ============================================================================
// Jacobian computation (variable — depends on xi, eta)
// ============================================================================


/// Compute 3D tangent vectors at (xi, eta) from initial 3D coordinates
fn compute_j3d(coords_3d: &[[f64; 3]; 4], xi: f64, eta: f64) -> (Vector3<f64>, Vector3<f64>) {
    let (dn_dxi, dn_deta) = shape_function_derivatives(xi, eta);
    let mut g_r = Vector3::zeros();
    let mut g_s = Vector3::zeros();

    for i in 0..4 {
        let x = Vector3::new(coords_3d[i][0], coords_3d[i][1], coords_3d[i][2]);
        g_r += dn_dxi[i] * x;
        g_s += dn_deta[i] * x;
    }
    (g_r, g_s)
}

// ============================================================================
// Local coordinate system
// ============================================================================

/// Compute local orthonormal frame (e1, e2, e3) at (xi, eta)
fn compute_local_frame_at(coords_3d: &[[f64; 3]; 4], xi: f64, eta: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let (g_r, g_s) = compute_j3d(coords_3d, xi, eta);
    
    // e3 is the surface normal
    let mut e3 = g_r.cross(&g_s);
    if e3.norm() < 1e-12 {
        // Fallback for degenerate elements
        e3 = Vector3::new(0.0, 0.0, 1.0);
    } else {
        e3 = e3.normalize();
    }

    // e1 is the projection of g_r onto the plane orthogonal to e3
    let mut e1 = g_r - g_r.dot(&e3) * e3;
    if e1.norm() < 1e-12 {
        // Fallback: use g_s if g_r is parallel to normal
        e1 = g_s - g_s.dot(&e3) * e3;
    }
    e1 = e1.normalize();

    let e2 = e3.cross(&e1).normalize();

    (e1, e2, e3)
}

/// Project 3D coords to local 2D using a specific frame
fn compute_local_coords_centroid(coords_3d: &[[f64; 3]; 4], e1: &Vector3<f64>, e2: &Vector3<f64>) -> [[f64; 2]; 4] {
    let mut local = [[0.0f64; 2]; 4];
    for i in 0..4 {
        let p = Vector3::new(coords_3d[i][0], coords_3d[i][1], coords_3d[i][2]);
        local[i][0] = p.dot(e1);
        local[i][1] = p.dot(e2);
    }
    local
}

// ============================================================================
// Characteristic vectors & MITC4+ membrane coefficients
// ============================================================================

fn compute_characteristic_vectors(
    coords_3d: &[[f64; 3]; 4],
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let nodes: [Vector3<f64>; 4] = [
        Vector3::new(coords_3d[0][0], coords_3d[0][1], coords_3d[0][2]),
        Vector3::new(coords_3d[1][0], coords_3d[1][1], coords_3d[1][2]),
        Vector3::new(coords_3d[2][0], coords_3d[2][1], coords_3d[2][2]),
        Vector3::new(coords_3d[3][0], coords_3d[3][1], coords_3d[3][2]),
    ];

    let x_r = 0.25 * (-nodes[0] + nodes[1] + nodes[2] - nodes[3]);
    let x_s = 0.25 * (-nodes[0] - nodes[1] + nodes[2] + nodes[3]);
    let x_d = 0.25 * ( nodes[0] - nodes[1] + nodes[2] - nodes[3]);

    let mut n_vec = x_r.cross(&x_s);
    if n_vec.norm() < 1e-12 {
        let v = Vector3::new(
            coords_3d[1][0] - coords_3d[0][0],
            coords_3d[1][1] - coords_3d[0][1],
            coords_3d[1][2] - coords_3d[0][2],
        );
        let w = Vector3::new(
            coords_3d[2][0] - coords_3d[0][0],
            coords_3d[2][1] - coords_3d[0][1],
            coords_3d[2][2] - coords_3d[0][2],
        );
        n_vec = v.cross(&w);
    }
    n_vec = n_vec.normalize();

    // Solve A * coeff = [1,0,0] and [0,1,0] for m_r, m_s
    let a_mat = Matrix3::new(
        x_r.dot(&x_r), x_r.dot(&x_s), x_r.dot(&n_vec),
        x_s.dot(&x_r), x_s.dot(&x_s), x_s.dot(&n_vec),
        n_vec.dot(&x_r), n_vec.dot(&x_s), n_vec.dot(&n_vec),
    );

    let (m_r, m_s) = if let Some(a_inv) = a_mat.try_inverse() {
        let cr = a_inv * Vector3::new(1.0, 0.0, 0.0);
        let cs = a_inv * Vector3::new(0.0, 1.0, 0.0);
        let mr = cr[0] * x_r + cr[1] * x_s + cr[2] * n_vec;
        let ms = cs[0] * x_r + cs[1] * x_s + cs[2] * n_vec;
        (mr, ms)
    } else {
        // Fallback
        let mr = x_r / x_r.dot(&x_r);
        let ms = x_s / x_s.dot(&x_s);
        (mr, ms)
    };

    (x_r, x_s, x_d, n_vec, m_r, m_s)
}

fn compute_membrane_coefficients(
    x_d: &Vector3<f64>,
    m_r: &Vector3<f64>,
    m_s: &Vector3<f64>,
) -> (f64, f64, f64, f64, f64) {
    let c_r = x_d.dot(m_r);
    let c_s = x_d.dot(m_s);

    let mut d = c_r * c_r + c_s * c_s - 1.0;
    if d.abs() < 1e-12 {
        d = if d >= 0.0 { 1e-12 } else { -1e-12 };
    }

    let a_a = c_r * (c_r - 1.0) / (2.0 * d);
    let a_b = c_r * (c_r + 1.0) / (2.0 * d);
    let a_c = c_s * (c_s - 1.0) / (2.0 * d);
    let a_d = c_s * (c_s + 1.0) / (2.0 * d);
    let a_e = -2.0 * c_r * c_s / d;

    (a_a, a_b, a_c, a_d, a_e)
}

#[inline(always)]
fn regularized_inverse_2x2(m: &Matrix2<f64>) -> Matrix2<f64> {
    let scale = m[(0, 0)]
        .abs()
        .max(m[(0, 1)].abs())
        .max(m[(1, 0)].abs())
        .max(m[(1, 1)].abs())
        .max(1.0);

    let mut reg = *m;
    let eps = scale * 1.0e-12;
    reg[(0, 0)] += eps;
    reg[(1, 1)] += eps;

    reg.try_inverse().unwrap_or_else(Matrix2::identity)
}

// ============================================================================
// Covariant membrane strain B-row at a single tying point
// ============================================================================

/// Compute a single covariant membrane strain B-row (1×24) at (xi, eta) for given component.
/// component: 0 = rr, 1 = ss, 2 = rs
fn compute_covariant_membrane_b_row(
    coords_3d: &[[f64; 3]; 4],
    _e1: &Vector3<f64>,
    _e2: &Vector3<f64>,
    _e3: &Vector3<f64>,
    xi: f64,
    eta: f64,
    component: usize,
) -> Vec24 {
    let (dn_dxi, dn_deta) = shape_function_derivatives(xi, eta);

    // 3D tangent vectors
    let (g_r_3d, g_s_3d) = compute_j3d(coords_3d, xi, eta);

    let mut b = Vec24::zeros();

    for i in 0..4 {
        let u_idx = 6 * i;
        let v_idx = 6 * i + 1;
        let w_idx = 6 * i + 2;

        match component {
            0 => {
                // e_rr = g_r · (∂u/∂r) = Σ (dN_i/dr) (g_r · u_i)
                b[u_idx] = g_r_3d[0] * dn_dxi[i];
                b[v_idx] = g_r_3d[1] * dn_dxi[i];
                b[w_idx] = g_r_3d[2] * dn_dxi[i];
            }
            1 => {
                // e_ss = g_s · (∂u/∂s) = Σ (dN_i/ds) (g_s · u_i)
                b[u_idx] = g_s_3d[0] * dn_deta[i];
                b[v_idx] = g_s_3d[1] * dn_deta[i];
                b[w_idx] = g_s_3d[2] * dn_deta[i];
            }
            _ => {
                // e_rs = 0.5 * (g_r · (∂u/∂s) + g_s · (∂u/∂r))
                b[u_idx] = 0.5 * (g_r_3d[0] * dn_deta[i] + g_s_3d[0] * dn_dxi[i]);
                b[v_idx] = 0.5 * (g_r_3d[1] * dn_deta[i] + g_s_3d[1] * dn_dxi[i]);
                b[w_idx] = 0.5 * (g_r_3d[2] * dn_deta[i] + g_s_3d[2] * dn_dxi[i]);
            }
        }
    }

    b
}

// ============================================================================
// Covariant-to-Cartesian strain transform
// ============================================================================

/// Compute 3×3 covariant-to-local transformation for membrane strains.
///
/// Transforms covariant strains (ε_rr, ε_ss, 2ε_rs) → local-frame strains
/// (ε_11, ε_22, 2ε_12) using the point-wise projection matrix j_loc.
fn covariant_to_local_mapping(j_loc: &Matrix2<f64>) -> Matrix3<f64> {
    let j_inv = regularized_inverse_2x2(j_loc);

    let j11 = j_inv[(0, 0)];
    let j12 = j_inv[(0, 1)];
    let j21 = j_inv[(1, 0)];
    let j22 = j_inv[(1, 1)];

    Matrix3::new(
        j11 * j11,           j21 * j21,           j11 * j21,
        j12 * j12,           j22 * j22,           j12 * j22,
        2.0 * j11 * j12,     2.0 * j21 * j22,     j11 * j22 + j12 * j21,
    )
}

/// Compute j_loc and its inverse at an arbitrary (xi, eta) from 3D geometry.
fn compute_j_loc_at(
    coords_3d: &[[f64; 3]; 4],
    e1: &Vector3<f64>,
    e2: &Vector3<f64>,
    xi: f64,
    eta: f64,
) -> (Matrix2<f64>, Matrix2<f64>) {
    let (g_r, g_s) = compute_j3d(coords_3d, xi, eta);
    let j_loc = Matrix2::new(
        g_r.dot(e1), g_s.dot(e1),
        g_r.dot(e2), g_s.dot(e2),
    );
    let j_inv = regularized_inverse_2x2(&j_loc);
    (j_loc, j_inv)
}

// ============================================================================
// Constructor
// ============================================================================

impl Mitc4Precomputed {
    /// Create precomputed element data from 12 nodal coordinates
    /// [x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4]
    pub fn new(
        node_coords: &[f64; 12],
        constitutive: ShellConstitutive,
        thickness: f64,
        e_mod: f64,
        drilling_scale: f64,
    ) -> Self {
        let mut coords_3d = [[0.0f64; 3]; 4];
        for i in 0..4 {
            coords_3d[i][0] = node_coords[3 * i];
            coords_3d[i][1] = node_coords[3 * i + 1];
            coords_3d[i][2] = node_coords[3 * i + 2];
        }

    // Characteristic vectors & membrane coefficients
    let (x_r, x_s, x_d, n_vec, m_r, m_s) = compute_characteristic_vectors(&coords_3d);
    let (a_a, a_b, a_c, a_d, a_e) = compute_membrane_coefficients(&x_d, &m_r, &m_s);

    // Precompute Jacobians at Gauss points (3D covariant)
    let mut gp_jacobians = [GpJacobian {
        j_loc: Matrix2::zeros(),
        j_inv: Matrix2::zeros(),
        sqrt_g: 0.0,
        dh: SMatrix::<f64, 2, 4>::zeros(),
    }; N_GAUSS];

    // We need a reference frame for the nodal DOFs (T matrix)
    // We use the frame at the centroid (xi=0, eta=0)
    let (e1_ref, e2_ref, e3_ref) = compute_local_frame_at(&coords_3d, 0.0, 0.0);
    let t3 = Matrix3::new(
        e1_ref[0], e1_ref[1], e1_ref[2],
        e2_ref[0], e2_ref[1], e2_ref[2],
        e3_ref[0], e3_ref[1], e3_ref[2],
    );

    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let (g_r, g_s) = compute_j3d(&coords_3d, xi, eta);
        let sqrt_g = g_r.cross(&g_s).norm();
        
        // Use the frame at this specific point to compute j_loc
        let (e1, e2, _) = compute_local_frame_at(&coords_3d, xi, eta);
        let j_loc = Matrix2::new(
            g_r.dot(&e1), g_s.dot(&e1),
            g_r.dot(&e2), g_s.dot(&e2),
        );
        let j_inv = j_loc.try_inverse().unwrap_or_else(|| Matrix2::identity());
        let (dn_dxi, dn_deta) = shape_function_derivatives(xi, eta);
        let mut dh = SMatrix::<f64, 2, 4>::zeros();
        for i in 0..4 {
            // dh = j_inv^T * [dN/dxi; dN/deta]
            dh[(0, i)] = j_inv[(0, 0)] * dn_dxi[i] + j_inv[(1, 0)] * dn_deta[i];
            dh[(1, i)] = j_inv[(0, 1)] * dn_dxi[i] + j_inv[(1, 1)] * dn_deta[i];
        }
        gp_jacobians[g] = GpJacobian { j_loc, j_inv, sqrt_g, dh };
    }

    // Precompute bubble data at Gauss points
    let mut gp_bubble = [GpBubble { nb: 0.0, dnb_dxi: 0.0, dnb_deta: 0.0 }; N_GAUSS];
    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let nb = bubble_function(xi, eta);
        let (dnb_dxi, dnb_deta) = bubble_derivatives(xi, eta);
        gp_bubble[g] = GpBubble { nb, dnb_dxi, dnb_deta };
    }

    // Precompute covariant membrane B-rows at 5 tying points
    let tying_points = [
        (0.0,  1.0, 0), // A
        (0.0, -1.0, 0), // B
        (1.0,  0.0, 1), // C
        (-1.0, 0.0, 1), // D
        (0.0,  0.0, 2), // E
    ];
    let mut b_rr = [Vec24::zeros(); 5];
    let mut b_ss = [Vec24::zeros(); 5];
    let mut b_rs = [Vec24::zeros(); 5];

    for (i, &(xi, eta, comp)) in tying_points.iter().enumerate() {
        // We need the row for EACH component at this point
        b_rr[i] = compute_covariant_membrane_b_row(&coords_3d, &e1_ref, &e2_ref, &e3_ref, xi, eta, 0);
        b_ss[i] = compute_covariant_membrane_b_row(&coords_3d, &e1_ref, &e2_ref, &e3_ref, xi, eta, 1);
        b_rs[i] = compute_covariant_membrane_b_row(&coords_3d, &e1_ref, &e2_ref, &e3_ref, xi, eta, 2);
    }

    let k_drill = e_mod * thickness * thickness * 0.15 * drilling_scale;

    Mitc4Precomputed {
        local_coords: compute_local_coords_centroid(&coords_3d, &e1_ref, &e2_ref),
        t3,
        constitutive,
        k_drill,
        thickness,
        a_a, a_b, a_c, a_d, a_e,
        x_r, x_s, x_d, n_vec, m_r, m_s,
        initial_coords_3d: coords_3d,
        gp_jacobians,
        gp_bubble,
        b_rr, b_ss, b_rs,

    }

    }
}

// ============================================================================
// B-matrices at a Gauss point
// ============================================================================

/// MITC4+ assumed membrane strain B-matrix (3×24) at (xi, eta)
fn b_m_mitc4_plus(
    pre: &Mitc4Precomputed,
    xi: f64,
    eta: f64,
    e1: &Vector3<f64>,
    e2: &Vector3<f64>,
) -> SMatrix<f64, 3, 24> {
    let r = xi;
    let s = eta;

    let a_a = pre.a_a;
    let a_b = pre.a_b;
    let a_c = pre.a_c;
    let a_d = pre.a_d;
    let a_e = pre.a_e;

    // Blended covariant B-rows (Ko et al. 2017, Eqs. 27a-c)
    let w_a = 0.5 * (1.0 - 2.0 * a_a + s + 2.0 * a_a * s * s);
    let w_b = 0.5 * (1.0 - 2.0 * a_b - s + 2.0 * a_b * s * s);
    let w_c = a_c * (-1.0 + s * s);
    let w_d = a_d * (-1.0 + s * s);
    let w_e = a_e * (-1.0 + s * s);
    let sum_w = w_a + w_b + w_c + w_d + w_e;
    let norm = if sum_w.abs() > 1e-12 { 1.0 / sum_w } else { 1.0 };

    let b_rr = norm * (
        w_a * pre.b_rr[0] + w_b * pre.b_rr[1] + w_c * pre.b_rr[2] + w_d * pre.b_rr[3] + w_e * pre.b_rr[4]
    );

    let w_a_s = a_a * (-1.0 + r * r);
    let w_b_s = a_b * (-1.0 + r * r);
    let w_c_s = 0.5 * (1.0 - 2.0 * a_c + r + 2.0 * a_c * r * r);
    let w_d_s = 0.5 * (1.0 - 2.0 * a_d - r + 2.0 * a_d * r * r);
    let w_e_s = a_e * (-1.0 + r * r);
    let sum_w_s = w_a_s + w_b_s + w_c_s + w_d_s + w_e_s;
    let norm_s = if sum_w_s.abs() > 1e-12 { 1.0 / sum_w_s } else { 1.0 };

    let b_ss = norm_s * (
        w_a_s * pre.b_ss[0] + w_b_s * pre.b_ss[1] + w_c_s * pre.b_ss[2] + w_d_s * pre.b_ss[3] + w_e_s * pre.b_ss[4]
    );

    let w_a_rs = 0.25 * (r + 4.0 * a_a * r * s);
    let w_b_rs = 0.25 * (-r + 4.0 * a_b * r * s);
    let w_c_rs = 0.25 * (s + 4.0 * a_c * r * s);
    let w_d_rs = 0.25 * (-s + 4.0 * a_d * r * s);
    let w_e_rs = (1.0 + a_e * r * s);
    let sum_w_rs = w_a_rs + w_b_rs + w_c_rs + w_d_rs + w_e_rs;
    let norm_rs = if sum_w_rs.abs() > 1e-12 { 1.0 / sum_w_rs } else { 1.0 };

    let b_rs = norm_rs * (
        w_a_rs * pre.b_rs[0] + w_b_rs * pre.b_rs[1] + w_c_rs * pre.b_rs[2] + w_d_rs * pre.b_rs[3] + w_e_rs * pre.b_rs[4]
    );

    // Stack: B_covariant = [B_rr; B_ss; 2*B_rs] (3×24)
    let mut b_cov = SMatrix::<f64, 3, 24>::zeros();
    for j in 0..24 {
        b_cov[(0, j)] = b_rr[j];
        b_cov[(1, j)] = b_ss[j];
        b_cov[(2, j)] = 2.0 * b_rs[j];
    }

    // Transform covariant → local orthonormal frame using the PROVIDED frame
    let (j_loc, _) = compute_j_loc_at(&pre.initial_coords_3d, e1, e2, xi, eta);
    let t = covariant_to_local_mapping(&j_loc);
    t * b_cov
}

/// Projected membrane B-matrix (3×24) that maps global displacements to local strains
fn b_m_projected(dh: &SMatrix<f64, 2, 4>, e1: &Vector3<f64>, e2: &Vector3<f64>) -> SMatrix<f64, 3, 24> {
    let mut bm = SMatrix::<f64, 3, 24>::zeros();
    for i in 0..4 {
        let u_idx = 6 * i;
        let d_n_dx1 = dh[(0, i)];
        let d_n_dx2 = dh[(1, i)];

        // eps_11 = dN/dx1 * (u . e1)
        bm[(0, u_idx)]     = d_n_dx1 * e1[0];
        bm[(0, u_idx + 1)] = d_n_dx1 * e1[1];
        bm[(0, u_idx + 2)] = d_n_dx1 * e1[2];

        // eps_22 = dN/dx2 * (u . e2)
        bm[(1, u_idx)]     = d_n_dx2 * e2[0];
        bm[(1, u_idx + 1)] = d_n_dx2 * e2[1];
        bm[(1, u_idx + 2)] = d_n_dx2 * e2[2];

        // gamma_12 = dN/dx1 * (u . e2) + dN/dx2 * (u . e1)
        bm[(2, u_idx)]     = d_n_dx1 * e2[0] + d_n_dx2 * e1[0];
        bm[(2, u_idx + 1)] = d_n_dx1 * e2[1] + d_n_dx2 * e1[1];
        bm[(2, u_idx + 2)] = d_n_dx1 * e2[2] + d_n_dx2 * e1[2];
    }
    bm
}

/// Bending curvature B-matrix (3×24) at a GP
fn b_kappa(dh: &SMatrix<f64, 2, 4>) -> SMatrix<f64, 3, 24> {
    let mut bk = SMatrix::<f64, 3, 24>::zeros();
    for i in 0..4 {
        let thx_idx = 6 * i + 3;
        let thy_idx = 6 * i + 4;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        bk[(0, thy_idx)] =  dni_dx;
        bk[(1, thx_idx)] = -dni_dy;
        bk[(2, thy_idx)] =  dni_dy;
        bk[(2, thx_idx)] = -dni_dx;
    }
    bk
}

/// Bending curvature contribution from bubble (3×2)
fn b_kappa_bubble(j_inv: &Matrix2<f64>, dnb_dxi: f64, dnb_deta: f64) -> SMatrix<f64, 3, 2> {
    let dnb_dx = j_inv[(0, 0)] * dnb_dxi + j_inv[(0, 1)] * dnb_deta;
    let dnb_dy = j_inv[(1, 0)] * dnb_dxi + j_inv[(1, 1)] * dnb_deta;

    let mut bkb = SMatrix::<f64, 3, 2>::zeros();
    bkb[(0, 1)] =  dnb_dx;
    bkb[(1, 0)] = -dnb_dy;
    bkb[(2, 0)] = -dnb_dx;
    bkb[(2, 1)] =  dnb_dy;
    bkb
}

/// MITC4 transverse shear B-matrix (2×24) — rotation-based formulation.
///
/// `area_measure` is the surface Jacobian |g_r × g_s| at the integration point.
fn b_gamma_mitc4(local_coords: &[[f64; 2]; 4], xi: f64, eta: f64, area_measure: f64) -> SMatrix<f64, 2, 24> {
    let xl = local_coords;

    let dx34 = xl[2][0] - xl[3][0];
    let dy34 = xl[2][1] - xl[3][1];
    let dx21 = xl[1][0] - xl[0][0];
    let dy21 = xl[1][1] - xl[0][1];
    let dx32 = xl[2][0] - xl[1][0];
    let dy32 = xl[2][1] - xl[1][1];
    let dx41 = xl[3][0] - xl[0][0];
    let dy41 = xl[3][1] - xl[0][1];

    let qtr = 0.25;

    // G matrix (4×12): edge-based shear strain interpolation
    let mut g = SMatrix::<f64, 4, 12>::zeros();

    // Edge 4-1
    g[(0, 0)] = -0.5;
    g[(0, 1)] = -dy41 * qtr;
    g[(0, 2)] =  dx41 * qtr;
    g[(0, 9)] =  0.5;
    g[(0, 10)] = -dy41 * qtr;
    g[(0, 11)] =  dx41 * qtr;

    // Edge 1-2
    g[(1, 0)] = -0.5;
    g[(1, 1)] = -dy21 * qtr;
    g[(1, 2)] =  dx21 * qtr;
    g[(1, 3)] =  0.5;
    g[(1, 4)] = -dy21 * qtr;
    g[(1, 5)] =  dx21 * qtr;

    // Edge 2-3
    g[(2, 3)] = -0.5;
    g[(2, 4)] = -dy32 * qtr;
    g[(2, 5)] =  dx32 * qtr;
    g[(2, 6)] =  0.5;
    g[(2, 7)] = -dy32 * qtr;
    g[(2, 8)] =  dx32 * qtr;

    // Edge 3-4
    g[(3, 6)] =  0.5;
    g[(3, 7)] = -dy34 * qtr;
    g[(3, 8)] =  dx34 * qtr;
    g[(3, 9)] = -0.5;
    g[(3, 10)] = -dy34 * qtr;
    g[(3, 11)] =  dx34 * qtr;

    // Ax, Bx, Cx, Ay, By, Cy from node coords
    let ax = -xl[0][0] + xl[1][0] + xl[2][0] - xl[3][0];
    let bx =  xl[0][0] - xl[1][0] + xl[2][0] - xl[3][0];
    let cx = -xl[0][0] - xl[1][0] + xl[2][0] + xl[3][0];

    let ay = -xl[0][1] + xl[1][1] + xl[2][1] - xl[3][1];
    let by =  xl[0][1] - xl[1][1] + xl[2][1] - xl[3][1];
    let cy = -xl[0][1] - xl[1][1] + xl[2][1] + xl[3][1];

    let alph = ay.atan2(ax);
    let beta = std::f64::consts::FRAC_PI_2 - cx.atan2(cy);

    let rot = Matrix2::new(
        beta.sin(),  -alph.sin(),
       -beta.cos(),   alph.cos(),
    );

    // Ms matrix (2×4)
    let mut ms = SMatrix::<f64, 2, 4>::zeros();
    ms[(1, 0)] = 1.0 - xi;
    ms[(0, 1)] = 1.0 - eta;
    ms[(1, 2)] = 1.0 + xi;
    ms[(0, 3)] = 1.0 + eta;

    // Bsv = Ms @ G  (2×12)
    let mut bsv = ms * g;

    // Scale factors
    let r1_vec = Vector2::new(cx + xi * bx, cy + xi * by);
    let r1 = r1_vec.norm();
    let r2_vec = Vector2::new(ax + eta * bx, ay + eta * by);
    let r2 = r2_vec.norm();

    for j in 0..12 {
        bsv[(0, j)] *= r1 / (8.0 * area_measure);
        bsv[(1, j)] *= r2 / (8.0 * area_measure);
    }

    // Apply rotation
    let bs_12 = rot * bsv;

    // Scatter from 12-DOF (w, θx, θy per node) to 24-DOF
    let mut bs = SMatrix::<f64, 2, 24>::zeros();
    for i in 0..4 {
        bs[(0, 6 * i + 2)] = bs_12[(0, 3 * i)];
        bs[(0, 6 * i + 3)] = bs_12[(0, 3 * i + 1)];
        bs[(0, 6 * i + 4)] = bs_12[(0, 3 * i + 2)];
        bs[(1, 6 * i + 2)] = bs_12[(1, 3 * i)];
        bs[(1, 6 * i + 3)] = bs_12[(1, 3 * i + 1)];
        bs[(1, 6 * i + 4)] = bs_12[(1, 3 * i + 2)];
    }

    bs
}

/// MITC4+ shear with bubble enrichment.
/// Returns (Bs_nodal [2×24], Bs_bubble [2×2])
fn b_gamma_mitc4_plus(
    local_coords: &[[f64; 2]; 4],
    xi: f64,
    eta: f64,
    area_measure: f64,
    nb: f64,
) -> (SMatrix<f64, 2, 24>, Matrix2<f64>) {
    let bs_nodal = b_gamma_mitc4(local_coords, xi, eta, area_measure);

    let mut bs_bubble = Matrix2::zeros();
    bs_bubble[(0, 1)] =  nb;
    bs_bubble[(1, 0)] = -nb;

    (bs_nodal, bs_bubble)
}

/// Drilling B-vector (Hughes & Brezzi, 1×24)
fn b_drill(dh: &SMatrix<f64, 2, 4>, n_vals: &[f64; 4]) -> Vec24 {
    let mut bd = Vec24::zeros();
    for i in 0..4 {
        let u_idx = 6 * i;
        let v_idx = 6 * i + 1;
        let thz_idx = 6 * i + 5;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        bd[u_idx]   = -0.5 * dni_dy;
        bd[v_idx]   =  0.5 * dni_dx;
        bd[thz_idx] = -n_vals[i];
    }
    bd
}

// ============================================================================
// Stiffness matrix computation
// ============================================================================

/// Compute element stiffness matrix K_e in LOCAL coordinates (24×24)
pub fn compute_ke_local(pre: &Mitc4Precomputed) -> Mat24 {
    let cm = &pre.constitutive.cm;
    let cb = &pre.constitutive.cb;
    let cs = &pre.constitutive.cs;

    // --- Membrane stiffness: Projected 3D linear membrane ---
    let mut k_m = Mat24::zeros();
    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let gj = &pre.gp_jacobians[g];
        let w = GAUSS_W[g];

        let (e1, e2, _) = compute_local_frame_at(&pre.initial_coords_3d, xi, eta);
        let bm = b_m_projected(&gj.dh, &e1, &e2);

        k_m += (bm.transpose() * cm * &bm) * (w * gj.sqrt_g);
    }

    // --- Bending + shear with bubble condensation ---
    let mut knn_b = Mat24::zeros();
    let mut knb_b = SMatrix::<f64, 24, 2>::zeros();
    let mut kbb_b = Matrix2::zeros();

    let mut knn_s = Mat24::zeros();
    let mut knb_s = SMatrix::<f64, 24, 2>::zeros();
    let mut kbb_s = Matrix2::zeros();

    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let gj = &pre.gp_jacobians[g];
        let gb = &pre.gp_bubble[g];
        let sqrt_g = gj.sqrt_g;
        let w = GAUSS_W[g];

        // Bending
        let bk = b_kappa(&gj.dh);
        let bkb = b_kappa_bubble(&gj.j_inv, gb.dnb_dxi, gb.dnb_deta);

        knn_b += (bk.transpose() * cb * &bk) * (w * sqrt_g);
        knb_b += (bk.transpose() * cb * &bkb) * (w * sqrt_g);
        kbb_b += (bkb.transpose() * cb * &bkb) * (w * sqrt_g);

        // Shear
        let (e1, e2, _) = compute_local_frame_at(&pre.initial_coords_3d, xi, eta);
        let local_coords = compute_local_coords_centroid(&pre.initial_coords_3d, &e1, &e2);
        let (bs_nodal, bs_bubble) = b_gamma_mitc4_plus(
            &local_coords, xi, eta, sqrt_g, gb.nb,
        );

        knn_s += (bs_nodal.transpose() * cs * &bs_nodal) * (w * sqrt_g);
        knb_s += (bs_nodal.transpose() * cs * &bs_bubble) * (w * sqrt_g);
        kbb_s += (bs_bubble.transpose() * cs * &bs_bubble) * (w * sqrt_g);
    }

    let knn = knn_b + knn_s;
    let knb = knb_b + knb_s;
    let kbb = kbb_b + kbb_s;

    // Static condensation: K = Knn - Knb @ Kbb^{-1} @ Knb^T
    // Regularize the 2×2 bubble block instead of dropping condensation entirely.
    let kbb_inv = regularized_inverse_2x2(&kbb);
    let k_bs_raw = knn - &knb * kbb_inv * knb.transpose();
    let k_bs = 0.5 * (&k_bs_raw + k_bs_raw.transpose());

    // --- Drilling stiffness ---
    let mut k_drill = Mat24::zeros();
    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let gj = &pre.gp_jacobians[g];
        let sqrt_g = gj.sqrt_g;
        let w = GAUSS_W[g];

        let n_vals = shape_functions(xi, eta);
        let bd = b_drill(&gj.dh, &n_vals);

        k_drill += (&bd * bd.transpose()) * (pre.k_drill * w * sqrt_g);
    }

    k_m + k_bs + k_drill
}

/// Compute element stiffness matrix in GLOBAL coordinates (24×24)
pub fn compute_ke_global(pre: &Mitc4Precomputed) -> Mat24 {
    let k_local = compute_ke_local(pre);
    transform_to_global(pre, &k_local)
}

// ============================================================================
// Mass matrix
// ============================================================================

/// Compute consistent mass matrix in GLOBAL coordinates (24×24)
pub fn compute_me_global(pre: &Mitc4Precomputed, rho: f64) -> Mat24 {
    let m_trans = rho * pre.thickness;
    let m_rot = rho * pre.thickness.powi(3) / 12.0;

    let mut m_local = Mat24::zeros();

    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let sqrt_g = pre.gp_jacobians[g].sqrt_g;
        let w = GAUSS_W[g];

        let n = shape_functions(xi, eta);

        for i in 0..4 {
            for j in 0..4 {
                let val_t = n[i] * n[j] * m_trans * w * sqrt_g;
                for k in 0..3 {
                    m_local[(6 * i + k, 6 * j + k)] += val_t;
                }

                let val_r = n[i] * n[j] * m_rot * w * sqrt_g;
                for k in 3..6 {
                    m_local[(6 * i + k, 6 * j + k)] += val_r;
                }
            }
        }
    }

    transform_to_global(pre, &m_local)
}

/// Compute consistent mass matrix for composite elements (global coords, 24×24).
///
/// Uses pre-computed ply-integrated mass parameters instead of ρ·h.
pub fn compute_me_composite_global(
    pre: &Mitc4Precomputed,
    mass_per_area: f64,
    rotational_inertia: f64,
) -> Mat24 {
    let m_trans = mass_per_area;
    let m_rot = rotational_inertia;

    let mut m_local = Mat24::zeros();

    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let sqrt_g = pre.gp_jacobians[g].sqrt_g;
        let w = GAUSS_W[g];

        let n = shape_functions(xi, eta);

        for i in 0..4 {
            for j in 0..4 {
                let val_t = n[i] * n[j] * m_trans * w * sqrt_g;
                for k in 0..3 {
                    m_local[(6 * i + k, 6 * j + k)] += val_t;
                }

                let val_r = n[i] * n[j] * m_rot * w * sqrt_g;
                for k in 3..6 {
                    m_local[(6 * i + k, 6 * j + k)] += val_r;
                }
            }
        }
    }

    transform_to_global(pre, &m_local)
}

// ============================================================================
// Nonlinear: displacement gradient, GL strain, tangent stiffness, forces
// ============================================================================

/// Compute displacement gradient H = du/dX at GP g
fn displacement_gradient(dh: &SMatrix<f64, 2, 4>, u: &Vec24) -> Matrix3<f64> {
    let mut u_nodes = [[0.0f64; 3]; 4];
    for i in 0..4 {
        u_nodes[i][0] = u[6 * i];
        u_nodes[i][1] = u[6 * i + 1];
        u_nodes[i][2] = u[6 * i + 2];
    }

    let mut h = Matrix3::zeros();
    for j in 0..2 {
        for comp in 0..3 {
            let mut val = 0.0;
            for nd in 0..4 {
                val += u_nodes[nd][comp] * dh[(j, nd)];
            }
            h[(comp, j)] = val;
        }
    }
    h
}

/// Green-Lagrange strain E = 0.5*(H + H^T + H^T @ H)
fn green_lagrange_strain(h: &Matrix3<f64>) -> Matrix3<f64> {
    0.5 * (h + h.transpose() + h.transpose() * h)
}

/// Linear strain-displacement B_L (6×24): [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]
fn compute_b_l(dh: &SMatrix<f64, 2, 4>) -> SMatrix<f64, 6, 24> {
    let mut bl = SMatrix::<f64, 6, 24>::zeros();
    for i in 0..4 {
        let col = 6 * i;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        bl[(0, col)]     = dni_dx;      // ε_xx = ∂u/∂x
        bl[(1, col + 1)] = dni_dy;      // ε_yy = ∂v/∂y
        bl[(3, col)]     = dni_dy;      // γ_xy = ∂u/∂y
        bl[(3, col + 1)] = dni_dx;      // γ_xy += ∂v/∂x
    }
    bl
}

/// Nonlinear strain-displacement B_NL (6×24)
fn compute_b_nl(dh: &SMatrix<f64, 2, 4>, h: &Matrix3<f64>) -> SMatrix<f64, 6, 24> {
    let mut bnl = SMatrix::<f64, 6, 24>::zeros();
    for i in 0..4 {
        let col = 6 * i;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        // E_xx nonlinear
        bnl[(0, col)]     = h[(0, 0)] * dni_dx;
        bnl[(0, col + 1)] = h[(1, 0)] * dni_dx;
        bnl[(0, col + 2)] = h[(2, 0)] * dni_dx;

        // E_yy nonlinear
        bnl[(1, col)]     = h[(0, 1)] * dni_dy;
        bnl[(1, col + 1)] = h[(1, 1)] * dni_dy;
        bnl[(1, col + 2)] = h[(2, 1)] * dni_dy;

        // 2*E_xy nonlinear
        bnl[(3, col)]     = h[(0, 0)] * dni_dy + h[(0, 1)] * dni_dx;
        bnl[(3, col + 1)] = h[(1, 0)] * dni_dy + h[(1, 1)] * dni_dx;
        bnl[(3, col + 2)] = h[(2, 0)] * dni_dy + h[(2, 1)] * dni_dx;
    }
    bnl
}

/// Geometric B matrix (6×24) for stress stiffness
fn compute_b_geometric(dh: &SMatrix<f64, 2, 4>) -> SMatrix<f64, 6, 24> {
    let mut bg = SMatrix::<f64, 6, 24>::zeros();
    for i in 0..4 {
        let col = 6 * i;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        bg[(0, col)]     = dni_dx;  // ∂u/∂x
        bg[(1, col)]     = dni_dy;  // ∂u/∂y
        bg[(2, col + 1)] = dni_dx;  // ∂v/∂x
        bg[(3, col + 1)] = dni_dy;  // ∂v/∂y
        bg[(4, col + 2)] = dni_dx;  // ∂w/∂x
        bg[(5, col + 2)] = dni_dy;  // ∂w/∂y
    }
    bg
}

/// Extract membrane rows [0,1,3] from a 6×24 matrix → 3×24
fn extract_membrane_rows(b6: &SMatrix<f64, 6, 24>) -> SMatrix<f64, 3, 24> {
    let mut b3 = SMatrix::<f64, 3, 24>::zeros();
    for j in 0..24 {
        b3[(0, j)] = b6[(0, j)];
        b3[(1, j)] = b6[(1, j)];
        b3[(2, j)] = b6[(3, j)];
    }
    b3
}

/// Compute tangent stiffness K_T in GLOBAL coordinates.
///
/// K_T = K_0 + K_L + K_sigma  (Total Lagrangian)
pub fn compute_kt_global(pre: &Mitc4Precomputed, u_global: &Vec24) -> Mat24 {
    // Transform displacements to local
    let t24 = build_t24(pre);
    let u_local = &t24 * u_global;

    // Linear stiffness (local)
    let k0 = compute_ke_local(pre);

    // Cm_raw = E/(1-ν²) * [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]
    let cm_raw = &pre.constitutive.cm_raw;

    // K_L: initial displacement stiffness
    let mut k_l = Mat24::zeros();

    for g in 0..N_GAUSS {
        let gj = &pre.gp_jacobians[g];
        let sqrt_g = gj.sqrt_g;
        let w = GAUSS_W[g];

        let h_mat = displacement_gradient(&gj.dh, &u_local);
        let bl = compute_b_l(&gj.dh);
        let bnl = compute_b_nl(&gj.dh, &h_mat);

        let bm_l = extract_membrane_rows(&bl);
        let bm_nl = extract_membrane_rows(&bnl);

        k_l += (
            bm_l.transpose() * cm_raw * &bm_nl
            + bm_nl.transpose() * cm_raw * &bm_l
            + bm_nl.transpose() * cm_raw * &bm_nl
        ) * (w * sqrt_g * pre.thickness);
    }

    // Compute stress from displacement for K_sigma
    let sigma = compute_membrane_stress(pre, &u_local);

    // K_sigma: geometric stiffness
    let k_sigma = compute_geometric_stiffness_local(pre, &sigma);

    let k_t = k0 + k_l + k_sigma;

    // Symmetrize & transform
    let k_t_sym = 0.5 * (&k_t + k_t.transpose());
    t24.transpose() * &k_t_sym * &t24
}

/// Compute membrane stress from local displacements at element center
fn compute_membrane_stress(pre: &Mitc4Precomputed, u_local: &Vec24) -> Vector3<f64> {
    let cm_raw = &pre.constitutive.cm_raw;

    // Evaluate at element center using the covariant MITC4+ membrane B-matrix.
    let (e1, e2, _) = compute_local_frame_at(&pre.initial_coords_3d, 0.0, 0.0);
    let bm = b_m_mitc4_plus(pre, 0.0, 0.0, &e1, &e2);
    let eps_m = bm * u_local;
    cm_raw * eps_m
}

/// Compute geometric stiffness K_sigma in LOCAL coordinates
fn compute_geometric_stiffness_local(pre: &Mitc4Precomputed, sigma: &Vector3<f64>) -> Mat24 {
    let s_m = Matrix2::new(
        sigma[0], sigma[2],
        sigma[2], sigma[1],
    ) * pre.thickness;

    let mut s_tilde = SMatrix::<f64, 6, 6>::zeros();
    for i in 0..2 {
        for j in 0..2 {
            s_tilde[(i, j)]     = s_m[(i, j)];
            s_tilde[(i + 2, j + 2)] = s_m[(i, j)];
            s_tilde[(i + 4, j + 4)] = s_m[(i, j)];
        }
    }

    let mut k_sigma = Mat24::zeros();
    for g in 0..N_GAUSS {
        let gj = &pre.gp_jacobians[g];
        let sqrt_g = gj.sqrt_g;
        let w = GAUSS_W[g];

        let bg = compute_b_geometric(&gj.dh);
        k_sigma += (bg.transpose() * &s_tilde * &bg) * (w * sqrt_g);
    }

    0.5 * (&k_sigma + k_sigma.transpose())
}

/// Compute internal forces in GLOBAL coordinates.
pub fn compute_fint_global(pre: &Mitc4Precomputed, u_global: &Vec24, nonlinear: bool) -> Vec24 {
    let t24 = build_t24(pre);
    let u_local = &t24 * u_global;

    let f_local = if !nonlinear {
        let k_local = compute_ke_local(pre);
        k_local * &u_local
    } else {
        let mut f = Vec24::zeros();

        // Bending+shear (linear, condensed) + drilling
        let k_bs = compute_bending_shear_condensed(pre);
        let k_dr = compute_drilling_stiffness(pre);
        f += (k_bs + k_dr) * &u_local;

        // Membrane with GL strain — use MITC4+ assumed B for consistency with K
        let cm_raw = &pre.constitutive.cm_raw;
        for g in 0..N_GAUSS {
            let gj = &pre.gp_jacobians[g];
            let xi = GAUSS_XI[g];
            let eta = GAUSS_ETA[g];
            let w = GAUSS_W[g];
            let sqrt_g = gj.sqrt_g;

            let h_mat = displacement_gradient(&gj.dh, &u_local);
            let e_gl = green_lagrange_strain(&h_mat);
            let eps_m = Vector3::new(e_gl[(0, 0)], e_gl[(1, 1)], 2.0 * e_gl[(0, 1)]);
            let sigma_m = cm_raw * eps_m;

            let (e1, e2, _) = compute_local_frame_at(&pre.initial_coords_3d, xi, eta);
            let bm_l = b_m_mitc4_plus(pre, xi, eta, &e1, &e2);
            let bnl = compute_b_nl(&gj.dh, &h_mat);
            let bm_nl = extract_membrane_rows(&bnl);
            let b_total = bm_l + bm_nl;

            f += b_total.transpose() * sigma_m * (w * sqrt_g * pre.thickness);
        }
        f
    };

    t24.transpose() * f_local
}

// ============================================================================
// Helper: bending+shear condensed (separate from ke for f_int)
// ============================================================================

fn compute_bending_shear_condensed(pre: &Mitc4Precomputed) -> Mat24 {
    let cb = &pre.constitutive.cb;
    let cs = &pre.constitutive.cs;

    let mut knn_b = Mat24::zeros();
    let mut knb_b = SMatrix::<f64, 24, 2>::zeros();
    let mut kbb_b = Matrix2::zeros();

    let mut knn_s = Mat24::zeros();
    let mut knb_s = SMatrix::<f64, 24, 2>::zeros();
    let mut kbb_s = Matrix2::zeros();

    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let gj = &pre.gp_jacobians[g];
        let gb = &pre.gp_bubble[g];
        let sqrt_g = gj.sqrt_g;
        let w = GAUSS_W[g];

        let bk = b_kappa(&gj.dh);
        let bkb = b_kappa_bubble(&gj.j_inv, gb.dnb_dxi, gb.dnb_deta);

        knn_b += (bk.transpose() * cb * &bk) * (w * sqrt_g);
        knb_b += (bk.transpose() * cb * &bkb) * (w * sqrt_g);
        kbb_b += (bkb.transpose() * cb * &bkb) * (w * sqrt_g);

        let (bs_nodal, bs_bubble) = b_gamma_mitc4_plus(
            &pre.local_coords, xi, eta, sqrt_g, gb.nb,
        );

        knn_s += (bs_nodal.transpose() * cs * &bs_nodal) * (w * sqrt_g);
        knb_s += (bs_nodal.transpose() * cs * &bs_bubble) * (w * sqrt_g);
        kbb_s += (bs_bubble.transpose() * cs * &bs_bubble) * (w * sqrt_g);
    }

    let knn = knn_b + knn_s;
    let knb = knb_b + knb_s;
    let kbb = kbb_b + kbb_s;

    let kbb_inv = regularized_inverse_2x2(&kbb);
    knn - &knb * kbb_inv * knb.transpose()
}

fn compute_drilling_stiffness(pre: &Mitc4Precomputed) -> Mat24 {
    let mut k_drill = Mat24::zeros();
    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let gj = &pre.gp_jacobians[g];
        let sqrt_g = gj.sqrt_g;
        let w = GAUSS_W[g];

        let n_vals = shape_functions(xi, eta);
        let bd = b_drill(&gj.dh, &n_vals);

        k_drill += (&bd * bd.transpose()) * (pre.k_drill * w * sqrt_g);
    }
    k_drill
}

// ============================================================================
// Transformation
// ============================================================================

/// Build 24×24 global-to-local transformation matrix
fn build_t24(pre: &Mitc4Precomputed) -> SMatrix<f64, 24, 24> {
    let mut t24 = SMatrix::<f64, 24, 24>::zeros();
    for i in 0..8 {
        let r = 3 * i;
        for a in 0..3 {
            for b in 0..3 {
                t24[(r + a, r + b)] = pre.t3[(a, b)];
            }
        }
    }
    t24
}

/// Transform a 24×24 local matrix to global coordinates: T^T @ M @ T
fn transform_to_global(pre: &Mitc4Precomputed, m_local: &Mat24) -> Mat24 {
    let t24 = build_t24(pre);
    let k_global = t24.transpose() * m_local * &t24;
    0.5 * (&k_global + k_global.transpose())
}

// ============================================================================
// Body load
// ============================================================================

/// Compute body load vector f_body (24) in GLOBAL coordinates.
///
/// Integrates `f = ∫ Nᵀ · (ρ · h · g) dA` using 2×2 Gauss quadrature.
/// Only translational DOFs (indices 0,1,2 of each node block) receive
/// contributions; rotational DOFs (3,4,5) are zero.
///
/// `gravity` is the body-force acceleration vector in global coordinates [gx, gy, gz].
pub fn compute_body_load_global(
    pre: &Mitc4Precomputed,
    rho: f64,
    gravity: &Vector3<f64>,
) -> Vec24 {
    let h = pre.thickness;
    let rho_h = rho * h;

    // Transform gravity to local coordinates once
    let g_local = pre.t3 * gravity;

    let mut f_local = Vec24::zeros();

    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let sqrt_g = pre.gp_jacobians[g].sqrt_g;
        let w = GAUSS_W[g];

        let n = shape_functions(xi, eta);

        for i in 0..4 {
            let contrib = n[i] * rho_h * w * sqrt_g;
            // Translational DOFs only (0,1,2 of each 6-DOF node block)
            for k in 0..3 {
                f_local[6 * i + k] += contrib * g_local[k];
            }
        }
    }

    // Transform to global using T24^T
    let t24 = build_t24(pre);
    t24.transpose() * f_local
}

// ============================================================================
// Geometric stiffness (global)
// ============================================================================

/// Compute geometric stiffness K_σ (24×24) in GLOBAL coordinates.
///
/// `sigma_membrane` = [σxx, σyy, σxy] in local coordinates.
pub fn compute_k_sigma_global(
    pre: &Mitc4Precomputed,
    sigma_membrane: &Vector3<f64>,
) -> Mat24 {
    let k_local = compute_geometric_stiffness_local(pre, sigma_membrane);
    transform_to_global(pre, &k_local)
}

// ============================================================================
// Centrifugal prestress
// ============================================================================

/// Compute centrifugal prestress [σxx, σyy, σxy] in local coordinates.
///
/// Returns the membrane stress state due to centrifugal loading:
///     σ_cf ≈ ρ · ω² · r_radial · L_char
///
/// where L_char = √(element area), computed from the sum of Gauss-point areas.
///
/// `omega`           : angular velocity (rad/s)
/// `rotation_axis`   : unit vector of rotation axis in global coords
/// `rotation_center` : a point on the rotation axis in global coords
/// `centroid`        : element centroid in global coords
/// `rho`             : material density (kg/m³)
pub fn compute_centrifugal_prestress(
    pre: &Mitc4Precomputed,
    omega: f64,
    rotation_axis: &Vector3<f64>,
    rotation_center: &Vector3<f64>,
    centroid: &Vector3<f64>,
    rho: f64,
) -> Vector3<f64> {
    // Normalize rotation axis
    let axis = rotation_axis.normalize();

    // Vector from rotation center to centroid
    let r_vec = centroid - rotation_center;

    // Project out the component along the axis → radial vector
    let r_parallel = r_vec.dot(&axis) * axis;
    let r_radial_vec = r_vec - r_parallel;
    let r_radial = r_radial_vec.norm();

    if r_radial < 1.0e-10 {
        // Element is on the rotation axis — no centrifugal stress
        return Vector3::zeros();
    }

    let radial_dir = r_radial_vec / r_radial;

    // Element area: sum of sqrt_g * w over all Gauss points
    let area: f64 = (0..N_GAUSS).map(|g| pre.gp_jacobians[g].sqrt_g * GAUSS_W[g]).sum();

    // Characteristic element length and stress magnitude
    let l_char = area.sqrt();
    let sigma_cf = rho * omega * omega * r_radial * l_char;

    // Transform radial direction to local coordinates using the 3×3 rotation part
    let radial_local = pre.t3 * radial_dir;
    let cos_theta = radial_local[0];
    let sin_theta = radial_local[1];

    Vector3::new(
        sigma_cf * cos_theta * cos_theta,
        sigma_cf * sin_theta * sin_theta,
        sigma_cf * cos_theta * sin_theta,
    )
}

// ============================================================================
// Stress / Strain Recovery
// ============================================================================

/// Compute element stress and strain at the element centroid (xi=0, eta=0).
///
/// Returns `([sx, sy, sxy, 0, 0, 0], [exx, eyy, exy, 0, 0, 0])` in the element's
/// LOCAL coordinate system.
///
/// # Arguments
/// * `u_global` - 24-DOF global displacement vector
/// * `z_factor` - Normalized through-thickness position: 0.0=mid, ±0.5=top/bottom
/// * `stress_type` - 0=membrane only, 1=bending only, 2=total
pub fn compute_element_stress(
    pre: &Mitc4Precomputed,
    u_global: &Vec24,
    z_factor: f64,
    stress_type: u8,
) -> ([f64; 6], [f64; 6]) {
    let t24 = build_t24(pre);
    let u_local = t24 * u_global;

    // Evaluate at element centroid (xi=0, eta=0) using 3D covariant geometry
    let xi = 0.0_f64;
    let eta = 0.0_f64;
    let (e1, e2, _) = compute_local_frame_at(&pre.initial_coords_3d, xi, eta);
    let (_, j_inv) = compute_j_loc_at(&pre.initial_coords_3d, &e1, &e2, xi, eta);
    let (dn_dxi, dn_deta) = shape_function_derivatives(xi, eta);
    let mut dh = SMatrix::<f64, 2, 4>::zeros();
    for i in 0..4 {
        dh[(0, i)] = j_inv[(0, 0)] * dn_dxi[i] + j_inv[(1, 0)] * dn_deta[i];
        dh[(1, i)] = j_inv[(0, 1)] * dn_dxi[i] + j_inv[(1, 1)] * dn_deta[i];
    }
    
    let cm_raw = &pre.constitutive.cm_raw;
    let h = pre.thickness;
    
    // Membrane
    let bm = b_m_mitc4_plus(pre, xi, eta, &e1, &e2);
    let eps_m = bm * &u_local;
    let sig_m = cm_raw * eps_m;


    // Bending
    let bk = b_kappa(&dh);
    let kappa = bk * &u_local;
    let z = z_factor * h;
    let sig_b = cm_raw * kappa * z;

    // Total stress
    let sig: nalgebra::Vector3<f64> = match stress_type {
        0 => sig_m,
        1 => sig_b,
        _ => sig_m + sig_b,
    };

    // Total strain at centroid (membrane + bending)
    let eps_total: nalgebra::Vector3<f64> = match stress_type {
        0 => eps_m,
        1 => kappa * z,
        _ => eps_m + kappa * z,
    };

    let sigma6 = [sig[0], sig[1], 0.0, sig[2], 0.0, 0.0];
    let eps6 = [eps_total[0], eps_total[1], 0.0, eps_total[2], 0.0, 0.0];
    (sigma6, eps6)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::isotropic::IsotropicMaterial;
    use crate::materials::Material;

    /// Build a unit-square quad element in the XY plane.
    /// Nodes: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
    fn make_pre() -> Mitc4Precomputed {
        let thickness = 0.01_f64;
        let mat = IsotropicMaterial::new(2.0e11, 0.3, 7800.0);
        let shell = mat.constitutive(thickness, 5.0 / 6.0);
        let node_coords: [f64; 12] = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        Mitc4Precomputed::new(&node_coords, shell, thickness, 2.0e11, 1.0)
    }

    #[test]
    fn test_body_load_global_zero_gravity() {
        let pre = make_pre();
        let g = Vector3::zeros();
        let f = compute_body_load_global(&pre, 7800.0, &g);
        assert!(f.norm() < 1e-12, "zero gravity → zero body load");
    }

    #[test]
    fn test_body_load_global_z_gravity() {
        let pre = make_pre();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let f = compute_body_load_global(&pre, 7800.0, &g);

        // Only translational z-DOFs should be non-zero
        for i in 0..4 {
            assert!(f[6 * i].abs() < 1e-10, "node {i} fx should be ~0");
            assert!(f[6 * i + 1].abs() < 1e-10, "node {i} fy should be ~0");
            assert!(f[6 * i + 2].abs() > 1e-6, "node {i} fz should be nonzero");
            for k in 3..6 {
                assert!(f[6 * i + k].abs() < 1e-12, "node {i} rotational dof {k} should be 0");
            }
        }

        // Total z-force = ρ·h·|g|·area (area = 1.0 for unit square)
        let area = 1.0_f64;
        let h = 0.01_f64;
        let rho = 7800.0_f64;
        let expected_total_fz = rho * h * (-9.81) * area;
        let total_fz: f64 = (0..4).map(|i| f[6 * i + 2]).sum();
        assert!(
            (total_fz - expected_total_fz).abs() < 1e-4,
            "total fz: got {total_fz}, expected {expected_total_fz}"
        );
    }

    #[test]
    fn test_k_sigma_global_zero_stress() {
        let pre = make_pre();
        let sigma = Vector3::zeros();
        let k = compute_k_sigma_global(&pre, &sigma);
        assert!(k.norm() < 1e-12, "zero stress → zero K_sigma");
    }

    #[test]
    fn test_k_sigma_global_symmetric() {
        let pre = make_pre();
        let sigma = Vector3::new(1.0e6, 0.5e6, 0.2e6);
        let k = compute_k_sigma_global(&pre, &sigma);
        let diff = k - k.transpose();
        assert!(
            diff.norm() < 1e-6 * k.norm().max(1.0),
            "K_sigma_global must be symmetric"
        );
    }

    #[test]
    fn test_k_sigma_global_matches_local_transformed() {
        let pre = make_pre();
        let sigma = Vector3::new(1.0e6, 0.5e6, 0.2e6);
        let k_local = compute_geometric_stiffness_local(&pre, &sigma);
        let k_global_direct = compute_k_sigma_global(&pre, &sigma);
        let k_global_manual = transform_to_global(&pre, &k_local);
        let diff = k_global_direct - k_global_manual;
        assert!(diff.norm() < 1e-6, "compute_k_sigma_global must equal transform(k_local)");
    }

    #[test]
    fn test_centrifugal_prestress_on_axis() {
        let pre = make_pre();
        let axis = Vector3::new(0.0, 0.0, 1.0);
        // Place center at the centroid (0.5, 0.5, 0) → r_radial ≈ 0
        let center = Vector3::new(0.5, 0.5, 0.0);
        let centroid = Vector3::new(0.5, 0.5, 0.0);
        let sigma = compute_centrifugal_prestress(&pre, 100.0, &axis, &center, &centroid, 7800.0);
        assert!(sigma.norm() < 1e-6, "element on axis → zero centrifugal stress");
    }

    #[test]
    fn test_centrifugal_prestress_nonzero() {
        let pre = make_pre();
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let center = Vector3::zeros();
        let centroid = Vector3::new(0.5, 0.5, 0.0);
        let sigma = compute_centrifugal_prestress(&pre, 100.0, &axis, &center, &centroid, 7800.0);
        assert!(sigma[0].is_finite());
        assert!(sigma[1].is_finite());
        assert!(sigma[2].is_finite());
        assert!(sigma[0] + sigma[1] >= 0.0, "centrifugal stress trace must be non-negative");
    }

    #[test]
    fn test_ke_local_flat_plate_parity() {
        let pre = make_pre();
        let ke = compute_ke_local(&pre);
        
        // We expect Ke to be symmetric
        let diff = &ke - ke.transpose();
        assert!(diff.norm() < 1e-10, "Ke must be symmetric");
        
        // For a unit square, the membrane part of Ke should be non-zero
        assert!(ke.norm() > 1e-6, "Ke should not be zero");
    }
}
