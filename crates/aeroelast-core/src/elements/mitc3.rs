// MITC3+ Shell Element Kernel
//
// High-performance implementation of the MITC3+ triangular shell element
// (Lee, Lee & Bathe, 2014).
//
// All geometry-constant quantities are precomputed once per element.
// No heap allocations in the hot path — fixed-size nalgebra matrices throughout.

use nalgebra::{Matrix2, Matrix3, SMatrix, SVector, Vector2, Vector3};

use crate::materials::ShellConstitutive;

// ============================================================================
// Type aliases for fixed-size element matrices
// ============================================================================

/// 18×18 element stiffness/mass matrix (3 nodes × 6 DOFs)
pub type Mat18 = SMatrix<f64, 18, 18>;
/// 20×20 extended matrix (18 nodal + 2 bubble DOFs)
pub type Mat20 = SMatrix<f64, 20, 20>;
/// 18-element force/displacement vector
pub type Vec18 = SVector<f64, 18>;
/// 20-element extended vector
pub type Vec20 = SVector<f64, 20>;

// ============================================================================
// Gauss quadrature (Hammer points, degree 2 on triangle)
// ============================================================================

const N_GAUSS: usize = 3;

const GAUSS_R: [f64; N_GAUSS] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
const GAUSS_S: [f64; N_GAUSS] = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
const GAUSS_W: [f64; N_GAUSS] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

// ============================================================================
// Tying points for MITC3+ assumed shear strain
// ============================================================================

const D_PARAM: f64 = 1.0e-4;

// Constant part: A, B, C
const TP_A: (f64, f64) = (1.0 / 6.0, 2.0 / 3.0);
const TP_B: (f64, f64) = (2.0 / 3.0, 1.0 / 6.0);
const TP_C: (f64, f64) = (1.0 / 6.0, 1.0 / 6.0);

// Linear part: D, E, F
const TP_D: (f64, f64) = (1.0 / 3.0 + D_PARAM, 1.0 / 3.0 - 2.0 * D_PARAM);
const TP_E: (f64, f64) = (1.0 / 3.0 - 2.0 * D_PARAM, 1.0 / 3.0 + D_PARAM);
const TP_F: (f64, f64) = (1.0 / 3.0 + D_PARAM, 1.0 / 3.0 + D_PARAM);

// ============================================================================
// Precomputed element data (geometry-constant, computed once in `new`)
// ============================================================================

/// All data that is constant for a given element geometry.
/// Computed once, reused for every assembly call.
pub struct Mitc3Precomputed {
    /// Element area
    pub area: f64,
    /// Jacobian matrix (constant for linear triangle)
    pub j_mat: Matrix2<f64>,
    /// Jacobian inverse
    pub j_inv: Matrix2<f64>,
    /// Jacobian determinant
    pub det_j: f64,
    /// Shape function derivatives in Cartesian coords: dH[0,i] = dhi/dx, dH[1,i] = dhi/dy
    pub dh: Matrix2x3,
    /// Local coordinates (x1,y1, x2,y2, x3,y3)
    pub local_coords: [f64; 6],
    /// Transformation matrix local<->global (3×3 rotation part)
    pub t3: Matrix3<f64>,
    /// Covariant base vectors (constant for linear triangle)
    pub g_r: Vector2<f64>,
    pub g_s: Vector2<f64>,
    /// Constitutive matrices (material + thickness dependent, but constant per element)
    pub constitutive: ShellConstitutive,
    /// Drilling stiffness factor
    pub k_drill: f64,
    /// Thickness
    pub thickness: f64,
    /// Precomputed covariant shear at the 6 tying points (extended 20-DOF space)
    /// Each is (B_ert, B_est) as Vec20
    pub tying_ext_a: (Vec20, Vec20),
    pub tying_ext_b: (Vec20, Vec20),
    pub tying_ext_c: (Vec20, Vec20),
    pub tying_ext_d: (Vec20, Vec20),
    pub tying_ext_e: (Vec20, Vec20),
    pub tying_ext_f: (Vec20, Vec20),
}

/// 2×3 matrix type for shape function derivatives
type Matrix2x3 = SMatrix<f64, 2, 3>;

// ============================================================================
// Shape functions (pure, no state)
// ============================================================================

/// Linear shape functions: h1 = 1-r-s, h2 = r, h3 = s
#[inline(always)]
fn shape_functions(r: f64, s: f64) -> Vector3<f64> {
    Vector3::new(1.0 - r - s, r, s)
}

/// Bubble function: f4 = 27·r·s·(1-r-s)
#[inline(always)]
fn bubble(r: f64, s: f64) -> f64 {
    27.0 * r * s * (1.0 - r - s)
}

/// Bubble derivatives: (df4/dr, df4/ds)
#[inline(always)]
fn bubble_deriv(r: f64, s: f64) -> (f64, f64) {
    (
        27.0 * s * (1.0 - 2.0 * r - s),
        27.0 * r * (1.0 - r - 2.0 * s),
    )
}

/// Enriched shape functions: [f1, f2, f3, f4]
/// fi = hi - f4/3 for i=1,2,3
#[inline(always)]
fn enriched_shape(r: f64, s: f64) -> [f64; 4] {
    let h = shape_functions(r, s);
    let f4 = bubble(r, s);
    let f4_3 = f4 / 3.0;
    [h[0] - f4_3, h[1] - f4_3, h[2] - f4_3, f4]
}

/// Enriched shape function derivatives in natural coords
/// Returns (df/dr[4], df/ds[4])
#[inline(always)]
fn enriched_shape_deriv(r: f64, s: f64) -> ([f64; 4], [f64; 4]) {
    let (df4_dr, df4_ds) = bubble_deriv(r, s);
    let df4_dr_3 = df4_dr / 3.0;
    let df4_ds_3 = df4_ds / 3.0;
    (
        [-1.0 - df4_dr_3, 1.0 - df4_dr_3, -df4_dr_3, df4_dr],
        [-1.0 - df4_ds_3, -df4_ds_3, 1.0 - df4_ds_3, df4_ds],
    )
}

// ============================================================================
// Precomputation
// ============================================================================

impl Mitc3Precomputed {
    /// Create precomputed data for a MITC3+ element.
    ///
    /// `coords` is [x1,y1,z1, x2,y2,z2, x3,y3,z3] (9 values, row-major 3×3)
    /// `material_constitutive` is the precomputed constitutive matrices.
    /// `thickness` is the shell thickness.
    /// `shear_correction` is the shear correction factor (default 5/6).
    pub fn new(
        coords: &[f64; 9],
        constitutive: ShellConstitutive,
        thickness: f64,
        e_modulus: f64,
    ) -> Self {
        let p1 = Vector3::new(coords[0], coords[1], coords[2]);
        let p2 = Vector3::new(coords[3], coords[4], coords[5]);
        let p3 = Vector3::new(coords[6], coords[7], coords[8]);

        // Local coordinate system
        let v12 = p2 - p1;
        let v13 = p3 - p1;
        let l12 = v12.norm();
        let e1 = v12 / l12;
        let n = v12.cross(&v13);
        let n_len = n.norm();
        let e3 = n / n_len;
        let e2 = e3.cross(&e1);

        // Rotation matrix (rows = local basis vectors)
        let t3 = Matrix3::new(
            e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
        );

        // Local 2D coordinates
        let x1 = 0.0;
        let y1 = 0.0;
        let x2 = l12;
        let y2 = 0.0;
        let x3 = v13.dot(&e1);
        let y3 = v13.dot(&e2);
        let local_coords = [x1, y1, x2, y2, x3, y3];

        // Jacobian (constant for linear triangle)
        let dx_dr = x2 - x1;
        let dy_dr = y2 - y1;
        let dx_ds = x3 - x1;
        let dy_ds = y3 - y1;
        let j_mat = Matrix2::new(dx_dr, dy_dr, dx_ds, dy_ds);
        let det_j = dx_dr * dy_ds - dy_dr * dx_ds;
        let inv_det = 1.0 / det_j;
        let j_inv = Matrix2::new(
            dy_ds * inv_det,
            -dy_dr * inv_det,
            -dx_ds * inv_det,
            dx_dr * inv_det,
        );

        // Area = |detJ| / 2
        let area = det_j.abs() / 2.0;

        // Shape function derivatives in Cartesian (constant)
        // dh/dr = [-1, 1, 0], dh/ds = [-1, 0, 1]
        let dh_dr = Vector3::new(-1.0, 1.0, 0.0);
        let dh_ds = Vector3::new(-1.0, 0.0, 1.0);
        let dh_dx = j_inv[(0, 0)] * dh_dr + j_inv[(0, 1)] * dh_ds;
        let dh_dy = j_inv[(1, 0)] * dh_dr + j_inv[(1, 1)] * dh_ds;
        let dh = Matrix2x3::from_rows(&[dh_dx.transpose(), dh_dy.transpose()]);

        // Covariant base vectors (2D, in local plane)
        let g_r = Vector2::new(x2 - x1, y2 - y1);
        let g_s = Vector2::new(x3 - x1, y3 - y1);

        // Drilling stiffness
        let k_drill = e_modulus * thickness * thickness * 0.15;

        // Precompute tying point shear evaluations (extended space)
        let tying_ext_a =
            eval_covariant_shear_ext(TP_A.0, TP_A.1, &g_r, &g_s);
        let tying_ext_b =
            eval_covariant_shear_ext(TP_B.0, TP_B.1, &g_r, &g_s);
        let tying_ext_c =
            eval_covariant_shear_ext(TP_C.0, TP_C.1, &g_r, &g_s);
        let tying_ext_d =
            eval_covariant_shear_ext(TP_D.0, TP_D.1, &g_r, &g_s);
        let tying_ext_e =
            eval_covariant_shear_ext(TP_E.0, TP_E.1, &g_r, &g_s);
        let tying_ext_f =
            eval_covariant_shear_ext(TP_F.0, TP_F.1, &g_r, &g_s);

        Self {
            area,
            j_mat,
            j_inv,
            det_j,
            dh,
            local_coords,
            t3,
            g_r,
            g_s,
            constitutive,
            k_drill,
            thickness,
            tying_ext_a,
            tying_ext_b,
            tying_ext_c,
            tying_ext_d,
            tying_ext_e,
            tying_ext_f,
        }
    }
}

// ============================================================================
// Covariant shear evaluation at a tying point (extended 20-DOF space)
// ============================================================================

/// Evaluate covariant transverse shear B-vectors (e_rt, e_st) at a parametric
/// point in the extended 20-DOF space (18 nodal + 2 bubble rotations).
///
/// Returns (B_ert, B_est) each as Vec20.
fn eval_covariant_shear_ext(
    r: f64,
    s: f64,
    g_r: &Vector2<f64>,
    g_s: &Vector2<f64>,
) -> (Vec20, Vec20) {
    let f = enriched_shape(r, s);

    let mut b_ert = Vec20::zeros();
    let mut b_est = Vec20::zeros();

    // Corner nodes
    for i in 0..3 {
        let w_idx = 6 * i + 2;
        let thx_idx = 6 * i + 3;
        let thy_idx = 6 * i + 4;

        // dh/dr = [-1, 1, 0], dh/ds = [-1, 0, 1]
        let dhi_dr = [-1.0, 1.0, 0.0][i];
        let dhi_ds = [-1.0, 0.0, 1.0][i];

        // w uses linear interpolation
        b_ert[w_idx] = dhi_dr;
        b_est[w_idx] = dhi_ds;

        // Rotations use enriched interpolation
        b_ert[thy_idx] = f[i] * g_r[0];
        b_ert[thx_idx] = -f[i] * g_r[1];

        b_est[thy_idx] = f[i] * g_s[0];
        b_est[thx_idx] = -f[i] * g_s[1];
    }

    // Bubble rotations (indices 18, 19)
    b_ert[19] = f[3] * g_r[0]; // thy4
    b_ert[18] = -f[3] * g_r[1]; // thx4

    b_est[19] = f[3] * g_s[0];
    b_est[18] = -f[3] * g_s[1];

    (b_ert, b_est)
}

// ============================================================================
// Extended bending B-matrix (3×20)
// ============================================================================

type Mat3x20 = SMatrix<f64, 3, 20>;

fn b_kappa_ext(r: f64, s: f64, j_inv: &Matrix2<f64>) -> Mat3x20 {
    let (df_dr, df_ds) = enriched_shape_deriv(r, s);

    // Convert to Cartesian
    let mut df_dx = [0.0; 4];
    let mut df_dy = [0.0; 4];
    for i in 0..4 {
        df_dx[i] = j_inv[(0, 0)] * df_dr[i] + j_inv[(0, 1)] * df_ds[i];
        df_dy[i] = j_inv[(1, 0)] * df_dr[i] + j_inv[(1, 1)] * df_ds[i];
    }

    let mut bk = Mat3x20::zeros();

    // Corner nodes
    for i in 0..3 {
        let thx = 6 * i + 3;
        let thy = 6 * i + 4;

        // κxx = ∂θy/∂x
        bk[(0, thy)] = df_dx[i];
        // κyy = -∂θx/∂y
        bk[(1, thx)] = -df_dy[i];
        // κxy = ∂θy/∂y - ∂θx/∂x
        bk[(2, thy)] = df_dy[i];
        bk[(2, thx)] = -df_dx[i];
    }

    // Bubble (indices 18=thx4, 19=thy4)
    bk[(0, 19)] = df_dx[3];
    bk[(1, 18)] = -df_dy[3];
    bk[(2, 19)] = df_dy[3];
    bk[(2, 18)] = -df_dx[3];

    bk
}

// ============================================================================
// Extended shear B-matrix (2×20) using precomputed tying point data
// ============================================================================

type Mat2x20 = SMatrix<f64, 2, 20>;

fn b_gamma_ext(
    r: f64,
    s: f64,
    pre: &Mitc3Precomputed,
) -> Mat2x20 {
    let (ref b_ert_a, ref b_est_a) = pre.tying_ext_a;
    let (ref b_ert_b, ref b_est_b) = pre.tying_ext_b;
    let (ref b_ert_c, ref b_est_c) = pre.tying_ext_c;
    let (ref b_ert_d, ref _b_est_d) = pre.tying_ext_d;
    let (ref _b_ert_e, ref b_est_e) = pre.tying_ext_e;
    let (ref b_ert_f, ref b_est_f) = pre.tying_ext_f;

    // Constant part (Eq. 15)
    let b_ert_const =
        (2.0 / 3.0) * (b_ert_b - 0.5 * b_est_b) + (1.0 / 3.0) * (b_ert_c + b_est_c);
    let b_est_const =
        (2.0 / 3.0) * (b_est_a - 0.5 * b_ert_a) + (1.0 / 3.0) * (b_ert_c + b_est_c);

    // Linear part (Eq. 16)
    let b_c_hat = (b_ert_f - b_ert_d) - (b_est_f - b_est_e);
    let b_ert_linear = (1.0 / 3.0) * &b_c_hat * (3.0 * s - 1.0);
    let b_est_linear = (1.0 / 3.0) * &b_c_hat * (1.0 - 3.0 * r);

    // Total assumed
    let b_ert = b_ert_const + b_ert_linear;
    let b_est = b_est_const + b_est_linear;

    // Transform to Cartesian shear
    let j_inv = &pre.j_inv;
    let b_xz = j_inv[(0, 0)] * &b_ert + j_inv[(0, 1)] * &b_est;
    let b_yz = j_inv[(1, 0)] * &b_ert + j_inv[(1, 1)] * &b_est;

    let mut bg = Mat2x20::zeros();
    for j in 0..20 {
        bg[(0, j)] = b_xz[j];
        bg[(1, j)] = b_yz[j];
    }
    bg
}

// ============================================================================
// Membrane B-matrix (3×18) — standard CST
// ============================================================================

type Mat3x18 = SMatrix<f64, 3, 18>;

fn b_membrane(dh: &Matrix2x3) -> Mat3x18 {
    let mut bm = Mat3x18::zeros();

    for i in 0..3 {
        let u_idx = 6 * i;
        let v_idx = 6 * i + 1;
        let dhi_dx = dh[(0, i)];
        let dhi_dy = dh[(1, i)];

        // εxx = ∂u/∂x
        bm[(0, u_idx)] = dhi_dx;
        // εyy = ∂v/∂y
        bm[(1, v_idx)] = dhi_dy;
        // γxy = ∂u/∂y + ∂v/∂x
        bm[(2, u_idx)] = dhi_dy;
        bm[(2, v_idx)] = dhi_dx;
    }

    bm
}

// ============================================================================
// Drilling B-matrix (row for θz only)
// ============================================================================

type RowVec18 = SMatrix<f64, 1, 18>;

fn b_drill(r: f64, s: f64, dh: &Matrix2x3) -> RowVec18 {
    let h = shape_functions(r, s);
    let mut bd = RowVec18::zeros();

    for i in 0..3 {
        let u_idx = 6 * i;
        let v_idx = 6 * i + 1;
        let thz_idx = 6 * i + 5;
        let dhi_dx = dh[(0, i)];
        let dhi_dy = dh[(1, i)];

        // (∂v/∂x - ∂u/∂y)/2 - θz
        bd[(0, u_idx)] = -0.5 * dhi_dy;
        bd[(0, v_idx)] = 0.5 * dhi_dx;
        bd[(0, thz_idx)] = -h[i];
    }

    bd
}

// ============================================================================
// Stiffness matrix computation
// ============================================================================

/// Compute the 18×18 element stiffness matrix in LOCAL coordinates.
///
/// This computes K = K_membrane + K_bending_shear_condensed + K_coupling.
/// The bending-shear part uses the MITC3+ bubble enrichment and is statically
/// condensed from a 20-DOF to 18-DOF space.
pub fn compute_ke_local(pre: &Mitc3Precomputed) -> Mat18 {
    let area = pre.area;
    let cm = &pre.constitutive.cm;
    let cb = &pre.constitutive.cb;
    let cs = &pre.constitutive.cs;
    let dh = &pre.dh;
    let j_inv = &pre.j_inv;

    // --- Membrane stiffness (constant B for CST) ---
    let bm = b_membrane(dh);
    // For CST, B is constant → K_m = area * Bm^T Cm Bm (weights sum to 1)
    let km = area * (bm.transpose() * cm * &bm);

    // --- Drilling stiffness ---
    let mut k_drill_total = Mat18::zeros();
    for gp in 0..N_GAUSS {
        let bd = b_drill(GAUSS_R[gp], GAUSS_S[gp], dh);
        k_drill_total += (GAUSS_W[gp] * area * pre.k_drill) * (bd.transpose() * &bd);
    }

    // --- Bending + shear with MITC3+ bubble (extended 20-DOF) ---
    let mut k_ext = Mat20::zeros();
    for gp in 0..N_GAUSS {
        let r = GAUSS_R[gp];
        let s = GAUSS_S[gp];
        let w = GAUSS_W[gp];

        let bk = b_kappa_ext(r, s, j_inv);
        let bg = b_gamma_ext(r, s, pre);

        k_ext += w * area * (bk.transpose() * cb * &bk + bg.transpose() * cs * &bg);
    }

    // Static condensation: K = K_uu - K_uq K_qq^{-1} K_qu
    let k_uu = k_ext.fixed_view::<18, 18>(0, 0).into_owned();
    let k_uq = k_ext.fixed_view::<18, 2>(0, 18).into_owned();
    let k_qu = k_ext.fixed_view::<2, 18>(18, 0).into_owned();
    let k_qq = k_ext.fixed_view::<2, 2>(18, 18).into_owned();

    // 2×2 inverse (explicit for performance)
    let det_qq = k_qq[(0, 0)] * k_qq[(1, 1)] - k_qq[(0, 1)] * k_qq[(1, 0)];
    let inv_qq = SMatrix::<f64, 2, 2>::new(
        k_qq[(1, 1)] / det_qq,
        -k_qq[(0, 1)] / det_qq,
        -k_qq[(1, 0)] / det_qq,
        k_qq[(0, 0)] / det_qq,
    );

    let k_bs_cond = k_uu - &k_uq * &inv_qq * &k_qu;

    // Total local stiffness
    let k_local = km + k_drill_total + k_bs_cond;

    // Symmetrize
    0.5 * (&k_local + k_local.transpose())
}

/// Compute the 18×18 element stiffness in GLOBAL coordinates.
pub fn compute_ke_global(pre: &Mitc3Precomputed) -> Mat18 {
    let k_local = compute_ke_local(pre);
    transform_to_global(&k_local, &pre.t3)
}

// ============================================================================
// Mass matrix computation
// ============================================================================

/// Compute the 18×18 consistent mass matrix in GLOBAL coordinates.
pub fn compute_me_global(pre: &Mitc3Precomputed, rho: f64) -> Mat18 {
    let area = pre.area;
    let h = pre.thickness;
    let m_trans = rho * h;
    let m_rot = rho * h * h * h / 12.0;

    let mut m_local = Mat18::zeros();

    for gp in 0..N_GAUSS {
        let hf = shape_functions(GAUSS_R[gp], GAUSS_S[gp]);
        let w = GAUSS_W[gp];

        for i in 0..3 {
            for j in 0..3 {
                let val_t = hf[i] * hf[j] * m_trans * w * area;
                let val_r = hf[i] * hf[j] * m_rot * w * area;

                for k in 0..3 {
                    m_local[(6 * i + k, 6 * j + k)] += val_t;
                }
                for k in 3..6 {
                    m_local[(6 * i + k, 6 * j + k)] += val_r;
                }
            }
        }
    }

    transform_to_global(&m_local, &pre.t3)
}

/// Compute the 18×18 consistent mass matrix for composite elements (global coords).
///
/// Uses pre-computed ply-integrated mass parameters instead of ρ·h.
pub fn compute_me_composite_global(
    pre: &Mitc3Precomputed,
    mass_per_area: f64,
    rotational_inertia: f64,
) -> Mat18 {
    let area = pre.area;
    let m_trans = mass_per_area;
    let m_rot = rotational_inertia;

    let mut m_local = Mat18::zeros();

    for gp in 0..N_GAUSS {
        let hf = shape_functions(GAUSS_R[gp], GAUSS_S[gp]);
        let w = GAUSS_W[gp];

        for i in 0..3 {
            for j in 0..3 {
                let val_t = hf[i] * hf[j] * m_trans * w * area;
                let val_r = hf[i] * hf[j] * m_rot * w * area;

                for k in 0..3 {
                    m_local[(6 * i + k, 6 * j + k)] += val_t;
                }
                for k in 3..6 {
                    m_local[(6 * i + k, 6 * j + k)] += val_r;
                }
            }
        }
    }

    transform_to_global(&m_local, &pre.t3)
}

// ============================================================================
// Transformation local <-> global
// ============================================================================

/// Transform an 18×18 matrix from local to global: T^T · K · T
fn transform_to_global(k_local: &Mat18, t3: &Matrix3<f64>) -> Mat18 {
    // Build block-diagonal 18×18 transformation matrix
    let mut t18 = Mat18::zeros();
    for i in 0..6 {
        let row = 3 * i;
        for a in 0..3 {
            for b in 0..3 {
                t18[(row + a, row + b)] = t3[(a, b)];
            }
        }
    }

    t18.transpose() * k_local * &t18
}

// ============================================================================
// Nonlinear: Displacement gradient, Green-Lagrange strain
// ============================================================================

/// Compute displacement gradient H = ∂u/∂X at centroid.
///
/// `u_local` is the 18-DOF displacement in local coordinates.
/// Returns 3×3 H matrix (only first 2 columns are non-zero for flat shell).
pub fn displacement_gradient(pre: &Mitc3Precomputed, u_local: &Vec18) -> Matrix3<f64> {
    let dh = &pre.dh;
    let mut h_mat = Matrix3::zeros();

    for i in 0..3 {
        let ux = u_local[6 * i];
        let uy = u_local[6 * i + 1];
        let uz = u_local[6 * i + 2];
        let dhi_dx = dh[(0, i)];
        let dhi_dy = dh[(1, i)];

        // H[comp, deriv_dir]
        h_mat[(0, 0)] += ux * dhi_dx;
        h_mat[(0, 1)] += ux * dhi_dy;
        h_mat[(1, 0)] += uy * dhi_dx;
        h_mat[(1, 1)] += uy * dhi_dy;
        h_mat[(2, 0)] += uz * dhi_dx;
        h_mat[(2, 1)] += uz * dhi_dy;
    }

    h_mat
}

/// Compute Green-Lagrange strain tensor: E = 0.5·(H + H^T + H^T·H)
pub fn green_lagrange_strain(h_mat: &Matrix3<f64>) -> Matrix3<f64> {
    0.5 * (h_mat + h_mat.transpose() + h_mat.transpose() * h_mat)
}

/// Green-Lagrange strain in membrane Voigt form: [Exx, Eyy, 2·Exy]
pub fn gl_strain_voigt(h_mat: &Matrix3<f64>) -> Vector3<f64> {
    let e = green_lagrange_strain(h_mat);
    Vector3::new(e[(0, 0)], e[(1, 1)], 2.0 * e[(0, 1)])
}

// ============================================================================
// Nonlinear: B_L, B_NL, B_geometric matrices
// ============================================================================

type Mat6x18 = SMatrix<f64, 6, 18>;

/// Linear strain-displacement matrix B_L (6×18) for TL formulation.
pub fn compute_b_l(dh: &Matrix2x3) -> Mat3x18 {
    // For membrane only (rows: εxx, εyy, γxy)
    b_membrane(dh)
}

/// Nonlinear strain-displacement matrix B_NL (3×18) depending on current displacement.
pub fn compute_b_nl(dh: &Matrix2x3, h_mat: &Matrix3<f64>) -> Mat3x18 {
    let mut b_nl = Mat3x18::zeros();

    for i in 0..3 {
        let col = 6 * i;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        // Exx nonlinear: (∂u/∂x)·δ(∂u/∂x) + (∂v/∂x)·δ(∂v/∂x) + (∂w/∂x)·δ(∂w/∂x)
        b_nl[(0, col)] = h_mat[(0, 0)] * dni_dx;
        b_nl[(0, col + 1)] = h_mat[(1, 0)] * dni_dx;
        b_nl[(0, col + 2)] = h_mat[(2, 0)] * dni_dx;

        // Eyy nonlinear
        b_nl[(1, col)] = h_mat[(0, 1)] * dni_dy;
        b_nl[(1, col + 1)] = h_mat[(1, 1)] * dni_dy;
        b_nl[(1, col + 2)] = h_mat[(2, 1)] * dni_dy;

        // 2*Exy nonlinear
        b_nl[(2, col)] = h_mat[(0, 0)] * dni_dy + h_mat[(0, 1)] * dni_dx;
        b_nl[(2, col + 1)] = h_mat[(1, 0)] * dni_dy + h_mat[(1, 1)] * dni_dx;
        b_nl[(2, col + 2)] = h_mat[(2, 0)] * dni_dy + h_mat[(2, 1)] * dni_dx;
    }

    b_nl
}

/// Geometric B-matrix (6×18) for K_σ computation.
pub fn compute_b_geometric(dh: &Matrix2x3) -> Mat6x18 {
    let mut bg = Mat6x18::zeros();

    for i in 0..3 {
        let col = 6 * i;
        let dni_dx = dh[(0, i)];
        let dni_dy = dh[(1, i)];

        bg[(0, col)] = dni_dx; // ∂u/∂x
        bg[(1, col)] = dni_dy; // ∂u/∂y
        bg[(2, col + 1)] = dni_dx; // ∂v/∂x
        bg[(3, col + 1)] = dni_dy; // ∂v/∂y
        bg[(4, col + 2)] = dni_dx; // ∂w/∂x
        bg[(5, col + 2)] = dni_dy; // ∂w/∂y
    }

    bg
}

// ============================================================================
// Nonlinear: Geometric stiffness K_σ
// ============================================================================

type Mat6 = SMatrix<f64, 6, 6>;

/// Compute geometric stiffness K_σ (18×18) in local coordinates.
///
/// `sigma_membrane` = [σxx, σyy, σxy]
pub fn compute_k_sigma_local(
    pre: &Mitc3Precomputed,
    sigma_membrane: &Vector3<f64>,
) -> Mat18 {
    let h = pre.thickness;

    // 2×2 stress block
    let s_m = Matrix2::new(
        sigma_membrane[0] * h,
        sigma_membrane[2] * h,
        sigma_membrane[2] * h,
        sigma_membrane[1] * h,
    );

    // 6×6 block diagonal
    let mut s_tilde = Mat6::zeros();
    s_tilde.fixed_view_mut::<2, 2>(0, 0).copy_from(&s_m);
    s_tilde.fixed_view_mut::<2, 2>(2, 2).copy_from(&s_m);
    s_tilde.fixed_view_mut::<2, 2>(4, 4).copy_from(&s_m);

    // For CST, B_G is constant → K_σ = area * B_G^T S̃ B_G
    let bg = compute_b_geometric(&pre.dh);
    let k_sigma = pre.area * (bg.transpose() * &s_tilde * &bg);

    0.5 * (&k_sigma + k_sigma.transpose())
}

// ============================================================================
// Nonlinear: Tangent stiffness K_T
// ============================================================================

/// Compute tangent stiffness K_T (18×18) in GLOBAL coordinates.
///
/// K_T = K_0 + K_L(u) + K_σ(u) for Total Lagrangian formulation.
///
/// `u_global` is the 18-DOF displacement vector in global coordinates.
pub fn compute_kt_global(pre: &Mitc3Precomputed, u_global: &Vec18) -> Mat18 {
    // Transform displacement to local
    let u_local = global_to_local_disp(u_global, &pre.t3);

    // Reference stiffness K_0
    let k0 = compute_ke_local(pre);

    // Displacement gradient
    let h_mat = displacement_gradient(pre, &u_local);

    // Current membrane stress
    let eps_gl = gl_strain_voigt(&h_mat);
    let sigma_m = &pre.constitutive.cm_raw * &eps_gl;

    // Geometric stiffness
    let k_sigma = compute_k_sigma_local(pre, &sigma_m);

    // Initial displacement stiffness K_L
    let dh = &pre.dh;
    let b_l = compute_b_l(dh);
    let b_nl = compute_b_nl(dh, &h_mat);
    let cm_raw = &pre.constitutive.cm_raw;
    let area = pre.area;
    let h = pre.thickness;

    // K_L = area·h·(B_L^T·C·B_NL + B_NL^T·C·B_L + B_NL^T·C·B_NL)
    let k_l = area
        * h
        * (b_l.transpose() * cm_raw * &b_nl
            + b_nl.transpose() * cm_raw * &b_l
            + b_nl.transpose() * cm_raw * &b_nl);

    let k_t_local = k0 + k_l + k_sigma;
    let k_t_local = 0.5 * (&k_t_local + k_t_local.transpose());

    transform_to_global(&k_t_local, &pre.t3)
}

// ============================================================================
// Nonlinear: Internal forces
// ============================================================================

/// Compute internal force vector f_int (18) in GLOBAL coordinates.
pub fn compute_fint_global(pre: &Mitc3Precomputed, u_global: &Vec18, nonlinear: bool) -> Vec18 {
    let u_local = global_to_local_disp(u_global, &pre.t3);
    let area = pre.area;
    let cm_raw = &pre.constitutive.cm_raw;
    let h = pre.thickness;
    let dh = &pre.dh;

    let mut f_int = Vec18::zeros();

    // Bending + shear (linear, condensed)
    let k_bs = compute_ke_local(pre) - {
        // Recompute just membrane + drilling to subtract
        let bm = b_membrane(dh);
        let km = area * (bm.transpose() * &pre.constitutive.cm * &bm);
        let mut kd = Mat18::zeros();
        for gp in 0..N_GAUSS {
            let bd = b_drill(GAUSS_R[gp], GAUSS_S[gp], dh);
            kd += (GAUSS_W[gp] * area * pre.k_drill) * (bd.transpose() * &bd);
        }
        km + kd
    };
    f_int += k_bs * &u_local;

    // Drilling
    for gp in 0..N_GAUSS {
        let bd = b_drill(GAUSS_R[gp], GAUSS_S[gp], dh);
        let drill_strain = &bd * &u_local;
        f_int += (pre.k_drill * GAUSS_W[gp] * area) * (bd.transpose() * &drill_strain);
    }

    // Membrane
    if nonlinear {
        let h_mat = displacement_gradient(pre, &u_local);
        let eps_gl = gl_strain_voigt(&h_mat);
        let sigma_m = cm_raw * &eps_gl;

        let b_l = compute_b_l(dh);
        let b_nl = compute_b_nl(dh, &h_mat);
        let b_total = b_l + b_nl;

        f_int += (area * h) * (b_total.transpose() * &sigma_m);
    } else {
        let bm = b_membrane(dh);
        let eps_m = &bm * &u_local;
        // Use cm_raw * eps → σ, then Bm^T * σ * h * area
        let sigma_m = cm_raw * &eps_m;
        f_int += (area * h) * (bm.transpose() * &sigma_m);
    }

    // Transform to global
    local_to_global_force(&f_int, &pre.t3)
}

// ============================================================================
// Displacement / force transformations
// ============================================================================

fn global_to_local_disp(u_global: &Vec18, t3: &Matrix3<f64>) -> Vec18 {
    let mut u_local = Vec18::zeros();
    for i in 0..6 {
        let g = u_global.fixed_rows::<3>(3 * i);
        let l = t3 * g;
        u_local.fixed_rows_mut::<3>(3 * i).copy_from(&l);
    }
    u_local
}

fn local_to_global_force(f_local: &Vec18, t3: &Matrix3<f64>) -> Vec18 {
    let mut f_global = Vec18::zeros();
    let t3t = t3.transpose();
    for i in 0..6 {
        let l = f_local.fixed_rows::<3>(3 * i);
        let g = &t3t * &l;
        f_global.fixed_rows_mut::<3>(3 * i).copy_from(&g);
    }
    f_global
}

// ============================================================================
// Body load
// ============================================================================

/// Compute body load vector f_body (18) in GLOBAL coordinates.
///
/// Integrates `f = ∫ Nᵀ · (ρ · h · g) dA` over the element area.
/// Only translational DOFs (indices 0,1,2 of each node block) receive
/// contributions; rotational DOFs (3,4,5) are zero.
///
/// `gravity` is the body-force acceleration vector in global coordinates [gx, gy, gz].
pub fn compute_body_load_global(
    pre: &Mitc3Precomputed,
    rho: f64,
    gravity: &Vector3<f64>,
) -> Vec18 {
    let area = pre.area;
    let h = pre.thickness;
    let rho_h = rho * h;

    let mut f_local = Vec18::zeros();

    for gp in 0..N_GAUSS {
        let hf = shape_functions(GAUSS_R[gp], GAUSS_S[gp]);
        let w = GAUSS_W[gp];

        // Transform gravity to local coordinates
        let g_local = pre.t3 * gravity;

        for i in 0..3 {
            let contrib = hf[i] * rho_h * w * area;
            // Translational DOFs only (0,1,2 of each 6-DOF node block)
            for k in 0..3 {
                f_local[6 * i + k] += contrib * g_local[k];
            }
        }
    }

    // Transform to global
    local_to_global_force(&f_local, &pre.t3)
}

// ============================================================================
// Geometric stiffness (global)
// ============================================================================

/// Compute geometric stiffness K_σ (18×18) in GLOBAL coordinates.
///
/// `sigma_membrane` = [σxx, σyy, σxy] in local coordinates.
pub fn compute_k_sigma_global(
    pre: &Mitc3Precomputed,
    sigma_membrane: &Vector3<f64>,
) -> Mat18 {
    let k_local = compute_k_sigma_local(pre, sigma_membrane);
    transform_to_global(&k_local, &pre.t3)
}

// ============================================================================
// Centrifugal prestress
// ============================================================================

/// Compute centrifugal prestress [σxx, σyy, σxy] in local coordinates.
///
/// Returns the membrane stress state due to centrifugal loading:
///     σ_cf ≈ ρ · ω² · r_radial · L_char
///
/// The stress is then projected into the local membrane plane using
/// the angle of the radial direction.
///
/// `omega`           : angular velocity (rad/s)
/// `rotation_axis`   : unit vector of rotation axis in global coords
/// `rotation_center` : a point on the rotation axis in global coords
/// `centroid`        : element centroid in global coords
/// `rho`             : material density (kg/m³)
pub fn compute_centrifugal_prestress(
    pre: &Mitc3Precomputed,
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

    // Characteristic element length and stress magnitude
    let l_char = pre.area.sqrt();
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::isotropic::IsotropicMaterial;
    use crate::materials::Material;

    /// Build a simple flat right-triangle in the XY plane.
    /// Nodes: (0,0,0), (1,0,0), (0,1,0)
    fn make_pre() -> Mitc3Precomputed {
        let thickness = 0.01_f64;
        let mat = IsotropicMaterial::new(2.0e11, 0.3, 7800.0);
        let shell = mat.constitutive(thickness, 5.0 / 6.0);
        let node_coords: [f64; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        Mitc3Precomputed::new(&node_coords, shell, thickness, 2.0e11)
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
        for i in 0..3 {
            // x, y translational and all rotational DOFs must be ~0
            assert!(f[6 * i].abs() < 1e-10, "node {i} fx should be ~0");
            assert!(f[6 * i + 1].abs() < 1e-10, "node {i} fy should be ~0");
            assert!(f[6 * i + 2].abs() > 1e-6, "node {i} fz should be nonzero");
            for k in 3..6 {
                assert!(f[6 * i + k].abs() < 1e-12, "node {i} rotational dof {k} should be 0");
            }
        }

        // Total z-force = ρ·h·g·area (area = 0.5 for this triangle)
        let area = 0.5_f64;
        let h = 0.01_f64;
        let rho = 7800.0_f64;
        let expected_total_fz = rho * h * (-9.81) * area;
        let total_fz: f64 = (0..3).map(|i| f[6 * i + 2]).sum();
        assert!(
            (total_fz - expected_total_fz).abs() < 1e-6,
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
        let k_local = compute_k_sigma_local(&pre, &sigma);
        let k_global_direct = compute_k_sigma_global(&pre, &sigma);
        let k_global_manual = transform_to_global(&k_local, &pre.t3);
        let diff = k_global_direct - k_global_manual;
        assert!(diff.norm() < 1e-6, "compute_k_sigma_global must equal transform(k_local)");
    }

    #[test]
    fn test_centrifugal_prestress_on_axis() {
        let pre = make_pre();
        // Centroid at (1/3, 1/3, 0); rotation axis = Z; center = origin
        // Element is at r_radial = sqrt((1/3)^2+(1/3)^2)
        // But if we put centroid ON the axis, expect zeros
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let center = Vector3::new(1.0 / 3.0, 1.0 / 3.0, 0.0); // same as centroid XY
        let centroid = Vector3::new(1.0 / 3.0, 1.0 / 3.0, 0.0);
        let sigma = compute_centrifugal_prestress(&pre, 100.0, &axis, &center, &centroid, 7800.0);
        assert!(sigma.norm() < 1e-6, "element on axis → zero centrifugal stress");
    }

    #[test]
    fn test_centrifugal_prestress_nonzero() {
        let pre = make_pre();
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let center = Vector3::zeros();
        let centroid = Vector3::new(1.0 / 3.0, 1.0 / 3.0, 0.0);
        let sigma = compute_centrifugal_prestress(&pre, 100.0, &axis, &center, &centroid, 7800.0);
        // All components should be finite and the trace (σxx + σyy) ≥ 0
        assert!(sigma[0].is_finite());
        assert!(sigma[1].is_finite());
        assert!(sigma[2].is_finite());
        assert!(sigma[0] + sigma[1] >= 0.0, "centrifugal stress trace must be non-negative");
    }
}
