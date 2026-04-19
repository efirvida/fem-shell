// 3D Solid Element Kernels: HEXA8, TETRA4, WEDGE6
//
// Isoparametric solid elements for linear elasticity.
// Formulation: K = ∫BᵀCB dΩ,  M = ∫ρNᵀN dΩ
//
// Strain ordering (Voigt): [εxx, εyy, εzz, γxy, γyz, γzx]
//
// All integration is done directly — no heap allocation in hot path.

use nalgebra::{Matrix3, SMatrix};

// ============================================================================
// Type aliases
// ============================================================================

/// 24×24 HEXA8 element matrix (8 nodes × 3 DOFs)
pub type MatH8 = SMatrix<f64, 24, 24>;
/// 12×12 TETRA4 element matrix (4 nodes × 3 DOFs)
pub type MatT4 = SMatrix<f64, 12, 12>;
/// 18×18 WEDGE6 element matrix (6 nodes × 3 DOFs)
pub type MatW6 = SMatrix<f64, 18, 18>;

// ============================================================================
// Isotropic 3D constitutive matrix C (6×6)
// ============================================================================

/// Build the 6×6 isotropic elasticity matrix.
///
/// λ = E·ν / ((1+ν)(1-2ν))
/// μ = E / (2(1+ν))
///
/// Voigt order: [σxx, σyy, σzz, τxy, τyz, τzx]
fn isotropic_c(e: f64, nu: f64) -> SMatrix<f64, 6, 6> {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let d = lambda + 2.0 * mu;

    SMatrix::<f64, 6, 6>::from_row_slice(&[
        d,      lambda, lambda, 0.0, 0.0, 0.0,
        lambda, d,      lambda, 0.0, 0.0, 0.0,
        lambda, lambda, d,      0.0, 0.0, 0.0,
        0.0,    0.0,    0.0,    mu,  0.0, 0.0,
        0.0,    0.0,    0.0,    0.0, mu,  0.0,
        0.0,    0.0,    0.0,    0.0, 0.0, mu,
    ])
}

// ============================================================================
// Jacobian helper
// ============================================================================

/// Compute 3D Jacobian: J[i,j] = Σ_k dNk/dξi * x_j[k]
/// Returns (J, det_J, J_inv).
fn jacobian_3d(
    dn_dxi: &[f64],
    dn_deta: &[f64],
    dn_dzeta: &[f64],
    coords: &[[f64; 3]],
) -> (Matrix3<f64>, f64, Matrix3<f64>) {
    let n = coords.len();
    let mut j = Matrix3::zeros();
    for k in 0..n {
        j[(0, 0)] += dn_dxi[k] * coords[k][0];
        j[(0, 1)] += dn_dxi[k] * coords[k][1];
        j[(0, 2)] += dn_dxi[k] * coords[k][2];
        j[(1, 0)] += dn_deta[k] * coords[k][0];
        j[(1, 1)] += dn_deta[k] * coords[k][1];
        j[(1, 2)] += dn_deta[k] * coords[k][2];
        j[(2, 0)] += dn_dzeta[k] * coords[k][0];
        j[(2, 1)] += dn_dzeta[k] * coords[k][1];
        j[(2, 2)] += dn_dzeta[k] * coords[k][2];
    }
    let det = j.determinant();
    let inv = j.try_inverse().expect("singular Jacobian in solid element");
    (j, det, inv)
}

// ============================================================================
// B-matrix builder (6 × 3*n_nodes)
// ============================================================================

/// Transform natural derivatives to global, then fill B matrix (6 × 3*n_nodes).
///
/// B layout per node i (column block 3i..3i+2):
///   row 0: dN/dx  (εxx)
///   row 1: dN/dy  (εyy)
///   row 2: dN/dz  (εzz)
///   row 3: dN/dy, dN/dx  (γxy)
///   row 4: dN/dz, dN/dy  (γyz)
///   row 5: dN/dz, dN/dx  (γzx)
fn fill_b<const NDOF: usize>(
    dn_dxi: &[f64],
    dn_deta: &[f64],
    dn_dzeta: &[f64],
    inv_j: &Matrix3<f64>,
    b: &mut SMatrix<f64, 6, NDOF>,
) {
    let n_nodes = dn_dxi.len();
    for i in 0..n_nodes {
        let dndx = inv_j[(0, 0)] * dn_dxi[i] + inv_j[(0, 1)] * dn_deta[i] + inv_j[(0, 2)] * dn_dzeta[i];
        let dndy = inv_j[(1, 0)] * dn_dxi[i] + inv_j[(1, 1)] * dn_deta[i] + inv_j[(1, 2)] * dn_dzeta[i];
        let dndz = inv_j[(2, 0)] * dn_dxi[i] + inv_j[(2, 1)] * dn_deta[i] + inv_j[(2, 2)] * dn_dzeta[i];
        let col = 3 * i;
        b[(0, col)]     = dndx;
        b[(1, col + 1)] = dndy;
        b[(2, col + 2)] = dndz;
        b[(3, col)]     = dndy;
        b[(3, col + 1)] = dndx;
        b[(4, col + 1)] = dndz;
        b[(4, col + 2)] = dndy;
        b[(5, col)]     = dndz;
        b[(5, col + 2)] = dndx;
    }
}

// ============================================================================
// HEXA8
// ============================================================================

/// Node natural coordinates for HEXA8.
///
/// Corner order: (±1, ±1, ±1)
const HEXA8_NODES: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0], // 0
    [ 1.0, -1.0, -1.0], // 1
    [ 1.0,  1.0, -1.0], // 2
    [-1.0,  1.0, -1.0], // 3
    [-1.0, -1.0,  1.0], // 4
    [ 1.0, -1.0,  1.0], // 5
    [ 1.0,  1.0,  1.0], // 6
    [-1.0,  1.0,  1.0], // 7
];

/// 2×2×2 Gauss rule for HEXA8.
const GP: f64 = 0.577_350_269_189_625_8; // 1/sqrt(3)

fn hexa8_gauss() -> ([[[f64; 3]; 2]; 2], [[[f64; 1]; 2]; 2]) {
    // This is just a helper to iterate; we return raw inline below.
    // Unused — inlined directly into compute functions.
    (
        [[[0.0; 3]; 2]; 2],
        [[[0.0; 1]; 2]; 2],
    )
}

/// Trilinear shape functions for HEXA8.
#[inline]
fn hexa8_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
    let mut n = [0.0f64; 8];
    for (i, nc) in HEXA8_NODES.iter().enumerate() {
        n[i] = 0.125 * (1.0 + nc[0] * xi) * (1.0 + nc[1] * eta) * (1.0 + nc[2] * zeta);
    }
    n
}

/// Shape function derivatives for HEXA8.
#[inline]
fn hexa8_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 8], [f64; 8], [f64; 8]) {
    let mut dxi = [0.0f64; 8];
    let mut deta = [0.0f64; 8];
    let mut dzeta = [0.0f64; 8];
    for (i, nc) in HEXA8_NODES.iter().enumerate() {
        let a = nc[0];
        let b = nc[1];
        let c = nc[2];
        dxi[i]   = 0.125 * a * (1.0 + b * eta)  * (1.0 + c * zeta);
        deta[i]  = 0.125 * (1.0 + a * xi)  * b  * (1.0 + c * zeta);
        dzeta[i] = 0.125 * (1.0 + a * xi)  * (1.0 + b * eta) * c;
    }
    (dxi, deta, dzeta)
}

/// Gauss points for 2×2×2 rule.
const HEXA8_GP: [f64; 2] = [-GP, GP];
const HEXA8_W: f64 = 1.0;

/// Compute HEXA8 stiffness matrix Ke = ∫BᵀCB dV.
pub fn hexa8_ke(coords: &[[f64; 3]; 8], e: f64, nu: f64) -> MatH8 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatH8::zeros();

    for &xi in &HEXA8_GP {
        for &eta in &HEXA8_GP {
            for &zeta in &HEXA8_GP {
                let (dxi, deta, dzeta) = hexa8_derivs(xi, eta, zeta);
                let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

                let mut b = SMatrix::<f64, 6, 24>::zeros();
                fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

                let btcb = b.transpose() * c_mat * b;
                ke += btcb * (det_j * HEXA8_W * HEXA8_W * HEXA8_W);
            }
        }
    }
    // Symmetrise for numerical precision
    0.5 * (ke + ke.transpose())
}

/// Compute HEXA8 consistent mass matrix Me = ∫ρNᵀN dV.
pub fn hexa8_me(coords: &[[f64; 3]; 8], rho: f64) -> MatH8 {
    let mut me = MatH8::zeros();

    for &xi in &HEXA8_GP {
        for &eta in &HEXA8_GP {
            for &zeta in &HEXA8_GP {
                let n_sf = hexa8_shape(xi, eta, zeta);
                let (dxi, deta, dzeta) = hexa8_derivs(xi, eta, zeta);
                let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

                // N_mat is 3×24; NᵀN is 24×24 but block-diagonal by component.
                // We accumulate the scalar NiNj and expand to 3×3 identity blocks.
                let fac = rho * det_j * HEXA8_W * HEXA8_W * HEXA8_W;
                for i in 0..8 {
                    for j in 0..8 {
                        let val = n_sf[i] * n_sf[j] * fac;
                        for d in 0..3 {
                            me[(3 * i + d, 3 * j + d)] += val;
                        }
                    }
                }
            }
        }
    }
    0.5 * (me + me.transpose())
}

// ============================================================================
// TETRA4
// ============================================================================

/// 1-point integration rule for TETRA4 (unit tetrahedron, volume 1/6).
const TETRA4_GP: [f64; 3] = [0.25, 0.25, 0.25];
const TETRA4_W: f64 = 1.0 / 6.0;

/// Shape functions for TETRA4.
/// Natural coords: ξ,η,ζ ∈ [0,1] with ξ+η+ζ ≤ 1.
#[inline]
fn tetra4_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 4] {
    [1.0 - xi - eta - zeta, xi, eta, zeta]
}

/// Shape function derivatives for TETRA4 (constant).
#[inline]
fn tetra4_derivs() -> ([f64; 4], [f64; 4], [f64; 4]) {
    (
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    )
}

/// Compute TETRA4 stiffness matrix.
pub fn tetra4_ke(coords: &[[f64; 3]; 4], e: f64, nu: f64) -> MatT4 {
    let c_mat = isotropic_c(e, nu);
    let (dxi, deta, dzeta) = tetra4_derivs();
    let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

    let mut b = SMatrix::<f64, 6, 12>::zeros();
    fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

    let ke = b.transpose() * c_mat * b * (det_j * TETRA4_W);
    0.5 * (ke + ke.transpose())
}

/// Compute TETRA4 consistent mass matrix.
pub fn tetra4_me(coords: &[[f64; 3]; 4], rho: f64) -> MatT4 {
    let (dxi, deta, dzeta) = tetra4_derivs();
    let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

    let xi = TETRA4_GP[0];
    let eta = TETRA4_GP[1];
    let zeta = TETRA4_GP[2];
    let n_sf = tetra4_shape(xi, eta, zeta);

    let fac = rho * det_j * TETRA4_W;
    let mut me = MatT4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            let val = n_sf[i] * n_sf[j] * fac;
            for d in 0..3 {
                me[(3 * i + d, 3 * j + d)] += val;
            }
        }
    }
    0.5 * (me + me.transpose())
}

// ============================================================================
// WEDGE6
// ============================================================================

/// Triangular 3-point rule × linear 2-point rule = 6 integration points.
///
/// Triangular points: (1/6,1/6), (2/3,1/6), (1/6,2/3), each weight 1/6.
/// Linear points: ±1/√3, each weight 1.0.
/// Combined weight per point: tri_w × lin_w × 2 = 1/6 × 1.0 × 2 = 1/3.
const TRI3_XI:  [f64; 3] = [1.0/6.0, 2.0/3.0, 1.0/6.0];
const TRI3_ETA: [f64; 3] = [1.0/6.0, 1.0/6.0, 2.0/3.0];
const TRI3_W:   f64 = 1.0/6.0;

const LIN2_ZETA: [f64; 2] = [-GP, GP];
const LIN2_W:    f64 = 1.0;

/// Combined Gauss weight for WEDGE6: tri_w × lin_w × 2.
const WEDGE6_W: f64 = TRI3_W * LIN2_W * 2.0;

/// Shape functions for WEDGE6.
#[inline]
fn wedge6_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 6] {
    let l1 = 1.0 - xi - eta;
    let l2 = xi;
    let l3 = eta;
    let lm = 0.5 * (1.0 - zeta);
    let lp = 0.5 * (1.0 + zeta);
    [l1*lm, l2*lm, l3*lm, l1*lp, l2*lp, l3*lp]
}

/// Shape function derivatives for WEDGE6.
#[inline]
fn wedge6_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let l1 = 1.0 - xi - eta;
    let l2 = xi;
    let l3 = eta;
    let lm = 0.5 * (1.0 - zeta);
    let lp = 0.5 * (1.0 + zeta);
    let dlm = -0.5;
    let dlp =  0.5;

    let dxi = [
        -lm, lm, 0.0,
        -lp, lp, 0.0,
    ];
    let deta = [
        -lm, 0.0, lm,
        -lp, 0.0, lp,
    ];
    let dzeta = [
        l1*dlm, l2*dlm, l3*dlm,
        l1*dlp, l2*dlp, l3*dlp,
    ];
    (dxi, deta, dzeta)
}

/// Compute WEDGE6 stiffness matrix.
pub fn wedge6_ke(coords: &[[f64; 3]; 6], e: f64, nu: f64) -> MatW6 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatW6::zeros();

    for t in 0..3 {
        let xi  = TRI3_XI[t];
        let eta = TRI3_ETA[t];
        for &zeta in &LIN2_ZETA {
            let (dxi, deta, dzeta) = wedge6_derivs(xi, eta, zeta);
            let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

            let mut b = SMatrix::<f64, 6, 18>::zeros();
            fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

            let btcb = b.transpose() * c_mat * b;
            ke += btcb * (det_j * WEDGE6_W);
        }
    }
    0.5 * (ke + ke.transpose())
}

/// Compute WEDGE6 consistent mass matrix.
pub fn wedge6_me(coords: &[[f64; 3]; 6], rho: f64) -> MatW6 {
    let mut me = MatW6::zeros();

    for t in 0..3 {
        let xi  = TRI3_XI[t];
        let eta = TRI3_ETA[t];
        for &zeta in &LIN2_ZETA {
            let n_sf = wedge6_shape(xi, eta, zeta);
            let (dxi, deta, dzeta) = wedge6_derivs(xi, eta, zeta);
            let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

            let fac = rho * det_j * WEDGE6_W;
            for i in 0..6 {
                for j in 0..6 {
                    let val = n_sf[i] * n_sf[j] * fac;
                    for d in 0..3 {
                        me[(3 * i + d, 3 * j + d)] += val;
                    }
                }
            }
        }
    }
    0.5 * (me + me.transpose())
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Unit cube HEXA8
    fn unit_cube() -> [[f64; 3]; 8] {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    }

    // Unit tetrahedron TETRA4
    fn unit_tetra() -> [[f64; 3]; 4] {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    // Unit wedge WEDGE6
    fn unit_wedge() -> [[f64; 3]; 6] {
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0,  1.0],
            [1.0, 0.0,  1.0],
            [0.0, 1.0,  1.0],
        ]
    }

    #[test]
    fn hexa8_ke_symmetry() {
        let coords = unit_cube();
        let ke = hexa8_ke(&coords, 2.1e11, 0.3);
        let diff = (ke - ke.transpose()).abs().max();
        assert!(diff < 1e-10, "HEXA8 Ke not symmetric: {}", diff);
    }

    #[test]
    fn hexa8_me_symmetry_positive() {
        let coords = unit_cube();
        let me = hexa8_me(&coords, 7800.0);
        let diff = (me - me.transpose()).abs().max();
        assert!(diff < 1e-10, "HEXA8 Me not symmetric: {}", diff);
        // All diagonal entries positive
        for i in 0..24 {
            assert!(me[(i, i)] > 0.0, "HEXA8 Me diagonal[{}] not positive", i);
        }
    }

    #[test]
    fn hexa8_me_total_mass() {
        // Sum of any single-direction rows = rho * vol (consistent mass property).
        // For unit cube vol = 1, so sum of x-DOF row sums = rho.
        let coords = unit_cube();
        let rho = 7800.0;
        let me = hexa8_me(&coords, rho);
        let mut sum_x = 0.0;
        for i in 0..24 {
            sum_x += me[(0, i)]; // row for first x-DOF sums to rho*vol / (8 nodes)? No —
        }
        // Correct check: sum of the full x-block row-sums = rho * vol.
        let mut total = 0.0;
        for i in 0..8 {
            for j in 0..8 {
                total += me[(3 * i, 3 * j)];
            }
        }
        let expected = rho * 1.0; // vol = 1
        let rel_err = (total - expected).abs() / expected;
        assert!(rel_err < 1e-10, "HEXA8 Me mass total error: {}", rel_err);
    }

    #[test]
    fn tetra4_ke_symmetry() {
        let coords = unit_tetra();
        let ke = tetra4_ke(&coords, 2.1e11, 0.3);
        let diff = (ke - ke.transpose()).abs().max();
        assert!(diff < 1e-10, "TETRA4 Ke not symmetric: {}", diff);
    }

    #[test]
    fn tetra4_me_total_mass() {
        // Unit tet volume = 1/6.
        let coords = unit_tetra();
        let rho = 7800.0;
        let me = tetra4_me(&coords, rho);
        let mut total = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                total += me[(3 * i, 3 * j)];
            }
        }
        let expected = rho / 6.0;
        let rel_err = (total - expected).abs() / expected;
        assert!(rel_err < 1e-10, "TETRA4 Me mass total error: {}", rel_err);
    }

    #[test]
    fn wedge6_ke_symmetry() {
        let coords = unit_wedge();
        let ke = wedge6_ke(&coords, 2.1e11, 0.3);
        let diff = (ke - ke.transpose()).abs().max();
        assert!(diff < 1e-10, "WEDGE6 Ke not symmetric: {}", diff);
    }

    #[test]
    fn wedge6_me_trace_matches_volume() {
        // Unit wedge with triangle base area 0.5 and height 2.0 → vol = 1.0.
        let coords = unit_wedge();
        let rho = 7800.0;
        let me = wedge6_me(&coords, rho);
        let mut trace_x = 0.0;
        for i in 0..6 {
            trace_x += me[(3 * i, 3 * i)];
        }
        let expected = rho * 1.0; // vol = 0.5 * 2 = 1.0
        let rel_err = (trace_x - expected).abs() / expected;
        assert!(rel_err < 1e-10, "WEDGE6 Me mass trace error: {}", rel_err);
    }
}
