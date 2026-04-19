// 3D Solid Element Kernels: HEXA8, HEXA20, TETRA4, TETRA10, WEDGE6, WEDGE15, PYRAMID5, PYRAMID13
//
// Isoparametric solid elements for linear elasticity.
// Formulation: K = ∫BᵀCB dΩ,  M = ∫ρNᵀN dΩ
//
// Strain ordering (Voigt): [εxx, εyy, εzz, γxy, γyz, γzx]
//
// All integration is done directly — no heap allocation in hot path.

use nalgebra::{Matrix3, SMatrix};

use crate::quadrature::{
    GAUSS2_PTS as HEXA8_GP_IMPORT, TETRA4_GP, TETRA4_W, TETRA10_GP, TETRA10_W,
    TRI3_XI, TRI3_ETA, TRI3_W, LIN2_ZETA, LIN2_W, WEDGE6_W,
    WEDGE15_TRI_XI, WEDGE15_TRI_ETA, WEDGE15_TRI_W, WEDGE15_LIN_ZETA, WEDGE15_LIN_W,
    PYRAMID5_GP, PYRAMID5_W, PYRAMID13_GP, PYRAMID13_W,
};

// ============================================================================
// Type aliases
// ============================================================================

/// 24×24 HEXA8 element matrix (8 nodes × 3 DOFs)
pub type MatH8 = SMatrix<f64, 24, 24>;
/// 60×60 HEXA20 element matrix (20 nodes × 3 DOFs)
pub type MatH20 = SMatrix<f64, 60, 60>;
/// 12×12 TETRA4 element matrix (4 nodes × 3 DOFs)
pub type MatT4 = SMatrix<f64, 12, 12>;
/// 30×30 TETRA10 element matrix (10 nodes × 3 DOFs)
pub type MatT10 = SMatrix<f64, 30, 30>;
/// 18×18 WEDGE6 element matrix (6 nodes × 3 DOFs)
pub type MatW6 = SMatrix<f64, 18, 18>;
/// 45×45 WEDGE15 element matrix (15 nodes × 3 DOFs)
pub type MatW15 = SMatrix<f64, 45, 45>;
/// 15×15 PYRAMID5 element matrix (5 nodes × 3 DOFs)
pub type MatP5 = SMatrix<f64, 15, 15>;
/// 39×39 PYRAMID13 element matrix (13 nodes × 3 DOFs)
pub type MatP13 = SMatrix<f64, 39, 39>;

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
const GP: f64 = 0.577_350_269_189_625_8; // 1/sqrt(3) — also available in crate::quadrature

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
// TETRA4_GP and TETRA4_W are imported from crate::quadrature above.


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
// TRI3_XI, TRI3_ETA, TRI3_W, LIN2_ZETA, LIN2_W, WEDGE6_W imported from crate::quadrature.

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
// TETRA10
// ============================================================================

/// 10-node quadratic tetrahedron element.
///
/// Node ordering (Gmsh convention):
///     Corners: 0, 1, 2, 3
///     Edge midpoints: 4 (0-1), 5 (1-2), 6 (0-2), 7 (0-3), 8 (2-3), 9 (1-3)
///
/// ```text
///                    v
///                  .
///                 ,/
///                /
///            2
///          ,/|`\
///        ,/  |  `\
///       ,6   '.   `5
///     ,/      8    `\
///   ,/        |      `\
///  0--------4--'.-------1 --> u
///   `\.        |      ,/
///      `\.     |    ,9
///         `7.  '. ,/
///            `\. |/
///               `3
///                  `\.
///                     ` w
/// ```
///
/// Natural coordinates: ξ, η, ζ ∈ [0, 1] with ξ + η + ζ ≤ 1
/// (volume coordinates: L1 = 1−ξ−η−ζ, L2 = ξ, L3 = η, L4 = ζ)

/// 4-point integration rule for TETRA10 (Keast degree-2 rule).
// TETRA10_GP and TETRA10_W are imported from crate::quadrature above.


/// Quadratic tetrahedral shape functions.
#[inline]
fn tetra10_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 10] {
    let l1 = 1.0 - xi - eta - zeta;
    let l2 = xi;
    let l3 = eta;
    let l4 = zeta;
    [
        l1 * (2.0 * l1 - 1.0), // N0
        l2 * (2.0 * l2 - 1.0), // N1
        l3 * (2.0 * l3 - 1.0), // N2
        l4 * (2.0 * l4 - 1.0), // N3
        4.0 * l1 * l2,         // N4 - edge 0-1
        4.0 * l2 * l3,         // N5 - edge 1-2
        4.0 * l3 * l1,         // N6 - edge 0-2 (Gmsh convention)
        4.0 * l1 * l4,         // N7 - edge 0-3
        4.0 * l3 * l4,         // N8 - edge 2-3 (Gmsh: swapped with N9)
        4.0 * l2 * l4,         // N9 - edge 1-3 (Gmsh: swapped with N8)
    ]
}

/// Shape function derivatives for TETRA10.
#[inline]
fn tetra10_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 10], [f64; 10], [f64; 10]) {
    let l1 = 1.0 - xi - eta - zeta;
    let l2 = xi;
    let l3 = eta;
    let l4 = zeta;
    // dL1/d(xi,eta,zeta) = (-1,-1,-1), dL2=(1,0,0), dL3=(0,1,0), dL4=(0,0,1)
    let dxi = [
        (4.0 * l1 - 1.0) * (-1.0), // dN0/dxi
        4.0 * l2 - 1.0,            // dN1/dxi
        0.0,                        // dN2/dxi
        0.0,                        // dN3/dxi
        4.0 * (l1 - l2) * (-1.0) + 4.0 * l2 * (-1.0), // N4: 4*(dL1*L2 + L1*dL2) = 4*(-L2 + L1*1) = 4*(L1-L2) but dL1/dxi=-1
        4.0 * l3,                   // dN5/dxi: 4*(dL2*L3) = 4*L3
        4.0 * l3 * (-1.0),          // dN6/dxi: 4*(L3*dL1) = -4*L3
        4.0 * l4 * (-1.0),          // dN7/dxi: 4*(L4*dL1) = -4*L4
        0.0,                        // dN8/dxi: 4*L3*L4, dL3/dxi=0, dL4/dxi=0
        4.0 * l4,                   // dN9/dxi: 4*(dL2*L4) = 4*L4
    ];
    let deta = [
        (4.0 * l1 - 1.0) * (-1.0), // dN0/deta
        0.0,                        // dN1/deta
        4.0 * l3 - 1.0,            // dN2/deta
        0.0,                        // dN3/deta
        4.0 * l2 * (-1.0),          // dN4/deta: 4*(L2*dL1) = -4*L2
        4.0 * l2,                   // dN5/deta: 4*(L2*dL3) = 4*L2
        4.0 * (l1 - l3),            // dN6/deta: 4*(L3*dL1 + L1*dL3) = 4*(-L3+L1)
        4.0 * l4 * (-1.0),          // dN7/deta: 4*(L4*dL1) = -4*L4
        4.0 * l4,                   // dN8/deta: 4*(L4*dL3) = 4*L4
        0.0,                        // dN9/deta
    ];
    let dzeta = [
        (4.0 * l1 - 1.0) * (-1.0), // dN0/dzeta
        0.0,                        // dN1/dzeta
        0.0,                        // dN2/dzeta
        4.0 * l4 - 1.0,            // dN3/dzeta
        4.0 * l2 * (-1.0),          // dN4/dzeta: -4*L2
        0.0,                        // dN5/dzeta
        4.0 * l3 * (-1.0),          // dN6/dzeta: -4*L3
        4.0 * (l1 - l4),            // dN7/dzeta: 4*(L4*dL1 + L1*dL4) = 4*(-L4+L1)
        4.0 * l3,                   // dN8/dzeta: 4*(L3*dL4) = 4*L3
        4.0 * l2,                   // dN9/dzeta: 4*(L2*dL4) = 4*L2
    ];
    (dxi, deta, dzeta)
}

/// Compute TETRA10 stiffness matrix Ke = ∫BᵀCB dV.
pub fn tetra10_ke(coords: &[[f64; 3]; 10], e: f64, nu: f64) -> MatT10 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatT10::zeros();

    for gp in &TETRA10_GP {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let (dxi, deta, dzeta) = tetra10_derivs(xi, eta, zeta);
        let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

        let mut b = SMatrix::<f64, 6, 30>::zeros();
        fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

        ke += b.transpose() * c_mat * b * (det_j * TETRA10_W);
    }
    0.5 * (ke + ke.transpose())
}

/// Compute TETRA10 consistent mass matrix Me = ∫ρNᵀN dV.
pub fn tetra10_me(coords: &[[f64; 3]; 10], rho: f64) -> MatT10 {
    let mut me = MatT10::zeros();

    for gp in &TETRA10_GP {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let n_sf = tetra10_shape(xi, eta, zeta);
        let (dxi, deta, dzeta) = tetra10_derivs(xi, eta, zeta);
        let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

        let fac = rho * det_j * TETRA10_W;
        for i in 0..10 {
            for j in 0..10 {
                let val = n_sf[i] * n_sf[j] * fac;
                for d in 0..3 {
                    me[(3 * i + d, 3 * j + d)] += val;
                }
            }
        }
    }
    0.5 * (me + me.transpose())
}

// ============================================================================
// WEDGE15
// ============================================================================

/// 15-node quadratic wedge/prism element.
///
/// Triangular base with quadratic interpolation.
///
/// Node ordering (Gmsh convention):
///
/// ```text
///                    3
///                  ,/|`\
///                12  |  13
///              ,/    |    `\
///             4------14-----5
///             |      8      |
///             |      |      |
///            10      |      11
///             |      0      |
///             |    ,/ `\    |
///             |  ,6     `7  |
///             |,/         `\|
///             1------9------2
/// ```
///
/// Corner nodes: 0,1,2 (bottom z=−1), 3,4,5 (top z=+1)
///
/// Edge mapping:
///  6: edge 0-1 (bottom),  7: edge 0-2 (bottom),  8: edge 0-3 (vertical)
///  9: edge 1-2 (bottom), 10: edge 1-4 (vertical), 11: edge 2-5 (vertical)
/// 12: edge 3-4 (top),    13: edge 3-5 (top),      14: edge 4-5 (top)
///
/// Natural coordinates: ξ ∈ [0,1], η ∈ [0,1] with ξ+η ≤ 1, ζ ∈ [−1,1]

/// 7-point triangular rule × 3-point Gauss in ζ = 21 integration points.
///
/// Triangular points from Dunavant degree-5 rule, weights scaled for unit triangle (×0.5).
/// Linear points: ±√(3/5), 0 with weights 5/9, 8/9, 5/9.
/// Combined weight: tri_w × lin_w × 2 (factor 2 for ζ ∈ [−1,1]).
// WEDGE15_TRI_XI, WEDGE15_TRI_ETA, WEDGE15_TRI_W, WEDGE15_LIN_ZETA, WEDGE15_LIN_W
// imported from crate::quadrature.


/// Shape functions for WEDGE15.
#[inline]
fn wedge15_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 15] {
    let l1 = 1.0 - xi - eta;
    let l2 = xi;
    let l3 = eta;
    let lm = 0.5 * zeta * (zeta - 1.0); // bottom quadratic
    let l0 = 1.0 - zeta * zeta;          // middle bubble
    let lp = 0.5 * zeta * (zeta + 1.0); // top quadratic
    [
        l1 * (2.0 * l1 - 1.0) * lm, // N0  corner bottom
        l2 * (2.0 * l2 - 1.0) * lm, // N1  corner bottom
        l3 * (2.0 * l3 - 1.0) * lm, // N2  corner bottom
        l1 * (2.0 * l1 - 1.0) * lp, // N3  corner top
        l2 * (2.0 * l2 - 1.0) * lp, // N4  corner top
        l3 * (2.0 * l3 - 1.0) * lp, // N5  corner top
        4.0 * l1 * l2 * lm,          // N6  edge 0-1 bottom
        4.0 * l1 * l3 * lm,          // N7  edge 0-2 bottom
        l1 * l0,                      // N8  edge 0-3 vertical
        4.0 * l2 * l3 * lm,          // N9  edge 1-2 bottom
        l2 * l0,                      // N10 edge 1-4 vertical
        l3 * l0,                      // N11 edge 2-5 vertical
        4.0 * l1 * l2 * lp,          // N12 edge 3-4 top
        4.0 * l1 * l3 * lp,          // N13 edge 3-5 top
        4.0 * l2 * l3 * lp,          // N14 edge 4-5 top
    ]
}

/// Shape function derivatives for WEDGE15.
#[inline]
fn wedge15_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 15], [f64; 15], [f64; 15]) {
    let l1 = 1.0 - xi - eta;
    let l2 = xi;
    let l3 = eta;
    let lm = 0.5 * zeta * (zeta - 1.0);
    let l0 = 1.0 - zeta * zeta;
    let lp = 0.5 * zeta * (zeta + 1.0);
    let dlm = zeta - 0.5;
    let dl0 = -2.0 * zeta;
    let dlp = zeta + 0.5;

    // dN/dxi  (dL1/dxi = -1, dL2/dxi = 1, dL3/dxi = 0)
    let dxi = [
        (4.0 * l1 - 1.0) * (-1.0) * lm, // N0
        (4.0 * l2 - 1.0) * lm,           // N1
        0.0,                               // N2
        (4.0 * l1 - 1.0) * (-1.0) * lp, // N3
        (4.0 * l2 - 1.0) * lp,           // N4
        0.0,                               // N5
        4.0 * (l1 - l2) * lm,            // N6: 4*(dL1*L2 + L1*dL2)*lm = 4*(−L2+L1)*lm
        -4.0 * l3 * lm,                   // N7: 4*(dL1*L3)*lm = −4*L3*lm
        -l0,                               // N8: dL1/dxi * l0
        4.0 * l3 * lm,                    // N9: 4*(dL2*L3)*lm = 4*L3*lm
        l0,                                // N10: dL2/dxi * l0
        0.0,                               // N11
        4.0 * (l1 - l2) * lp,            // N12
        -4.0 * l3 * lp,                   // N13
        4.0 * l3 * lp,                    // N14
    ];

    // dN/deta (dL1/deta = -1, dL2/deta = 0, dL3/deta = 1)
    let deta = [
        (4.0 * l1 - 1.0) * (-1.0) * lm, // N0
        0.0,                               // N1
        (4.0 * l3 - 1.0) * lm,           // N2
        (4.0 * l1 - 1.0) * (-1.0) * lp, // N3
        0.0,                               // N4
        (4.0 * l3 - 1.0) * lp,           // N5
        -4.0 * l2 * lm,                   // N6: 4*(dL1*L2)*lm = −4*L2*lm
        4.0 * (l1 - l3) * lm,            // N7: 4*(dL1*L3 + L1*dL3)*lm = 4*(−L3+L1)*lm
        -l0,                               // N8: dL1/deta * l0
        4.0 * l2 * lm,                    // N9: 4*(dL3*L2)*lm = 4*L2*lm
        0.0,                               // N10
        l0,                                // N11: dL3/deta * l0
        -4.0 * l2 * lp,                   // N12
        4.0 * (l1 - l3) * lp,            // N13
        4.0 * l2 * lp,                    // N14
    ];

    // dN/dzeta
    let dzeta = [
        l1 * (2.0 * l1 - 1.0) * dlm, // N0
        l2 * (2.0 * l2 - 1.0) * dlm, // N1
        l3 * (2.0 * l3 - 1.0) * dlm, // N2
        l1 * (2.0 * l1 - 1.0) * dlp, // N3
        l2 * (2.0 * l2 - 1.0) * dlp, // N4
        l3 * (2.0 * l3 - 1.0) * dlp, // N5
        4.0 * l1 * l2 * dlm,          // N6
        4.0 * l1 * l3 * dlm,          // N7
        l1 * dl0,                      // N8
        4.0 * l2 * l3 * dlm,          // N9
        l2 * dl0,                      // N10
        l3 * dl0,                      // N11
        4.0 * l1 * l2 * dlp,          // N12
        4.0 * l1 * l3 * dlp,          // N13
        4.0 * l2 * l3 * dlp,          // N14
    ];

    (dxi, deta, dzeta)
}

/// Compute WEDGE15 stiffness matrix Ke = ∫BᵀCB dV.
pub fn wedge15_ke(coords: &[[f64; 3]; 15], e: f64, nu: f64) -> MatW15 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatW15::zeros();

    for t in 0..7 {
        let xi  = WEDGE15_TRI_XI[t];
        let eta = WEDGE15_TRI_ETA[t];
        let tw  = WEDGE15_TRI_W[t];
        for z in 0..3 {
            let zeta = WEDGE15_LIN_ZETA[z];
            let lw   = WEDGE15_LIN_W[z];
            let (dxi, deta, dzeta) = wedge15_derivs(xi, eta, zeta);
            let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

            let mut b = SMatrix::<f64, 6, 45>::zeros();
            fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

            ke += b.transpose() * c_mat * b * (det_j * tw * lw * 2.0);
        }
    }
    0.5 * (ke + ke.transpose())
}

/// Compute WEDGE15 consistent mass matrix Me = ∫ρNᵀN dV.
pub fn wedge15_me(coords: &[[f64; 3]; 15], rho: f64) -> MatW15 {
    let mut me = MatW15::zeros();

    for t in 0..7 {
        let xi  = WEDGE15_TRI_XI[t];
        let eta = WEDGE15_TRI_ETA[t];
        let tw  = WEDGE15_TRI_W[t];
        for z in 0..3 {
            let zeta = WEDGE15_LIN_ZETA[z];
            let lw   = WEDGE15_LIN_W[z];
            let n_sf = wedge15_shape(xi, eta, zeta);
            let (dxi, deta, dzeta) = wedge15_derivs(xi, eta, zeta);
            let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

            let fac = rho * det_j * tw * lw * 2.0;
            for i in 0..15 {
                for j in 0..15 {
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
// PYRAMID5
// ============================================================================

/// 5-node linear pyramid element.
///
/// Square base with apex. Useful for transitioning between hex and tet meshes.
///
/// ```text
///                 4 (apex)
///                /|\\
///               / | \\
///              /  |  \\
///             3-------2
///            /|       |
///           / |       |
///          0---------1
/// ```
///
/// Base at ζ = 0, apex at ζ = 1.
/// Natural coordinates: ξ, η ∈ [−1, 1], ζ ∈ [0, 1]
///
/// Shape functions (Bedrosian singularity-free form, r = 1 − ζ):
///   Nᵢ = (r ± ξ)(r ± η) / (4r)   for base nodes
///   N₄ = ζ                          for apex

/// 8-point Felippa integration rule for PYRAMID5 (precision 3).
///
/// Reference: Felippa, C.A. (2004). "A compendium of FEM integration formulas
/// for symbolic work," Engineering Computation, 21(8), 867-890.
// PYRAMID5_GP and PYRAMID5_W imported from crate::quadrature.
const PYRAMID5_EPS: f64 = 1e-14;

/// Shape functions for PYRAMID5 (singularity-free formulation).
#[inline]
fn pyramid5_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 5] {
    let r = if 1.0 - zeta > PYRAMID5_EPS { 1.0 - zeta } else { PYRAMID5_EPS };
    [
        0.25 * (r - xi) * (r - eta) / r, // N0
        0.25 * (r + xi) * (r - eta) / r, // N1
        0.25 * (r + xi) * (r + eta) / r, // N2
        0.25 * (r - xi) * (r + eta) / r, // N3
        zeta,                              // N4 apex
    ]
}

/// Shape function derivatives for PYRAMID5.
///
/// Uses quotient rule on Nᵢ = (r ± ξ)(r ± η) / (4r), r = 1 − ζ, dr/dζ = −1.
/// dNᵢ/dζ = −[(r ± η) + (r ± ξ)] / (4r) + (r ± ξ)(r ± η) / (4r²)
#[inline]
fn pyramid5_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 5], [f64; 5], [f64; 5]) {
    let r = if 1.0 - zeta > PYRAMID5_EPS { 1.0 - zeta } else { PYRAMID5_EPS };
    let r2 = r * r;
    let rm_xi  = r - xi;
    let rp_xi  = r + xi;
    let rm_eta = r - eta;
    let rp_eta = r + eta;

    let dxi = [
        -0.25 * rm_eta / r,
         0.25 * rm_eta / r,
         0.25 * rp_eta / r,
        -0.25 * rp_eta / r,
        0.0,
    ];
    let deta = [
        -0.25 * rm_xi / r,
        -0.25 * rp_xi / r,
         0.25 * rp_xi / r,
         0.25 * rm_xi / r,
        0.0,
    ];
    let dzeta = [
        -0.25 * (rm_eta + rm_xi) / r + 0.25 * rm_xi * rm_eta / r2,
        -0.25 * (rm_eta + rp_xi) / r + 0.25 * rp_xi * rm_eta / r2,
        -0.25 * (rp_eta + rp_xi) / r + 0.25 * rp_xi * rp_eta / r2,
        -0.25 * (rp_eta + rm_xi) / r + 0.25 * rm_xi * rp_eta / r2,
        1.0,
    ];
    (dxi, deta, dzeta)
}

/// Compute PYRAMID5 stiffness matrix Ke = ∫BᵀCB dV.
pub fn pyramid5_ke(coords: &[[f64; 3]; 5], e: f64, nu: f64) -> MatP5 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatP5::zeros();

    for (gp, &w) in PYRAMID5_GP.iter().zip(PYRAMID5_W.iter()) {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let (dxi, deta, dzeta) = pyramid5_derivs(xi, eta, zeta);
        let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

        let mut b = SMatrix::<f64, 6, 15>::zeros();
        fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

        ke += b.transpose() * c_mat * b * (det_j * w);
    }
    0.5 * (ke + ke.transpose())
}

/// Compute PYRAMID5 consistent mass matrix Me = ∫ρNᵀN dV.
pub fn pyramid5_me(coords: &[[f64; 3]; 5], rho: f64) -> MatP5 {
    let mut me = MatP5::zeros();

    for (gp, &w) in PYRAMID5_GP.iter().zip(PYRAMID5_W.iter()) {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let n_sf = pyramid5_shape(xi, eta, zeta);
        let (dxi, deta, dzeta) = pyramid5_derivs(xi, eta, zeta);
        let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

        let fac = rho * det_j * w;
        for i in 0..5 {
            for j in 0..5 {
                let val = n_sf[i] * n_sf[j] * fac;
                for d in 0..3 {
                    me[(3 * i + d, 3 * j + d)] += val;
                }
            }
        }
    }
    0.5 * (me + me.transpose())
}

// ============================================================================
// PYRAMID13
// ============================================================================

/// 13-node quadratic pyramid element.
///
/// Square base with apex and mid-edge nodes.
///
/// Gmsh Node ordering (element type 19):
///
/// ```text
///                  4
///                ,/|\
///              ,/ .'|\
///            ,/   | | \
///          ,/    .' | `.
///        ,7      |  12  \
///      ,/       .'   |   \
///    ,/         9    |    11
///   0--------6-.'----3    `.
///    `\        |      `\    \
///      `5     .'        10   \
///        `\   |           `\  \
///          `\.'             `\`
///             1--------8-------2
/// ```
///
/// Base corners: 0, 1, 2, 3 — Apex: 4
///
/// Edge midpoints (Gmsh convention):
///  5: edge 0-1 (base front),     6: edge 0-3 (base left)
///  7: edge 0-4 (lateral to apex), 8: edge 1-2 (base right)
///  9: edge 1-4 (lateral),        10: edge 2-3 (base back)
/// 11: edge 2-4 (lateral),        12: edge 3-4 (lateral)
///
/// Natural coordinates: ξ, η ∈ [−1, 1], ζ ∈ [0, 1]

/// 18-point Felippa integration rule for PYRAMID13 (precision 5).
///
/// Reference: Felippa, C.A. (2004). "A compendium of FEM integration formulas
/// for symbolic work," Engineering Computation, 21(8), 867-890.
// PYRAMID13_GP and PYRAMID13_W imported from crate::quadrature.


/// Shape functions for PYRAMID13 (rational blending, Gmsh convention).
///
/// Uses scaled coordinates ξₛ = ξ/r, ηₛ = η/r where r = 1 − ζ.
/// Base corners use 8-node serendipity on the scaled square × vertical blend.
/// Lateral edge midpoints use bilinear × bubble vertical blend.
#[inline]
fn pyramid13_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 13] {
    let r = if 1.0 - zeta > PYRAMID5_EPS { 1.0 - zeta } else { PYRAMID5_EPS };
    let xi_s  = (xi  / r).clamp(-1.0, 1.0);
    let eta_s = (eta / r).clamp(-1.0, 1.0);

    // 8-node serendipity corner functions on scaled base
    let f0 = 0.25 * (1.0 - xi_s) * (1.0 - eta_s) * (-xi_s - eta_s - 1.0);
    let f1 = 0.25 * (1.0 + xi_s) * (1.0 - eta_s) * ( xi_s - eta_s - 1.0);
    let f2 = 0.25 * (1.0 + xi_s) * (1.0 + eta_s) * ( xi_s + eta_s - 1.0);
    let f3 = 0.25 * (1.0 - xi_s) * (1.0 + eta_s) * (-xi_s + eta_s - 1.0);

    // Edge bubble functions on scaled base
    let g5  = 0.5 * (1.0 - xi_s * xi_s)  * (1.0 - eta_s); // edge 0-1
    let g6  = 0.5 * (1.0 - xi_s)          * (1.0 - eta_s * eta_s); // edge 0-3
    let g8  = 0.5 * (1.0 + xi_s)          * (1.0 - eta_s * eta_s); // edge 1-2
    let g10 = 0.5 * (1.0 - xi_s * xi_s)  * (1.0 + eta_s); // edge 2-3

    // Bilinear functions for lateral edge blending
    let phi0 = 0.25 * (1.0 - xi_s) * (1.0 - eta_s);
    let phi1 = 0.25 * (1.0 + xi_s) * (1.0 - eta_s);
    let phi2 = 0.25 * (1.0 + xi_s) * (1.0 + eta_s);
    let phi3 = 0.25 * (1.0 - xi_s) * (1.0 + eta_s);

    let h_base    = r * (1.0 - 2.0 * zeta);     // (1-ζ)(1-2ζ)
    let h_lateral = 4.0 * zeta * r;              // 4ζ(1-ζ) — =1 at ζ=0.5
    let h_apex    = zeta * (2.0 * zeta - 1.0);  // quadratic, =1 at ζ=1

    let mut n = [0.0f64; 13];
    // Base corners
    n[0] = f0 * h_base;
    n[1] = f1 * h_base;
    n[2] = f2 * h_base;
    n[3] = f3 * h_base;
    // Apex
    n[4] = h_apex;
    // Base edge midpoints
    n[5]  = g5  * h_base;
    n[6]  = g6  * h_base;
    n[8]  = g8  * h_base;
    n[10] = g10 * h_base;
    // Lateral edge midpoints
    n[7]  = phi0 * h_lateral;
    n[9]  = phi1 * h_lateral;
    n[11] = phi2 * h_lateral;
    n[12] = phi3 * h_lateral;
    n
}

/// Shape function derivatives for PYRAMID13.
///
/// Chain rule: dξₛ/dζ = ξ/r², dηₛ/dζ = η/r²
/// dN/dxi  = (∂f/∂ξₛ) * h / r
/// dN/deta = (∂f/∂ηₛ) * h / r
/// dN/dζ   = (∂f/∂ξₛ * ξ + ∂f/∂ηₛ * η) / r² * h + f * dh/dζ
#[inline]
fn pyramid13_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 13], [f64; 13], [f64; 13]) {
    let r = if 1.0 - zeta > PYRAMID5_EPS { 1.0 - zeta } else { PYRAMID5_EPS };
    let r2 = r * r;
    let xi_s  = (xi  / r).clamp(-1.0, 1.0);
    let eta_s = (eta / r).clamp(-1.0, 1.0);

    // Vertical blending functions and their derivatives w.r.t. ζ
    let h_base    = r * (1.0 - 2.0 * zeta);
    let h_lateral = 4.0 * zeta * r;
    let h_apex    = zeta * (2.0 * zeta - 1.0);
    let dh_base_dzeta    = -3.0 + 4.0 * zeta;
    let dh_lateral_dzeta = 4.0 * (1.0 - 2.0 * zeta);
    let dh_apex_dzeta    = 4.0 * zeta - 1.0;

    // Serendipity corner functions and derivatives w.r.t. (ξₛ, ηₛ)
    let f = [
        0.25 * (1.0 - xi_s) * (1.0 - eta_s) * (-xi_s - eta_s - 1.0),
        0.25 * (1.0 + xi_s) * (1.0 - eta_s) * ( xi_s - eta_s - 1.0),
        0.25 * (1.0 + xi_s) * (1.0 + eta_s) * ( xi_s + eta_s - 1.0),
        0.25 * (1.0 - xi_s) * (1.0 + eta_s) * (-xi_s + eta_s - 1.0),
    ];
    let df_dxi_s = [
        0.25 * (1.0 - eta_s) * (2.0 * xi_s + eta_s),
        0.25 * (1.0 - eta_s) * (2.0 * xi_s - eta_s),
        0.25 * (1.0 + eta_s) * (2.0 * xi_s + eta_s),
        0.25 * (1.0 + eta_s) * (2.0 * xi_s - eta_s),
    ];
    let df_deta_s = [
        0.25 * (1.0 - xi_s) * ( xi_s + 2.0 * eta_s),
        0.25 * (1.0 + xi_s) * (-xi_s + 2.0 * eta_s),
        0.25 * (1.0 + xi_s) * ( xi_s + 2.0 * eta_s),
        0.25 * (1.0 - xi_s) * (-xi_s + 2.0 * eta_s),
    ];

    // Edge bubble functions and derivatives
    let g5  = 0.5 * (1.0 - xi_s * xi_s)  * (1.0 - eta_s);
    let g6  = 0.5 * (1.0 - xi_s)          * (1.0 - eta_s * eta_s);
    let g8  = 0.5 * (1.0 + xi_s)          * (1.0 - eta_s * eta_s);
    let g10 = 0.5 * (1.0 - xi_s * xi_s)  * (1.0 + eta_s);

    let dg5_xi  = -xi_s * (1.0 - eta_s);
    let dg5_eta = -0.5 * (1.0 - xi_s * xi_s);
    let dg6_xi  = -0.5 * (1.0 - eta_s * eta_s);
    let dg6_eta = -(1.0 - xi_s) * eta_s;
    let dg8_xi  =  0.5 * (1.0 - eta_s * eta_s);
    let dg8_eta = -(1.0 + xi_s) * eta_s;
    let dg10_xi  = -xi_s * (1.0 + eta_s);
    let dg10_eta =  0.5 * (1.0 - xi_s * xi_s);

    // Bilinear functions for lateral edges
    let phi = [
        0.25 * (1.0 - xi_s) * (1.0 - eta_s),
        0.25 * (1.0 + xi_s) * (1.0 - eta_s),
        0.25 * (1.0 + xi_s) * (1.0 + eta_s),
        0.25 * (1.0 - xi_s) * (1.0 + eta_s),
    ];
    let dphi_xi  = [-0.25*(1.0-eta_s),  0.25*(1.0-eta_s),  0.25*(1.0+eta_s), -0.25*(1.0+eta_s)];
    let dphi_eta = [-0.25*(1.0-xi_s),  -0.25*(1.0+xi_s),   0.25*(1.0+xi_s),   0.25*(1.0-xi_s)];

    // Helper: compute dN/dxi, dN/deta, dN/dzeta for N = func * h
    // where func = func(xi_s, eta_s), h = h(zeta)
    let derivs = |func: f64, df_xi_s: f64, df_eta_s: f64, h: f64, dh: f64| -> (f64, f64, f64) {
        let d_xi   = df_xi_s  * h / r;
        let d_eta  = df_eta_s * h / r;
        let d_zeta = (df_xi_s * xi + df_eta_s * eta) / r2 * h + func * dh;
        (d_xi, d_eta, d_zeta)
    };

    let mut dxi   = [0.0f64; 13];
    let mut deta  = [0.0f64; 13];
    let mut dzeta = [0.0f64; 13];

    // Base corner nodes (0-3)
    for i in 0..4 {
        let (dx, de, dz) = derivs(f[i], df_dxi_s[i], df_deta_s[i], h_base, dh_base_dzeta);
        dxi[i] = dx; deta[i] = de; dzeta[i] = dz;
    }
    // Apex (4)
    dxi[4] = 0.0; deta[4] = 0.0; dzeta[4] = dh_apex_dzeta;
    // Base edge midpoints
    let (dx, de, dz) = derivs(g5,  dg5_xi,   dg5_eta,  h_base, dh_base_dzeta);  dxi[5]=dx; deta[5]=de; dzeta[5]=dz;
    let (dx, de, dz) = derivs(g6,  dg6_xi,   dg6_eta,  h_base, dh_base_dzeta);  dxi[6]=dx; deta[6]=de; dzeta[6]=dz;
    let (dx, de, dz) = derivs(g8,  dg8_xi,   dg8_eta,  h_base, dh_base_dzeta);  dxi[8]=dx; deta[8]=de; dzeta[8]=dz;
    let (dx, de, dz) = derivs(g10, dg10_xi,  dg10_eta, h_base, dh_base_dzeta);  dxi[10]=dx; deta[10]=de; dzeta[10]=dz;
    // Lateral edge midpoints (7, 9, 11, 12) ↔ corners (0, 1, 2, 3)
    let lateral = [(7usize, 0usize), (9, 1), (11, 2), (12, 3)];
    for (node, corner) in lateral {
        let (dx, de, dz) = derivs(phi[corner], dphi_xi[corner], dphi_eta[corner], h_lateral, dh_lateral_dzeta);
        dxi[node]=dx; deta[node]=de; dzeta[node]=dz;
    }

    (dxi, deta, dzeta)
}

/// Compute PYRAMID13 stiffness matrix Ke = ∫BᵀCB dV.
pub fn pyramid13_ke(coords: &[[f64; 3]; 13], e: f64, nu: f64) -> MatP13 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatP13::zeros();

    for (gp, &w) in PYRAMID13_GP.iter().zip(PYRAMID13_W.iter()) {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let (dxi, deta, dzeta) = pyramid13_derivs(xi, eta, zeta);
        let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

        let mut b = SMatrix::<f64, 6, 39>::zeros();
        fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

        ke += b.transpose() * c_mat * b * (det_j * w);
    }
    0.5 * (ke + ke.transpose())
}

/// Compute PYRAMID13 consistent mass matrix Me = ∫ρNᵀN dV.
pub fn pyramid13_me(coords: &[[f64; 3]; 13], rho: f64) -> MatP13 {
    let mut me = MatP13::zeros();

    for (gp, &w) in PYRAMID13_GP.iter().zip(PYRAMID13_W.iter()) {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let n_sf = pyramid13_shape(xi, eta, zeta);
        let (dxi, deta, dzeta) = pyramid13_derivs(xi, eta, zeta);
        let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

        let fac = rho * det_j * w;
        for i in 0..13 {
            for j in 0..13 {
                let val = n_sf[i] * n_sf[j] * fac;
                for d in 0..3 {
                    me[(3 * i + d, 3 * j + d)] += val;
                }
            }
        }
    }
    0.5 * (me + me.transpose())
}

// ============================================================================
// HEXA20
// ============================================================================

/// 20-node quadratic hexahedron (serendipity brick) element.
///
/// Node ordering (Gmsh convention):
///
/// ```text
///    3----13----2
///    |\\         |\\
///    | 15       | 14
///    9  \\       11 \\
///    |   7----19+---6
///    |   |      |   |
///    0---+-8----1   |
///     \\  17      \\  18
///     10 |       12|
///       \\|         \\|
///        4----16----5
/// ```
///
/// Corner nodes: 0–7 (same ordering as HEXA8)
///
/// Edge midpoints:
///  8: edge 0-1,  9: edge 0-3, 10: edge 0-4, 11: edge 1-2
/// 12: edge 1-5, 13: edge 2-3, 14: edge 2-6, 15: edge 3-7
/// 16: edge 4-5, 17: edge 4-7, 18: edge 5-6, 19: edge 6-7
///
/// Natural coordinates: ξ, η, ζ ∈ [−1, 1]

/// 3×3×3 Gauss quadrature for HEXA20.
// SQRT35 and 3-pt 1D rule available via crate::quadrature::GAUSS3_PTS / GAUSS3_W.
// Using local aliases below for the HEXA20 loops to keep code readable.
const SQRT35: f64 = 0.774_596_669_241_483_4; // √(3/5)
const HEXA20_GP1D: [f64; 3] = [-SQRT35, 0.0, SQRT35];
const HEXA20_W1D:  [f64; 3] = [5.0/9.0, 8.0/9.0, 5.0/9.0];


/// Natural coordinates of the 8 corner nodes of a hex.
const HEX_CORNERS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0], // 0
    [ 1.0, -1.0, -1.0], // 1
    [ 1.0,  1.0, -1.0], // 2
    [-1.0,  1.0, -1.0], // 3
    [-1.0, -1.0,  1.0], // 4
    [ 1.0, -1.0,  1.0], // 5
    [ 1.0,  1.0,  1.0], // 6
    [-1.0,  1.0,  1.0], // 7
];

/// Serendipity shape functions for HEXA20.
#[inline]
fn hexa20_shape(xi: f64, eta: f64, zeta: f64) -> [f64; 20] {
    let mut n = [0.0f64; 20];

    // Corner nodes (0-7): serendipity formula
    // Nᵢ = 1/8 * (1 + ξᵢξ)(1 + ηᵢη)(1 + ζᵢζ)(ξᵢξ + ηᵢη + ζᵢζ − 2)
    for (i, c) in HEX_CORNERS.iter().enumerate() {
        n[i] = 0.125
            * (1.0 + c[0] * xi)
            * (1.0 + c[1] * eta)
            * (1.0 + c[2] * zeta)
            * (c[0] * xi + c[1] * eta + c[2] * zeta - 2.0);
    }

    // Mid-edge nodes (8-19): one quadratic, two linear factors
    n[8]  = 0.25 * (1.0 - xi*xi)   * (1.0 - eta)  * (1.0 - zeta); // edge 0-1
    n[9]  = 0.25 * (1.0 - xi)      * (1.0 - eta*eta) * (1.0 - zeta); // edge 0-3
    n[10] = 0.25 * (1.0 - xi)      * (1.0 - eta)  * (1.0 - zeta*zeta); // edge 0-4
    n[11] = 0.25 * (1.0 + xi)      * (1.0 - eta*eta) * (1.0 - zeta); // edge 1-2
    n[12] = 0.25 * (1.0 + xi)      * (1.0 - eta)  * (1.0 - zeta*zeta); // edge 1-5
    n[13] = 0.25 * (1.0 - xi*xi)   * (1.0 + eta)  * (1.0 - zeta); // edge 2-3
    n[14] = 0.25 * (1.0 + xi)      * (1.0 + eta)  * (1.0 - zeta*zeta); // edge 2-6
    n[15] = 0.25 * (1.0 - xi)      * (1.0 + eta)  * (1.0 - zeta*zeta); // edge 3-7
    n[16] = 0.25 * (1.0 - xi*xi)   * (1.0 - eta)  * (1.0 + zeta); // edge 4-5
    n[17] = 0.25 * (1.0 - xi)      * (1.0 - eta*eta) * (1.0 + zeta); // edge 4-7
    n[18] = 0.25 * (1.0 + xi)      * (1.0 - eta*eta) * (1.0 + zeta); // edge 5-6
    n[19] = 0.25 * (1.0 - xi*xi)   * (1.0 + eta)  * (1.0 + zeta); // edge 6-7
    n
}

/// Analytical shape function derivatives for HEXA20.
#[inline]
fn hexa20_derivs(xi: f64, eta: f64, zeta: f64) -> ([f64; 20], [f64; 20], [f64; 20]) {
    let mut dxi   = [0.0f64; 20];
    let mut deta  = [0.0f64; 20];
    let mut dzeta = [0.0f64; 20];

    // Corner nodes
    for (i, c) in HEX_CORNERS.iter().enumerate() {
        let (a, b, g) = (c[0], c[1], c[2]);
        let u = 1.0 + a * xi;
        let v = 1.0 + b * eta;
        let w = 1.0 + g * zeta;
        let s = a * xi + b * eta + g * zeta - 2.0;
        dxi[i]   = 0.125 * a * v * w * s + 0.125 * u * v * w * a;
        deta[i]  = 0.125 * u * b * w * s + 0.125 * u * v * w * b;
        dzeta[i] = 0.125 * u * v * g * s + 0.125 * u * v * w * g;
    }

    // Mid-edge nodes (analytical derivatives)
    // n[8] = 0.25*(1-xi²)*(1-eta)*(1-zeta)
    dxi[8]   = 0.25 * (-2.0*xi)   * (1.0-eta)  * (1.0-zeta);
    deta[8]  = 0.25 * (1.0-xi*xi) * (-1.0)      * (1.0-zeta);
    dzeta[8] = 0.25 * (1.0-xi*xi) * (1.0-eta)   * (-1.0);

    // n[9] = 0.25*(1-xi)*(1-eta²)*(1-zeta)
    dxi[9]   = 0.25 * (-1.0)      * (1.0-eta*eta) * (1.0-zeta);
    deta[9]  = 0.25 * (1.0-xi)    * (-2.0*eta)    * (1.0-zeta);
    dzeta[9] = 0.25 * (1.0-xi)    * (1.0-eta*eta) * (-1.0);

    // n[10] = 0.25*(1-xi)*(1-eta)*(1-zeta²)
    dxi[10]   = 0.25 * (-1.0)     * (1.0-eta)     * (1.0-zeta*zeta);
    deta[10]  = 0.25 * (1.0-xi)   * (-1.0)        * (1.0-zeta*zeta);
    dzeta[10] = 0.25 * (1.0-xi)   * (1.0-eta)     * (-2.0*zeta);

    // n[11] = 0.25*(1+xi)*(1-eta²)*(1-zeta)
    dxi[11]   = 0.25 * (1.0)      * (1.0-eta*eta) * (1.0-zeta);
    deta[11]  = 0.25 * (1.0+xi)   * (-2.0*eta)    * (1.0-zeta);
    dzeta[11] = 0.25 * (1.0+xi)   * (1.0-eta*eta) * (-1.0);

    // n[12] = 0.25*(1+xi)*(1-eta)*(1-zeta²)
    dxi[12]   = 0.25 * (1.0)      * (1.0-eta)     * (1.0-zeta*zeta);
    deta[12]  = 0.25 * (1.0+xi)   * (-1.0)        * (1.0-zeta*zeta);
    dzeta[12] = 0.25 * (1.0+xi)   * (1.0-eta)     * (-2.0*zeta);

    // n[13] = 0.25*(1-xi²)*(1+eta)*(1-zeta)
    dxi[13]   = 0.25 * (-2.0*xi)  * (1.0+eta)     * (1.0-zeta);
    deta[13]  = 0.25 * (1.0-xi*xi) * (1.0)        * (1.0-zeta);
    dzeta[13] = 0.25 * (1.0-xi*xi) * (1.0+eta)    * (-1.0);

    // n[14] = 0.25*(1+xi)*(1+eta)*(1-zeta²)
    dxi[14]   = 0.25 * (1.0)      * (1.0+eta)     * (1.0-zeta*zeta);
    deta[14]  = 0.25 * (1.0+xi)   * (1.0)         * (1.0-zeta*zeta);
    dzeta[14] = 0.25 * (1.0+xi)   * (1.0+eta)     * (-2.0*zeta);

    // n[15] = 0.25*(1-xi)*(1+eta)*(1-zeta²)
    dxi[15]   = 0.25 * (-1.0)     * (1.0+eta)     * (1.0-zeta*zeta);
    deta[15]  = 0.25 * (1.0-xi)   * (1.0)         * (1.0-zeta*zeta);
    dzeta[15] = 0.25 * (1.0-xi)   * (1.0+eta)     * (-2.0*zeta);

    // n[16] = 0.25*(1-xi²)*(1-eta)*(1+zeta)
    dxi[16]   = 0.25 * (-2.0*xi)  * (1.0-eta)     * (1.0+zeta);
    deta[16]  = 0.25 * (1.0-xi*xi) * (-1.0)       * (1.0+zeta);
    dzeta[16] = 0.25 * (1.0-xi*xi) * (1.0-eta)    * (1.0);

    // n[17] = 0.25*(1-xi)*(1-eta²)*(1+zeta)
    dxi[17]   = 0.25 * (-1.0)     * (1.0-eta*eta) * (1.0+zeta);
    deta[17]  = 0.25 * (1.0-xi)   * (-2.0*eta)    * (1.0+zeta);
    dzeta[17] = 0.25 * (1.0-xi)   * (1.0-eta*eta) * (1.0);

    // n[18] = 0.25*(1+xi)*(1-eta²)*(1+zeta)
    dxi[18]   = 0.25 * (1.0)      * (1.0-eta*eta) * (1.0+zeta);
    deta[18]  = 0.25 * (1.0+xi)   * (-2.0*eta)    * (1.0+zeta);
    dzeta[18] = 0.25 * (1.0+xi)   * (1.0-eta*eta) * (1.0);

    // n[19] = 0.25*(1-xi²)*(1+eta)*(1+zeta)
    dxi[19]   = 0.25 * (-2.0*xi)  * (1.0+eta)     * (1.0+zeta);
    deta[19]  = 0.25 * (1.0-xi*xi) * (1.0)        * (1.0+zeta);
    dzeta[19] = 0.25 * (1.0-xi*xi) * (1.0+eta)    * (1.0);

    (dxi, deta, dzeta)
}

/// Compute HEXA20 stiffness matrix Ke = ∫BᵀCB dV.
pub fn hexa20_ke(coords: &[[f64; 3]; 20], e: f64, nu: f64) -> MatH20 {
    let c_mat = isotropic_c(e, nu);
    let mut ke = MatH20::zeros();

    for (ki, &xi) in HEXA20_GP1D.iter().enumerate() {
        for (kj, &eta) in HEXA20_GP1D.iter().enumerate() {
            for (kk, &zeta) in HEXA20_GP1D.iter().enumerate() {
                let w = HEXA20_W1D[ki] * HEXA20_W1D[kj] * HEXA20_W1D[kk];
                let (dxi, deta, dzeta) = hexa20_derivs(xi, eta, zeta);
                let (_, det_j, inv_j) = jacobian_3d(&dxi, &deta, &dzeta, coords);

                let mut b = SMatrix::<f64, 6, 60>::zeros();
                fill_b(&dxi, &deta, &dzeta, &inv_j, &mut b);

                ke += b.transpose() * c_mat * b * (det_j * w);
            }
        }
    }
    0.5 * (ke + ke.transpose())
}

/// Compute HEXA20 consistent mass matrix Me = ∫ρNᵀN dV.
pub fn hexa20_me(coords: &[[f64; 3]; 20], rho: f64) -> MatH20 {
    let mut me = MatH20::zeros();

    for (ki, &xi) in HEXA20_GP1D.iter().enumerate() {
        for (kj, &eta) in HEXA20_GP1D.iter().enumerate() {
            for (kk, &zeta) in HEXA20_GP1D.iter().enumerate() {
                let w = HEXA20_W1D[ki] * HEXA20_W1D[kj] * HEXA20_W1D[kk];
                let n_sf = hexa20_shape(xi, eta, zeta);
                let (dxi, deta, dzeta) = hexa20_derivs(xi, eta, zeta);
                let (_, det_j, _) = jacobian_3d(&dxi, &deta, &dzeta, coords);

                let fac = rho * det_j * w;
                for i in 0..20 {
                    for j in 0..20 {
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
