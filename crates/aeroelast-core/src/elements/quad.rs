// QUAD Plane Elasticity Element Kernels
//
// Implements QUAD4, QUAD8, and QUAD9 isoparametric elements for plane
// elasticity (plane strain formulation matching the Python reference).
//
// Formulation:
//   K = ∫ Bᵀ·C·B dΩ
//   M = ∫ ρ·Nᵀ·N dΩ
//   f = ∫ Nᵀ·b dΩ
//
// Constitutive matrix: plane strain with Lamé coefficients (matches Python).
// No thickness multiplier (matching Python reference).

use nalgebra::SMatrix;

use crate::quadrature::{GAUSS2_PTS, GAUSS2_W, GAUSS3_PTS, GAUSS3_W};

// ============================================================================
// Type aliases
// ============================================================================

type Mat8 = SMatrix<f64, 8, 8>;
type Vec8 = nalgebra::SVector<f64, 8>;
type Mat16 = SMatrix<f64, 16, 16>;
type Vec16 = nalgebra::SVector<f64, 16>;
type Mat18 = SMatrix<f64, 18, 18>;
type Vec18 = nalgebra::SVector<f64, 18>;

// ============================================================================
// Constitutive matrix (plane strain — matches Python)
// ============================================================================

/// C matrix for plane strain: C[row][col]
#[inline]
fn constitutive(e: f64, nu: f64) -> [[f64; 3]; 3] {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    [
        [lambda + 2.0 * mu, lambda, 0.0],
        [lambda, lambda + 2.0 * mu, 0.0],
        [0.0, 0.0, mu],
    ]
}

// ============================================================================
// Jacobian
// ============================================================================

/// Returns (J, det_J, J_inv)
#[inline]
fn jacobian(dn_dxi: &[f64], dn_deta: &[f64], x: &[f64], y: &[f64]) -> ([[f64; 2]; 2], f64, [[f64; 2]; 2]) {
    let j00: f64 = dn_dxi.iter().zip(x).map(|(d, xi)| d * xi).sum();
    let j01: f64 = dn_dxi.iter().zip(y).map(|(d, yi)| d * yi).sum();
    let j10: f64 = dn_deta.iter().zip(x).map(|(d, xi)| d * xi).sum();
    let j11: f64 = dn_deta.iter().zip(y).map(|(d, yi)| d * yi).sum();
    let det = j00 * j11 - j01 * j10;
    let inv_det = 1.0 / det;
    (
        [[j00, j01], [j10, j11]],
        det,
        [[j11 * inv_det, -j01 * inv_det], [-j10 * inv_det, j00 * inv_det]],
    )
}

#[inline]
fn phys_derivs(dn_dxi: &[f64], dn_deta: &[f64], j_inv: &[[f64; 2]; 2]) -> (Vec<f64>, Vec<f64>) {
    let n = dn_dxi.len();
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; n];
    for i in 0..n {
        dx[i] = j_inv[0][0] * dn_dxi[i] + j_inv[0][1] * dn_deta[i];
        dy[i] = j_inv[1][0] * dn_dxi[i] + j_inv[1][1] * dn_deta[i];
    }
    (dx, dy)
}

// ============================================================================
// B-matrix contribution: Bᵀ·C·B, returns (ndof x ndof) contribution
// Adds scale * Bᵀ·C·B to ke (row-major flat, n_dof × n_dof)
// ============================================================================

#[inline]
fn accumulate_ke_flat(dn_dx: &[f64], dn_dy: &[f64], c: &[[f64; 3]; 3], det_j: f64, w: f64, ke: &mut [f64]) {
    let n = dn_dx.len();
    let n_dof = 2 * n;
    let scale = det_j * w;

    for i in 0..n {
        let bi = [dn_dx[i], 0.0, dn_dy[i]]; // column 2i of B
        let bi1 = [0.0, dn_dy[i], dn_dx[i]]; // column 2i+1 of B

        for j in 0..n {
            let bj = [dn_dx[j], 0.0, dn_dy[j]];
            let bj1 = [0.0, dn_dy[j], dn_dx[j]];

            let cbj = dot3x3(c, &bj);
            let cbj1 = dot3x3(c, &bj1);

            ke[(2 * i) * n_dof + 2 * j] += scale * dot3(&bi, &cbj);
            ke[(2 * i) * n_dof + 2 * j + 1] += scale * dot3(&bi, &cbj1);
            ke[(2 * i + 1) * n_dof + 2 * j] += scale * dot3(&bi1, &cbj);
            ke[(2 * i + 1) * n_dof + 2 * j + 1] += scale * dot3(&bi1, &cbj1);
        }
    }
}

#[inline(always)]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn dot3x3(c: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        c[0][0] * v[0] + c[0][1] * v[1] + c[0][2] * v[2],
        c[1][0] * v[0] + c[1][1] * v[1] + c[1][2] * v[2],
        c[2][0] * v[0] + c[2][1] * v[1] + c[2][2] * v[2],
    ]
}

fn symmetrize_flat(ke: &mut [f64], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            let avg = 0.5 * (ke[i * n + j] + ke[j * n + i]);
            ke[i * n + j] = avg;
            ke[j * n + i] = avg;
        }
    }
}

// ============================================================================
// QUAD4 shape functions
// ============================================================================

#[inline]
fn quad4_shape(xi: f64, eta: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]
}

#[inline]
fn quad4_derivs(xi: f64, eta: f64) -> ([f64; 4], [f64; 4]) {
    (
        [
            0.25 * (-(1.0 - eta)),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            0.25 * (-(1.0 + eta)),
        ],
        [
            0.25 * (-(1.0 - xi)),
            0.25 * (-(1.0 + xi)),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ],
    )
}

// ============================================================================
// QUAD8 shape functions (serendipity)
// ============================================================================

#[inline]
fn quad8_shape(xi: f64, eta: f64) -> [f64; 8] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta) * (-1.0 - xi - eta),
        0.25 * (1.0 + xi) * (1.0 - eta) * (-1.0 + xi - eta),
        0.25 * (1.0 + xi) * (1.0 + eta) * (-1.0 + xi + eta),
        0.25 * (1.0 - xi) * (1.0 + eta) * (-1.0 - xi + eta),
        0.5 * (1.0 - xi * xi) * (1.0 - eta),
        0.5 * (1.0 + xi) * (1.0 - eta * eta),
        0.5 * (1.0 - xi * xi) * (1.0 + eta),
        0.5 * (1.0 - xi) * (1.0 - eta * eta),
    ]
}

#[inline]
fn quad8_derivs(xi: f64, eta: f64) -> ([f64; 8], [f64; 8]) {
    (
        [
            0.25 * (1.0 - eta) * (2.0 * xi + eta),
            0.25 * (1.0 - eta) * (2.0 * xi - eta),
            0.25 * (1.0 + eta) * (2.0 * xi + eta),
            0.25 * (1.0 + eta) * (2.0 * xi - eta),
            -xi * (1.0 - eta),
            0.5 * (1.0 - eta * eta),
            -xi * (1.0 + eta),
            -0.5 * (1.0 - eta * eta),
        ],
        [
            0.25 * (1.0 - xi) * (2.0 * eta + xi),
            0.25 * (1.0 + xi) * (2.0 * eta - xi),
            0.25 * (1.0 + xi) * (2.0 * eta + xi),
            0.25 * (1.0 - xi) * (2.0 * eta - xi),
            -0.5 * (1.0 - xi * xi),
            -(1.0 + xi) * eta,
            0.5 * (1.0 - xi * xi),
            -(1.0 - xi) * eta,
        ],
    )
}

// ============================================================================
// QUAD9 shape functions (biquadratic Lagrange)
// ============================================================================

#[inline]
fn quad9_shape(xi: f64, eta: f64) -> [f64; 9] {
    [
        0.25 * xi * eta * (xi - 1.0) * (eta - 1.0),
        0.25 * xi * eta * (xi + 1.0) * (eta - 1.0),
        0.25 * xi * eta * (xi + 1.0) * (eta + 1.0),
        0.25 * xi * eta * (xi - 1.0) * (eta + 1.0),
        0.5 * (1.0 - xi * xi) * eta * (eta - 1.0),
        0.5 * xi * (xi + 1.0) * (1.0 - eta * eta),
        0.5 * (1.0 - xi * xi) * eta * (eta + 1.0),
        0.5 * xi * (xi - 1.0) * (1.0 - eta * eta),
        (1.0 - xi * xi) * (1.0 - eta * eta),
    ]
}

#[inline]
fn quad9_derivs(xi: f64, eta: f64) -> ([f64; 9], [f64; 9]) {
    (
        [
            0.25 * eta * (eta - 1.0) * (2.0 * xi - 1.0),
            0.25 * eta * (eta - 1.0) * (2.0 * xi + 1.0),
            0.25 * eta * (eta + 1.0) * (2.0 * xi + 1.0),
            0.25 * eta * (eta + 1.0) * (2.0 * xi - 1.0),
            -xi * eta * (eta - 1.0),
            0.5 * (1.0 - eta * eta) * (2.0 * xi + 1.0),
            -xi * eta * (eta + 1.0),
            0.5 * (1.0 - eta * eta) * (2.0 * xi - 1.0),
            -2.0 * xi * (1.0 - eta * eta),
        ],
        [
            0.25 * xi * (xi - 1.0) * (2.0 * eta - 1.0),
            0.25 * xi * (xi + 1.0) * (2.0 * eta - 1.0),
            0.25 * xi * (xi + 1.0) * (2.0 * eta + 1.0),
            0.25 * xi * (xi - 1.0) * (2.0 * eta + 1.0),
            0.5 * (1.0 - xi * xi) * (2.0 * eta - 1.0),
            -xi * (xi + 1.0) * eta,
            0.5 * (1.0 - xi * xi) * (2.0 * eta + 1.0),
            -xi * (xi - 1.0) * eta,
            -2.0 * eta * (1.0 - xi * xi),
        ],
    )
}

// ============================================================================
// QUAD4 Precomputed
// ============================================================================

/// All geometry for a QUAD4 element (4 nodes × 2 DOFs = 8 DOFs).
pub struct Quad4Precomputed {
    pub x: [f64; 4],
    pub y: [f64; 4],
}

impl Quad4Precomputed {
    /// Create from node coordinates.
    pub fn new(coords: &[[f64; 2]; 4]) -> Self {
        let mut x = [0.0; 4];
        let mut y = [0.0; 4];
        for i in 0..4 {
            x[i] = coords[i][0];
            y[i] = coords[i][1];
        }
        Self { x, y }
    }

    /// 8×8 stiffness matrix (plane strain, 2×2 Gauss).
    pub fn compute_ke_global(&self, e: f64, nu: f64) -> [f64; 64] {
        let c = constitutive(e, nu);
        let mut ke = [0.0f64; 64];

        for i in 0..2 {
            for j in 0..2 {
                let xi = GAUSS2_PTS[i];
                let eta = GAUSS2_PTS[j];
                let w = GAUSS2_W[i] * GAUSS2_W[j];
                let (dn_xi, dn_eta) = quad4_derivs(xi, eta);
                let (_, det_j, j_inv) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let (dx, dy) = phys_derivs(&dn_xi, &dn_eta, &j_inv);
                accumulate_ke_flat(&dx, &dy, &c, det_j, w, &mut ke);
            }
        }
        symmetrize_flat(&mut ke, 8);
        ke
    }

    /// 8×8 consistent mass matrix (2×2 Gauss).
    pub fn compute_me_global(&self, rho: f64) -> [f64; 64] {
        let mut me = [0.0f64; 64];

        for i in 0..2 {
            for j in 0..2 {
                let xi = GAUSS2_PTS[i];
                let eta = GAUSS2_PTS[j];
                let w = GAUSS2_W[i] * GAUSS2_W[j];
                let n = quad4_shape(xi, eta);
                let (dn_xi, dn_eta) = quad4_derivs(xi, eta);
                let (_, det_j, _) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let scale = rho * det_j * w;
                for p in 0..4 {
                    for q in 0..4 {
                        let v = scale * n[p] * n[q];
                        me[(2 * p) * 8 + 2 * q] += v;
                        me[(2 * p + 1) * 8 + 2 * q + 1] += v;
                    }
                }
            }
        }
        symmetrize_flat(&mut me, 8);
        me
    }

    /// 8-element body load vector (2×2 Gauss).
    pub fn compute_body_load_global(&self, b: [f64; 2]) -> [f64; 8] {
        let mut f = [0.0f64; 8];
        for i in 0..2 {
            for j in 0..2 {
                let xi = GAUSS2_PTS[i];
                let eta = GAUSS2_PTS[j];
                let w = GAUSS2_W[i] * GAUSS2_W[j];
                let n = quad4_shape(xi, eta);
                let (dn_xi, dn_eta) = quad4_derivs(xi, eta);
                let (_, det_j, _) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let scale = det_j * w;
                for p in 0..4 {
                    f[2 * p] += scale * n[p] * b[0];
                    f[2 * p + 1] += scale * n[p] * b[1];
                }
            }
        }
        f
    }
}

// ============================================================================
// QUAD8 Precomputed
// ============================================================================

/// All geometry for a QUAD8 element (8 nodes × 2 DOFs = 16 DOFs).
pub struct Quad8Precomputed {
    pub x: [f64; 8],
    pub y: [f64; 8],
}

impl Quad8Precomputed {
    pub fn new(coords: &[[f64; 2]; 8]) -> Self {
        let mut x = [0.0; 8];
        let mut y = [0.0; 8];
        for i in 0..8 {
            x[i] = coords[i][0];
            y[i] = coords[i][1];
        }
        Self { x, y }
    }

    /// 16×16 stiffness matrix (3×3 Gauss).
    pub fn compute_ke_global(&self, e: f64, nu: f64) -> [f64; 256] {
        let c = constitutive(e, nu);
        let mut ke = [0.0f64; 256];
        for i in 0..3 {
            for j in 0..3 {
                let xi = GAUSS3_PTS[i];
                let eta = GAUSS3_PTS[j];
                let w = GAUSS3_W[i] * GAUSS3_W[j];
                let (dn_xi, dn_eta) = quad8_derivs(xi, eta);
                let (_, det_j, j_inv) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let (dx, dy) = phys_derivs(&dn_xi, &dn_eta, &j_inv);
                accumulate_ke_flat(&dx, &dy, &c, det_j, w, &mut ke);
            }
        }
        symmetrize_flat(&mut ke, 16);
        ke
    }

    /// 16×16 consistent mass matrix (3×3 Gauss).
    pub fn compute_me_global(&self, rho: f64) -> [f64; 256] {
        let mut me = [0.0f64; 256];
        for i in 0..3 {
            for j in 0..3 {
                let xi = GAUSS3_PTS[i];
                let eta = GAUSS3_PTS[j];
                let w = GAUSS3_W[i] * GAUSS3_W[j];
                let n = quad8_shape(xi, eta);
                let (dn_xi, dn_eta) = quad8_derivs(xi, eta);
                let (_, det_j, _) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let scale = rho * det_j * w;
                for p in 0..8 {
                    for q in 0..8 {
                        let v = scale * n[p] * n[q];
                        me[(2 * p) * 16 + 2 * q] += v;
                        me[(2 * p + 1) * 16 + 2 * q + 1] += v;
                    }
                }
            }
        }
        symmetrize_flat(&mut me, 16);
        me
    }

    /// 16-element body load vector (3×3 Gauss).
    pub fn compute_body_load_global(&self, b: [f64; 2]) -> [f64; 16] {
        let mut f = [0.0f64; 16];
        for i in 0..3 {
            for j in 0..3 {
                let xi = GAUSS3_PTS[i];
                let eta = GAUSS3_PTS[j];
                let w = GAUSS3_W[i] * GAUSS3_W[j];
                let n = quad8_shape(xi, eta);
                let (dn_xi, dn_eta) = quad8_derivs(xi, eta);
                let (_, det_j, _) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let scale = det_j * w;
                for p in 0..8 {
                    f[2 * p] += scale * n[p] * b[0];
                    f[2 * p + 1] += scale * n[p] * b[1];
                }
            }
        }
        f
    }
}

// ============================================================================
// QUAD9 Precomputed
// ============================================================================

/// All geometry for a QUAD9 element (9 nodes × 2 DOFs = 18 DOFs).
pub struct Quad9Precomputed {
    pub x: [f64; 9],
    pub y: [f64; 9],
}

impl Quad9Precomputed {
    pub fn new(coords: &[[f64; 2]; 9]) -> Self {
        let mut x = [0.0; 9];
        let mut y = [0.0; 9];
        for i in 0..9 {
            x[i] = coords[i][0];
            y[i] = coords[i][1];
        }
        Self { x, y }
    }

    /// 18×18 stiffness matrix (3×3 Gauss).
    pub fn compute_ke_global(&self, e: f64, nu: f64) -> [f64; 324] {
        let c = constitutive(e, nu);
        let mut ke = [0.0f64; 324];
        for i in 0..3 {
            for j in 0..3 {
                let xi = GAUSS3_PTS[i];
                let eta = GAUSS3_PTS[j];
                let w = GAUSS3_W[i] * GAUSS3_W[j];
                let (dn_xi, dn_eta) = quad9_derivs(xi, eta);
                let (_, det_j, j_inv) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let (dx, dy) = phys_derivs(&dn_xi, &dn_eta, &j_inv);
                accumulate_ke_flat(&dx, &dy, &c, det_j, w, &mut ke);
            }
        }
        symmetrize_flat(&mut ke, 18);
        ke
    }

    /// 18×18 consistent mass matrix (3×3 Gauss).
    pub fn compute_me_global(&self, rho: f64) -> [f64; 324] {
        let mut me = [0.0f64; 324];
        for i in 0..3 {
            for j in 0..3 {
                let xi = GAUSS3_PTS[i];
                let eta = GAUSS3_PTS[j];
                let w = GAUSS3_W[i] * GAUSS3_W[j];
                let n = quad9_shape(xi, eta);
                let (dn_xi, dn_eta) = quad9_derivs(xi, eta);
                let (_, det_j, _) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let scale = rho * det_j * w;
                for p in 0..9 {
                    for q in 0..9 {
                        let v = scale * n[p] * n[q];
                        me[(2 * p) * 18 + 2 * q] += v;
                        me[(2 * p + 1) * 18 + 2 * q + 1] += v;
                    }
                }
            }
        }
        symmetrize_flat(&mut me, 18);
        me
    }

    /// 18-element body load vector (3×3 Gauss).
    pub fn compute_body_load_global(&self, b: [f64; 2]) -> [f64; 18] {
        let mut f = [0.0f64; 18];
        for i in 0..3 {
            for j in 0..3 {
                let xi = GAUSS3_PTS[i];
                let eta = GAUSS3_PTS[j];
                let w = GAUSS3_W[i] * GAUSS3_W[j];
                let n = quad9_shape(xi, eta);
                let (dn_xi, dn_eta) = quad9_derivs(xi, eta);
                let (_, det_j, _) = jacobian(&dn_xi, &dn_eta, &self.x, &self.y);
                let scale = det_j * w;
                for p in 0..9 {
                    f[2 * p] += scale * n[p] * b[0];
                    f[2 * p + 1] += scale * n[p] * b[1];
                }
            }
        }
        f
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quad4() -> Quad4Precomputed {
        Quad4Precomputed::new(&[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    }

    fn make_quad8() -> Quad8Precomputed {
        Quad8Precomputed::new(&[
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
            [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        ])
    }

    fn make_quad9() -> Quad9Precomputed {
        Quad9Precomputed::new(&[
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
            [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
            [0.0, 0.0],
        ])
    }

    fn is_symmetric(flat: &[f64], n: usize, tol: f64) -> bool {
        for i in 0..n {
            for j in 0..n {
                if (flat[i * n + j] - flat[j * n + i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn diagonal_positive(flat: &[f64], n: usize) -> bool {
        (0..n).all(|i| flat[i * n + i] > 0.0)
    }

    fn all_finite(flat: &[f64]) -> bool {
        flat.iter().all(|v| v.is_finite())
    }

    #[test]
    fn test_quad4_ke_symmetric_positive_finite() {
        let pre = make_quad4();
        let ke = pre.compute_ke_global(2.0e11, 0.3);
        assert!(all_finite(&ke), "QUAD4 Ke: not finite");
        assert!(is_symmetric(&ke, 8, 1e-8), "QUAD4 Ke: not symmetric");
        assert!(diagonal_positive(&ke, 8), "QUAD4 Ke: diagonal not positive");
    }

    #[test]
    fn test_quad4_me_symmetric_positive_finite() {
        let pre = make_quad4();
        let me = pre.compute_me_global(7800.0);
        assert!(all_finite(&me), "QUAD4 Me: not finite");
        assert!(is_symmetric(&me, 8, 1e-12), "QUAD4 Me: not symmetric");
        assert!(diagonal_positive(&me, 8), "QUAD4 Me: diagonal not positive");
    }

    #[test]
    fn test_quad4_body_load_gravity() {
        // Unit square [-1,1]^2 has area 4.
        // b = [0, rho*g] → total fy = rho*g*area
        let pre = make_quad4();
        let rho = 7800.0f64;
        let g = 9.81f64;
        let f = pre.compute_body_load_global([0.0, -rho * g]);
        let total_fy: f64 = (0..4).map(|i| f[2 * i + 1]).sum();
        let expected = -rho * g * 4.0; // area = 4
        assert!(
            (total_fy - expected).abs() < 1e-4,
            "QUAD4 body load: got {total_fy}, expected {expected}"
        );
    }

    #[test]
    fn test_quad8_ke_symmetric_finite() {
        let pre = make_quad8();
        let ke = pre.compute_ke_global(2.0e11, 0.3);
        assert!(all_finite(&ke), "QUAD8 Ke: not finite");
        assert!(is_symmetric(&ke, 16, 1e-8), "QUAD8 Ke: not symmetric");
    }

    #[test]
    fn test_quad8_me_symmetric_finite() {
        let pre = make_quad8();
        let me = pre.compute_me_global(7800.0);
        assert!(all_finite(&me), "QUAD8 Me: not finite");
        assert!(is_symmetric(&me, 16, 1e-12), "QUAD8 Me: not symmetric");
    }

    #[test]
    fn test_quad9_ke_symmetric_finite() {
        let pre = make_quad9();
        let ke = pre.compute_ke_global(2.0e11, 0.3);
        assert!(all_finite(&ke), "QUAD9 Ke: not finite");
        assert!(is_symmetric(&ke, 18, 1e-8), "QUAD9 Ke: not symmetric");
    }

    #[test]
    fn test_quad9_me_symmetric_finite() {
        let pre = make_quad9();
        let me = pre.compute_me_global(7800.0);
        assert!(all_finite(&me), "QUAD9 Me: not finite");
        assert!(is_symmetric(&me, 18, 1e-12), "QUAD9 Me: not symmetric");
    }

    #[test]
    fn test_quad9_body_load_gravity() {
        let pre = make_quad9();
        let rho = 7800.0f64;
        let g = 9.81f64;
        let f = pre.compute_body_load_global([0.0, -rho * g]);
        let total_fy: f64 = (0..9).map(|i| f[2 * i + 1]).sum();
        let area = 4.0f64;
        let expected = -rho * g * area;
        assert!(
            (total_fy - expected).abs() < 1e-3,
            "QUAD9 body load: got {total_fy}, expected {expected}"
        );
    }

    /// Suppress dead-code warnings for the nalgebra type aliases
    #[allow(dead_code)]
    fn _use_types() {
        let _: Mat8 = Mat8::zeros();
        let _: Vec8 = Vec8::zeros();
        let _: Mat16 = Mat16::zeros();
        let _: Vec16 = Vec16::zeros();
        let _: Mat18 = Mat18::zeros();
        let _: Vec18 = Vec18::zeros();
    }
}
