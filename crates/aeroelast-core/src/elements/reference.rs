// Reference Element Traits
//
// Defines `ReferenceElement2D` and `ReferenceElement3D` traits plus generic
// Gauss integration helpers for stiffness (Ke) and consistent mass (Me).
//
// Design decisions:
// - Return `&'static [[f64; D]]` slices for Gauss points/weights — these are
//   static references to const arrays, zero heap allocation in the hot path.
// - Avoid GATs / const-generic arithmetic in return types — requires unstable
//   `generic_const_exprs`. Callers pass pre-allocated `&mut [f64]` slices,
//   which they stack-allocate using fixed-size arrays.
// - MITC3/MITC4 and composite elements are intentionally excluded — they have
//   special shear-correction terms that do not fit the plain Bᵀ·C·B pattern.

// ============================================================================
// 2D trait (plane stress / plane strain quad elements)
// ============================================================================

/// A 2D reference element defined on [−1,1]² (quads).
///
/// `NODES`: number of nodes (e.g. 4 for QUAD4, 8 for QUAD8, 9 for QUAD9).
pub trait ReferenceElement2D<const NODES: usize> {
    /// Shape function values at reference point (xi, eta).
    fn shape_functions(&self, xi: f64, eta: f64) -> [f64; NODES];

    /// Shape function derivatives w.r.t. (xi, eta).
    /// Returns (dN/dxi, dN/deta) each of length NODES.
    fn shape_derivs(&self, xi: f64, eta: f64) -> ([f64; NODES], [f64; NODES]);

    /// Gauss integration points as (xi, eta) pairs.
    fn gauss_points(&self) -> &'static [[f64; 2]];

    /// Gauss weights (same length as `gauss_points`).
    fn gauss_weights(&self) -> &'static [f64];
}

// ============================================================================
// 3D trait (solid elements)
// ============================================================================

/// A 3D reference element defined on [−1,1]³ or a tetrahedral/prismatic domain.
///
/// `NODES`: number of nodes (e.g. 8 for HEXA8).
pub trait ReferenceElement3D<const NODES: usize> {
    /// Shape function values at reference point (xi, eta, zeta).
    fn shape_functions(&self, xi: f64, eta: f64, zeta: f64) -> [f64; NODES];

    /// Shape function derivatives w.r.t. (xi, eta, zeta).
    /// Returns (dN/dxi, dN/deta, dN/dzeta) each of length NODES.
    fn shape_derivs(
        &self,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> ([f64; NODES], [f64; NODES], [f64; NODES]);

    /// Gauss integration points as (xi, eta, zeta) triples.
    fn gauss_points(&self) -> &'static [[f64; 3]];

    /// Gauss weights (same length as `gauss_points`).
    fn gauss_weights(&self) -> &'static [f64];
}

// ============================================================================
// Internal helpers
// ============================================================================

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

#[inline(always)]
fn dot6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5]
}

/// C (6×6 nalgebra) times a 6-vector.
#[inline(always)]
fn cmat_dot6(c: &nalgebra::SMatrix<f64, 6, 6>, v: &[f64; 6]) -> [f64; 6] {
    let mut r = [0.0f64; 6];
    for row in 0..6 {
        for col in 0..6 {
            r[row] += c[(row, col)] * v[col];
        }
    }
    r
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
// Generic 2D integrators
// ============================================================================

/// Compute a plane-strain 2D stiffness matrix using the generic Gauss loop.
///
/// Accumulates into `out` (flat row-major, length `(NODES*2)²`), then symmetrises.
/// Caller must pass a zeroed slice.
pub fn integrate_ke_2d<const NODES: usize, E: ReferenceElement2D<NODES>>(
    elem: &E,
    x: &[f64; NODES],
    y: &[f64; NODES],
    c: &[[f64; 3]; 3],
    out: &mut [f64],
) {
    let n_dof = 2 * NODES;
    debug_assert_eq!(out.len(), n_dof * n_dof);

    for (gp, &w) in elem.gauss_points().iter().zip(elem.gauss_weights()) {
        let xi  = gp[0];
        let eta = gp[1];

        let (dn_xi, dn_eta) = elem.shape_derivs(xi, eta);

        let j00: f64 = (0..NODES).map(|k| dn_xi[k]  * x[k]).sum();
        let j01: f64 = (0..NODES).map(|k| dn_xi[k]  * y[k]).sum();
        let j10: f64 = (0..NODES).map(|k| dn_eta[k] * x[k]).sum();
        let j11: f64 = (0..NODES).map(|k| dn_eta[k] * y[k]).sum();

        let det_j   = j00 * j11 - j01 * j10;
        let inv_det = 1.0 / det_j;
        let ji00 =  j11 * inv_det;
        let ji01 = -j01 * inv_det;
        let ji10 = -j10 * inv_det;
        let ji11 =  j00 * inv_det;

        let mut dn_dx = [0.0f64; NODES];
        let mut dn_dy = [0.0f64; NODES];
        for k in 0..NODES {
            dn_dx[k] = ji00 * dn_xi[k] + ji01 * dn_eta[k];
            dn_dy[k] = ji10 * dn_xi[k] + ji11 * dn_eta[k];
        }

        let scale = det_j * w;

        for i in 0..NODES {
            let bi  = [dn_dx[i], 0.0,      dn_dy[i]];
            let bi1 = [0.0,      dn_dy[i], dn_dx[i]];

            for j in 0..NODES {
                let bj  = [dn_dx[j], 0.0,      dn_dy[j]];
                let bj1 = [0.0,      dn_dy[j], dn_dx[j]];

                let cbj  = dot3x3(c, &bj);
                let cbj1 = dot3x3(c, &bj1);

                out[(2 * i) * n_dof + 2 * j]             += scale * dot3(&bi,  &cbj);
                out[(2 * i) * n_dof + 2 * j + 1]         += scale * dot3(&bi,  &cbj1);
                out[(2 * i + 1) * n_dof + 2 * j]         += scale * dot3(&bi1, &cbj);
                out[(2 * i + 1) * n_dof + 2 * j + 1]     += scale * dot3(&bi1, &cbj1);
            }
        }
    }

    symmetrize_flat(out, n_dof);
}

/// Compute a 2D consistent mass matrix using the generic Gauss loop.
///
/// Caller must pass a zeroed slice of length `(NODES*2)²`.
pub fn integrate_me_2d<const NODES: usize, E: ReferenceElement2D<NODES>>(
    elem: &E,
    x: &[f64; NODES],
    y: &[f64; NODES],
    rho: f64,
    out: &mut [f64],
) {
    let n_dof = 2 * NODES;
    debug_assert_eq!(out.len(), n_dof * n_dof);

    for (gp, &w) in elem.gauss_points().iter().zip(elem.gauss_weights()) {
        let xi  = gp[0];
        let eta = gp[1];

        let n_sf            = elem.shape_functions(xi, eta);
        let (dn_xi, dn_eta) = elem.shape_derivs(xi, eta);

        let j00: f64 = (0..NODES).map(|k| dn_xi[k]  * x[k]).sum();
        let j01: f64 = (0..NODES).map(|k| dn_xi[k]  * y[k]).sum();
        let j10: f64 = (0..NODES).map(|k| dn_eta[k] * x[k]).sum();
        let j11: f64 = (0..NODES).map(|k| dn_eta[k] * y[k]).sum();

        let det_j = j00 * j11 - j01 * j10;
        let scale = rho * det_j * w;

        for p in 0..NODES {
            for q in 0..NODES {
                let v = scale * n_sf[p] * n_sf[q];
                out[(2 * p) * n_dof + 2 * q]         += v;
                out[(2 * p + 1) * n_dof + 2 * q + 1] += v;
            }
        }
    }

    symmetrize_flat(out, n_dof);
}

/// Compute a 2D body load vector using the generic Gauss loop.
///
/// Caller must pass a zeroed slice of length `NODES*2`.
pub fn integrate_fext_2d<const NODES: usize, E: ReferenceElement2D<NODES>>(
    elem: &E,
    x: &[f64; NODES],
    y: &[f64; NODES],
    b: [f64; 2],
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), 2 * NODES);

    for (gp, &w) in elem.gauss_points().iter().zip(elem.gauss_weights()) {
        let xi  = gp[0];
        let eta = gp[1];

        let n_sf            = elem.shape_functions(xi, eta);
        let (dn_xi, dn_eta) = elem.shape_derivs(xi, eta);

        let j00: f64 = (0..NODES).map(|k| dn_xi[k]  * x[k]).sum();
        let j01: f64 = (0..NODES).map(|k| dn_xi[k]  * y[k]).sum();
        let j10: f64 = (0..NODES).map(|k| dn_eta[k] * x[k]).sum();
        let j11: f64 = (0..NODES).map(|k| dn_eta[k] * y[k]).sum();

        let det_j = j00 * j11 - j01 * j10;
        let scale = det_j * w;

        for p in 0..NODES {
            out[2 * p]     += scale * n_sf[p] * b[0];
            out[2 * p + 1] += scale * n_sf[p] * b[1];
        }
    }
}

// ============================================================================
// Generic 3D integrators (flat slice variants)
//
// We avoid `SMatrix<f64, {3*NODES}, {3*NODES}>` in function signatures to stay
// on stable Rust (no generic_const_exprs). Callers use concrete nalgebra types.
// ============================================================================

/// Compute a 3D stiffness matrix, writing flat row-major output.
///
/// `out` must be pre-zeroed, length `(NODES*3)²`.
/// `c_mat` is the 6×6 isotropic elasticity matrix.
pub fn integrate_ke_3d_flat<const NODES: usize, E: ReferenceElement3D<NODES>>(
    elem: &E,
    coords: &[[f64; 3]; NODES],
    c_mat: &nalgebra::SMatrix<f64, 6, 6>,
    out: &mut [f64],
) {
    use nalgebra::Matrix3;

    let n_dof = 3 * NODES;
    debug_assert_eq!(out.len(), n_dof * n_dof);

    for (gp, &w) in elem.gauss_points().iter().zip(elem.gauss_weights()) {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let (dn_xi, dn_eta, dn_dzeta) = elem.shape_derivs(xi, eta, zeta);

        let mut jmat = Matrix3::<f64>::zeros();
        for k in 0..NODES {
            jmat[(0, 0)] += dn_xi[k]     * coords[k][0];
            jmat[(0, 1)] += dn_xi[k]     * coords[k][1];
            jmat[(0, 2)] += dn_xi[k]     * coords[k][2];
            jmat[(1, 0)] += dn_eta[k]    * coords[k][0];
            jmat[(1, 1)] += dn_eta[k]    * coords[k][1];
            jmat[(1, 2)] += dn_eta[k]    * coords[k][2];
            jmat[(2, 0)] += dn_dzeta[k]  * coords[k][0];
            jmat[(2, 1)] += dn_dzeta[k]  * coords[k][1];
            jmat[(2, 2)] += dn_dzeta[k]  * coords[k][2];
        }
        let det_j = jmat.determinant();
        let inv_j = jmat.try_inverse().expect("singular Jacobian in generic ke_3d");

        // Compute physical derivatives per node
        let mut bn = [[0.0f64; 3]; NODES]; // [dndx, dndy, dndz]
        for i in 0..NODES {
            bn[i][0] = inv_j[(0,0)]*dn_xi[i] + inv_j[(0,1)]*dn_eta[i] + inv_j[(0,2)]*dn_dzeta[i];
            bn[i][1] = inv_j[(1,0)]*dn_xi[i] + inv_j[(1,1)]*dn_eta[i] + inv_j[(1,2)]*dn_dzeta[i];
            bn[i][2] = inv_j[(2,0)]*dn_xi[i] + inv_j[(2,1)]*dn_eta[i] + inv_j[(2,2)]*dn_dzeta[i];
        }

        let scale = det_j * w;

        for i in 0..NODES {
            let [dx_i, dy_i, dz_i] = bn[i];
            // B columns for node i (3 columns: x, y, z DOFs)
            // B[:,3i]   = [dx, 0,  0,  dy, 0,  dz]
            // B[:,3i+1] = [0,  dy, 0,  dx, dz, 0 ]
            // B[:,3i+2] = [0,  0,  dz, 0,  dy, dx]
            let bi_x = [dx_i, 0.0,  0.0,  dy_i, 0.0,  dz_i];
            let bi_y = [0.0,  dy_i, 0.0,  dx_i, dz_i, 0.0 ];
            let bi_z = [0.0,  0.0,  dz_i, 0.0,  dy_i, dx_i];

            for j in 0..NODES {
                let [dx_j, dy_j, dz_j] = bn[j];
                let bj_x = [dx_j, 0.0,  0.0,  dy_j, 0.0,  dz_j];
                let bj_y = [0.0,  dy_j, 0.0,  dx_j, dz_j, 0.0 ];
                let bj_z = [0.0,  0.0,  dz_j, 0.0,  dy_j, dx_j];

                let cbj_x = cmat_dot6(c_mat, &bj_x);
                let cbj_y = cmat_dot6(c_mat, &bj_y);
                let cbj_z = cmat_dot6(c_mat, &bj_z);

                let row = 3 * i;
                let col = 3 * j;
                out[ row      * n_dof + col]     += scale * dot6(&bi_x, &cbj_x);
                out[ row      * n_dof + col + 1] += scale * dot6(&bi_x, &cbj_y);
                out[ row      * n_dof + col + 2] += scale * dot6(&bi_x, &cbj_z);
                out[(row + 1) * n_dof + col]     += scale * dot6(&bi_y, &cbj_x);
                out[(row + 1) * n_dof + col + 1] += scale * dot6(&bi_y, &cbj_y);
                out[(row + 1) * n_dof + col + 2] += scale * dot6(&bi_y, &cbj_z);
                out[(row + 2) * n_dof + col]     += scale * dot6(&bi_z, &cbj_x);
                out[(row + 2) * n_dof + col + 1] += scale * dot6(&bi_z, &cbj_y);
                out[(row + 2) * n_dof + col + 2] += scale * dot6(&bi_z, &cbj_z);
            }
        }
    }

    symmetrize_flat(out, n_dof);
}

/// Compute a 3D consistent mass matrix, writing flat row-major output.
///
/// `out` must be pre-zeroed, length `(NODES*3)²`.
pub fn integrate_me_3d_flat<const NODES: usize, E: ReferenceElement3D<NODES>>(
    elem: &E,
    coords: &[[f64; 3]; NODES],
    rho: f64,
    out: &mut [f64],
) {
    use nalgebra::Matrix3;

    let n_dof = 3 * NODES;
    debug_assert_eq!(out.len(), n_dof * n_dof);

    for (gp, &w) in elem.gauss_points().iter().zip(elem.gauss_weights()) {
        let (xi, eta, zeta) = (gp[0], gp[1], gp[2]);
        let n_sf = elem.shape_functions(xi, eta, zeta);
        let (dn_xi, dn_eta, dn_dzeta) = elem.shape_derivs(xi, eta, zeta);

        let mut jmat = Matrix3::<f64>::zeros();
        for k in 0..NODES {
            jmat[(0, 0)] += dn_xi[k]    * coords[k][0];
            jmat[(0, 1)] += dn_xi[k]    * coords[k][1];
            jmat[(0, 2)] += dn_xi[k]    * coords[k][2];
            jmat[(1, 0)] += dn_eta[k]   * coords[k][0];
            jmat[(1, 1)] += dn_eta[k]   * coords[k][1];
            jmat[(1, 2)] += dn_eta[k]   * coords[k][2];
            jmat[(2, 0)] += dn_dzeta[k] * coords[k][0];
            jmat[(2, 1)] += dn_dzeta[k] * coords[k][1];
            jmat[(2, 2)] += dn_dzeta[k] * coords[k][2];
        }
        let det_j = jmat.determinant();
        let fac = rho * det_j * w;

        for i in 0..NODES {
            for j in 0..NODES {
                let v = n_sf[i] * n_sf[j] * fac;
                for d in 0..3 {
                    out[(3*i+d) * n_dof + (3*j+d)] += v;
                }
            }
        }
    }

    symmetrize_flat(out, n_dof);
}

// ============================================================================
// Tests: verify generic integrators match specialized element implementations
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::quad::{Quad4Precomputed, Quad8Precomputed, Quad9Precomputed};
    use crate::elements::quad::{Quad4Ref, Quad8Ref, Quad9Ref};
    use crate::elements::solid::{hexa8_ke, hexa8_me, Hexa8Ref};

    fn constitutive_plane_strain(e: f64, nu: f64) -> [[f64; 3]; 3] {
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        [
            [lambda + 2.0 * mu, lambda, 0.0],
            [lambda, lambda + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ]
    }

    fn isotropic_c_6x6(e: f64, nu: f64) -> nalgebra::SMatrix<f64, 6, 6> {
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        let d = lambda + 2.0 * mu;
        nalgebra::SMatrix::<f64, 6, 6>::from_row_slice(&[
            d,      lambda, lambda, 0.0, 0.0, 0.0,
            lambda, d,      lambda, 0.0, 0.0, 0.0,
            lambda, lambda, d,      0.0, 0.0, 0.0,
            0.0,    0.0,    0.0,    mu,  0.0, 0.0,
            0.0,    0.0,    0.0,    0.0, mu,  0.0,
            0.0,    0.0,    0.0,    0.0, 0.0, mu,
        ])
    }

    #[test]
    fn quad4_ke_generic_matches_original() {
        let coords: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let e = 2.0e11;
        let nu = 0.3;

        // Original
        let pre = Quad4Precomputed::new(&coords);
        let orig = pre.compute_ke_global(e, nu);

        // Generic
        let mut x = [0.0f64; 4];
        let mut y = [0.0f64; 4];
        for i in 0..4 { x[i] = coords[i][0]; y[i] = coords[i][1]; }
        let c = constitutive_plane_strain(e, nu);
        let mut ke_generic = [0.0f64; 64];
        integrate_ke_2d(&Quad4Ref, &x, &y, &c, &mut ke_generic);

        for i in 0..64 {
            let diff = (orig[i] - ke_generic[i]).abs();
            let scale = orig[i].abs().max(1.0);
            assert!(diff / scale < 1e-10, "QUAD4 Ke mismatch at [{},{}]: orig={}, generic={}", i/8, i%8, orig[i], ke_generic[i]);
        }
    }

    #[test]
    fn quad4_me_generic_matches_original() {
        let coords: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let rho = 7800.0;

        let pre = Quad4Precomputed::new(&coords);
        let orig = pre.compute_me_global(rho);

        let mut x = [0.0f64; 4];
        let mut y = [0.0f64; 4];
        for i in 0..4 { x[i] = coords[i][0]; y[i] = coords[i][1]; }
        let mut me_generic = [0.0f64; 64];
        integrate_me_2d(&Quad4Ref, &x, &y, rho, &mut me_generic);

        for i in 0..64 {
            let diff = (orig[i] - me_generic[i]).abs();
            assert!(diff < 1e-8, "QUAD4 Me mismatch at [{},{}]: {} vs {}", i/8, i%8, orig[i], me_generic[i]);
        }
    }

    #[test]
    fn hexa8_ke_generic_matches_original() {
        let coords: [[f64; 3]; 8] = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let e = 2.1e11;
        let nu = 0.3;

        let orig = hexa8_ke(&coords, e, nu);

        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 576]; // 24*24
        integrate_ke_3d_flat(&Hexa8Ref, &coords, &c_mat, &mut ke_flat);

        for i in 0..24 {
            for j in 0..24 {
                let diff = (orig[(i, j)] - ke_flat[i * 24 + j]).abs();
                assert!(diff < 1e-4, "HEXA8 Ke mismatch at [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*24+j]);
            }
        }
    }

    #[test]
    fn hexa8_me_generic_matches_original() {
        let coords: [[f64; 3]; 8] = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let rho = 7800.0;

        let orig = hexa8_me(&coords, rho);
        let mut me_flat = [0.0f64; 576];
        integrate_me_3d_flat(&Hexa8Ref, &coords, rho, &mut me_flat);

        for i in 0..24 {
            for j in 0..24 {
                let diff = (orig[(i, j)] - me_flat[i * 24 + j]).abs();
                assert!(diff < 1e-8, "HEXA8 Me mismatch at [{i},{j}]: {} vs {}", orig[(i,j)], me_flat[i*24+j]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Higher-order solid elements
    // -----------------------------------------------------------------------

    #[test]
    fn hexa20_ke_generic_matches_original() {
        use crate::elements::solid::{hexa20_ke, Hexa20Ref};
        let coords: [[f64; 3]; 20] = [
            // corners: same as HEXA8
            [-1.0,-1.0,-1.0],[1.0,-1.0,-1.0],[1.0,1.0,-1.0],[-1.0,1.0,-1.0],
            [-1.0,-1.0, 1.0],[1.0,-1.0, 1.0],[1.0,1.0, 1.0],[-1.0,1.0, 1.0],
            // bottom face edge midpoints
            [0.0,-1.0,-1.0],[1.0,0.0,-1.0],[0.0,1.0,-1.0],[-1.0,0.0,-1.0],
            // top face edge midpoints
            [0.0,-1.0, 1.0],[1.0,0.0, 1.0],[0.0,1.0, 1.0],[-1.0,0.0, 1.0],
            // vertical edge midpoints
            [-1.0,-1.0, 0.0],[1.0,-1.0, 0.0],[1.0,1.0, 0.0],[-1.0,1.0, 0.0],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = hexa20_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 3600]; // 60*60
        integrate_ke_3d_flat(&Hexa20Ref, &coords, &c_mat, &mut ke_flat);
        for i in 0..60 { for j in 0..60 {
            let diff = (orig[(i,j)] - ke_flat[i*60+j]).abs();
            let scale = orig[(i,j)].abs().max(1.0);
            assert!(diff/scale < 1e-6, "HEXA20 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*60+j]);
        }}
    }

    #[test]
    fn tetra4_ke_generic_matches_original() {
        use crate::elements::solid::{tetra4_ke, Tetra4Ref};
        let coords: [[f64; 3]; 4] = [
            [0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = tetra4_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 144]; // 12*12
        integrate_ke_3d_flat(&Tetra4Ref, &coords, &c_mat, &mut ke_flat);
        for i in 0..12 { for j in 0..12 {
            let diff = (orig[(i,j)] - ke_flat[i*12+j]).abs();
            let scale = orig[(i,j)].abs().max(1.0);
            assert!(diff/scale < 1e-6, "TETRA4 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*12+j]);
        }}
    }

    #[test]
    fn tetra10_ke_generic_matches_original() {
        use crate::elements::solid::{tetra10_ke, Tetra10Ref};
        let coords: [[f64; 3]; 10] = [
            [0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],
            [0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.5,0.0],
            [0.0,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = tetra10_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 900]; // 30*30
        integrate_ke_3d_flat(&Tetra10Ref, &coords, &c_mat, &mut ke_flat);
        for i in 0..30 { for j in 0..30 {
            let diff = (orig[(i,j)] - ke_flat[i*30+j]).abs();
            let scale = orig[(i,j)].abs().max(1.0);
            assert!(diff/scale < 1e-6, "TETRA10 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*30+j]);
        }}
    }

    #[test]
    fn wedge6_ke_generic_matches_original() {
        use crate::elements::solid::{wedge6_ke, Wedge6Ref};
        let coords: [[f64; 3]; 6] = [
            [0.0,0.0,-1.0],[1.0,0.0,-1.0],[0.0,1.0,-1.0],
            [0.0,0.0, 1.0],[1.0,0.0, 1.0],[0.0,1.0, 1.0],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = wedge6_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 324]; // 18*18
        integrate_ke_3d_flat(&Wedge6Ref, &coords, &c_mat, &mut ke_flat);
        for i in 0..18 { for j in 0..18 {
            let diff = (orig[(i,j)] - ke_flat[i*18+j]).abs();
            let scale = orig[(i,j)].abs().max(1.0);
            assert!(diff/scale < 1e-6, "WEDGE6 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*18+j]);
        }}
    }

    #[test]
    fn wedge15_ke_generic_matches_original() {
        use crate::elements::solid::{wedge15_ke, Wedge15Ref};
        // Use reference element geometry (identity-ish Jacobian) for robustness
        let coords: [[f64; 3]; 15] = [
            // bottom corners
            [0.0,0.0,-1.0],[1.0,0.0,-1.0],[0.0,1.0,-1.0],
            // top corners
            [0.0,0.0, 1.0],[1.0,0.0, 1.0],[0.0,1.0, 1.0],
            // bottom edge midpoints: N6=edge01, N7=edge02, N8=vert0, N9=edge12
            [0.5,0.0,-1.0],[0.0,0.5,-1.0],[0.0,0.0,0.0],[0.5,0.5,-1.0],
            // vertical edge midpoints: N10=vert1, N11=vert2
            [1.0,0.0,0.0],[0.0,1.0,0.0],
            // top edge midpoints: N12=edge34, N13=edge35, N14=edge45
            [0.5,0.0,1.0],[0.0,0.5,1.0],[0.5,0.5,1.0],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = wedge15_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 2025]; // 45*45
        integrate_ke_3d_flat(&Wedge15Ref, &coords, &c_mat, &mut ke_flat);
        // Use relative tolerance w.r.t. max diagonal for near-zero entries
        let diag_max: f64 = (0..45).map(|i| orig[(i,i)].abs()).fold(0.0f64, f64::max);
        for i in 0..45 { for j in 0..45 {
            let diff = (orig[(i,j)] - ke_flat[i*45+j]).abs();
            let scale = orig[(i,j)].abs().max(diag_max * 1e-6).max(1.0);
            assert!(diff/scale < 1e-4, "WEDGE15 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*45+j]);
        }}
    }

    #[test]
    fn pyramid5_ke_generic_matches_original() {
        use crate::elements::solid::{pyramid5_ke, Pyramid5Ref};
        let coords: [[f64; 3]; 5] = [
            [-1.0,-1.0,0.0],[1.0,-1.0,0.0],[1.0,1.0,0.0],[-1.0,1.0,0.0],
            [0.0,0.0,1.0],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = pyramid5_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 225]; // 15*15
        integrate_ke_3d_flat(&Pyramid5Ref, &coords, &c_mat, &mut ke_flat);
        for i in 0..15 { for j in 0..15 {
            let diff = (orig[(i,j)] - ke_flat[i*15+j]).abs();
            let scale = orig[(i,j)].abs().max(1.0);
            assert!(diff/scale < 1e-6, "PYRAMID5 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*15+j]);
        }}
    }

    #[test]
    fn pyramid13_ke_generic_matches_original() {
        use crate::elements::solid::{pyramid13_ke, Pyramid13Ref};
        let coords: [[f64; 3]; 13] = [
            [-1.0,-1.0,0.0],[1.0,-1.0,0.0],[1.0,1.0,0.0],[-1.0,1.0,0.0],
            [0.0,0.0,1.0],
            // base edge midpoints
            [0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0],
            // lateral edge midpoints
            [-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5],
        ];
        let e = 2.1e11; let nu = 0.3;
        let orig = pyramid13_ke(&coords, e, nu);
        let c_mat = isotropic_c_6x6(e, nu);
        let mut ke_flat = [0.0f64; 1521]; // 39*39
        integrate_ke_3d_flat(&Pyramid13Ref, &coords, &c_mat, &mut ke_flat);
        for i in 0..39 { for j in 0..39 {
            let diff = (orig[(i,j)] - ke_flat[i*39+j]).abs();
            let scale = orig[(i,j)].abs().max(1.0);
            assert!(diff/scale < 1e-6, "PYRAMID13 Ke [{i},{j}]: {} vs {}", orig[(i,j)], ke_flat[i*39+j]);
        }}
    }
}
