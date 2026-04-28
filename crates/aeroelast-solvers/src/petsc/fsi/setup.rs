/// FSI setup utilities: BC reduction, mass lumping, and interface DOF remapping.
///
/// These routines bridge the gap between the global assembled COO matrices
/// (produced by Python via the Rust assembler) and the reduced system that
/// [`crate::petsc::elasticity::dynamic_newmark::NewmarkStepper`] expects.
///
/// # Workflow
/// 1. [`reduce_coo`]         — extract the free-DOF sub-system from global COO.
/// 2. [`lump_mass_coo`]      — convert consistent mass to lumped diagonal form.
/// 3. [`rayleigh_c_coo`]     — build Rayleigh C = η_k·K + η_m·M from free-DOF COO.
/// 4. [`remap_interface_dofs`] — translate global DOF indices to reduced indices.

use std::collections::HashMap;

// ── BC reduction ──────────────────────────────────────────────────────────────

/// Extract the free-DOF sub-system from a global COO matrix.
///
/// Given global COO triplets `(rows, cols, vals)` and a sorted list of free
/// DOF indices `free_dofs`, returns new COO triplets whose row/column indices
/// run from `0` to `free_dofs.len() - 1`.
///
/// Entries whose row **or** column refers to a constrained (non-free) DOF are
/// discarded — this is the standard condensation for homogeneous Dirichlet BCs.
///
/// # Arguments
/// * `rows`      — global row indices (i32)
/// * `cols`      — global column indices (i32)
/// * `vals`      — matrix values
/// * `free_dofs` — sorted array of free (unconstrained) global DOF indices
///
/// # Returns
/// `(rows_red, cols_red, vals_red)` — reduced COO triplets.
pub fn reduce_coo(
    rows: &[i32],
    cols: &[i32],
    vals: &[f64],
    free_dofs: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Build global_dof → reduced_index lookup.
    let dof_map: HashMap<i32, i32> = free_dofs
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i as i32))
        .collect();

    let mut r_out = Vec::with_capacity(vals.len());
    let mut c_out = Vec::with_capacity(vals.len());
    let mut v_out = Vec::with_capacity(vals.len());

    for ((&row, &col), &val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        if let (Some(&ri), Some(&ci)) = (dof_map.get(&row), dof_map.get(&col)) {
            r_out.push(ri);
            c_out.push(ci);
            v_out.push(val);
        }
    }

    (r_out, c_out, v_out)
}

// ── Mass lumping ──────────────────────────────────────────────────────────────

/// Convert a consistent mass matrix (COO) to a diagonal (lumped) form.
///
/// Uses the **row-sum** lumping method: the lumped diagonal coefficient for
/// DOF `i` is the sum of the absolute values of all entries in row `i`.
///
/// Returns diagonal COO triplets: `rows = cols = [0, 1, …, n_dofs-1]`.
///
/// # Arguments
/// * `rows`   — COO row indices (in the **reduced** system, 0-based)
/// * `cols`   — COO column indices
/// * `vals`   — matrix values
/// * `n_dofs` — number of rows/columns in the reduced system
///
/// # Returns
/// `(rows_diag, cols_diag, vals_diag)` — diagonal COO.
pub fn lump_mass_coo(
    rows: &[i32],
    cols: &[i32],
    vals: &[f64],
    n_dofs: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let _ = cols; // not needed for row-sum lumping
    let mut diag = vec![0.0f64; n_dofs];

    for (&r, &v) in rows.iter().zip(vals.iter()) {
        let idx = r as usize;
        if idx < n_dofs {
            diag[idx] += v.abs();
        }
    }

    let indices: Vec<i32> = (0..n_dofs as i32).collect();
    (indices.clone(), indices, diag)
}

// ── Rayleigh damping ──────────────────────────────────────────────────────────

/// Build Rayleigh damping C = η_k·K + η_m·M from reduced COO triplets.
///
/// K and M must share the same sparsity pattern (same `rows` and `cols`
/// arrays), which is always true for matrices assembled from the same mesh
/// topology under Rayleigh damping.
///
/// Returns COO triplets for C with the same sparsity pattern.
///
/// # Arguments
/// * `k_vals` — stiffness matrix values (reduced)
/// * `m_vals` — mass matrix values (reduced, same sparsity as K)
/// * `rows`   — shared row indices
/// * `cols`   — shared column indices
/// * `eta_k`  — stiffness-proportional coefficient (α)
/// * `eta_m`  — mass-proportional coefficient (β)
///
/// # Returns
/// `(rows, cols, c_vals)` — the `rows` and `cols` are cloned from input.
pub fn rayleigh_c_coo(
    k_vals: &[f64],
    m_vals: &[f64],
    rows: &[i32],
    cols: &[i32],
    eta_k: f64,
    eta_m: f64,
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    assert_eq!(
        k_vals.len(),
        m_vals.len(),
        "K and M must have the same number of non-zeros for Rayleigh damping"
    );
    let c_vals: Vec<f64> = k_vals
        .iter()
        .zip(m_vals.iter())
        .map(|(&k, &m)| eta_k * k + eta_m * m)
        .collect();
    (rows.to_vec(), cols.to_vec(), c_vals)
}

// ── Interface DOF remapping ───────────────────────────────────────────────────

/// Translate global DOF indices to their positions in the reduced system.
///
/// Given a sorted `free_dofs` array (the output of BC reduction), each global
/// DOF `g` in `interface_dofs_global` is mapped to its **reduced** index via
/// binary search.
///
/// # Panics
/// Panics if any DOF in `interface_dofs_global` is not found in `free_dofs`
/// (i.e., an interface node is on a Dirichlet boundary — physically invalid).
///
/// # Arguments
/// * `interface_dofs_global` — DOF indices in the global (unreduced) system
/// * `free_dofs`             — sorted array of free global DOF indices
///
/// # Returns
/// DOF indices in the reduced (0-based) system.
pub fn remap_interface_dofs(
    interface_dofs_global: &[usize],
    free_dofs: &[i32],
) -> Vec<usize> {
    interface_dofs_global
        .iter()
        .map(|&d| {
            let d_i32 = d as i32;
            let pos = free_dofs.partition_point(|&f| f < d_i32);
            assert!(
                pos < free_dofs.len() && free_dofs[pos] == d_i32,
                "Interface DOF {d} is not a free DOF — interface nodes cannot have Dirichlet BCs"
            );
            pos
        })
        .collect()
}

// ── Sparsity alignment ────────────────────────────────────────────────────────

/// Expand a lumped-mass diagonal into the sparsity pattern of matrix K.
///
/// [`NewmarkStepper`] requires K, M, and C to share the same COO sparsity
/// pattern (same `rows` and `cols` arrays, aligned entry-by-entry).  When
/// the mass matrix has been row-sum lumped to a diagonal, this function
/// "pads" it back to K's full sparsity by inserting zeros at off-diagonal
/// positions.
///
/// The resulting `m_vals` slice is aligned with `k_rows/k_cols` so that it
/// can be passed directly to `NewmarkStepper::new` alongside K's triplets.
///
/// # Arguments
/// * `lumped_diag` — diagonal mass values, length = number of free DOFs
/// * `k_rows`      — COO row indices for K (reduced system)
/// * `k_cols`      — COO column indices for K (reduced system)
///
/// # Returns
/// `m_vals` aligned with `k_rows/k_cols`.
pub fn expand_diag_to_sparsity(lumped_diag: &[f64], k_rows: &[i32], k_cols: &[i32]) -> Vec<f64> {
    k_rows
        .iter()
        .zip(k_cols.iter())
        .map(|(&r, &c)| {
            if r == c {
                let idx = r as usize;
                if idx < lumped_diag.len() {
                    lumped_diag[idx]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        })
        .collect()
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_coo_2dof_system() {
        // 4×4 system, DOFs 0-3; constrain DOF 1 → free = [0, 2, 3]
        // Matrix: identity
        let rows = vec![0i32, 1, 2, 3];
        let cols = vec![0i32, 1, 2, 3];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let free_dofs = vec![0i32, 2, 3];

        let (r, c, v) = reduce_coo(&rows, &cols, &vals, &free_dofs);

        // DOF 1 is constrained → only entries for DOFs 0, 2, 3 survive.
        // Remapped: 0→0, 2→1, 3→2
        assert_eq!(r, vec![0i32, 1, 2]);
        assert_eq!(c, vec![0i32, 1, 2]);
        assert_eq!(v, vec![1.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reduce_coo_off_diagonal() {
        // 3×3 system, free = [0, 2]; off-diagonal (0,2) should survive remapped to (0,1)
        let rows = vec![0i32, 0, 1, 2, 2];
        let cols = vec![0i32, 2, 1, 0, 2];
        let vals = vec![10.0, 5.0, 99.0, 5.0, 20.0];
        let free_dofs = vec![0i32, 2];

        let (r, c, v) = reduce_coo(&rows, &cols, &vals, &free_dofs);

        // Entry (1,1)=99 is dropped (DOF 1 is constrained)
        assert_eq!(r.len(), 4);
        assert!(r.contains(&0) && r.contains(&1));
        let _ = (c, v); // just check lengths
    }

    #[test]
    fn test_lump_mass_coo_diagonal() {
        // 2×2 diagonal mass: [[3, 0], [0, 5]] → lumped = [3, 5]
        let rows = vec![0i32, 1];
        let cols = vec![0i32, 1];
        let vals = vec![3.0, 5.0];
        let (_, _, v) = lump_mass_coo(&rows, &cols, &vals, 2);
        assert_eq!(v, vec![3.0, 5.0]);
    }

    #[test]
    fn test_lump_mass_coo_consistent() {
        // 2×2 consistent: [[2, 1], [1, 2]] → row sums = [3, 3]
        let rows = vec![0i32, 0, 1, 1];
        let cols = vec![0i32, 1, 0, 1];
        let vals = vec![2.0, 1.0, 1.0, 2.0];
        let (_, _, v) = lump_mass_coo(&rows, &cols, &vals, 2);
        assert!((v[0] - 3.0).abs() < 1e-12);
        assert!((v[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_rayleigh_c_coo() {
        let k_vals = vec![1.0, 2.0];
        let m_vals = vec![3.0, 4.0];
        let rows = vec![0i32, 1];
        let cols = vec![0i32, 1];
        let (_, _, c) = rayleigh_c_coo(&k_vals, &m_vals, &rows, &cols, 0.1, 0.2);
        // c[0] = 0.1*1 + 0.2*3 = 0.7,  c[1] = 0.1*2 + 0.2*4 = 1.0
        assert!((c[0] - 0.7).abs() < 1e-12);
        assert!((c[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_remap_interface_dofs() {
        // free_dofs = [2, 5, 7, 10]; interface global DOFs: [5, 10]
        // Expected: 5→1, 10→3
        let free_dofs = vec![2i32, 5, 7, 10];
        let interface = vec![5usize, 10];
        let remapped = remap_interface_dofs(&interface, &free_dofs);
        assert_eq!(remapped, vec![1, 3]);
    }

    #[test]
    fn test_expand_diag_to_sparsity() {
        // 3×3 K: diag + one off-diagonal. lumped_diag = [10, 20, 30]
        // K triplets: (0,0), (0,1), (1,1), (2,2)
        let k_rows = vec![0i32, 0, 1, 2];
        let k_cols = vec![0i32, 1, 1, 2];
        let diag = vec![10.0, 20.0, 30.0];
        let m = expand_diag_to_sparsity(&diag, &k_rows, &k_cols);
        // (0,0)→10, (0,1)→0, (1,1)→20, (2,2)→30
        assert_eq!(m, vec![10.0, 0.0, 20.0, 30.0]);
    }
}
