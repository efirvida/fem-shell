/// COO (Coordinate format) batch assembly.
///
/// Generates sparse matrix triplets (row, col, value) from element matrices
/// and DOF connectivity arrays. This is the bridge between per-element
/// computations and global sparse matrix assembly in PETSc.

/// Generate COO triplets from batch element matrices.
///
/// # Arguments
/// * `dofs` - Flat DOF connectivity: [n_elem × dofs_per_elem]
/// * `ke_flat` - Flat element matrices: [n_elem × dofs_per_elem × dofs_per_elem]
/// * `n_elem` - Number of elements
/// * `dof_per_elem` - DOFs per element
///
/// # Returns
/// (rows, cols, values) each of length n_elem × dof_per_elem²
pub fn coo_from_batch(
    dofs: &[i64],
    ke_flat: &[f64],
    n_elem: usize,
    dof_per_elem: usize,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let entries_per_elem = dof_per_elem * dof_per_elem;
    let total = n_elem * entries_per_elem;

    let mut rows = vec![0i64; total];
    let mut cols = vec![0i64; total];
    let mut vals = vec![0.0f64; total];

    for e in 0..n_elem {
        let dof_offset = e * dof_per_elem;
        let ke_offset = e * entries_per_elem;
        let out_offset = e * entries_per_elem;

        for i in 0..dof_per_elem {
            let row_dof = dofs[dof_offset + i];
            for j in 0..dof_per_elem {
                let col_dof = dofs[dof_offset + j];
                let idx = out_offset + i * dof_per_elem + j;
                rows[idx] = row_dof;
                cols[idx] = col_dof;
                vals[idx] = ke_flat[ke_offset + i * dof_per_elem + j];
            }
        }
    }

    (rows, cols, vals)
}

/// Compute the number of non-zeros per row for preallocation.
///
/// # Arguments
/// * `dofs` - Flat DOF connectivity: [n_elem × dofs_per_elem]
/// * `n_elem` - Number of elements
/// * `dof_per_elem` - DOFs per element
/// * `n_dof_total` - Total number of DOFs in the system
///
/// # Returns
/// Vector of length n_dof_total with nnz count per row
pub fn compute_nnz_per_row(
    dofs: &[i64],
    n_elem: usize,
    dof_per_elem: usize,
    n_dof_total: usize,
) -> Vec<i64> {
    // Use sets to track unique column indices per row
    let mut row_cols: Vec<std::collections::HashSet<i64>> =
        vec![std::collections::HashSet::new(); n_dof_total];

    for e in 0..n_elem {
        let offset = e * dof_per_elem;
        let elem_dofs = &dofs[offset..offset + dof_per_elem];

        for &row_dof in elem_dofs {
            let row = row_dof as usize;
            if row < n_dof_total {
                for &col_dof in elem_dofs {
                    row_cols[row].insert(col_dof);
                }
            }
        }
    }

    row_cols.iter().map(|s| s.len() as i64).collect()
}
