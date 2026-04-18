//! Dense generalized eigenvalue solver for modal analysis.
//!
//! Solves K·φ = λ·M·φ for the lowest modes using Cholesky transformation
//! to standard symmetric form, then `nalgebra::SymmetricEigen`.
//!
//! Suitable for problems up to ~15 000 free DOFs (dense storage).

use nalgebra::{Cholesky, DMatrix, Dyn, SymmetricEigen};
use std::collections::HashMap;

/// Maximum free DOFs for the dense solver.
const MAX_DENSE_DOFS: usize = 15_000;

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Build mapping from global DOF index → local free-DOF index.
fn build_free_map(free_dofs: &[i64]) -> HashMap<i64, usize> {
    free_dofs
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i))
        .collect()
}

/// Assemble element matrices directly into dense free-DOF matrices.
///
/// Layout:  ke_flat\[e * ndof² + i*ndof + j\] = K^e_{i,j}
///          dofs_flat\[e * ndof + k\]          = global DOF of local DOF k
fn assemble_dense_from_elements(
    ke_flat: &[f64],
    me_flat: &[f64],
    dofs_flat: &[i64],
    ndof: usize,
    free_map: &HashMap<i64, usize>,
    n_free: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n_elem = dofs_flat.len() / ndof;
    let ndof2 = ndof * ndof;

    let mut k = DMatrix::<f64>::zeros(n_free, n_free);
    let mut m = DMatrix::<f64>::zeros(n_free, n_free);

    for e in 0..n_elem {
        let d0 = e * ndof;
        let m0 = e * ndof2;
        for i in 0..ndof {
            let gi = dofs_flat[d0 + i];
            if let Some(&li) = free_map.get(&gi) {
                for j in 0..ndof {
                    let gj = dofs_flat[d0 + j];
                    if let Some(&lj) = free_map.get(&gj) {
                        let val_idx = m0 + i * ndof + j;
                        k[(li, lj)] += ke_flat[val_idx];
                        m[(li, lj)] += me_flat[val_idx];
                    }
                }
            }
        }
    }

    (k, m)
}

/// Build dense matrix from COO triplets restricted to free DOFs.
fn coo_to_dense(
    rows: &[i64],
    cols: &[i64],
    vals: &[f64],
    free_map: &HashMap<i64, usize>,
    n_free: usize,
) -> DMatrix<f64> {
    let mut mat = DMatrix::<f64>::zeros(n_free, n_free);
    for idx in 0..rows.len() {
        if let (Some(&lr), Some(&lc)) = (free_map.get(&rows[idx]), free_map.get(&cols[idx])) {
            mat[(lr, lc)] += vals[idx];
        }
    }
    mat
}

/// Cholesky factorisation with optional diagonal regularisation.
fn robust_cholesky(m: &DMatrix<f64>) -> Result<Cholesky<f64, Dyn>, String> {
    if let Some(c) = m.clone().cholesky() {
        return Ok(c);
    }
    // Add small regularisation
    let n = m.nrows();
    let trace: f64 = (0..n).map(|i| m[(i, i)]).sum();
    let eps = trace.abs() / n as f64 * 1e-10;
    let mut m_reg = m.clone();
    for i in 0..n {
        m_reg[(i, i)] += eps;
    }
    m_reg.cholesky().ok_or_else(|| {
        format!(
            "Mass matrix not positive definite (trace={trace:.3e}, eps={eps:.3e})"
        )
    })
}

/// Core: solve K_free·φ = λ·M_free·φ via Cholesky transformation.
///
/// Returns `(eigenvalues, eigenvectors_free)`, sorted ascending,
/// filtered to λ > 1e-8, limited to `num_modes`.
fn solve_generalized_eigenproblem(
    k_free: DMatrix<f64>,
    m_free: DMatrix<f64>,
    num_modes: usize,
) -> Result<(Vec<f64>, DMatrix<f64>), String> {
    let n = k_free.nrows();

    // M = L·Lᵀ
    let chol = robust_cholesky(&m_free)?;
    let l = chol.l();

    // L⁻¹ via forward substitution: solve L·X = I
    let id = DMatrix::<f64>::identity(n, n);
    let l_inv = l
        .solve_lower_triangular(&id)
        .ok_or("L⁻¹ computation failed")?;
    let l_inv_t = l_inv.transpose();

    // A = L⁻¹·K·L⁻ᵀ  (standard symmetric eigenproblem)
    let a = &l_inv * &k_free * &l_inv_t;

    // Symmetrise (numerical round-off)
    let a_sym = (&a + a.transpose()) * 0.5;

    // Eigendecomposition
    let eigen = SymmetricEigen::new(a_sym);

    // Sort ascending by eigenvalue
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter λ > 1e-8, take num_modes
    let valid: Vec<usize> = idx
        .into_iter()
        .filter(|&i| eigen.eigenvalues[i] > 1e-8)
        .take(num_modes)
        .collect();

    let nm = valid.len();
    if nm == 0 {
        return Err("No positive eigenvalues found (all <= 1e-8)".into());
    }

    let eigenvalues: Vec<f64> = valid.iter().map(|&i| eigen.eigenvalues[i]).collect();

    // Transform eigenvectors back: φ_free = L⁻ᵀ·y
    let mut phi = DMatrix::<f64>::zeros(n, nm);
    for (col, &eig_i) in valid.iter().enumerate() {
        let y = eigen.eigenvectors.column(eig_i);
        let p = &l_inv_t * y;
        phi.set_column(col, &p);
    }

    Ok((eigenvalues, phi))
}

/// Convert eigenvalues → Hz and expand mode shapes to full DOF space.
///
/// Output layout: `modes_flat[mode * n_total + dof]`
fn postprocess(
    eigenvalues: &[f64],
    phi_free: &DMatrix<f64>,
    free_dofs: &[i64],
    n_total: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nm = eigenvalues.len();

    let freq: Vec<f64> = eigenvalues
        .iter()
        .map(|&lam| lam.sqrt() / (2.0 * std::f64::consts::PI))
        .collect();

    // Expand: each mode occupies [mode*n_total .. (mode+1)*n_total)
    let mut modes = vec![0.0f64; nm * n_total];
    for mode in 0..nm {
        let off = mode * n_total;
        for (local, &global) in free_dofs.iter().enumerate() {
            modes[off + global as usize] = phi_free[(local, mode)];
        }
    }

    (freq, modes)
}

fn validate(n_free: usize, num_modes: usize) -> Result<(), String> {
    if n_free > MAX_DENSE_DOFS {
        return Err(format!(
            "Dense solver limit exceeded: {n_free} free DOFs (max {MAX_DENSE_DOFS})"
        ));
    }
    if n_free == 0 {
        return Err("No free DOFs — all DOFs are constrained".into());
    }
    if num_modes == 0 {
        return Err("num_modes must be > 0".into());
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Modal solve from pre-computed element matrices and DOF connectivity.
///
/// # Arguments
/// * `ke_flat`  – flattened element stiffness (n_elem × ndof²)
/// * `me_flat`  – flattened element mass     (n_elem × ndof²)
/// * `dofs_flat` – flattened DOF connectivity (n_elem × ndof)
/// * `ndof`     – DOFs per element (18 for MITC3, 24 for MITC4)
/// * `n_total`  – total system DOFs
/// * `free_dofs` – indices of unconstrained DOFs
/// * `num_modes` – number of lowest modes to return
///
/// # Returns
/// `(frequencies_hz, mode_shapes_flat)` where mode_shapes is
/// row-major `[mode * n_total + dof]`.
pub fn modal_solve_elements(
    ke_flat: &[f64],
    me_flat: &[f64],
    dofs_flat: &[i64],
    ndof: usize,
    n_total: usize,
    free_dofs: &[i64],
    num_modes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_free = free_dofs.len();
    validate(n_free, num_modes)?;

    let free_map = build_free_map(free_dofs);
    let (k, m) = assemble_dense_from_elements(
        ke_flat, me_flat, dofs_flat, ndof, &free_map, n_free,
    );
    let (eigs, phi) = solve_generalized_eigenproblem(k, m, num_modes)?;
    Ok(postprocess(&eigs, &phi, free_dofs, n_total))
}

/// Modal solve from COO sparse triplets (flexible, supports mixed meshes).
pub fn modal_solve_coo(
    k_rows: &[i64],
    k_cols: &[i64],
    k_vals: &[f64],
    m_rows: &[i64],
    m_cols: &[i64],
    m_vals: &[f64],
    n_total: usize,
    free_dofs: &[i64],
    num_modes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_free = free_dofs.len();
    validate(n_free, num_modes)?;

    let free_map = build_free_map(free_dofs);
    let k = coo_to_dense(k_rows, k_cols, k_vals, &free_map, n_free);
    let m = coo_to_dense(m_rows, m_cols, m_vals, &free_map, n_free);
    let (eigs, phi) = solve_generalized_eigenproblem(k, m, num_modes)?;
    Ok(postprocess(&eigs, &phi, free_dofs, n_total))
}
