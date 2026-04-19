use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use rayon::prelude::*;

use aeroelast_core::elements::mitc3::{self, Mitc3Precomputed};
use aeroelast_core::elements::mitc4::{self, Mitc4Precomputed};
use aeroelast_core::materials::composite::composite_constitutive;
use aeroelast_core::materials::isotropic::IsotropicMaterial;
use aeroelast_core::materials::Material;
use aeroelast_core::assembly;
use aeroelast_core::assembly::{ElemType, MaterialSpec, MeshAssembler, MeshTopology};

// ============================================================================
// PyO3 exposed functions
// ============================================================================

/// Batch-compute MITC3+ element stiffness matrices (global coords).
#[pyfunction]
#[pyo3(signature = (coords, e_mod, nu, thickness, shear_correction=5.0/6.0))]
fn batch_ke_mitc3<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    thickness: f64,
    shear_correction: f64,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 9];
            for i in 0..9 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc3Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);
            let ke = mitc3::compute_ke_global(&pre);
            let mut flat = [0.0f64; 324];
            for i in 0..18 {
                for j in 0..18 {
                    flat[i * 18 + j] = ke[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 324;
    let mut output = Vec::with_capacity(total);
    for ke_flat in &results {
        output.extend_from_slice(ke_flat);
    }

    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC3+ element mass matrices (global coords).
#[pyfunction]
#[pyo3(signature = (coords, e_mod, nu, rho, thickness, shear_correction=5.0/6.0))]
fn batch_me_mitc3<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    rho: f64,
    thickness: f64,
    shear_correction: f64,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, rho);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 9];
            for i in 0..9 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc3Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);
            let me = mitc3::compute_me_global(&pre, rho);
            let mut flat = [0.0f64; 324];
            for i in 0..18 {
                for j in 0..18 {
                    flat[i * 18 + j] = me[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 324;
    let mut output = Vec::with_capacity(total);
    for me_flat in &results {
        output.extend_from_slice(me_flat);
    }

    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC3+ tangent stiffness matrices (global coords, nonlinear).
#[pyfunction]
#[pyo3(signature = (coords, displacements, e_mod, nu, thickness, shear_correction=5.0/6.0))]
fn batch_kt_mitc3<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    displacements: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    thickness: f64,
    shear_correction: f64,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let disp_arr = displacements.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 9];
            for i in 0..9 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc3Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);

            let mut u = mitc3::Vec18::zeros();
            for i in 0..18 {
                u[i] = disp_arr[[e, i]];
            }

            let kt = mitc3::compute_kt_global(&pre, &u);
            let mut flat = [0.0f64; 324];
            for i in 0..18 {
                for j in 0..18 {
                    flat[i * 18 + j] = kt[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 324;
    let mut output = Vec::with_capacity(total);
    for kt_flat in &results {
        output.extend_from_slice(kt_flat);
    }

    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC3+ internal force vectors (global coords).
#[pyfunction]
#[pyo3(signature = (coords, displacements, e_mod, nu, thickness, shear_correction=5.0/6.0, nonlinear=true))]
fn batch_fint_mitc3<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    displacements: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    thickness: f64,
    shear_correction: f64,
    nonlinear: bool,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let disp_arr = displacements.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 18]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 9];
            for i in 0..9 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc3Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);

            let mut u = mitc3::Vec18::zeros();
            for i in 0..18 {
                u[i] = disp_arr[[e, i]];
            }

            let f = mitc3::compute_fint_global(&pre, &u, nonlinear);
            let mut flat = [0.0f64; 18];
            for i in 0..18 {
                flat[i] = f[i];
            }
            flat
        })
        .collect();

    let total = n_elem * 18;
    let mut output = Vec::with_capacity(total);
    for f_flat in &results {
        output.extend_from_slice(f_flat);
    }

    Array1::from(output).into_pyarray(py)
}

// ============================================================================
// MITC4 batch functions
// ============================================================================

/// Batch-compute MITC4+ element stiffness matrices (global coords).
#[pyfunction]
#[pyo3(signature = (coords, e_mod, nu, thickness, shear_correction=5.0/6.0))]
fn batch_ke_mitc4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    thickness: f64,
    shear_correction: f64,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 12];
            for i in 0..12 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc4Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);
            let ke = mitc4::compute_ke_global(&pre);
            let mut flat = [0.0f64; 576];
            for i in 0..24 {
                for j in 0..24 {
                    flat[i * 24 + j] = ke[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 576;
    let mut output = Vec::with_capacity(total);
    for ke_flat in &results {
        output.extend_from_slice(ke_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC4+ element mass matrices (global coords).
#[pyfunction]
#[pyo3(signature = (coords, e_mod, nu, rho, thickness, shear_correction=5.0/6.0))]
fn batch_me_mitc4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    rho: f64,
    thickness: f64,
    shear_correction: f64,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, rho);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 12];
            for i in 0..12 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc4Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);
            let me = mitc4::compute_me_global(&pre, rho);
            let mut flat = [0.0f64; 576];
            for i in 0..24 {
                for j in 0..24 {
                    flat[i * 24 + j] = me[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 576;
    let mut output = Vec::with_capacity(total);
    for me_flat in &results {
        output.extend_from_slice(me_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC4+ tangent stiffness matrices (global coords, nonlinear).
#[pyfunction]
#[pyo3(signature = (coords, displacements, e_mod, nu, thickness, shear_correction=5.0/6.0))]
fn batch_kt_mitc4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    displacements: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    thickness: f64,
    shear_correction: f64,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let disp_arr = displacements.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 12];
            for i in 0..12 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc4Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);

            let mut u = mitc4::Vec24::zeros();
            for i in 0..24 {
                u[i] = disp_arr[[e, i]];
            }

            let kt = mitc4::compute_kt_global(&pre, &u);
            let mut flat = [0.0f64; 576];
            for i in 0..24 {
                for j in 0..24 {
                    flat[i * 24 + j] = kt[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 576;
    let mut output = Vec::with_capacity(total);
    for kt_flat in &results {
        output.extend_from_slice(kt_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC4+ internal force vectors (global coords).
#[pyfunction]
#[pyo3(signature = (coords, displacements, e_mod, nu, thickness, shear_correction=5.0/6.0, nonlinear=true))]
fn batch_fint_mitc4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    displacements: PyReadonlyArray2<'py, f64>,
    e_mod: f64,
    nu: f64,
    thickness: f64,
    shear_correction: f64,
    nonlinear: bool,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let disp_arr = displacements.as_array();
    let n_elem = coords_arr.nrows();

    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    let results: Vec<[f64; 24]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 12];
            for i in 0..12 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let pre = Mitc4Precomputed::new(&node_coords, constitutive.clone(), thickness, e_mod);

            let mut u = mitc4::Vec24::zeros();
            for i in 0..24 {
                u[i] = disp_arr[[e, i]];
            }

            let f = mitc4::compute_fint_global(&pre, &u, nonlinear);
            let mut flat = [0.0f64; 24];
            for i in 0..24 {
                flat[i] = f[i];
            }
            flat
        })
        .collect();

    let total = n_elem * 24;
    let mut output = Vec::with_capacity(total);
    for f_flat in &results {
        output.extend_from_slice(f_flat);
    }
    Array1::from(output).into_pyarray(py)
}

// ============================================================================
// Composite batch functions
// ============================================================================

/// Batch-compute MITC3+ composite element stiffness matrices (global coords).
#[pyfunction]
fn batch_ke_mitc3_composite<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    cm_flat: PyReadonlyArray2<'py, f64>,
    cb_flat: PyReadonlyArray2<'py, f64>,
    cs_flat: PyReadonlyArray2<'py, f64>,
    thickness: PyReadonlyArray1<'py, f64>,
    e_equiv: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let cm_arr = cm_flat.as_array();
    let cb_arr = cb_flat.as_array();
    let cs_arr = cs_flat.as_array();
    let h_arr = thickness.as_array();
    let e_arr = e_equiv.as_array();
    let n_elem = coords_arr.nrows();

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 9];
            for i in 0..9 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let mut a = [0.0f64; 9];
            let mut d = [0.0f64; 9];
            let mut cs = [0.0f64; 4];
            for i in 0..9 {
                a[i] = cm_arr[[e, i]];
                d[i] = cb_arr[[e, i]];
            }
            for i in 0..4 {
                cs[i] = cs_arr[[e, i]];
            }
            let h = h_arr[e];
            let e_eq = e_arr[e];

            let constitutive = composite_constitutive(&a, &d, &cs, h);
            let pre = Mitc3Precomputed::new(&node_coords, constitutive, h, e_eq);
            let ke = mitc3::compute_ke_global(&pre);
            let mut flat = [0.0f64; 324];
            for i in 0..18 {
                for j in 0..18 {
                    flat[i * 18 + j] = ke[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 324;
    let mut output = Vec::with_capacity(total);
    for ke_flat in &results {
        output.extend_from_slice(ke_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC3+ composite element mass matrices (global coords).
#[pyfunction]
fn batch_me_mitc3_composite<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    mass_per_area: PyReadonlyArray1<'py, f64>,
    rotational_inertia: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let mpa_arr = mass_per_area.as_array();
    let ri_arr = rotational_inertia.as_array();
    let n_elem = coords_arr.nrows();

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 9];
            for i in 0..9 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let m_trans = mpa_arr[e];
            let m_rot = ri_arr[e];

            let dummy = composite_constitutive(
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[1.0, 0.0, 0.0, 1.0],
                1.0,
            );
            let pre = Mitc3Precomputed::new(&node_coords, dummy, 1.0, 1.0);
            let me = mitc3::compute_me_composite_global(&pre, m_trans, m_rot);
            let mut flat = [0.0f64; 324];
            for i in 0..18 {
                for j in 0..18 {
                    flat[i * 18 + j] = me[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 324;
    let mut output = Vec::with_capacity(total);
    for me_flat in &results {
        output.extend_from_slice(me_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC4+ composite element stiffness matrices (global coords).
#[pyfunction]
fn batch_ke_mitc4_composite<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    cm_flat: PyReadonlyArray2<'py, f64>,
    cb_flat: PyReadonlyArray2<'py, f64>,
    cs_flat: PyReadonlyArray2<'py, f64>,
    thickness: PyReadonlyArray1<'py, f64>,
    e_equiv: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let cm_arr = cm_flat.as_array();
    let cb_arr = cb_flat.as_array();
    let cs_arr = cs_flat.as_array();
    let h_arr = thickness.as_array();
    let e_arr = e_equiv.as_array();
    let n_elem = coords_arr.nrows();

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 12];
            for i in 0..12 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let mut a = [0.0f64; 9];
            let mut d = [0.0f64; 9];
            let mut cs = [0.0f64; 4];
            for i in 0..9 {
                a[i] = cm_arr[[e, i]];
                d[i] = cb_arr[[e, i]];
            }
            for i in 0..4 {
                cs[i] = cs_arr[[e, i]];
            }
            let h = h_arr[e];
            let e_eq = e_arr[e];

            let constitutive = composite_constitutive(&a, &d, &cs, h);
            let pre = Mitc4Precomputed::new(&node_coords, constitutive, h, e_eq);
            let ke = mitc4::compute_ke_global(&pre);
            let mut flat = [0.0f64; 576];
            for i in 0..24 {
                for j in 0..24 {
                    flat[i * 24 + j] = ke[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 576;
    let mut output = Vec::with_capacity(total);
    for ke_flat in &results {
        output.extend_from_slice(ke_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Batch-compute MITC4+ composite element mass matrices (global coords).
#[pyfunction]
fn batch_me_mitc4_composite<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    mass_per_area: PyReadonlyArray1<'py, f64>,
    rotational_inertia: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let coords_arr = coords.as_array();
    let mpa_arr = mass_per_area.as_array();
    let ri_arr = rotational_inertia.as_array();
    let n_elem = coords_arr.nrows();

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|e| {
            let mut node_coords = [0.0f64; 12];
            for i in 0..12 {
                node_coords[i] = coords_arr[[e, i]];
            }
            let m_trans = mpa_arr[e];
            let m_rot = ri_arr[e];

            let dummy = composite_constitutive(
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[1.0, 0.0, 0.0, 1.0],
                1.0,
            );
            let pre = Mitc4Precomputed::new(&node_coords, dummy, 1.0, 1.0);
            let me = mitc4::compute_me_composite_global(&pre, m_trans, m_rot);
            let mut flat = [0.0f64; 576];
            for i in 0..24 {
                for j in 0..24 {
                    flat[i * 24 + j] = me[(i, j)];
                }
            }
            flat
        })
        .collect();

    let total = n_elem * 576;
    let mut output = Vec::with_capacity(total);
    for me_flat in &results {
        output.extend_from_slice(me_flat);
    }
    Array1::from(output).into_pyarray(py)
}

/// Generate COO triplets for sparse assembly from element matrices.
#[pyfunction]
fn coo_assembly<'py>(
    py: Python<'py>,
    dofs: PyReadonlyArray2<'py, i64>,
    ke_flat: PyReadonlyArray1<'py, f64>,
    n_dof_per_elem: usize,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let dofs_arr = dofs.as_array();
    let ke_arr = ke_flat.as_array();
    let n_elem = dofs_arr.nrows();

    let dofs_flat: Vec<i64> = dofs_arr.iter().copied().collect();
    let ke_flat_slice: Vec<f64> = ke_arr.iter().copied().collect();

    let (rows, cols, vals) =
        assembly::coo_from_batch(&dofs_flat, &ke_flat_slice, n_elem, n_dof_per_elem);

    (
        Array1::from(rows).into_pyarray(py),
        Array1::from(cols).into_pyarray(py),
        Array1::from(vals).into_pyarray(py),
    )
}

/// Compute NNZ per row for PETSc preallocation.
#[pyfunction]
fn compute_nnz<'py>(
    py: Python<'py>,
    dofs: PyReadonlyArray2<'py, i64>,
    n_dof_total: usize,
) -> Bound<'py, PyArray1<i64>> {
    let dofs_arr = dofs.as_array();
    let n_elem = dofs_arr.nrows();
    let dof_per_elem = dofs_arr.ncols();

    let dofs_flat: Vec<i64> = dofs_arr.iter().copied().collect();
    let nnz = assembly::compute_nnz_per_row(&dofs_flat, n_elem, dof_per_elem, n_dof_total);

    Array1::from(nnz).into_pyarray(py)
}

// ============================================================================
// PETSc pipeline: assemble + modal solve via SLEPc
// ============================================================================

/// Assemble a global stiffness or mass matrix into a PETSc Mat (AIJ sequential).
///
/// Accepts COO triplets produced by `coo_assembly` and returns an opaque
/// Python capsule wrapping the assembled `PetscMat`. The capsule is consumed
/// by `petsc_modal_solve`.
///
/// Args:
///   rows:        COO row indices, dtype=int32
///   cols:        COO col indices, dtype=int32
///   vals:        COO values, dtype=float64
///   n_dof:       total number of DOFs (matrix is n_dof × n_dof)
///
/// Returns:
///   A Python capsule wrapping the PETSc Mat handle.
///   The handle is freed automatically when the capsule is garbage-collected.
#[pyfunction]
fn petsc_assemble_matrix<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
    vals: PyReadonlyArray1<'py, f64>,
    n_dof: usize,
) -> PyResult<Bound<'py, PyCapsule>> {
    let rows_s = rows.as_slice()?;
    let cols_s = cols.as_slice()?;
    let vals_s = vals.as_slice()?;

    let mat = aeroelast_solvers::petsc::assembler::assemble_seq_aij(rows_s, cols_s, vals_s, n_dof)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Move `mat` into the capsule. PyO3 boxes it internally; Drop runs MatDestroy.
    PyCapsule::new(py, mat, None)
}

/// Modal solve using SLEPc EPS on two PETSc matrices (K and M).
///
/// Args:
///   k_capsule:  Python capsule wrapping the stiffness PetscMat
///   m_capsule:  Python capsule wrapping the mass PetscMat
///   n_modes:    number of modes to compute
///
/// Returns:
///   (eigenvalues, eigenvectors_flat)
///   eigenvalues:      np.ndarray shape (n_conv,)  — ω² (rad²/s²)
///   eigenvectors_flat: np.ndarray shape (n_conv * n_dof,) — row-major mode shapes
#[pyfunction]
fn petsc_modal_solve<'py>(
    py: Python<'py>,
    k_capsule: &Bound<'py, PyCapsule>,
    m_capsule: &Bound<'py, PyCapsule>,
    n_modes: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    // SAFETY: both capsules were created by `petsc_assemble_matrix` in this module,
    // so they contain valid `PetscMat` values. `pointer_checked(None)` returns a
    // NonNull<c_void> pointing to the boxed value; we cast and reborrow as shared ref.
    let k = unsafe {
        k_capsule
            .pointer_checked(None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .cast::<aeroelast_solvers::PetscMat>()
            .as_ref()
    };
    let m = unsafe {
        m_capsule
            .pointer_checked(None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .cast::<aeroelast_solvers::PetscMat>()
            .as_ref()
    };

    let result = aeroelast_solvers::petsc::modal::modal_solve(k, m, n_modes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let eigenvalues = Array1::from(result.eigenvalues).into_pyarray(py);
    let eigenvectors_flat: Vec<f64> = result.eigenvectors.into_iter().flatten().collect();
    let eigenvectors = Array1::from(eigenvectors_flat).into_pyarray(py);

    Ok((eigenvalues, eigenvectors))
}

/// All-in-one modal solve from full-system COO triplets + free DOF list.
///
/// Restricts the COO system to the free DOFs submatrix, assembles the
/// reduced PETSc Mats, solves the generalized eigenvalue problem K·x = λ·M·x,
/// and returns natural frequencies in Hz (sorted ascending).
///
/// Args:
///   k_rows, k_cols, k_vals  – COO triplets for the full stiffness matrix (i64 indices)
///   m_rows, m_cols, m_vals  – COO triplets for the full mass matrix (i64 indices)
///   n_dof_total             – total DOF count (full system size)
///   free_dofs               – 1-D array of free DOF indices (i64, sorted)
///   n_modes                 – number of modes to compute
///
/// Returns:
///   (frequencies_hz, modes_flat)
///   frequencies_hz:  np.ndarray shape (n_conv,)  — natural frequencies in Hz
///   modes_flat:      np.ndarray shape (n_conv * n_free,) — row-major, reduced space
#[pyfunction]
fn modal_solve_coo<'py>(
    py: Python<'py>,
    k_rows: PyReadonlyArray1<'py, i64>,
    k_cols: PyReadonlyArray1<'py, i64>,
    k_vals: PyReadonlyArray1<'py, f64>,
    m_rows: PyReadonlyArray1<'py, i64>,
    m_cols: PyReadonlyArray1<'py, i64>,
    m_vals: PyReadonlyArray1<'py, f64>,
    n_dof_total: usize,
    free_dofs: PyReadonlyArray1<'py, i64>,
    n_modes: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let kr = k_rows.as_slice()?;
    let kc = k_cols.as_slice()?;
    let kv = k_vals.as_slice()?;
    let mr = m_rows.as_slice()?;
    let mc = m_cols.as_slice()?;
    let mv = m_vals.as_slice()?;
    let free = free_dofs.as_slice()?;

    // Build free DOF set and remapping: old global → new reduced index
    let mut free_sorted: Vec<i64> = free.to_vec();
    free_sorted.sort_unstable();
    let n_free = free_sorted.len();
    // HashMap: global dof → reduced index
    let free_map: std::collections::HashMap<i64, i32> = free_sorted
        .iter()
        .enumerate()
        .map(|(new_idx, &old_dof)| (old_dof, new_idx as i32))
        .collect();

    // Restrict COO to free×free submatrix and remap indices to [0, n_free)
    let restrict = |rows: &[i64], cols: &[i64], vals: &[f64]| -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut rr = Vec::new();
        let mut cc = Vec::new();
        let mut vv = Vec::new();
        for ((r, c), v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            if let (Some(&ri), Some(&ci)) = (free_map.get(r), free_map.get(c)) {
                rr.push(ri);
                cc.push(ci);
                vv.push(*v);
            }
        }
        (rr, cc, vv)
    };

    let (rk, ck, vk) = restrict(kr, kc, kv);
    let (rm, cm, vm) = restrict(mr, mc, mv);

    // Assemble reduced PETSc matrices
    let mat_k = aeroelast_solvers::petsc::assembler::assemble_seq_aij(&rk, &ck, &vk, n_free)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mat_m = aeroelast_solvers::petsc::assembler::assemble_seq_aij(&rm, &cm, &vm, n_free)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Solve eigenvalue problem
    let result = aeroelast_solvers::petsc::modal::modal_solve(&mat_k, &mat_m, n_modes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Convert eigenvalues ω² → frequencies in Hz, filter positive, sort
    let mut pairs: Vec<(f64, Vec<f64>)> = result
        .eigenvalues
        .into_iter()
        .zip(result.eigenvectors.into_iter())
        .filter(|(lam, _)| *lam > 1e-8)
        .map(|(lam, vec)| (lam.sqrt() / (2.0 * std::f64::consts::PI), vec))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.truncate(n_modes);

    let frequencies: Vec<f64> = pairs.iter().map(|(f, _)| *f).collect();
    let modes_flat: Vec<f64> = pairs.into_iter().flat_map(|(_, v)| v).collect();

    let _ = n_dof_total; // available if caller needs expansion; not used here

    Ok((
        Array1::from(frequencies).into_pyarray(py),
        Array1::from(modes_flat).into_pyarray(py),
    ))
}

// ============================================================================
// PyMeshAssembler: PyO3 class wrapping aeroelast_core::assembly::MeshAssembler
// ============================================================================

/// Python-accessible mesh assembler backed by the Rust `MeshAssembler`.
///
/// Construct from flat numpy arrays (matching the existing Python patterns),
/// then call assembly methods that return COO triplets as numpy arrays.
/// The Python shim (`assembler.py`) calls `_coo_to_petsc()` to convert.
#[pyclass]
pub struct PyMeshAssembler {
    inner: MeshAssembler,
}

/// Parse a Python material dict into a `MaterialSpec`.
///
/// Expected keys (isotropic): type="isotropic", e, nu, rho, thickness, shear_correction
/// Expected keys (composite): type="composite", cm=[9], cb=[9], cs=[4],
///                            thickness, e_equiv, mass_per_area, rotational_inertia
fn parse_material(py: Python, obj: &Py<PyAny>) -> PyResult<MaterialSpec> {
    let dict = obj.bind(py);
    let mat_type: String = dict.get_item("type")
        .map_err(|_| pyo3::exceptions::PyKeyError::new_err("material dict missing 'type'"))?
        .extract()?;

    match mat_type.as_str() {
        "isotropic" => {
            let e: f64 = dict.get_item("e")?.extract()?;
            let nu: f64 = dict.get_item("nu")?.extract()?;
            let rho: f64 = dict.get_item("rho")?.extract()?;
            let thickness: f64 = dict.get_item("thickness")?.extract()?;
            let shear_correction: f64 = dict.get_item("shear_correction")
                .ok()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(5.0 / 6.0);
            Ok(MaterialSpec::Isotropic { e, nu, rho, thickness, shear_correction })
        }
        "composite" => {
            let cm_list: Vec<f64> = dict.get_item("cm")?.extract()?;
            let cb_list: Vec<f64> = dict.get_item("cb")?.extract()?;
            let cs_list: Vec<f64> = dict.get_item("cs")?.extract()?;
            let thickness: f64 = dict.get_item("thickness")?.extract()?;
            let e_equiv: f64 = dict.get_item("e_equiv")?.extract()?;
            let mass_per_area: f64 = dict.get_item("mass_per_area")?.extract()?;
            let rotational_inertia: f64 = dict.get_item("rotational_inertia")?.extract()?;

            if cm_list.len() != 9 || cb_list.len() != 9 || cs_list.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "composite material: cm must have 9 elements, cb 9, cs 4"
                ));
            }
            let mut cm = [0.0f64; 9];
            let mut cb = [0.0f64; 9];
            let mut cs = [0.0f64; 4];
            cm.copy_from_slice(&cm_list);
            cb.copy_from_slice(&cb_list);
            cs.copy_from_slice(&cs_list);

            Ok(MaterialSpec::Composite { cm, cb, cs, thickness, e_equiv, mass_per_area, rotational_inertia })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown material type '{}', expected 'isotropic' or 'composite'", other
        ))),
    }
}

#[pymethods]
impl PyMeshAssembler {
    /// Construct a `PyMeshAssembler` from Python mesh data.
    ///
    /// Parameters
    /// ----------
    /// node_coords : np.ndarray shape (n_nodes, 3)
    ///     Node coordinates in global frame.
    /// connectivity : list[list[int]]
    ///     Per-element node index lists (0-based).
    /// elem_types : list[int]
    ///     Element type per element: 3 = MITC3, 4 = MITC4.
    /// materials : list[dict]
    ///     Per-element material dicts with keys:
    ///     isotropic — {type, e, nu, rho, thickness, shear_correction}
    ///     composite — {type, cm, cb, cs, thickness, e_equiv, mass_per_area, rotational_inertia}
    #[new]
    pub fn new(
        node_coords: PyReadonlyArray2<f64>,
        connectivity: Vec<Vec<usize>>,
        elem_types: Vec<u8>,
        materials: Vec<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Self> {
        let coords_arr = node_coords.as_array();
        let n_nodes = coords_arr.nrows();

        // Build flat node_coords vec
        let mut flat_coords = Vec::with_capacity(n_nodes * 3);
        for i in 0..n_nodes {
            flat_coords.push(coords_arr[[i, 0]]);
            flat_coords.push(coords_arr[[i, 1]]);
            flat_coords.push(coords_arr[[i, 2]]);
        }

        // Build elem_types vec
        let rust_elem_types: Vec<ElemType> = elem_types
            .iter()
            .map(|&t| match t {
                3 => Ok(ElemType::Mitc3),
                4 => Ok(ElemType::Mitc4),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown elem_type {}: expected 3 (MITC3) or 4 (MITC4)", other
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Build topology
        let topology = MeshTopology::new(flat_coords, connectivity, rust_elem_types);

        // Parse material specs
        let rust_materials: Vec<MaterialSpec> = materials
            .iter()
            .map(|m| parse_material(py, m))
            .collect::<PyResult<Vec<_>>>()?;

        let inner = MeshAssembler::new(topology, rust_materials);
        Ok(PyMeshAssembler { inner })
    }

    /// Total number of DOFs in the system.
    #[getter]
    pub fn dofs_count(&self) -> usize {
        self.inner.dofs_count
    }

    /// Assemble the global elastic stiffness matrix K.
    ///
    /// Returns (rows, cols, vals) as numpy int64/float64 arrays.
    pub fn assemble_k<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<f64>>,
    ) {
        let (rows, cols, vals) = self.inner.assemble_k();
        (
            Array1::from(rows).into_pyarray(py),
            Array1::from(cols).into_pyarray(py),
            Array1::from(vals).into_pyarray(py),
        )
    }

    /// Assemble the global consistent mass matrix M.
    ///
    /// Returns (rows, cols, vals) as numpy int64/float64 arrays.
    pub fn assemble_m<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<f64>>,
    ) {
        let (rows, cols, vals) = self.inner.assemble_m();
        (
            Array1::from(rows).into_pyarray(py),
            Array1::from(cols).into_pyarray(py),
            Array1::from(vals).into_pyarray(py),
        )
    }

    /// Assemble the global body load vector.
    ///
    /// Parameters
    /// ----------
    /// gravity : list[f64] of length 3 — body acceleration [gx, gy, gz] m/s²
    ///
    /// Returns np.ndarray of length dofs_count.
    pub fn assemble_f_body<'py>(
        &self,
        py: Python<'py>,
        gravity: [f64; 3],
    ) -> pyo3::Bound<'py, PyArray1<f64>> {
        let f = self.inner.assemble_f_body(gravity);
        Array1::from(f).into_pyarray(py)
    }

    /// Assemble the global geometric stiffness matrix K_σ.
    ///
    /// Parameters
    /// ----------
    /// sigma : np.ndarray shape (n_elems, 3) — membrane stress [σxx, σyy, σxy] per element
    ///
    /// Returns (rows, cols, vals) as numpy int64/float64 arrays.
    pub fn assemble_geometric_k<'py>(
        &self,
        py: Python<'py>,
        sigma: PyReadonlyArray2<f64>,
    ) -> PyResult<(
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<f64>>,
    )> {
        let sigma_arr = sigma.as_array();
        let n_elems = sigma_arr.nrows();
        if n_elems != self.inner.topology.n_elems {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sigma must have shape (n_elems, 3); got ({}, {})",
                n_elems, sigma_arr.ncols()
            )));
        }
        let sigma_vecs: Vec<[f64; 3]> = (0..n_elems)
            .map(|e| [sigma_arr[[e, 0]], sigma_arr[[e, 1]], sigma_arr[[e, 2]]])
            .collect();
        let (rows, cols, vals) = self.inner.assemble_geometric_k(&sigma_vecs);
        Ok((
            Array1::from(rows).into_pyarray(py),
            Array1::from(cols).into_pyarray(py),
            Array1::from(vals).into_pyarray(py),
        ))
    }

    /// Assemble the nonlinear tangent stiffness matrix K_T(u).
    ///
    /// Parameters
    /// ----------
    /// u : np.ndarray shape (dofs_count,) — global displacement vector
    ///
    /// Returns (rows, cols, vals) as numpy int64/float64 arrays.
    pub fn assemble_kt<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray1<f64>,
    ) -> PyResult<(
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<f64>>,
    )> {
        let u_slice = u.as_slice()?;
        let (rows, cols, vals) = self.inner.assemble_kt(u_slice);
        Ok((
            Array1::from(rows).into_pyarray(py),
            Array1::from(cols).into_pyarray(py),
            Array1::from(vals).into_pyarray(py),
        ))
    }

    /// Assemble the global internal force vector f_int(u).
    ///
    /// Parameters
    /// ----------
    /// u : np.ndarray shape (dofs_count,) — global displacement vector
    /// nonlinear : bool — include geometric-nonlinear contributions
    ///
    /// Returns np.ndarray of length dofs_count.
    pub fn assemble_fint<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray1<f64>,
        nonlinear: bool,
    ) -> PyResult<pyo3::Bound<'py, PyArray1<f64>>> {
        let u_slice = u.as_slice()?;
        let f = self.inner.assemble_fint(u_slice, nonlinear);
        Ok(Array1::from(f).into_pyarray(py))
    }

    /// NNZ per row for PETSc preallocation.
    ///
    /// Returns np.ndarray of length dofs_count (int64).
    pub fn nnz_per_row<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<i64>> {
        Array1::from(self.inner.nnz_per_row().to_vec()).into_pyarray(py)
    }
}

/// Register all aeroelast functions into a PyModule.
///
/// Called by both `aeroelast` and `fem_shell_core` modules to expose the
/// same Python API under different module names.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_ke_mitc3, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_mitc3, m)?)?;
    m.add_function(wrap_pyfunction!(batch_kt_mitc3, m)?)?;
    m.add_function(wrap_pyfunction!(batch_fint_mitc3, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_mitc4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_mitc4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_kt_mitc4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_fint_mitc4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_mitc3_composite, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_mitc3_composite, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_mitc4_composite, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_mitc4_composite, m)?)?;
    m.add_function(wrap_pyfunction!(coo_assembly, m)?)?;
    m.add_function(wrap_pyfunction!(compute_nnz, m)?)?;
    m.add_function(wrap_pyfunction!(petsc_assemble_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(petsc_modal_solve, m)?)?;
    m.add_function(wrap_pyfunction!(modal_solve_coo, m)?)?;
    m.add_class::<PyMeshAssembler>()?;
    Ok(())
}

// ============================================================================
// Module entry point (named `aeroelast`)
// ============================================================================

#[pymodule]
fn aeroelast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(m)
}
