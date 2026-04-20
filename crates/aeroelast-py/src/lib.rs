use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use rayon::prelude::*;

use aeroelast_core::elements::mitc3::{self, Mitc3Precomputed};
use aeroelast_core::elements::mitc4::{self, Mitc4Precomputed};
use aeroelast_core::elements::quad::{Quad4Precomputed, Quad8Precomputed, Quad9Precomputed};
use aeroelast_core::elements::solid;
use aeroelast_core::materials::composite::composite_constitutive;
use aeroelast_core::materials::isotropic::IsotropicMaterial;
use aeroelast_core::materials::laminate::{Laminate, Ply};
use aeroelast_core::materials::orthotropic::OrthotropicMaterial;
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
/// Expected keys (plane_stress): type="plane_stress", e, nu, rho, thickness
/// Expected keys (solid_3d): type="solid_3d", e, nu, rho
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
        "plane_stress" => {
            let e: f64 = dict.get_item("e")?.extract()?;
            let nu: f64 = dict.get_item("nu")?.extract()?;
            let rho: f64 = dict.get_item("rho")?.extract()?;
            let thickness: f64 = dict.get_item("thickness")
                .ok()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(1.0);
            Ok(MaterialSpec::PlaneStress { e, nu, rho, thickness })
        }
        "solid_3d" => {
            let e: f64 = dict.get_item("e")?.extract()?;
            let nu: f64 = dict.get_item("nu")?.extract()?;
            let rho: f64 = dict.get_item("rho")?.extract()?;
            Ok(MaterialSpec::Solid3D { e, nu, rho })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown material type '{}', expected 'isotropic', 'composite', 'plane_stress', or 'solid_3d'", other
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
    ///     Element type code per element:
    ///       3   = MITC3,       4   = MITC4
    ///       33  = MITC3Composite, 44 = MITC4Composite
    ///       104 = QUAD4,       108 = QUAD8,       109 = QUAD9
    ///       208 = HEXA8,       220 = HEXA20
    ///       304 = TETRA4,      310 = TETRA10
    ///       306 = WEDGE6,      315 = WEDGE15
    ///       305 = PYRAMID5,    313 = PYRAMID13
    /// materials : list[dict]
    ///     Per-element material dicts.  Supported types:
    ///     isotropic   — {type, e, nu, rho, thickness, shear_correction}
    ///     composite   — {type, cm, cb, cs, thickness, e_equiv, mass_per_area, rotational_inertia}
    ///     plane_stress — {type, e, nu, rho, thickness}
    ///     solid_3d    — {type, e, nu, rho}
    #[new]
    pub fn new(
        node_coords: PyReadonlyArray2<f64>,
        connectivity: Vec<Vec<usize>>,
        elem_types: Vec<u16>,
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
                3   => Ok(ElemType::Mitc3),
                4   => Ok(ElemType::Mitc4),
                33  => Ok(ElemType::Mitc3Composite),
                44  => Ok(ElemType::Mitc4Composite),
                104 => Ok(ElemType::Quad4),
                108 => Ok(ElemType::Quad8),
                109 => Ok(ElemType::Quad9),
                208 => Ok(ElemType::Hexa8),
                220 => Ok(ElemType::Hexa20),
                304 => Ok(ElemType::Tetra4),
                310 => Ok(ElemType::Tetra10),
                306 => Ok(ElemType::Wedge6),
                315 => Ok(ElemType::Wedge15),
                305 => Ok(ElemType::Pyramid5),
                313 => Ok(ElemType::Pyramid13),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown elem_type code {}: see PyMeshAssembler docstring for valid codes", other
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

    /// Construct a `PyMeshAssembler` from a `MeshModel` already loaded in Rust.
    ///
    /// This is the fast path: topology (node_coords, connectivity, elem_type codes)
    /// is extracted from Rust-native `MeshModel` without any Python loop.
    /// Only the `materials` list still comes from Python.
    ///
    /// Parameters
    /// ----------
    /// mesh : MeshModel
    ///     A `MeshModel` loaded via `MeshModel.from_hdf5(...)`.
    /// materials : list[dict]
    ///     Per-element material dicts in the same element order as `mesh.element_ids()`.
    ///     Same format as `PyMeshAssembler.__init__` (isotropic / composite / ...).
    #[staticmethod]
    pub fn from_mesh_model(
        mesh: &mut PyMeshModel,
        materials: Vec<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Self> {
        // Extract flat coords from Rust MeshModel — no Python loop needed
        let flat_coords = mesh.inner.node_coords_flat().to_vec();

        // Extract connectivity and elem_type codes — fully in Rust
        let (connectivity_usize, code_i32) = mesh.inner.build_connectivity_arrays();

        // Convert i32 codes to ElemType
        let rust_elem_types: Vec<ElemType> = code_i32
            .iter()
            .map(|&t| match t as u16 {
                3   => Ok(ElemType::Mitc3),
                4   => Ok(ElemType::Mitc4),
                33  => Ok(ElemType::Mitc3Composite),
                44  => Ok(ElemType::Mitc4Composite),
                104 => Ok(ElemType::Quad4),
                108 => Ok(ElemType::Quad8),
                109 => Ok(ElemType::Quad9),
                208 => Ok(ElemType::Hexa8),
                220 => Ok(ElemType::Hexa20),
                304 => Ok(ElemType::Tetra4),
                310 => Ok(ElemType::Tetra10),
                306 => Ok(ElemType::Wedge6),
                315 => Ok(ElemType::Wedge15),
                305 => Ok(ElemType::Pyramid5),
                313 => Ok(ElemType::Pyramid13),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown elem_type code {other} from MeshModel::build_connectivity_arrays"
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?;

        let topology = MeshTopology::new(flat_coords, connectivity_usize, rust_elem_types);

        // Parse material specs (still from Python)
        let rust_materials: Vec<MaterialSpec> = materials
            .iter()
            .map(|m| parse_material(py, m))
            .collect::<PyResult<Vec<_>>>()?;

        let inner = MeshAssembler::new(topology, rust_materials);
        Ok(PyMeshAssembler { inner })
    }

    /// Build a `PyMeshAssembler` directly from a `PyMeshModel` + Python properties map.
    ///
    /// This replaces `_build_py_mesh_assembler` entirely — no Python loops for
    /// node_coords, connectivity, or angle offsets. Only the properties dict is
    /// still passed from Python.
    ///
    /// Parameters
    /// ----------
    /// mesh : MeshModel
    ///     Rust-native mesh loaded via `MeshModel.from_hdf5(...)`.
    /// properties : dict[str, PyLaminate | dict]
    ///     Mapping from element-set name to property. Supported values:
    ///     - `PyLaminate` instance → composite MITC3/4 element
    ///     - dict with keys `type="isotropic"`, `e`, `nu`, `rho`, `thickness`[, `shear_correction`]
    /// span_direction : list[float] | None
    ///     3-component span direction for fibre angle correction. None to skip.
    /// fallback_material : dict | None
    ///     Isotropic material dict used for elements not in any property set.
    ///     If None, elements without properties raise an error.
    #[staticmethod]
    #[pyo3(signature = (mesh, properties, span_direction=None, fallback_material=None))]
    pub fn from_model(
        mesh: &mut PyMeshModel,
        properties: &pyo3::Bound<'_, pyo3::types::PyDict>,
        span_direction: Option<Vec<f64>>,
        fallback_material: Option<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Self> {
        use aeroelast_mesh::geometry::batch_angle_offsets;
        use aeroelast_core::materials::laminate::Laminate;
        use std::collections::HashMap;

        // ── topology (all in Rust, no Python loops) ──────────────────────────
        let flat_coords = mesh.inner.node_coords_flat().to_vec();
        let (connectivity, code_i32) = mesh.inner.build_connectivity_arrays();

        // ── parse Python properties dict: set_name → MaterialSpec template ───
        // We pre-compute ABD for each (laminate, angle_bucket) pair (the cache)
        // Properties can be PyLaminate or dict
        let mut set_to_prop: HashMap<String, Py<PyAny>> = HashMap::new();
        for (key, val) in properties.iter() {
            let set_name: String = key.extract()?;
            set_to_prop.insert(set_name, val.unbind());
        }

        // ── per-element property lookup: elem_id → (set_name, property) ──────
        // We build this from element_sets in MeshModel (already in Rust).
        // An element may appear in multiple sets (e.g. "allOuterShellEls" and
        // "05_00_HP_LE"). Only sets that have a corresponding property matter;
        // sets without a property are skipped so they don't overwrite a valid
        // mapping with a last-write-wins collision.
        let mut elem_to_set: HashMap<u64, String> = HashMap::new();
        for (set_name, eset) in &mesh.inner.element_sets {
            if !set_to_prop.contains_key(set_name) {
                continue; // skip sets that have no property — avoids last-write-wins clobber
            }
            for &eid in &eset.element_ids {
                elem_to_set.insert(eid, set_name.clone());
            }
        }

        // ── Phase 1: batch angle offsets (Rayon parallel, pure Rust) ─────────
        let span_dir: Option<[f64; 3]> = span_direction.as_ref().and_then(|v| {
            if v.len() >= 3 { Some([v[0], v[1], v[2]]) } else { None }
        });

        // Only compute for composite quad4 elements that have a laminate property
        let mut elem_angle_offsets: HashMap<u64, f64> = HashMap::new();
        if let Some(sd) = span_dir {
            // Identify composite quad4 elements
            let quad_indices: Vec<usize> = mesh.inner.elements.iter().enumerate()
                .filter_map(|(i, elem)| {
                    let set = elem_to_set.get(&elem.id)?;
                    let prop = set_to_prop.get(set)?;
                    // Check if it's a PyLaminate (composite) and has 4 nodes
                    let is_composite = prop.bind(py).is_instance_of::<PyLaminate>();
                    if is_composite && elem.node_ids.len() == 4 { Some(i) } else { None }
                })
                .collect();

            if !quad_indices.is_empty() {
                let quad_conns: Vec<Vec<usize>> = quad_indices.iter()
                    .map(|&i| connectivity[i].clone())
                    .collect();
                let offsets = batch_angle_offsets(&flat_coords, &quad_conns, sd);
                for (i_local, &i_elem) in quad_indices.iter().enumerate() {
                    elem_angle_offsets.insert(mesh.inner.elements[i_elem].id, offsets[i_local]);
                }
            }
        }

        // ── Phase 2a: pre-compute MaterialSpec per (set, angle_bucket) ──────────
        // This avoids PyO3 downcast + GIL ops inside the hot 86k-element loop.
        // We extract all Python objects here (with GIL), compute ABD matrices,
        // and cache the results in a pure-Rust HashMap indexed by (set_name, bucket).
        //
        // Key insight: there are only O(N_sets × N_angle_buckets) unique specs,
        // not O(N_elements). For a blade with 722 sets and typical angle spread,
        // this is ~722 cache misses total instead of ~86k PyO3 ops per run.

        // Helper closure: compute MaterialSpec from a Laminate + angle_bucket_tenths
        let compute_composite_spec = |lam: &Laminate, bucket_tenths: i32| -> PyResult<MaterialSpec> {
            let corrected_lam = if bucket_tenths == 0 {
                lam.clone()
            } else {
                let angle_deg = bucket_tenths as f64 / 10.0;
                let corrected_plies: Vec<_> = lam.plies.iter().map(|p| {
                    aeroelast_core::materials::laminate::Ply {
                        material: p.material.clone(),
                        thickness: p.thickness,
                        angle: p.angle + angle_deg,
                        z_bottom: p.z_bottom,
                        z_top: p.z_top,
                    }
                }).collect();
                Laminate::new(corrected_plies, lam.shear_correction_factor)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
            };

            let h = corrected_lam.total_thickness;
            let a_trace = corrected_lam.a[(0,0)] + corrected_lam.a[(1,1)] + corrected_lam.a[(2,2)];
            let e_equiv = if h > 0.0 { a_trace / (3.0 * h) } else { 0.0 };
            let mass_per_area: f64 = corrected_lam.plies.iter()
                .map(|p| p.material.density() * p.thickness)
                .sum();
            let rotational_inertia: f64 = corrected_lam.plies.iter()
                .map(|p| p.material.density() * (p.z_top.powi(3) - p.z_bottom.powi(3)) / 3.0)
                .sum();

            let mut cm = [0.0f64; 9];
            let mut cb = [0.0f64; 9];
            let mut cs = [0.0f64; 4];
            for ii in 0..3 { for jj in 0..3 {
                cm[ii*3+jj] = corrected_lam.a[(ii, jj)];
                cb[ii*3+jj] = corrected_lam.d[(ii, jj)];
            }}
            cs[0] = corrected_lam.cs[(0,0)];
            cs[1] = corrected_lam.cs[(0,1)];
            cs[2] = corrected_lam.cs[(1,0)];
            cs[3] = corrected_lam.cs[(1,1)];

            Ok(MaterialSpec::Composite { cm, cb, cs, thickness: h, e_equiv, mass_per_area, rotational_inertia })
        };

        // Enum to distinguish composite vs isotropic without PyO3 objects in hot path
        #[derive(Clone)]
        enum SetSpec {
            Composite(Laminate),  // raw Laminate for angle-corrected lookup
            Isotropic(MaterialSpec), // pre-parsed, no angle correction needed
        }

        // Pre-compute SetSpec per set (O(N_sets) PyO3 ops, not O(N_elements))
        let mut set_to_setspec: HashMap<String, SetSpec> = HashMap::with_capacity(set_to_prop.len());
        for (set_name, prop) in &set_to_prop {
            let bound = prop.bind(py);
            if let Ok(lam_ref) = bound.downcast::<PyLaminate>() {
                let lam = lam_ref.borrow().inner.clone();
                set_to_setspec.insert(set_name.clone(), SetSpec::Composite(lam));
            } else {
                let spec = parse_material(py, prop)?;
                set_to_setspec.insert(set_name.clone(), SetSpec::Isotropic(spec));
            }
        }

        // Pre-compute fallback spec if provided (avoids per-element parse)
        let fallback_spec: Option<MaterialSpec> = fallback_material.as_ref()
            .map(|fb| parse_material(py, fb))
            .transpose()?;

        // ABD cache: (set_name, angle_bucket_tenths) → MaterialSpec (composite only)
        // bucket_tenths=0 covers the vast majority of elements when no span_direction.
        let mut abd_cache: HashMap<(String, i32), MaterialSpec> = HashMap::new();

        // ── Phase 2b: build MaterialSpec per element (pure Rust, no PyO3) ──────
        let n_elems = mesh.inner.elements.len();
        let mut rust_materials: Vec<MaterialSpec> = Vec::with_capacity(n_elems);
        let mut rust_elem_types: Vec<ElemType> = Vec::with_capacity(n_elems);

        for (i, elem) in mesh.inner.elements.iter().enumerate() {
            let n_nodes_elem = elem.node_ids.len();

            let set_name_opt: Option<&String> = elem_to_set.get(&elem.id);

            let (etype, mat_spec) = if let Some(set_name) = set_name_opt {
                match set_to_setspec.get(set_name) {
                    Some(SetSpec::Composite(lam)) => {
                        // ── Composite shell ───────────────────────────────────
                        let etype = if n_nodes_elem == 3 { ElemType::Mitc3Composite } else { ElemType::Mitc4Composite };

                        let angle_offset = elem_angle_offsets.get(&elem.id).copied().unwrap_or(0.0);
                        let bucket_tenths = (angle_offset * 10.0).round() as i32;

                        // Use (set_name, bucket) as cache key — pure Rust, no PyO3
                        let spec = if let Some(cached) = abd_cache.get(&(set_name.clone(), bucket_tenths)) {
                            cached.clone()
                        } else {
                            let spec = compute_composite_spec(lam, bucket_tenths)?;
                            abd_cache.insert((set_name.clone(), bucket_tenths), spec.clone());
                            spec
                        };
                        (etype, spec)
                    }
                    Some(SetSpec::Isotropic(spec)) => {
                        // ── Isotropic dict ────────────────────────────────────
                        let etype = if n_nodes_elem == 3 { ElemType::Mitc3 } else { ElemType::Mitc4 };
                        (etype, spec.clone())
                    }
                    None => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "element {} belongs to set '{}' which has no property",
                            elem.id, set_name
                        )));
                    }
                }
            } else if let Some(ref fb_spec) = fallback_spec {
                // ── Fallback isotropic ─────────────────────────────────────────
                let etype = if n_nodes_elem == 3 { ElemType::Mitc3 } else { ElemType::Mitc4 };
                (etype, fb_spec.clone())
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "element {} has no property and no fallback_material was provided",
                    elem.id
                )));
            };

            // Override elem_type from mesh entities (for solid elements etc.)
            let final_etype = {
                let mesh_code = code_i32[i];
                match mesh_code as u16 {
                    3   => ElemType::Mitc3,
                    4   => ElemType::Mitc4,
                    33  => ElemType::Mitc3Composite,
                    44  => ElemType::Mitc4Composite,
                    104 => ElemType::Quad4,
                    108 => ElemType::Quad8,
                    109 => ElemType::Quad9,
                    208 => ElemType::Hexa8,
                    220 => ElemType::Hexa20,
                    304 => ElemType::Tetra4,
                    310 => ElemType::Tetra10,
                    306 => ElemType::Wedge6,
                    315 => ElemType::Wedge15,
                    305 => ElemType::Pyramid5,
                    313 => ElemType::Pyramid13,
                    // For shell elements loaded from HDF5 without composite info,
                    // use the composite/isotropic determination from property
                    _ => etype,
                }
            };

            rust_elem_types.push(final_etype);
            rust_materials.push(mat_spec);
        }
        let topology = MeshTopology::new(flat_coords, connectivity, rust_elem_types);
        let inner = MeshAssembler::new(topology, rust_materials);
        Ok(PyMeshAssembler { inner })
    }

    /// Total number of DOFs in the system.
    #[getter]
    pub fn dofs_count(&self) -> usize {
        self.inner.dofs_count
    }

    #[getter]
    pub fn n_elems(&self) -> usize {
        self.inner.topology.n_elems
    }

    #[getter]
    pub fn n_nodes(&self) -> usize {
        self.inner.topology.n_nodes
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

    /// Return mass-per-area (kg/m²) for each element.
    ///
    /// For composite elements this is Σ(ρ_k · t_k) over all plies.
    /// For isotropic shell elements this is ρ (kg/m³) — the caller is
    /// responsible for multiplying by thickness when needed.
    ///
    /// Returns
    /// -------
    /// np.ndarray shape (n_elems,), dtype float64
    pub fn rho_per_elem<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<f64>> {
        let rho: Vec<f64> = self.inner.materials.iter().map(|m| match m {
            MaterialSpec::Composite { mass_per_area, .. } => *mass_per_area,
            MaterialSpec::Isotropic { rho, .. } => *rho,
            _ => 0.0,
        }).collect();
        Array1::from(rho).into_pyarray(py)
    }

    /// Assemble the geometric stiffness matrix K_σ from centrifugal loading.
    ///
    /// Computes centrifugal prestress per element from first principles and
    /// assembles the geometric stiffness matrix in a single Rust pass.
    ///
    /// Parameters
    /// ----------
    /// omega : float
    ///     Angular velocity magnitude (rad/s).
    /// rotation_axis : list[float] of length 3
    ///     Unit vector of the rotation axis [ax, ay, az].
    /// rotation_center : list[float] of length 3
    ///     A point on the rotation axis [cx, cy, cz] (m).
    /// rho_per_elem : np.ndarray shape (n_elems,)
    ///     Density (kg/m³) or mass-per-area (kg/m²) per element.
    ///     For isotropic shell elements use ρ (kg/m³);
    ///     for composite laminates use mass_per_area (kg/m²).
    ///
    /// Returns (rows, cols, vals) as numpy int64/float64 arrays.
    pub fn assemble_centrifugal_k<'py>(
        &self,
        py: Python<'py>,
        omega: f64,
        rotation_axis: [f64; 3],
        rotation_center: [f64; 3],
        rho_per_elem: PyReadonlyArray1<f64>,
    ) -> PyResult<(
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<i64>>,
        pyo3::Bound<'py, PyArray1<f64>>,
    )> {
        let rho_slice = rho_per_elem.as_slice()?;
        if rho_slice.len() != self.inner.topology.n_elems {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "rho_per_elem must have length n_elems={}; got {}",
                self.inner.topology.n_elems,
                rho_slice.len()
            )));
        }
        let (rows, cols, vals) = self.inner.assemble_centrifugal_k(
            omega,
            rotation_axis,
            rotation_center,
            rho_slice,
        );
        Ok((
            Array1::from(rows).into_pyarray(py),
            Array1::from(cols).into_pyarray(py),
            Array1::from(vals).into_pyarray(py),
        ))
    }

    /// Compute element-centroid stress and strain for every element.
    ///
    /// Parameters
    /// ----------
    /// u : np.ndarray shape (dofs_count,) — global displacement vector
    /// z_factor : float — non-dimensional through-thickness location for shell
    ///     elements: z = z_factor × h.  Typical values: -0.5 (bottom),
    ///     0.0 (mid-surface, default), +0.5 (top).  Ignored for plane and
    ///     solid elements.
    /// stress_type : int — shell stress contribution (ignored for non-shell):
    ///     0 = membrane only, 1 = bending only, 2 = total (default).
    ///
    /// Returns
    /// -------
    /// (sigma, epsilon) : tuple of np.ndarray, each shape (n_elems, 6)
    ///     Voigt [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx] per element.
    ///     Out-of-plane entries are zero for shell / plane elements.
    pub fn compute_stress_field<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray1<f64>,
        z_factor: f64,
        stress_type: u8,
    ) -> PyResult<(
        pyo3::Bound<'py, PyArray2<f64>>,
        pyo3::Bound<'py, PyArray2<f64>>,
    )> {
        let u_slice = u.as_slice()?;
        let (sigma, epsilon) = self.inner.compute_stress_field(u_slice, z_factor, stress_type);
        let n = sigma.len();
        // Flatten into (n_elems, 6) arrays
        let sigma_flat: Vec<f64> = sigma.into_iter().flat_map(|s| s.into_iter()).collect();
        let eps_flat: Vec<f64> = epsilon.into_iter().flat_map(|e| e.into_iter()).collect();
        let sigma_arr = Array2::from_shape_vec((n, 6), sigma_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py);
        let eps_arr = Array2::from_shape_vec((n, 6), eps_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((sigma_arr, eps_arr))
    }
}

// ============================================================================
// Solid element batch functions: HEXA8, TETRA4, WEDGE6
// ============================================================================

/// Batch-compute HEXA8 element stiffness matrices.
///
/// coords: shape (n_elem, 24) — each row: 8 nodes × [x,y,z]
/// Returns flat array (n_elem × 576).
#[pyfunction]
fn batch_ke_hexa8<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 8];
            for n in 0..8 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::hexa8_ke(&c, e, nu);
            let mut flat = [0.0f64; 576];
            for r in 0..24 {
                for col in 0..24 {
                    flat[r * 24 + col] = ke[(r, col)];
                }
            }
            flat
        })
        .collect();

    let mut out = Vec::with_capacity(n_elem * 576);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute HEXA8 element mass matrices.
///
/// coords: shape (n_elem, 24)
/// Returns flat array (n_elem × 576).
#[pyfunction]
fn batch_me_hexa8<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();

    let results: Vec<[f64; 576]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 8];
            for n in 0..8 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::hexa8_me(&c, rho);
            let mut flat = [0.0f64; 576];
            for r in 0..24 {
                for col in 0..24 {
                    flat[r * 24 + col] = me[(r, col)];
                }
            }
            flat
        })
        .collect();

    let mut out = Vec::with_capacity(n_elem * 576);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute TETRA4 element stiffness matrices.
///
/// coords: shape (n_elem, 12) — each row: 4 nodes × [x,y,z]
/// Returns flat array (n_elem × 144).
#[pyfunction]
fn batch_ke_tetra4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();

    let results: Vec<[f64; 144]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 4];
            for n in 0..4 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::tetra4_ke(&c, e, nu);
            let mut flat = [0.0f64; 144];
            for r in 0..12 {
                for col in 0..12 {
                    flat[r * 12 + col] = ke[(r, col)];
                }
            }
            flat
        })
        .collect();

    let mut out = Vec::with_capacity(n_elem * 144);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute TETRA4 element mass matrices.
///
/// coords: shape (n_elem, 12)
/// Returns flat array (n_elem × 144).
#[pyfunction]
fn batch_me_tetra4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();

    let results: Vec<[f64; 144]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 4];
            for n in 0..4 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::tetra4_me(&c, rho);
            let mut flat = [0.0f64; 144];
            for r in 0..12 {
                for col in 0..12 {
                    flat[r * 12 + col] = me[(r, col)];
                }
            }
            flat
        })
        .collect();

    let mut out = Vec::with_capacity(n_elem * 144);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute WEDGE6 element stiffness matrices.
///
/// coords: shape (n_elem, 18) — each row: 6 nodes × [x,y,z]
/// Returns flat array (n_elem × 324).
#[pyfunction]
fn batch_ke_wedge6<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 6];
            for n in 0..6 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::wedge6_ke(&c, e, nu);
            let mut flat = [0.0f64; 324];
            for r in 0..18 {
                for col in 0..18 {
                    flat[r * 18 + col] = ke[(r, col)];
                }
            }
            flat
        })
        .collect();

    let mut out = Vec::with_capacity(n_elem * 324);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute WEDGE6 element mass matrices.
///
/// coords: shape (n_elem, 18)
/// Returns flat array (n_elem × 324).
#[pyfunction]
fn batch_me_wedge6<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();

    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 6];
            for n in 0..6 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::wedge6_me(&c, rho);
            let mut flat = [0.0f64; 324];
            for r in 0..18 {
                for col in 0..18 {
                    flat[r * 18 + col] = me[(r, col)];
                }
            }
            flat
        })
        .collect();

    let mut out = Vec::with_capacity(n_elem * 324);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

// ============================================================================
// Higher-order solid element batch functions
// ============================================================================

/// Batch-compute TETRA10 element stiffness matrices.
///
/// coords: shape (n_elem, 30) — each row: 10 nodes × [x, y, z]
/// Returns flat array (n_elem × 900).
#[pyfunction]
fn batch_ke_tetra10<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 900]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 10];
            for n in 0..10 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::tetra10_ke(&c, e, nu);
            let mut flat = [0.0f64; 900];
            for r in 0..30 { for col in 0..30 { flat[r * 30 + col] = ke[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 900);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute TETRA10 consistent mass matrices.
///
/// coords: shape (n_elem, 30), rho: density
/// Returns flat array (n_elem × 900).
#[pyfunction]
fn batch_me_tetra10<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 900]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 10];
            for n in 0..10 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::tetra10_me(&c, rho);
            let mut flat = [0.0f64; 900];
            for r in 0..30 { for col in 0..30 { flat[r * 30 + col] = me[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 900);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute WEDGE15 element stiffness matrices.
///
/// coords: shape (n_elem, 45) — each row: 15 nodes × [x, y, z]
/// Returns flat array (n_elem × 2025).
#[pyfunction]
fn batch_ke_wedge15<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 2025]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 15];
            for n in 0..15 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::wedge15_ke(&c, e, nu);
            let mut flat = [0.0f64; 2025];
            for r in 0..45 { for col in 0..45 { flat[r * 45 + col] = ke[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 2025);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute WEDGE15 consistent mass matrices.
///
/// coords: shape (n_elem, 45), rho: density
/// Returns flat array (n_elem × 2025).
#[pyfunction]
fn batch_me_wedge15<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 2025]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 15];
            for n in 0..15 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::wedge15_me(&c, rho);
            let mut flat = [0.0f64; 2025];
            for r in 0..45 { for col in 0..45 { flat[r * 45 + col] = me[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 2025);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute HEXA20 element stiffness matrices.
///
/// coords: shape (n_elem, 60) — each row: 20 nodes × [x, y, z]
/// Returns flat array (n_elem × 3600).
#[pyfunction]
fn batch_ke_hexa20<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 3600]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 20];
            for n in 0..20 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::hexa20_ke(&c, e, nu);
            let mut flat = [0.0f64; 3600];
            for r in 0..60 { for col in 0..60 { flat[r * 60 + col] = ke[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 3600);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute HEXA20 consistent mass matrices.
///
/// coords: shape (n_elem, 60), rho: density
/// Returns flat array (n_elem × 3600).
#[pyfunction]
fn batch_me_hexa20<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 3600]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 20];
            for n in 0..20 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::hexa20_me(&c, rho);
            let mut flat = [0.0f64; 3600];
            for r in 0..60 { for col in 0..60 { flat[r * 60 + col] = me[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 3600);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute PYRAMID5 element stiffness matrices.
///
/// coords: shape (n_elem, 15) — each row: 5 nodes × [x, y, z]
/// Returns flat array (n_elem × 225).
#[pyfunction]
fn batch_ke_pyramid5<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 225]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 5];
            for n in 0..5 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::pyramid5_ke(&c, e, nu);
            let mut flat = [0.0f64; 225];
            for r in 0..15 { for col in 0..15 { flat[r * 15 + col] = ke[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 225);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute PYRAMID5 consistent mass matrices.
///
/// coords: shape (n_elem, 15), rho: density
/// Returns flat array (n_elem × 225).
#[pyfunction]
fn batch_me_pyramid5<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 225]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 5];
            for n in 0..5 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::pyramid5_me(&c, rho);
            let mut flat = [0.0f64; 225];
            for r in 0..15 { for col in 0..15 { flat[r * 15 + col] = me[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 225);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute PYRAMID13 element stiffness matrices.
///
/// coords: shape (n_elem, 39) — each row: 13 nodes × [x, y, z]
/// Returns flat array (n_elem × 1521).
#[pyfunction]
fn batch_ke_pyramid13<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 1521]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 13];
            for n in 0..13 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let ke = solid::pyramid13_ke(&c, e, nu);
            let mut flat = [0.0f64; 1521];
            for r in 0..39 { for col in 0..39 { flat[r * 39 + col] = ke[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 1521);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute PYRAMID13 consistent mass matrices.
///
/// coords: shape (n_elem, 39), rho: density
/// Returns flat array (n_elem × 1521).
#[pyfunction]
fn batch_me_pyramid13<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 1521]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let mut c = [[0.0f64; 3]; 13];
            for n in 0..13 {
                c[n][0] = arr[[i, 3 * n]];
                c[n][1] = arr[[i, 3 * n + 1]];
                c[n][2] = arr[[i, 3 * n + 2]];
            }
            let me = solid::pyramid13_me(&c, rho);
            let mut flat = [0.0f64; 1521];
            for r in 0..39 { for col in 0..39 { flat[r * 39 + col] = me[(r, col)]; } }
            flat
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 1521);
    for f in &results { out.extend_from_slice(f); }
    Array1::from(out).into_pyarray(py)
}

// ============================================================================
// QUAD plane element batch functions
// ============================================================================

/// Batch-compute QUAD4 element stiffness matrices (plane strain).
///
/// coords: shape (n_elem, 8) — each row: 4 nodes × [x, y]
/// Returns flat array (n_elem × 64).
#[pyfunction]
fn batch_ke_quad4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 64]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let c = [
                [arr[[i, 0]], arr[[i, 1]]],
                [arr[[i, 2]], arr[[i, 3]]],
                [arr[[i, 4]], arr[[i, 5]]],
                [arr[[i, 6]], arr[[i, 7]]],
            ];
            Quad4Precomputed::new(&c).compute_ke_global(e, nu)
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 64);
    for r in &results { out.extend_from_slice(r); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute QUAD4 consistent mass matrices.
///
/// coords: shape (n_elem, 8), rho: density
/// Returns flat array (n_elem × 64).
#[pyfunction]
fn batch_me_quad4<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 64]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let c = [
                [arr[[i, 0]], arr[[i, 1]]],
                [arr[[i, 2]], arr[[i, 3]]],
                [arr[[i, 4]], arr[[i, 5]]],
                [arr[[i, 6]], arr[[i, 7]]],
            ];
            Quad4Precomputed::new(&c).compute_me_global(rho)
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 64);
    for r in &results { out.extend_from_slice(r); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute QUAD8 element stiffness matrices (plane strain).
///
/// coords: shape (n_elem, 16) — 8 nodes × [x, y]
/// Returns flat array (n_elem × 256).
#[pyfunction]
fn batch_ke_quad8<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 256]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let c = [
                [arr[[i, 0]], arr[[i, 1]]],
                [arr[[i, 2]], arr[[i, 3]]],
                [arr[[i, 4]], arr[[i, 5]]],
                [arr[[i, 6]], arr[[i, 7]]],
                [arr[[i, 8]], arr[[i, 9]]],
                [arr[[i, 10]], arr[[i, 11]]],
                [arr[[i, 12]], arr[[i, 13]]],
                [arr[[i, 14]], arr[[i, 15]]],
            ];
            Quad8Precomputed::new(&c).compute_ke_global(e, nu)
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 256);
    for r in &results { out.extend_from_slice(r); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute QUAD8 consistent mass matrices.
///
/// coords: shape (n_elem, 16), rho: density
/// Returns flat array (n_elem × 256).
#[pyfunction]
fn batch_me_quad8<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 256]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let c = [
                [arr[[i, 0]], arr[[i, 1]]],
                [arr[[i, 2]], arr[[i, 3]]],
                [arr[[i, 4]], arr[[i, 5]]],
                [arr[[i, 6]], arr[[i, 7]]],
                [arr[[i, 8]], arr[[i, 9]]],
                [arr[[i, 10]], arr[[i, 11]]],
                [arr[[i, 12]], arr[[i, 13]]],
                [arr[[i, 14]], arr[[i, 15]]],
            ];
            Quad8Precomputed::new(&c).compute_me_global(rho)
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 256);
    for r in &results { out.extend_from_slice(r); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute QUAD9 element stiffness matrices (plane strain).
///
/// coords: shape (n_elem, 18) — 9 nodes × [x, y]
/// Returns flat array (n_elem × 324).
#[pyfunction]
fn batch_ke_quad9<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    e: f64,
    nu: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let c = [
                [arr[[i, 0]], arr[[i, 1]]],
                [arr[[i, 2]], arr[[i, 3]]],
                [arr[[i, 4]], arr[[i, 5]]],
                [arr[[i, 6]], arr[[i, 7]]],
                [arr[[i, 8]], arr[[i, 9]]],
                [arr[[i, 10]], arr[[i, 11]]],
                [arr[[i, 12]], arr[[i, 13]]],
                [arr[[i, 14]], arr[[i, 15]]],
                [arr[[i, 16]], arr[[i, 17]]],
            ];
            Quad9Precomputed::new(&c).compute_ke_global(e, nu)
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 324);
    for r in &results { out.extend_from_slice(r); }
    Array1::from(out).into_pyarray(py)
}

/// Batch-compute QUAD9 consistent mass matrices.
///
/// coords: shape (n_elem, 18), rho: density
/// Returns flat array (n_elem × 324).
#[pyfunction]
fn batch_me_quad9<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rho: f64,
) -> Bound<'py, PyArray1<f64>> {
    let arr = coords.as_array();
    let n_elem = arr.nrows();
    let results: Vec<[f64; 324]> = (0..n_elem)
        .into_par_iter()
        .map(|i| {
            let c = [
                [arr[[i, 0]], arr[[i, 1]]],
                [arr[[i, 2]], arr[[i, 3]]],
                [arr[[i, 4]], arr[[i, 5]]],
                [arr[[i, 6]], arr[[i, 7]]],
                [arr[[i, 8]], arr[[i, 9]]],
                [arr[[i, 10]], arr[[i, 11]]],
                [arr[[i, 12]], arr[[i, 13]]],
                [arr[[i, 14]], arr[[i, 15]]],
                [arr[[i, 16]], arr[[i, 17]]],
            ];
            Quad9Precomputed::new(&c).compute_me_global(rho)
        })
        .collect();
    let mut out = Vec::with_capacity(n_elem * 324);
    for r in &results { out.extend_from_slice(r); }
    Array1::from(out).into_pyarray(py)
}

// ============================================================================
// Python-exposed material types: OrthotropicMaterial, Ply, Laminate
// ============================================================================

// ============================================================================
// PyMeshModel — wraps aeroelast_mesh::MeshModel
// ============================================================================

/// Python wrapper for `MeshModel` (aeroelast-mesh crate).
///
/// Holds the full mesh topology (nodes, elements, sets) loaded from HDF5.
/// Use `PyMeshModel.from_hdf5(path)` to load, then pass to
/// `PyMeshAssembler.from_mesh_model(mesh, properties)`.
#[pyclass(name = "MeshModel")]
struct PyMeshModel {
    inner: aeroelast_mesh::model::MeshModel,
}

#[pymethods]
impl PyMeshModel {
    /// Load a MeshModel from an HDF5 file written by the Python writer.
    #[staticmethod]
    fn from_hdf5(path: &str) -> PyResult<Self> {
        let inner = aeroelast_mesh::io::load_hdf5(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyMeshModel { inner })
    }

    /// Number of nodes in the mesh.
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of elements in the mesh.
    #[getter]
    fn element_count(&self) -> usize {
        self.inner.element_count()
    }

    /// Node names of sets as a list of strings.
    fn node_set_names(&self) -> Vec<String> {
        self.inner.node_sets.keys().cloned().collect()
    }

    /// Element set names as a list of strings.
    fn element_set_names(&self) -> Vec<String> {
        self.inner.element_sets.keys().cloned().collect()
    }

    /// Node coordinates as a flat (n_nodes × 3) numpy array (row-major).
    fn node_coords_flat<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.node_count();
        let flat = self.inner.node_coords_flat().to_vec();
        Array2::from_shape_vec((n, 3), flat)
            .expect("node_coords_flat shape mismatch")
            .into_pyarray(py)
    }

    /// Node IDs in index order (matches row order of node_coords_flat).
    fn node_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let ids: Vec<u64> = self.inner.nodes.iter().map(|n| n.id).collect();
        Array1::from(ids).into_pyarray(py)
    }

    /// Element IDs in index order.
    fn element_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let ids: Vec<u64> = self.inner.elements.iter().map(|e| e.id).collect();
        Array1::from(ids).into_pyarray(py)
    }

    /// Returns a list of node-id lists, one per element, in element order.
    fn element_node_ids(&self) -> Vec<Vec<u64>> {
        self.inner.elements.iter().map(|e| e.node_ids.clone()).collect()
    }

    /// Node IDs for elements in a named element set. Returns flat list.
    fn element_set_node_ids(&self, set_name: &str) -> PyResult<Vec<u64>> {
        let set = self.inner.element_sets.get(set_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("element set '{set_name}' not found"))
        })?;
        Ok(set.element_ids.clone())
    }

    /// Node IDs for a named node set.
    fn node_set_node_ids(&self, set_name: &str) -> PyResult<Vec<u64>> {
        let set = self.inner.node_sets.get(set_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("node set '{set_name}' not found"))
        })?;
        Ok(set.node_ids.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshModel(nodes={}, elements={}, node_sets={}, element_sets={})",
            self.inner.node_count(),
            self.inner.element_count(),
            self.inner.node_sets.len(),
            self.inner.element_sets.len(),
        )
    }

    /// Construct a MeshModel from raw Python data.
    ///
    /// Parameters
    /// ----------
    /// node_ids : list[int]
    ///     Node IDs in index order.
    /// node_coords_flat : list[float]
    ///     Flat (n_nodes × 3) coordinate array — row-major.
    /// element_ids : list[int]
    ///     Element IDs in index order.
    /// element_node_ids : list[list[int]]
    ///     Per-element node-ID lists (variable length).
    /// element_type_codes : list[int]
    ///     Assembler type code per element (3, 4, 33, 44, 208, …).
    /// element_sets : dict[str, list[int]]
    ///     Named element sets — values are lists of element IDs.
    /// node_sets : dict[str, list[int]]
    ///     Named node sets — values are lists of node IDs.
    #[staticmethod]
    #[pyo3(signature = (node_ids, node_coords_flat, element_ids, element_node_ids, element_type_codes, element_sets, node_sets))]
    fn from_raw_data(
        node_ids: Vec<u64>,
        node_coords_flat: Vec<f64>,
        element_ids: Vec<u64>,
        element_node_ids: Vec<Vec<u64>>,
        element_type_codes: Vec<i32>,
        element_sets: std::collections::HashMap<String, Vec<u64>>,
        node_sets: std::collections::HashMap<String, Vec<u64>>,
    ) -> PyResult<Self> {
        use aeroelast_mesh::entities::{Element, ElementSet, ElementType, Node, NodeSet};
        use aeroelast_mesh::model::MeshModel;

        let n_nodes = node_ids.len();
        if node_coords_flat.len() != n_nodes * 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node_coords_flat length {} != node_ids.len() * 3 = {}",
                node_coords_flat.len(), n_nodes * 3
            )));
        }

        let mut model = MeshModel::new();

        // Build nodes
        for (i, &nid) in node_ids.iter().enumerate() {
            let x = node_coords_flat[i * 3];
            let y = node_coords_flat[i * 3 + 1];
            let z = node_coords_flat[i * 3 + 2];
            model.add_node(Node::new(nid, x, y, z))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        // Build elements
        let elem_type_from_code = |code: i32, n_nodes_e: usize| -> ElementType {
            match code {
                3  => ElementType::Triangle3,
                33 => ElementType::CompTri3,
                4  => ElementType::Quad4,
                44 => ElementType::CompQuad4,
                104 => ElementType::Quad4,
                108 => ElementType::Quad8,
                109 => ElementType::Quad9,
                208 => ElementType::Hexa8,
                220 => ElementType::Hexa20,
                304 => ElementType::Tetra4,
                310 => ElementType::Tetra10,
                306 => ElementType::Wedge6,
                315 => ElementType::Wedge15,
                305 => ElementType::Pyramid5,
                313 => ElementType::Pyramid13,
                _ => if n_nodes_e == 3 { ElementType::Triangle3 } else { ElementType::Quad4 },
            }
        };

        for (i, &eid) in element_ids.iter().enumerate() {
            let node_ids_elem = element_node_ids[i].clone();
            let code = element_type_codes.get(i).copied().unwrap_or(0);
            let etype = elem_type_from_code(code, node_ids_elem.len());
            model.add_element(Element::new(eid, node_ids_elem, etype))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        // Build element sets
        for (set_name, eids) in element_sets {
            model.element_sets.insert(set_name.clone(), ElementSet::with_ids(set_name, eids));
        }

        // Build node sets
        for (set_name, nids) in node_sets {
            model.node_sets.insert(set_name.clone(), NodeSet::with_ids(set_name, nids));
        }

        Ok(PyMeshModel { inner: model })
    }
}

// ============================================================================
// PyOrthotropicMaterial, PyPly, PyLaminate
// ============================================================================

// ============================================================================
// ElementFamily
// ============================================================================

/// Element family classification exposed to Python.
///
/// Mirrors `aeroelast.elements.ElementFamily` (IntEnum).
/// Values are kept identical for drop-in compatibility:
///   SHELL = 2, PLANE = 3, SOLID = 4
#[pyclass(name = "ElementFamily", eq, eq_int, hash, frozen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PyElementFamily {
    #[pyo3(name = "SHELL")]
    Shell = 2,
    #[pyo3(name = "PLANE")]
    Plane = 3,
    #[pyo3(name = "SOLID")]
    Solid = 4,
}

/// Python wrapper for `OrthotropicMaterial`.
///
/// Mirrors the Python `OrthotropicMaterial` dataclass from `material.py`.
#[pyclass(name = "OrthotropicMaterial")]
#[derive(Clone)]
struct PyOrthotropicMaterial {
    inner: OrthotropicMaterial,
}

#[pymethods]
impl PyOrthotropicMaterial {
    #[new]
    #[pyo3(signature = (e1, e2, e3, g12, g23, g13, nu12, nu23, nu31, rho))]
    fn new(
        e1: f64, e2: f64, e3: f64,
        g12: f64, g23: f64, g13: f64,
        nu12: f64, nu23: f64, nu31: f64,
        rho: f64,
    ) -> Self {
        Self {
            inner: OrthotropicMaterial::new(e1, e2, e3, g12, g23, g13, nu12, nu23, nu31, rho),
        }
    }

    #[getter] fn e1(&self) -> f64 { self.inner.e1 }
    #[getter] fn e2(&self) -> f64 { self.inner.e2 }
    #[getter] fn e3(&self) -> f64 { self.inner.e3 }
    #[getter] fn g12(&self) -> f64 { self.inner.g12 }
    #[getter] fn g23(&self) -> f64 { self.inner.g23 }
    #[getter] fn g13(&self) -> f64 { self.inner.g13 }
    #[getter] fn nu12(&self) -> f64 { self.inner.nu12 }
    #[getter] fn nu23(&self) -> f64 { self.inner.nu23 }
    #[getter] fn nu31(&self) -> f64 { self.inner.nu31 }
    #[getter] fn rho(&self) -> f64 { self.inner.rho }

    fn __repr__(&self) -> String {
        format!(
            "OrthotropicMaterial(E1={:.3e}, E2={:.3e}, G12={:.3e}, nu12={:.3}, rho={:.1})",
            self.inner.e1, self.inner.e2, self.inner.g12, self.inner.nu12, self.inner.rho
        )
    }
}

/// Python wrapper for a single `Ply`.
#[pyclass(name = "Ply")]
#[derive(Clone)]
struct PyPly {
    inner: Ply,
}

#[pymethods]
impl PyPly {
    #[new]
    #[pyo3(signature = (material, thickness, angle))]
    fn new(material: &PyOrthotropicMaterial, thickness: f64, angle: f64) -> Self {
        Self { inner: Ply::new(material.inner, thickness, angle) }
    }

    #[getter] fn thickness(&self) -> f64 { self.inner.thickness }
    #[getter] fn angle(&self) -> f64 { self.inner.angle }
    #[getter] fn z_bottom(&self) -> f64 { self.inner.z_bottom }
    #[getter] fn z_top(&self) -> f64 { self.inner.z_top }

    fn __repr__(&self) -> String {
        format!("Ply(angle={:.1}°, t={:.4e}m)", self.inner.angle, self.inner.thickness)
    }
}

/// Python wrapper for `Laminate` (CLT — Classical Lamination Theory).
///
/// Computes A, B, D and Cs matrices from the ply stack.
#[pyclass(name = "Laminate")]
struct PyLaminate {
    inner: Laminate,
}

#[pymethods]
impl PyLaminate {
    #[new]
    #[pyo3(signature = (plies, shear_correction_factor=0.75))]
    fn new(plies: Vec<PyPly>, shear_correction_factor: f64) -> PyResult<Self> {
        let rust_plies: Vec<Ply> = plies.into_iter().map(|p| p.inner).collect();
        let lam = Laminate::new(rust_plies, shear_correction_factor)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner: lam })
    }

    /// 3×3 extensional stiffness matrix A [N/m] — row-major flat array (9 values).
    fn a_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut arr = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                arr[[i, j]] = self.inner.a[(i, j)];
            }
        }
        arr.into_pyarray(py)
    }

    /// 3×3 coupling stiffness matrix B [N] — row-major flat array (9 values).
    fn b_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut arr = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                arr[[i, j]] = self.inner.b[(i, j)];
            }
        }
        arr.into_pyarray(py)
    }

    /// 3×3 bending stiffness matrix D [N·m] — row-major flat array (9 values).
    fn d_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut arr = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                arr[[i, j]] = self.inner.d[(i, j)];
            }
        }
        arr.into_pyarray(py)
    }

    /// 2×2 transverse shear stiffness matrix Cs [N/m].
    fn cs_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut arr = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                arr[[i, j]] = self.inner.cs[(i, j)];
            }
        }
        arr.into_pyarray(py)
    }

    /// Full 6×6 ABD matrix [[A, B], [B, D]] as a numpy array.
    fn abd_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat = self.inner.abd_matrix_flat();
        let mut arr = Array2::<f64>::zeros((6, 6));
        for i in 0..6 {
            for j in 0..6 {
                arr[[i, j]] = flat[i * 6 + j];
            }
        }
        arr.into_pyarray(py)
    }

    #[getter] fn total_thickness(&self) -> f64 { self.inner.total_thickness }
    #[getter] fn n_plies(&self) -> usize { self.inner.plies.len() }
    #[getter] fn shear_correction_factor(&self) -> f64 { self.inner.shear_correction_factor }
    #[getter] fn is_symmetric(&self) -> bool { self.inner.is_symmetric() }
    #[getter] fn is_balanced(&self) -> bool { self.inner.is_balanced() }

    fn __repr__(&self) -> String {
        let angles: Vec<String> = self.inner.plies.iter()
            .map(|p| format!("{:.0}", p.angle))
            .collect();
        format!(
            "Laminate([{}], h={:.4}mm)",
            angles.join("/"),
            self.inner.total_thickness * 1000.0
        )
    }
}

/// Register all aeroelast functions into a PyModule.
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
    m.add_function(wrap_pyfunction!(batch_ke_hexa8, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_hexa8, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_tetra4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_tetra4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_wedge6, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_wedge6, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_tetra10, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_tetra10, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_wedge15, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_wedge15, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_hexa20, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_hexa20, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_pyramid5, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_pyramid5, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_pyramid13, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_pyramid13, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_quad4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_quad4, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_quad8, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_quad8, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ke_quad9, m)?)?;
    m.add_function(wrap_pyfunction!(batch_me_quad9, m)?)?;
    m.add_function(wrap_pyfunction!(coo_assembly, m)?)?;
    m.add_function(wrap_pyfunction!(compute_nnz, m)?)?;
    m.add_function(wrap_pyfunction!(petsc_assemble_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(petsc_modal_solve, m)?)?;
    m.add_function(wrap_pyfunction!(modal_solve_coo, m)?)?;
    m.add_class::<PyMeshModel>()?;
    m.add_class::<PyMeshAssembler>()?;
    m.add_class::<PyOrthotropicMaterial>()?;
    m.add_class::<PyPly>()?;
    m.add_class::<PyLaminate>()?;
    m.add_class::<PyElementFamily>()?;
    Ok(())
}

// ============================================================================
// Module entry point (named `aeroelast`)
// ============================================================================

#[pymodule]
fn _aeroelast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(m)
}
