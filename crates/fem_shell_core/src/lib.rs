use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod assembly;
pub mod elements;
pub mod materials;
pub mod solver;

use elements::mitc3::{self, Mitc3Precomputed};
use elements::mitc4::{self, Mitc4Precomputed};
use materials::composite::composite_constitutive;
use materials::isotropic::IsotropicMaterial;
use materials::Material;

// ============================================================================
// PyO3 exposed functions
// ============================================================================

/// Batch-compute MITC3+ element stiffness matrices (global coords).
///
/// Parameters
/// ----------
/// coords : ndarray (n_elem, 9)
///     Node coordinates [x1,y1,z1, x2,y2,z2, x3,y3,z3] per element.
/// e_mod : float
///     Young's modulus.
/// nu : float
///     Poisson's ratio.
/// thickness : float
///     Shell thickness.
/// shear_correction : float
///     Shear correction factor (default 5/6).
///
/// Returns
/// -------
/// ndarray (n_elem, 18, 18)
///     Element stiffness matrices in global coordinates.
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

    // Precompute material (shared across all elements)
    let mat = IsotropicMaterial::new(e_mod, nu, 0.0);
    let constitutive = mat.constitutive(thickness, shear_correction);

    // Parallel element computation
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

    // Flatten into output array
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

/// Generate COO triplets for sparse assembly from element matrices.

// ============================================================================
// Composite batch functions — per-element ABD matrices from CLT
// ============================================================================

/// Batch-compute MITC3+ composite element stiffness matrices (global coords).
///
/// Parameters
/// ----------
/// coords : ndarray (n_elem, 9)
///     Node coordinates per element.
/// cm_flat : ndarray (n_elem, 9)
///     A matrix (membrane, 3×3 row-major) per element.
/// cb_flat : ndarray (n_elem, 9)
///     D matrix (bending, 3×3 row-major) per element.
/// cs_flat : ndarray (n_elem, 4)
///     Cs matrix (shear, 2×2 row-major) per element.
/// thickness : ndarray (n_elem,)
///     Total laminate thickness per element.
/// e_equiv : ndarray (n_elem,)
///     Equivalent modulus per element (for drilling stiffness).
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
///
/// Parameters
/// ----------
/// coords : ndarray (n_elem, 9)
///     Node coordinates per element.
/// mass_per_area : ndarray (n_elem,)
///     Ply-integrated mass per unit area [kg/m²] per element.
/// rotational_inertia : ndarray (n_elem,)
///     Ply-integrated rotational inertia per unit area [kg·m] per element.
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

            // We need a minimal precomputed for geometry only (area + T matrix)
            // Use dummy constitutive — mass doesn't use it
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

/// Generate COO triplets for sparse assembly from element matrices (original).
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
// Modal solver
// ============================================================================

/// Solve the generalized eigenvalue problem K·φ = λ·M·φ from element matrices.
///
/// Assembles element K and M directly into dense free-DOF matrices, solves
/// via Cholesky transformation + symmetric eigendecomposition (nalgebra).
///
/// Parameters
/// ----------
/// ke_flat : ndarray (n_elem * ndof * ndof,)
///     Flattened element stiffness matrices (row-major per element).
/// me_flat : ndarray (n_elem * ndof * ndof,)
///     Flattened element mass matrices.
/// dofs : ndarray (n_elem, ndof)
///     Element DOF connectivity (global DOF indices).
/// n_total_dofs : int
///     Total DOFs in the system.
/// free_dofs : ndarray (n_free,)
///     Indices of unconstrained DOFs.
/// num_modes : int
///     Number of lowest modes to extract.
///
/// Returns
/// -------
/// (frequencies, mode_shapes)
///     frequencies : ndarray (n_modes,) — natural frequencies in Hz.
///     mode_shapes : ndarray (n_modes * n_total_dofs,) — flat column data,
///         reshape in Python via `.reshape((n_modes, n_total_dofs)).T`.
#[pyfunction]
fn modal_solve<'py>(
    py: Python<'py>,
    ke_flat: PyReadonlyArray1<'py, f64>,
    me_flat: PyReadonlyArray1<'py, f64>,
    dofs: PyReadonlyArray2<'py, i64>,
    n_total_dofs: usize,
    free_dofs: PyReadonlyArray1<'py, i64>,
    num_modes: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let ke = ke_flat.as_slice()?;
    let me = me_flat.as_slice()?;
    let dofs_arr = dofs.as_array();
    let ndof = dofs_arr.ncols();
    let dofs_flat: Vec<i64> = dofs_arr.iter().copied().collect();
    let free: Vec<i64> = free_dofs.as_slice()?.to_vec();

    let (freq, modes) = solver::modal::modal_solve_elements(
        ke, me, &dofs_flat, ndof, n_total_dofs, &free, num_modes,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok((
        Array1::from(freq).into_pyarray(py),
        Array1::from(modes).into_pyarray(py),
    ))
}

/// Solve the generalized eigenvalue problem from COO sparse triplets.
///
/// Accepts pre-assembled COO data (e.g. from `coo_assembly`). Supports
/// mixed meshes where element types have different DOF counts.
#[pyfunction]
fn modal_solve_coo<'py>(
    py: Python<'py>,
    k_rows: PyReadonlyArray1<'py, i64>,
    k_cols: PyReadonlyArray1<'py, i64>,
    k_vals: PyReadonlyArray1<'py, f64>,
    m_rows: PyReadonlyArray1<'py, i64>,
    m_cols: PyReadonlyArray1<'py, i64>,
    m_vals: PyReadonlyArray1<'py, f64>,
    n_total_dofs: usize,
    free_dofs: PyReadonlyArray1<'py, i64>,
    num_modes: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let kr = k_rows.as_slice()?;
    let kc = k_cols.as_slice()?;
    let kv = k_vals.as_slice()?;
    let mr = m_rows.as_slice()?;
    let mc = m_cols.as_slice()?;
    let mv = m_vals.as_slice()?;
    let free: Vec<i64> = free_dofs.as_slice()?.to_vec();

    let (freq, modes) = solver::modal::modal_solve_coo(
        kr, kc, kv, mr, mc, mv, n_total_dofs, &free, num_modes,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok((
        Array1::from(freq).into_pyarray(py),
        Array1::from(modes).into_pyarray(py),
    ))
}

// ============================================================================
// Module registration
// ============================================================================

#[pymodule]
fn fem_shell_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(modal_solve, m)?)?;
    m.add_function(wrap_pyfunction!(modal_solve_coo, m)?)?;
    Ok(())
}
