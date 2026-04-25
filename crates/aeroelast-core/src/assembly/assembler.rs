/// MeshAssembler: global stiffness/mass/force assembly for mixed element meshes.
///
/// All outputs are COO triplets (rows, cols, vals) — caller passes to PETSc or
/// any sparse solver. No PETSc dependency here.

use nalgebra::Vector3;

use crate::assembly::topology::{ElemType, MeshTopology};
use crate::elements::mitc3::{self, Mitc3Precomputed};
use crate::elements::mitc4::{self, Mitc4Precomputed};
use crate::elements::quad::{Quad4Precomputed, Quad8Precomputed, Quad9Precomputed};
use crate::elements::solid;
use crate::materials::{
    composite::composite_constitutive, isotropic::IsotropicMaterial, Material,
};

// ============================================================================
// MaterialSpec: per-element material descriptor
// ============================================================================

/// Material specification for a single element (any family).
#[derive(Clone)]
pub enum MaterialSpec {
    /// Isotropic linear elastic shell material.
    Isotropic {
        /// Young's modulus (Pa)
        e: f64,
        /// Poisson's ratio
        nu: f64,
        /// Density (kg/m³)
        rho: f64,
        /// Shell thickness (m)
        thickness: f64,
        /// Shear correction factor (typically 5/6)
        shear_correction: f64,
        /// Scaling factor for MITC drilling stabilization term.
        /// 1.0 keeps the baseline 0.15 coefficient used by MITC3/MITC4.
        drilling_scale: f64,
    },
    /// Composite laminate shell material (ABD matrices from CLT).
    Composite {
        /// A matrix (3×3 membrane stiffness), row-major [N/m]
        cm: [f64; 9],
        /// B matrix (3×3 coupling stiffness), row-major [N]
        /// Set to zeros for symmetric laminates
        cb_coupling: [f64; 9],
        /// D matrix (3×3 bending stiffness), row-major [N·m]
        cb: [f64; 9],
        /// Cs matrix (2×2 shear stiffness), row-major [N/m]
        cs: [f64; 4],
        /// Total laminate thickness (m)
        thickness: f64,
        /// Equivalent E for drilling DOF stabilization
        e_equiv: f64,
        /// Mass per unit area (kg/m²)
        mass_per_area: f64,
        /// Rotational inertia per unit area (kg·m²/m²)
        rotational_inertia: f64,
    },
    /// Plane-stress isotropic material (for QUAD4/8/9 elements).
    PlaneStress {
        /// Young's modulus (Pa)
        e: f64,
        /// Poisson's ratio
        nu: f64,
        /// Density (kg/m³)
        rho: f64,
        /// Element thickness (m) — used for mass scaling
        thickness: f64,
    },
    /// 3D solid isotropic material (for HEXA/TETRA/WEDGE/PYRAMID elements).
    Solid3D {
        /// Young's modulus (Pa)
        e: f64,
        /// Poisson's ratio
        nu: f64,
        /// Density (kg/m³)
        rho: f64,
    },
}

// ============================================================================
// PrecomputedElem: per-element precomputed geometric/constitutive data
// ============================================================================

enum PrecomputedElem {
    Tri(Mitc3Precomputed),
    Quad(Mitc4Precomputed),
    // QUAD plane elements: store 2D coords
    Plane4([[f64; 2]; 4]),
    Plane8([[f64; 2]; 8]),
    Plane9([[f64; 2]; 9]),
    // 3D solid elements: store 3D coords
    Hexa8([[f64; 3]; 8]),
    Hexa20([[f64; 3]; 20]),
    Tetra4([[f64; 3]; 4]),
    Tetra10([[f64; 3]; 10]),
    Wedge6([[f64; 3]; 6]),
    Wedge15([[f64; 3]; 15]),
    Pyramid5([[f64; 3]; 5]),
    Pyramid13([[f64; 3]; 13]),
}

// ============================================================================
// MeshAssembler
// ============================================================================

/// Global assembler for mixed-element meshes (shell, plane, solid).
///
/// Precomputes all element-level data on construction; assembly methods are
/// pure read-only passes that accumulate COO triplets.
pub struct MeshAssembler {
    /// Mesh topology (nodes, connectivity, element types)
    pub topology: MeshTopology,
    /// Material spec per element
    pub materials: Vec<MaterialSpec>,
    /// Total DOFs = n_nodes × dofs_per_node
    pub dofs_count: usize,
    /// Per-element DOF index lists (length = n_dof_per_elem per element)
    dof_connectivity: Vec<Vec<usize>>,
    /// Per-element precomputed geometric + constitutive data
    precomputed: Vec<PrecomputedElem>,
    /// NNZ per row for PETSc preallocation — computed lazily on first access
    nnz_per_row_data: std::sync::OnceLock<Vec<i64>>,
}

impl MeshAssembler {
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// Create a new `MeshAssembler`.
    ///
    /// # Arguments
    /// * `topology` - Mesh topology (takes ownership)
    /// * `materials` - One `MaterialSpec` per element
    pub fn new(topology: MeshTopology, materials: Vec<MaterialSpec>) -> Self {
        assert_eq!(
            topology.n_elems,
            materials.len(),
            "one MaterialSpec required per element"
        );

        let dofs_count = topology.dofs_count();
        let n_elems = topology.n_elems;

        // Build per-element precomputed data and DOF connectivity
        let mut precomputed = Vec::with_capacity(n_elems);
        let mut dof_connectivity = Vec::with_capacity(n_elems);

        for e in 0..n_elems {
            let dofs = topology.global_dof_indices(e);
            dof_connectivity.push(dofs);

            let coords = topology.elem_coords(e);
            let mat = &materials[e];

            let pre = match topology.elem_types[e] {
                ElemType::Mitc3 | ElemType::Mitc3Composite => {
                    assert_eq!(coords.len(), 9, "MITC3 needs 3 nodes × 3 coords");
                    let mut c9 = [0.0f64; 9];
                    c9.copy_from_slice(&coords);
                    let (constitutive, thickness, e_mod, drilling_scale) = build_constitutive_mitc3(mat);
                    PrecomputedElem::Tri(Mitc3Precomputed::new(
                        &c9,
                        constitutive,
                        thickness,
                        e_mod,
                        drilling_scale,
                    ))
                }
                ElemType::Mitc4 | ElemType::Mitc4Composite => {
                    assert_eq!(coords.len(), 12, "MITC4 needs 4 nodes × 3 coords");
                    let mut c12 = [0.0f64; 12];
                    c12.copy_from_slice(&coords);
                    let (constitutive, thickness, e_mod, drilling_scale) = build_constitutive_mitc4(mat);
                    PrecomputedElem::Quad(Mitc4Precomputed::new(
                        &c12,
                        constitutive,
                        thickness,
                        e_mod,
                        drilling_scale,
                    ))
                }
                ElemType::Quad4 => {
                    assert_eq!(coords.len(), 12, "QUAD4 needs 4 nodes × 3 coords");
                    let c = nodes3d_to_2d_4(&coords);
                    PrecomputedElem::Plane4(c)
                }
                ElemType::Quad8 => {
                    assert_eq!(coords.len(), 24, "QUAD8 needs 8 nodes × 3 coords");
                    let c = nodes3d_to_2d_8(&coords);
                    PrecomputedElem::Plane8(c)
                }
                ElemType::Quad9 => {
                    assert_eq!(coords.len(), 27, "QUAD9 needs 9 nodes × 3 coords");
                    let c = nodes3d_to_2d_9(&coords);
                    PrecomputedElem::Plane9(c)
                }
                ElemType::Hexa8 => {
                    assert_eq!(coords.len(), 24, "HEXA8 needs 8 nodes × 3 coords");
                    let c = coords_to_3d_8(&coords);
                    PrecomputedElem::Hexa8(c)
                }
                ElemType::Hexa20 => {
                    assert_eq!(coords.len(), 60, "HEXA20 needs 20 nodes × 3 coords");
                    let c = coords_to_3d_20(&coords);
                    PrecomputedElem::Hexa20(c)
                }
                ElemType::Tetra4 => {
                    assert_eq!(coords.len(), 12, "TETRA4 needs 4 nodes × 3 coords");
                    let c = coords_to_3d_4(&coords);
                    PrecomputedElem::Tetra4(c)
                }
                ElemType::Tetra10 => {
                    assert_eq!(coords.len(), 30, "TETRA10 needs 10 nodes × 3 coords");
                    let c = coords_to_3d_10(&coords);
                    PrecomputedElem::Tetra10(c)
                }
                ElemType::Wedge6 => {
                    assert_eq!(coords.len(), 18, "WEDGE6 needs 6 nodes × 3 coords");
                    let c = coords_to_3d_6(&coords);
                    PrecomputedElem::Wedge6(c)
                }
                ElemType::Wedge15 => {
                    assert_eq!(coords.len(), 45, "WEDGE15 needs 15 nodes × 3 coords");
                    let c = coords_to_3d_15(&coords);
                    PrecomputedElem::Wedge15(c)
                }
                ElemType::Pyramid5 => {
                    assert_eq!(coords.len(), 15, "PYRAMID5 needs 5 nodes × 3 coords");
                    let c = coords_to_3d_5(&coords);
                    PrecomputedElem::Pyramid5(c)
                }
                ElemType::Pyramid13 => {
                    assert_eq!(coords.len(), 39, "PYRAMID13 needs 13 nodes × 3 coords");
                    let c = coords_to_3d_13(&coords);
                    PrecomputedElem::Pyramid13(c)
                }
            };
            precomputed.push(pre);
        }

        // Compute NNZ per row for PETSc preallocation.
        //
        // Use Rayon parallel reduction: each thread builds a local
        // Vec<HashSet<usize>> for a chunk of elements, then we merge.
        // This is O(E × d²) total work but with ~N_cpus parallelism.
        // REMOVED from constructor — computed lazily via nnz_per_row().

        MeshAssembler {
            topology,
            materials,
            dofs_count,
            dof_connectivity,
            precomputed,
            nnz_per_row_data: std::sync::OnceLock::new(),
        }
    }

    // -----------------------------------------------------------------------
    // NNZ accessor
    // -----------------------------------------------------------------------

    /// NNZ count per row (for PETSc preallocation).
    /// Computed lazily on first call — O(E × d²) with HashSet dedup.
    pub fn nnz_per_row(&self) -> &[i64] {
        self.nnz_per_row_data.get_or_init(|| {
            let dofs_count = self.dofs_count;
            let dofs_per_elem = self.dof_connectivity.first().map_or(0, |d| d.len());
            let mut nnz: Vec<std::collections::HashSet<usize>> =
                (0..dofs_count).map(|_| std::collections::HashSet::with_capacity(dofs_per_elem)).collect();
            for elem_dofs in &self.dof_connectivity {
                for &row_dof in elem_dofs {
                    for &col_dof in elem_dofs {
                        nnz[row_dof].insert(col_dof);
                    }
                }
            }
            nnz.iter().map(|s| s.len() as i64).collect()
        })
    }

    // -----------------------------------------------------------------------
    // assemble_k: global elastic stiffness matrix as COO
    // -----------------------------------------------------------------------

    /// Assemble the global elastic stiffness matrix K.
    ///
    /// Returns `(rows, cols, vals)` COO triplets. Duplicate (i,j) entries are
    /// summed by the sparse assembler (PETSc `MatSetValues` with `ADD_VALUES`).
    pub fn assemble_k(&self) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for e in 0..self.topology.n_elems {
            let ke_flat: Vec<f64> = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let ke = mitc3::compute_ke_global(pre);
                    ke.as_slice().to_vec()
                }
                PrecomputedElem::Quad(pre) => {
                    let ke = mitc4::compute_ke_global(pre);
                    ke.as_slice().to_vec()
                }
                PrecomputedElem::Plane4(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    Quad4Precomputed::new(c).compute_ke_global(e_mod, nu).to_vec()
                }
                PrecomputedElem::Plane8(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    Quad8Precomputed::new(c).compute_ke_global(e_mod, nu).to_vec()
                }
                PrecomputedElem::Plane9(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    Quad9Precomputed::new(c).compute_ke_global(e_mod, nu).to_vec()
                }
                PrecomputedElem::Hexa8(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::hexa8_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Hexa20(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::hexa20_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Tetra4(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::tetra4_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Tetra10(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::tetra10_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Wedge6(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::wedge6_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Wedge15(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::wedge15_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Pyramid5(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::pyramid5_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Pyramid13(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::pyramid13_ke(c, e_mod, nu).as_slice().to_vec()
                }
            };

            scatter_elem_matrix(&self.dof_connectivity[e], &ke_flat, &mut rows, &mut cols, &mut vals);
        }

        (rows, cols, vals)
    }

    // -----------------------------------------------------------------------
    // assemble_m: global consistent mass matrix as COO
    // -----------------------------------------------------------------------

    /// Assemble the global consistent mass matrix M.
    ///
    /// Returns `(rows, cols, vals)` COO triplets.
    pub fn assemble_m(&self) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for e in 0..self.topology.n_elems {
            let me_flat: Vec<f64> = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let me = match &self.materials[e] {
                        MaterialSpec::Isotropic { rho, .. } => {
                            mitc3::compute_me_global(pre, *rho)
                        }
                        MaterialSpec::Composite { mass_per_area, rotational_inertia, .. } => {
                            mitc3::compute_me_composite_global(pre, *mass_per_area, *rotational_inertia)
                        }
                        _ => panic!("MITC3 element requires Isotropic or Composite material"),
                    };
                    me.as_slice().to_vec()
                }
                PrecomputedElem::Quad(pre) => {
                    let me = match &self.materials[e] {
                        MaterialSpec::Isotropic { rho, .. } => {
                            mitc4::compute_me_global(pre, *rho)
                        }
                        MaterialSpec::Composite { mass_per_area, rotational_inertia, .. } => {
                            mitc4::compute_me_composite_global(pre, *mass_per_area, *rotational_inertia)
                        }
                        _ => panic!("MITC4 element requires Isotropic or Composite material"),
                    };
                    me.as_slice().to_vec()
                }
                PrecomputedElem::Plane4(c) => {
                    let (rho, thickness) = plane_stress_rho_h(&self.materials[e]);
                    let _ = thickness;
                    Quad4Precomputed::new(c).compute_me_global(rho).to_vec()
                }
                PrecomputedElem::Plane8(c) => {
                    let (rho, _thickness) = plane_stress_rho_h(&self.materials[e]);
                    Quad8Precomputed::new(c).compute_me_global(rho).to_vec()
                }
                PrecomputedElem::Plane9(c) => {
                    let (rho, _thickness) = plane_stress_rho_h(&self.materials[e]);
                    Quad9Precomputed::new(c).compute_me_global(rho).to_vec()
                }
                PrecomputedElem::Hexa8(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::hexa8_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Hexa20(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::hexa20_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Tetra4(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::tetra4_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Tetra10(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::tetra10_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Wedge6(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::wedge6_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Wedge15(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::wedge15_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Pyramid5(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::pyramid5_me(c, rho).as_slice().to_vec()
                }
                PrecomputedElem::Pyramid13(c) => {
                    let rho = solid3d_rho(&self.materials[e]);
                    solid::pyramid13_me(c, rho).as_slice().to_vec()
                }
            };

            scatter_elem_matrix(&self.dof_connectivity[e], &me_flat, &mut rows, &mut cols, &mut vals);
        }

        (rows, cols, vals)
    }

    // -----------------------------------------------------------------------
    // assemble_f_body: global body load vector
    // -----------------------------------------------------------------------

    /// Assemble the global body load vector.
    ///
    /// # Arguments
    /// * `gravity` - Body acceleration vector [gx, gy, gz] (m/s²)
    ///
    /// Returns `Vec<f64>` of length `dofs_count`.
    pub fn assemble_f_body(&self, gravity: [f64; 3]) -> Vec<f64> {
        let g = Vector3::new(gravity[0], gravity[1], gravity[2]);
        let mut f = vec![0.0f64; self.dofs_count];

        for e in 0..self.topology.n_elems {
            let fe: Vec<f64> = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let rho = material_rho(&self.materials[e]);
                    let fvec = mitc3::compute_body_load_global(pre, rho, &g);
                    fvec.as_slice().to_vec()
                }
                PrecomputedElem::Quad(pre) => {
                    let rho = material_rho(&self.materials[e]);
                    let fvec = mitc4::compute_body_load_global(pre, rho, &g);
                    fvec.as_slice().to_vec()
                }
                // For plane/solid elements body load is not implemented yet;
                // return zeros of appropriate size.
                PrecomputedElem::Plane4(_) => vec![0.0; 8],
                PrecomputedElem::Plane8(_) => vec![0.0; 16],
                PrecomputedElem::Plane9(_) => vec![0.0; 18],
                PrecomputedElem::Hexa8(_) => vec![0.0; 24],
                PrecomputedElem::Hexa20(_) => vec![0.0; 60],
                PrecomputedElem::Tetra4(_) => vec![0.0; 12],
                PrecomputedElem::Tetra10(_) => vec![0.0; 30],
                PrecomputedElem::Wedge6(_) => vec![0.0; 18],
                PrecomputedElem::Wedge15(_) => vec![0.0; 45],
                PrecomputedElem::Pyramid5(_) => vec![0.0; 15],
                PrecomputedElem::Pyramid13(_) => vec![0.0; 39],
            };

            let dofs = &self.dof_connectivity[e];
            for (local_i, &global_i) in dofs.iter().enumerate() {
                f[global_i] += fe[local_i];
            }
        }

        f
    }

    // -----------------------------------------------------------------------
    // assemble_geometric_k: global geometric stiffness as COO
    // -----------------------------------------------------------------------

    /// Assemble the global geometric stiffness matrix K_σ.
    ///
    /// # Arguments
    /// * `sigma` - Membrane stress per element: `sigma[e] = [σxx, σyy, σxy]`
    ///   in the element's local coordinate system.
    ///   For plane/solid elements this is a no-op (returns zero).
    ///
    /// Returns `(rows, cols, vals)` COO triplets.
    pub fn assemble_geometric_k(&self, sigma: &[[f64; 3]]) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        assert_eq!(
            sigma.len(),
            self.topology.n_elems,
            "one stress tensor required per element"
        );

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for e in 0..self.topology.n_elems {
            let sv = Vector3::new(sigma[e][0], sigma[e][1], sigma[e][2]);

            let kg_flat: Vec<f64> = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let kg = mitc3::compute_k_sigma_global(pre, &sv);
                    kg.as_slice().to_vec()
                }
                PrecomputedElem::Quad(pre) => {
                    let kg = mitc4::compute_k_sigma_global(pre, &sv);
                    kg.as_slice().to_vec()
                }
                // Geometric stiffness not implemented for plane/solid in this assembler
                PrecomputedElem::Plane4(_) => vec![0.0; 64],
                PrecomputedElem::Plane8(_) => vec![0.0; 256],
                PrecomputedElem::Plane9(_) => vec![0.0; 324],
                PrecomputedElem::Hexa8(_) => vec![0.0; 576],
                PrecomputedElem::Hexa20(_) => vec![0.0; 3600],
                PrecomputedElem::Tetra4(_) => vec![0.0; 144],
                PrecomputedElem::Tetra10(_) => vec![0.0; 900],
                PrecomputedElem::Wedge6(_) => vec![0.0; 324],
                PrecomputedElem::Wedge15(_) => vec![0.0; 2025],
                PrecomputedElem::Pyramid5(_) => vec![0.0; 225],
                PrecomputedElem::Pyramid13(_) => vec![0.0; 1521],
            };

            scatter_elem_matrix(&self.dof_connectivity[e], &kg_flat, &mut rows, &mut cols, &mut vals);
        }

        (rows, cols, vals)
    }

    // -----------------------------------------------------------------------
    // assemble_kt: nonlinear tangent stiffness as COO
    // -----------------------------------------------------------------------

    /// Assemble the nonlinear tangent stiffness matrix K_T.
    ///
    /// # Arguments
    /// * `u` - Global displacement vector of length `dofs_count`
    ///
    /// Returns `(rows, cols, vals)` COO triplets.
    pub fn assemble_kt(&self, u: &[f64]) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        assert_eq!(u.len(), self.dofs_count, "displacement vector length mismatch");

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for e in 0..self.topology.n_elems {
            let dofs = &self.dof_connectivity[e];
            let kt_flat: Vec<f64> = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let ue = extract_elem_disp_18(u, dofs);
                    let kt = mitc3::compute_kt_global(pre, &ue);
                    kt.as_slice().to_vec()
                }
                PrecomputedElem::Quad(pre) => {
                    let ue = extract_elem_disp_24(u, dofs);
                    let kt = mitc4::compute_kt_global(pre, &ue);
                    kt.as_slice().to_vec()
                }
                // For plane/solid elements: K_T = K_e (linear only)
                PrecomputedElem::Plane4(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    Quad4Precomputed::new(c).compute_ke_global(e_mod, nu).to_vec()
                }
                PrecomputedElem::Plane8(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    Quad8Precomputed::new(c).compute_ke_global(e_mod, nu).to_vec()
                }
                PrecomputedElem::Plane9(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    Quad9Precomputed::new(c).compute_ke_global(e_mod, nu).to_vec()
                }
                PrecomputedElem::Hexa8(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::hexa8_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Hexa20(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::hexa20_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Tetra4(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::tetra4_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Tetra10(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::tetra10_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Wedge6(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::wedge6_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Wedge15(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::wedge15_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Pyramid5(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::pyramid5_ke(c, e_mod, nu).as_slice().to_vec()
                }
                PrecomputedElem::Pyramid13(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    solid::pyramid13_ke(c, e_mod, nu).as_slice().to_vec()
                }
            };

            scatter_elem_matrix(dofs, &kt_flat, &mut rows, &mut cols, &mut vals);
        }

        (rows, cols, vals)
    }

    // -----------------------------------------------------------------------
    // assemble_fint: internal force vector
    // -----------------------------------------------------------------------

    /// Assemble the global internal force vector f_int.
    ///
    /// # Arguments
    /// * `u` - Global displacement vector of length `dofs_count`
    /// * `nonlinear` - If true, use Green-Lagrange (total-Lagrangian) formulation
    ///
    /// Returns `Vec<f64>` of length `dofs_count`.
    pub fn assemble_fint(&self, u: &[f64], nonlinear: bool) -> Vec<f64> {
        assert_eq!(u.len(), self.dofs_count, "displacement vector length mismatch");

        let mut f = vec![0.0f64; self.dofs_count];

        for e in 0..self.topology.n_elems {
            let dofs = &self.dof_connectivity[e];
            let fe: Vec<f64> = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let ue = extract_elem_disp_18(u, dofs);
                    let fvec = mitc3::compute_fint_global(pre, &ue, nonlinear);
                    fvec.as_slice().to_vec()
                }
                PrecomputedElem::Quad(pre) => {
                    let ue = extract_elem_disp_24(u, dofs);
                    let fvec = mitc4::compute_fint_global(pre, &ue, nonlinear);
                    fvec.as_slice().to_vec()
                }
                // For plane/solid: f_int = K · u_e (linear only)
                PrecomputedElem::Plane4(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    let ke = Quad4Precomputed::new(c).compute_ke_global(e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(&ke, &ue)
                }
                PrecomputedElem::Plane8(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    let ke = Quad8Precomputed::new(c).compute_ke_global(e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(&ke, &ue)
                }
                PrecomputedElem::Plane9(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    let ke = Quad9Precomputed::new(c).compute_ke_global(e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(&ke, &ue)
                }
                PrecomputedElem::Hexa8(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::hexa8_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Hexa20(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::hexa20_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Tetra4(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::tetra4_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Tetra10(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::tetra10_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Wedge6(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::wedge6_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Wedge15(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::wedge15_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Pyramid5(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::pyramid5_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
                PrecomputedElem::Pyramid13(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ke = solid::pyramid13_ke(c, e_mod, nu);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    ke_times_ue(ke.as_slice(), &ue)
                }
            };

            for (local_i, &global_i) in dofs.iter().enumerate() {
                f[global_i] += fe[local_i];
            }
        }

        f
    }

    // -----------------------------------------------------------------------
    // assemble_centrifugal_k: geometric stiffness from centrifugal prestress
    // -----------------------------------------------------------------------

    /// Assemble the geometric stiffness matrix K_σ from centrifugal loading.
    ///
    /// Computes the centrifugal prestress for every shell element from first
    /// principles (centroid, radial distance, area, local axes), then calls
    /// `assemble_geometric_k` with the resulting per-element σ array.
    ///
    /// Only MITC3/MITC4 shell elements contribute; plane and solid elements
    /// are skipped (σ = 0).
    ///
    /// # Arguments
    /// * `omega`         - Angular velocity magnitude (rad/s)
    /// * `rotation_axis` - Unit vector of the rotation axis `[ax, ay, az]`
    /// * `rotation_center` - A point on the rotation axis `[cx, cy, cz]` (m)
    /// * `rho_per_elem`  - Density (or mass-per-area for composite) per element
    ///
    /// # Returns
    /// `(rows, cols, vals)` COO triplets for K_σ.
    pub fn assemble_centrifugal_k(
        &self,
        omega: f64,
        rotation_axis: [f64; 3],
        rotation_center: [f64; 3],
        rho_per_elem: &[f64],
    ) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        assert_eq!(
            rho_per_elem.len(),
            self.topology.n_elems,
            "rho_per_elem length must equal n_elems"
        );

        let axis = {
            let a = Vector3::new(rotation_axis[0], rotation_axis[1], rotation_axis[2]);
            let n = a.norm();
            if n > 1e-30 { a / n } else { Vector3::new(0.0, 0.0, 1.0) }
        };
        let center = Vector3::new(rotation_center[0], rotation_center[1], rotation_center[2]);

        let n_elems = self.topology.n_elems;
        let mut sigma: Vec<[f64; 3]> = vec![[0.0; 3]; n_elems];

        for e in 0..n_elems {
            // Only shell elements get centrifugal prestress
            let coords_flat = self.topology.elem_coords(e);
            let n_nodes = coords_flat.len() / 3;

            let is_shell = matches!(
                self.precomputed[e],
                PrecomputedElem::Tri(_) | PrecomputedElem::Quad(_)
            );
            if !is_shell {
                continue;
            }

            // Centroid
            let mut cx = 0.0f64;
            let mut cy = 0.0f64;
            let mut cz = 0.0f64;
            for n in 0..n_nodes {
                cx += coords_flat[3 * n];
                cy += coords_flat[3 * n + 1];
                cz += coords_flat[3 * n + 2];
            }
            let inv_n = 1.0 / n_nodes as f64;
            let centroid = Vector3::new(cx * inv_n, cy * inv_n, cz * inv_n);

            // Radial distance from rotation axis
            let r_vec = centroid - center;
            let r_parallel = r_vec.dot(&axis) * axis;
            let r_radial_vec = r_vec - r_parallel;
            let r_radial = r_radial_vec.norm();

            if r_radial < 1e-10 {
                continue; // on rotation axis — no centrifugal stress
            }
            let radial_dir = r_radial_vec / r_radial;

            // Element area (via cross product of diagonals)
            let area = if n_nodes >= 4 {
                // Quad: diagonals p2-p0 and p3-p1
                let p0 = Vector3::new(coords_flat[0], coords_flat[1], coords_flat[2]);
                let p2 = Vector3::new(coords_flat[6], coords_flat[7], coords_flat[8]);
                let p3 = Vector3::new(coords_flat[9], coords_flat[10], coords_flat[11]);
                let p1 = Vector3::new(coords_flat[3], coords_flat[4], coords_flat[5]);
                let v1 = p2 - p0;
                let v2 = p3 - p1;
                0.5 * v1.cross(&v2).norm()
            } else {
                // Tri: cross product of two edges
                let p0 = Vector3::new(coords_flat[0], coords_flat[1], coords_flat[2]);
                let p1 = Vector3::new(coords_flat[3], coords_flat[4], coords_flat[5]);
                let p2 = Vector3::new(coords_flat[6], coords_flat[7], coords_flat[8]);
                0.5 * (p1 - p0).cross(&(p2 - p0)).norm()
            };
            let l_char = area.max(1e-20).sqrt();

            // Local axes (e1, e2, e3) — same logic as Python _shell_local_axes
            let p0 = Vector3::new(coords_flat[0], coords_flat[1], coords_flat[2]);
            let p1 = Vector3::new(coords_flat[3], coords_flat[4], coords_flat[5]);
            let p2 = Vector3::new(coords_flat[6], coords_flat[7], coords_flat[8]);
            let v1 = p1 - p0;
            let v2 = p2 - p0;
            let n1 = v1.cross(&v2);

            let e3 = if n_nodes >= 4 {
                let p3 = Vector3::new(coords_flat[9], coords_flat[10], coords_flat[11]);
                let v1b = p2 - p0;
                let v2b = p3 - p0;
                let n2 = v1b.cross(&v2b);
                let n1n = n1.norm();
                let n2n = n2.norm();
                let avg = match (n1n > 1e-12, n2n > 1e-12) {
                    (true, true)  => n1 / n1n + n2 / n2n,
                    (true, false) => n1 / n1n,
                    (false, true) => n2 / n2n,
                    _             => Vector3::new(0.0, 0.0, 1.0),
                };
                let an = avg.norm();
                if an > 1e-12 { avg / an } else { Vector3::new(0.0, 0.0, 1.0) }
            } else {
                let n1n = n1.norm();
                if n1n > 1e-12 { n1 / n1n } else { Vector3::new(0.0, 0.0, 1.0) }
            };

            let e1_raw = v1 - v1.dot(&e3) * e3;
            let e1_raw = if e1_raw.norm() < 1e-12 {
                let fb = v2 - v2.dot(&e3) * e3;
                fb
            } else {
                e1_raw
            };
            let e1n = e1_raw.norm();
            let e1 = if e1n > 1e-12 { e1_raw / e1n } else { Vector3::new(1.0, 0.0, 0.0) };
            let e2_raw = e3.cross(&e1);
            let e2 = {
                let n = e2_raw.norm();
                if n > 1e-12 { e2_raw / n } else { Vector3::new(0.0, 1.0, 0.0) }
            };

            // Centrifugal stress magnitude
            let rho = rho_per_elem[e];
            let sigma_cf = rho * omega * omega * r_radial * l_char;

            // Project radial direction onto local x-y plane
            let cos_theta = radial_dir.dot(&e1);
            let sin_theta = radial_dir.dot(&e2);

            sigma[e][0] = sigma_cf * cos_theta * cos_theta;       // σ_xx
            sigma[e][1] = sigma_cf * sin_theta * sin_theta;       // σ_yy
            sigma[e][2] = sigma_cf * cos_theta * sin_theta;       // σ_xy
        }

        self.assemble_geometric_k(&sigma)
    }

    // -----------------------------------------------------------------------
    // compute_stress_field: element-centroid stress and strain recovery
    // -----------------------------------------------------------------------

    /// Compute the centroid stress and strain for every element in the mesh.
    ///
    /// This method is the Rust backbone of the post-processing pipeline.  For
    /// each element it evaluates the constitutive relationship C·B·u at the
    /// element centroid (natural coordinates (0,0) for quads / shells,
    /// (1/3,1/3) for triangles, and the equivalent parametric centroid for
    /// solid elements) and returns the Voigt 6-component stress and strain
    /// vectors.
    ///
    /// # Arguments
    /// * `u`           - Global displacement vector of length `dofs_count`.
    /// * `z_factor`    - Non-dimensional through-thickness location for shell
    ///                   elements: `z = z_factor × h`.  Typical values:
    ///                   `−0.5` (bottom), `0.0` (mid-surface), `+0.5` (top).
    ///                   Ignored for plane and solid elements.
    /// * `stress_type` - Shell stress contribution flag (ignored for non-shell
    ///                   elements):
    ///                   `0` = membrane only, `1` = bending only,
    ///                   `2` (or any other value) = total (membrane + bending).
    ///
    /// # Returns
    /// A tuple `(sigma, epsilon)` where each is a `Vec` of length `n_elems`.
    /// Every entry is a `[f64; 6]` Voigt vector:
    /// `[σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx]` (stress)
    /// `[ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]` (engineering strain).
    ///
    /// For shell and plane elements the out-of-plane components (indices 2, 4,
    /// 5) are zero (plane-stress assumption).
    pub fn compute_stress_field(
        &self,
        u: &[f64],
        z_factor: f64,
        stress_type: u8,
    ) -> (Vec<[f64; 6]>, Vec<[f64; 6]>) {
        assert_eq!(u.len(), self.dofs_count, "displacement vector length mismatch");

        let n = self.topology.n_elems;
        let mut sigma_out = vec![[0.0f64; 6]; n];
        let mut eps_out = vec![[0.0f64; 6]; n];

        for e in 0..n {
            let dofs = &self.dof_connectivity[e];
            let (sig6, eps6): ([f64; 6], [f64; 6]) = match &self.precomputed[e] {
                PrecomputedElem::Tri(pre) => {
                    let ue = extract_elem_disp_18(u, dofs);
                    mitc3::compute_element_stress(pre, &ue, z_factor, stress_type)
                }
                PrecomputedElem::Quad(pre) => {
                    let ue = extract_elem_disp_24(u, dofs);
                    mitc4::compute_element_stress(pre, &ue, z_factor, stress_type)
                }
                // Plane (2-D) elements: recover via K·u is not stress — use B·u directly.
                // We delegate to the solid-style centroid stress for quad plane elements
                // by extracting in-plane (x,y) DOFs only.
                PrecomputedElem::Plane4(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    Quad4Precomputed::new(c).compute_centroid_stress(e_mod, nu, &ue)
                }
                PrecomputedElem::Plane8(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    Quad8Precomputed::new(c).compute_centroid_stress(e_mod, nu, &ue)
                }
                PrecomputedElem::Plane9(c) => {
                    let (e_mod, nu) = plane_stress_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    Quad9Precomputed::new(c).compute_centroid_stress(e_mod, nu, &ue)
                }
                PrecomputedElem::Hexa8(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::hexa8_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Hexa20(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::hexa20_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Tetra4(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::tetra4_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Tetra10(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::tetra10_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Wedge6(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::wedge6_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Wedge15(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::wedge15_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Pyramid5(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::pyramid5_centroid_stress(c, e_mod, nu, &ue)
                }
                PrecomputedElem::Pyramid13(c) => {
                    let (e_mod, nu) = solid3d_en(&self.materials[e]);
                    let ue: Vec<f64> = dofs.iter().map(|&d| u[d]).collect();
                    solid::pyramid13_centroid_stress(c, e_mod, nu, &ue)
                }
            };
            sigma_out[e] = sig6;
            eps_out[e] = eps6;
        }

        (sigma_out, eps_out)
    }
}

// ============================================================================
// Helper: build constitutive data from MaterialSpec
// ============================================================================

fn build_constitutive_mitc3(
    mat: &MaterialSpec,
) -> (crate::materials::ShellConstitutive, f64, f64, f64) {
    match mat {
        MaterialSpec::Isotropic {
            e,
            nu,
            rho,
            thickness,
            shear_correction,
            drilling_scale,
        } => {
            let iso = IsotropicMaterial::new(*e, *nu, *rho);
            let constitutive = iso.constitutive(*thickness, *shear_correction);
            (constitutive, *thickness, *e, *drilling_scale)
        }
        MaterialSpec::Composite { cm, cb_coupling, cb, cs, thickness, e_equiv, .. } => {
            let constitutive = composite_constitutive(cm, cb_coupling, cb, cs, *thickness);
            (constitutive, *thickness, *e_equiv, 1.0)
        }
        _ => panic!("Shell element requires Isotropic or Composite MaterialSpec"),
    }
}

fn build_constitutive_mitc4(
    mat: &MaterialSpec,
) -> (crate::materials::ShellConstitutive, f64, f64, f64) {
    build_constitutive_mitc3(mat) // same signature
}

fn material_rho(mat: &MaterialSpec) -> f64 {
    match mat {
        MaterialSpec::Isotropic { rho, .. } => *rho,
        MaterialSpec::Composite { mass_per_area, .. } => *mass_per_area,
        MaterialSpec::PlaneStress { rho, .. } => *rho,
        MaterialSpec::Solid3D { rho, .. } => *rho,
    }
}

fn plane_stress_en(mat: &MaterialSpec) -> (f64, f64) {
    match mat {
        MaterialSpec::PlaneStress { e, nu, .. } => (*e, *nu),
        MaterialSpec::Isotropic { e, nu, .. } => (*e, *nu),
        _ => panic!("QUAD element requires PlaneStress or Isotropic MaterialSpec"),
    }
}

fn plane_stress_rho_h(mat: &MaterialSpec) -> (f64, f64) {
    match mat {
        MaterialSpec::PlaneStress { rho, thickness, .. } => (*rho, *thickness),
        MaterialSpec::Isotropic { rho, thickness, .. } => (*rho, *thickness),
        _ => panic!("QUAD element requires PlaneStress or Isotropic MaterialSpec"),
    }
}

fn solid3d_en(mat: &MaterialSpec) -> (f64, f64) {
    match mat {
        MaterialSpec::Solid3D { e, nu, .. } => (*e, *nu),
        MaterialSpec::Isotropic { e, nu, .. } => (*e, *nu),
        _ => panic!("Solid element requires Solid3D or Isotropic MaterialSpec"),
    }
}

fn solid3d_rho(mat: &MaterialSpec) -> f64 {
    match mat {
        MaterialSpec::Solid3D { rho, .. } => *rho,
        MaterialSpec::Isotropic { rho, .. } => *rho,
        _ => panic!("Solid element requires Solid3D or Isotropic MaterialSpec"),
    }
}

// ============================================================================
// Coord extraction helpers
// ============================================================================

fn nodes3d_to_2d_4(coords: &[f64]) -> [[f64; 2]; 4] {
    let mut c = [[0.0f64; 2]; 4];
    for n in 0..4 { c[n][0] = coords[3*n]; c[n][1] = coords[3*n+1]; }
    c
}
fn nodes3d_to_2d_8(coords: &[f64]) -> [[f64; 2]; 8] {
    let mut c = [[0.0f64; 2]; 8];
    for n in 0..8 { c[n][0] = coords[3*n]; c[n][1] = coords[3*n+1]; }
    c
}
fn nodes3d_to_2d_9(coords: &[f64]) -> [[f64; 2]; 9] {
    let mut c = [[0.0f64; 2]; 9];
    for n in 0..9 { c[n][0] = coords[3*n]; c[n][1] = coords[3*n+1]; }
    c
}
fn coords_to_3d_4(coords: &[f64]) -> [[f64; 3]; 4] {
    let mut c = [[0.0f64; 3]; 4];
    for n in 0..4 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_5(coords: &[f64]) -> [[f64; 3]; 5] {
    let mut c = [[0.0f64; 3]; 5];
    for n in 0..5 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_6(coords: &[f64]) -> [[f64; 3]; 6] {
    let mut c = [[0.0f64; 3]; 6];
    for n in 0..6 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_8(coords: &[f64]) -> [[f64; 3]; 8] {
    let mut c = [[0.0f64; 3]; 8];
    for n in 0..8 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_10(coords: &[f64]) -> [[f64; 3]; 10] {
    let mut c = [[0.0f64; 3]; 10];
    for n in 0..10 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_13(coords: &[f64]) -> [[f64; 3]; 13] {
    let mut c = [[0.0f64; 3]; 13];
    for n in 0..13 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_15(coords: &[f64]) -> [[f64; 3]; 15] {
    let mut c = [[0.0f64; 3]; 15];
    for n in 0..15 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}
fn coords_to_3d_20(coords: &[f64]) -> [[f64; 3]; 20] {
    let mut c = [[0.0f64; 3]; 20];
    for n in 0..20 { c[n].copy_from_slice(&coords[3*n..3*n+3]); }
    c
}

// ============================================================================
// Scatter helpers
// ============================================================================

/// Scatter element matrix entries into COO triplet vectors.
fn scatter_elem_matrix(
    dofs: &[usize],
    ke_flat: &[f64],
    rows: &mut Vec<i64>,
    cols: &mut Vec<i64>,
    vals: &mut Vec<f64>,
) {
    let n = dofs.len();
    for i in 0..n {
        for j in 0..n {
            rows.push(dofs[i] as i64);
            cols.push(dofs[j] as i64);
            vals.push(ke_flat[i * n + j]);
        }
    }
}

/// Extract 18-DOF element displacement from global vector.
fn extract_elem_disp_18(u: &[f64], dofs: &[usize]) -> mitc3::Vec18 {
    let mut ue = mitc3::Vec18::zeros();
    for (local, &global) in dofs.iter().enumerate() {
        ue[local] = u[global];
    }
    ue
}

/// Extract 24-DOF element displacement from global vector.
fn extract_elem_disp_24(u: &[f64], dofs: &[usize]) -> mitc4::Vec24 {
    let mut ue = mitc4::Vec24::zeros();
    for (local, &global) in dofs.iter().enumerate() {
        ue[local] = u[global];
    }
    ue
}

/// Compute K_e * u_e for linear elements (f_int = K·u).
fn ke_times_ue(ke_flat: &[f64], ue: &[f64]) -> Vec<f64> {
    let n = ue.len();
    let mut fe = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            fe[i] += ke_flat[i * n + j] * ue[j];
        }
    }
    fe
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::topology::MeshTopology;

    /// Build a 2-element MITC3 patch.
    /// Nodes: (0,0,0), (1,0,0), (0,1,0), (1,1,0)
    /// Elems: [0,1,2], [1,3,2]
    fn two_tri_assembler() -> MeshAssembler {
        let node_coords = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
        ];
        let connectivity = vec![vec![0usize, 1, 2], vec![1, 3, 2]];
        let elem_types = vec![ElemType::Mitc3, ElemType::Mitc3];
        let topology = MeshTopology::new(node_coords, connectivity, elem_types);

        let mat = MaterialSpec::Isotropic {
            e: 2.0e11,
            nu: 0.3,
            rho: 7800.0,
            thickness: 0.01,
            shear_correction: 5.0 / 6.0,
            drilling_scale: 1.0,
        };
        let materials = vec![mat.clone(), mat];
        MeshAssembler::new(topology, materials)
    }

    #[test]
    fn test_assembler_new_no_panic() {
        let asm = two_tri_assembler();
        assert_eq!(asm.dofs_count, 4 * 6);
        assert_eq!(asm.topology.n_elems, 2);
    }

    #[test]
    fn test_assemble_k_finite() {
        let asm = two_tri_assembler();
        let (rows, cols, vals) = asm.assemble_k();
        assert!(!vals.is_empty(), "K should have entries");
        for &v in &vals {
            assert!(v.is_finite(), "K entry is not finite: {v}");
        }
        // rows and cols should be in valid range
        for &r in &rows {
            assert!((r as usize) < asm.dofs_count, "row index out of range");
        }
        for &c in &cols {
            assert!((c as usize) < asm.dofs_count, "col index out of range");
        }
    }

    #[test]
    fn test_assemble_m_finite() {
        let asm = two_tri_assembler();
        let (rows, cols, vals) = asm.assemble_m();
        assert!(!vals.is_empty(), "M should have entries");
        for &v in &vals {
            assert!(v.is_finite(), "M entry is not finite: {v}");
        }
        // Mass diagonal entries should be non-negative
        // (diagonal check: find entries where row==col)
        let diag_sum: f64 = rows
            .iter()
            .zip(cols.iter())
            .zip(vals.iter())
            .filter(|((r, c), _)| r == c)
            .map(|(_, &v)| v)
            .sum();
        assert!(diag_sum > 0.0, "diagonal mass entries should be positive; got {diag_sum}");
    }

    #[test]
    fn test_assemble_f_body_finite() {
        let asm = two_tri_assembler();
        let f = asm.assemble_f_body([0.0, 0.0, -9.81]);
        assert_eq!(f.len(), asm.dofs_count);
        for &v in &f {
            assert!(v.is_finite(), "body force entry is not finite: {v}");
        }
        // Total z-force: ρ·h·|g|·total_area (2 triangles of area 0.5 each = 1.0)
        let total_fz: f64 = (0..4).map(|i| f[6 * i + 2]).sum();
        let expected = 7800.0 * 0.01 * (-9.81) * 1.0;
        assert!(
            (total_fz - expected).abs() < 1e-4,
            "total fz: {total_fz} expected {expected}"
        );
    }

    #[test]
    fn test_assemble_geometric_k_zero_stress() {
        let asm = two_tri_assembler();
        let sigma = vec![[0.0f64; 3]; 2];
        let (_r, _c, vals) = asm.assemble_geometric_k(&sigma);
        let norm: f64 = vals.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "zero stress → zero geometric K");
    }

    #[test]
    fn test_assemble_geometric_k_finite() {
        let asm = two_tri_assembler();
        let sigma = vec![[1.0e6f64, 0.5e6, 0.2e6]; 2];
        let (_, _, vals) = asm.assemble_geometric_k(&sigma);
        for &v in &vals {
            assert!(v.is_finite(), "geometric K entry is not finite");
        }
    }

    #[test]
    fn test_assemble_kt_linear_matches_k_at_zero() {
        let asm = two_tri_assembler();
        let u = vec![0.0f64; asm.dofs_count];

        // At zero displacement, K_T should equal K_e
        let (_, _, vals_k) = asm.assemble_k();
        let (_, _, vals_kt) = asm.assemble_kt(&u);
        assert_eq!(vals_k.len(), vals_kt.len(), "K and KT should have same entries at u=0");
        for (v1, v2) in vals_k.iter().zip(vals_kt.iter()) {
            assert!((v1 - v2).abs() < 1e-6 * v1.abs().max(1.0), "K and KT differ at zero disp");
        }
    }

    #[test]
    fn test_assemble_fint_zero_at_zero_displacement() {
        let asm = two_tri_assembler();
        let u = vec![0.0f64; asm.dofs_count];
        let f = asm.assemble_fint(&u, false);
        let norm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "f_int should be zero at zero displacement");
    }
}
