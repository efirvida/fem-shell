from typing import Dict, Iterable, Optional

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.mesh import MeshModel
from fem_shell.core.properties import ShellPropertyType
from fem_shell.elements import ElementFactory, ElementFamily, FemElement


class MeshAssembler:
    def __init__(self, mesh: MeshModel, model: Dict):
        """
        Finite Element assembler using PETSc for distributed sparse matrices.

        Parameters
        ----------
        mesh : MeshModel
            The computational mesh containing nodes and elements
        model : Dict
            Material and element configuration dictionary

        Attributes
        ----------
        dofs_count : int
            Total number of degrees of freedom in the system
        _dofs_array : np.ndarray
            Element-to-DOF connectivity array
        _ke_array : np.ndarray
            Precomputed local stiffness matrices
        _me_array : np.ndarray
            Precomputed local mass matrices
        """
        self.mesh = mesh
        self.model = model["elements"]
        self.comm = MPI.COMM_WORLD
        self._element_map: Dict[int, FemElement] = {}
        self._dofs_array: np.ndarray = None
        self._node_dofs_map: Dict[int, Iterable] = {}
        self._ke_array: np.ndarray = None
        self._me_array: np.ndarray = None
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self._row_nnz: Optional[np.ndarray] = None
        self._precompute_elements()
        self._compute_sparsity_pattern()
        self._prepare_rust_batch_data()
        self._rust_batch_overwrite_ke_me()
        self._build_py_mesh_assembler()

    # Mapping from ElementFamily to (dofs_per_node, spatial_dim).
    # This avoids instantiating every element twice just to query these constants.
    _FAMILY_PROPERTIES = {
        ElementFamily.SHELL: (6, 3),
        ElementFamily.PLANE: (2, 2),
        ElementFamily.SOLID: (3, 3),
    }

    def _precompute_elements(self):
        """Precompute element matrices and DOF connectivity arrays.

        Uses mesh.node_id_to_index mapping to ensure DOF indices are consecutive
        starting from 0, regardless of the original node IDs in the mesh.
        
        CRITICAL FIX FOR MIXED MESHES:
        We must first determine the MAXIMUM DOFs per node required by any element type
        in the model. This defines the "global stride".
        
        If we used variable strides (e.g. node_idx * 3 for solids, node_idx * 6 for shells),
        we would get index collisions (aliasing) where different nodes map to the
        same DOF index.
        """
        elements = self.mesh.elements
        if not elements:
            return

        # --- Determine global maximum stride from element family ---
        # For uniform-family meshes (the common case) this is O(1).
        # For mixed meshes we inspect all families present in the model.
        element_family = self.model.get("element_family")
        if element_family is not None and element_family in self._FAMILY_PROPERTIES:
            # Fast path: single known family
            self.dofs_per_node, self.spatial_dim = self._FAMILY_PROPERTIES[element_family]
        else:
            # Mixed / unknown family: probe with a single element per distinct node count
            max_dofs_per_node = 0
            max_spatial_dim = 0
            seen_node_counts = set()
            probe_model = {k: v for k, v in self.model.items() if k != "properties"}
            for element in elements:
                nc = element.node_count
                if nc in seen_node_counts:
                    continue
                seen_node_counts.add(nc)
                temp_elem = ElementFactory.get_element(mesh_element=element, **probe_model)
                if temp_elem:
                    max_dofs_per_node = max(max_dofs_per_node, temp_elem.dofs_per_node)
                    max_spatial_dim = max(max_spatial_dim, temp_elem.spatial_dimmension)
            self.dofs_per_node = max_dofs_per_node
            self.spatial_dim = max_spatial_dim

        # --- Assemble element data (single pass) ---
        dofs_list = []
        ke_list = []
        me_list = []
        dof_sizes = set()

        node_id_to_index = self.mesh.node_id_to_index
        n_elements = len(elements)
        progress_interval = max(n_elements // 10, 1)

        # Build per-element property lookup from properties map (if provided).
        # Maps element_id -> ShellPropertyType for O(1) access in the loop.
        properties_map: Optional[Dict[str, ShellPropertyType]] = self.model.get("properties")
        element_property_lookup: Dict[int, ShellPropertyType] = {}
        if properties_map is not None:
            for set_name, prop in properties_map.items():
                if set_name in self.mesh.element_sets:
                    for elem in self.mesh.element_sets[set_name].elements:
                        element_property_lookup[elem.id] = prop

        for idx, element in enumerate(elements):
            if idx % progress_interval == 0:
                print(
                    f"\r  Precomputing element matrices... {idx}/{n_elements}"
                    f" ({100 * idx // n_elements}%)",
                    end="",
                    flush=True,
                )

            shell_property = None
            # Base model for the factory: always strip the 'properties' key
            # since it is consumed by the assembler, not the element constructor.
            element_model = {k: v for k, v in self.model.items() if k != "properties"}

            if element.id in element_property_lookup:
                # New path: per-element-set property via ShellPropertyType
                shell_property = element_property_lookup[element.id]
                # Strip legacy keys that conflict with shell_property kwargs
                element_model = {
                    k: v
                    for k, v in element_model.items()
                    if k not in ("material", "thickness", "laminate")
                }
            elif (
                element.thickness is not None
                and element_model.get("element_family") == ElementFamily.SHELL
            ):
                # Deprecated path: per-element thickness override from mesh
                import warnings

                warnings.warn(
                    "Per-element thickness via MeshElement.thickness is deprecated. "
                    "Use a 'properties' dict mapping element-set names to "
                    "ShellProperty / CompositeShellProperty instead.",
                    DeprecationWarning,
                    stacklevel=1,
                )
                element_model = {**element_model, "thickness": element.thickness}

            fem_element = ElementFactory.get_element(
                mesh_element=element, shell_property=shell_property, **element_model
            )
            if not fem_element:
                continue

            # Remap DOFs using the GLOBAL stride to avoid aliasing
            remapped_dof_indices = {}
            for node_id in fem_element.node_ids:
                node_index = node_id_to_index[node_id]
                
                # ALWAYS use the global max stride
                start_dof = node_index * self.dofs_per_node
                
                # The element takes only the DOFs it needs (e.g. 3) from the block of 6
                # This leaves the angular DOFs (3-5) "empty" for solid nodes, which is fine.
                element_dofs_count = fem_element.dofs_per_node
                
                # We simply map to the first N slots of the node's block
                end_dof = start_dof + element_dofs_count
                remapped_dof_indices[node_id] = tuple(range(start_dof, end_dof))

            self._node_dofs_map.update(remapped_dof_indices)

            dofs = np.array(
                [dof for node_id in fem_element.node_ids for dof in remapped_dof_indices[node_id]],
                dtype=np.int64,
            )
            dof_sizes.add(len(dofs))

            self._element_map[element.id] = fem_element
            dofs_list.append(dofs)
            ke_list.append(fem_element.K)
            me_list.append(fem_element.M)

        print(
            f"\r  Precomputing element matrices... {n_elements}/{n_elements} (100%)",
            flush=True,
        )

        # Check if all elements have the same DOF count (uniform mesh)
        self._is_mixed_mesh = len(dof_sizes) > 1

        if self._is_mixed_mesh:
            # Mixed mesh: store as lists (variable-size arrays not supported by numpy)
            self._dofs_list = dofs_list
            self._ke_list = ke_list
            self._me_list = me_list
            self._dofs_array = None
            self._ke_array = None
            self._me_array = None
        else:
            # Uniform mesh: store as numpy arrays for efficiency
            self._dofs_array = np.array(dofs_list, dtype=np.int64)
            self._ke_array = np.array(ke_list, dtype=np.float64)
            self._me_array = np.array(me_list, dtype=np.float64)
            self._dofs_list = None
            self._ke_list = None
            self._me_list = None

        self.dofs_count = self.mesh.node_count * self.dofs_per_node

    def _compute_sparsity_pattern(self):
        """
        Compute the sparse matrix non-zero pattern for efficient preallocation.

        Notes
        -----
        Determines the number of non-zeros per matrix row using element
        connectivity information. Critical for PETSc matrix performance.
        Supports both uniform and mixed-element meshes.
        """
        nnz = [set() for _ in range(self.dofs_count)]

        # Get the appropriate DOF data (list for mixed, array for uniform)
        dofs_data = self._dofs_list if self._is_mixed_mesh else self._dofs_array

        for elem_dofs in dofs_data:
            for dof_i in elem_dofs:
                nnz[dof_i].update(dof_j for dof_j in elem_dofs)

        self._row_nnz = np.array([len(s) for s in nnz], dtype=PETSc.IntType)

    def _prepare_rust_batch_data(self):
        """Prepare element groups for Rust batch computation.

        Groups elements by (element_type, E, nu, thickness, shear_correction)
        for efficient parallel batch processing via the ``fem_shell_core`` Rust
        backend.  Each group maps to a single parallel Rust call that processes
        all elements of the same type and material at once.

        For composite elements (MITC3Composite / MITC4Composite), per-element
        ABD matrices are extracted and passed to dedicated composite batch
        functions.
        """
        self._rust_groups = []
        self._rust_composite_groups = []
        self._has_rust = False
        self._all_elements_rust = False

        try:
            import fem_shell_core

            self._fsc = fem_shell_core
            self._has_rust = True
        except ImportError:
            return

        from fem_shell.elements.MITC3 import MITC3
        from fem_shell.elements.MITC3_composite import MITC3Composite
        from fem_shell.elements.MITC4 import MITC4
        from fem_shell.elements.MITC4_composite import MITC4Composite

        groups: dict = {}
        composite_groups: dict = {}  # keyed by ("MITC3_composite",) or ("MITC4_composite",)
        elem_list = list(self._element_map.values())

        for e, fem_elem in enumerate(elem_list):
            # Composite elements (must check before isotropic since Composite inherits from MITC3/4)
            if isinstance(fem_elem, MITC3Composite):
                key = "MITC3_composite"
                if key not in composite_groups:
                    composite_groups[key] = {
                        "indices": [], "coords": [],
                        "cm": [], "cb": [], "cs": [],
                        "thickness": [], "e_equiv": [],
                        "mass_per_area": [], "rotational_inertia": [],
                        "etype": key,
                    }
                g = composite_groups[key]
                g["indices"].append(e)
                g["coords"].append(fem_elem._initial_coords.ravel())
                g["cm"].append(fem_elem._A_matrix.ravel())
                g["cb"].append(fem_elem._D_matrix.ravel())
                g["cs"].append(fem_elem._Cs_matrix.ravel())
                h = fem_elem.thickness
                g["thickness"].append(h)
                # e_equiv so that e_equiv * h² * 0.15 matches Python drilling
                a_trace = np.trace(fem_elem._A_matrix)
                g["e_equiv"].append(a_trace / (3.0 * h) if h > 0 else 0.0)
                g["mass_per_area"].append(fem_elem._mass_per_area())
                g["rotational_inertia"].append(fem_elem._rotational_inertia())
                continue

            if isinstance(fem_elem, MITC4Composite):
                key = "MITC4_composite"
                if key not in composite_groups:
                    composite_groups[key] = {
                        "indices": [], "coords": [],
                        "cm": [], "cb": [], "cs": [],
                        "thickness": [], "e_equiv": [],
                        "mass_per_area": [], "rotational_inertia": [],
                        "etype": key,
                    }
                g = composite_groups[key]
                g["indices"].append(e)
                g["coords"].append(fem_elem._initial_coords.ravel())
                g["cm"].append(fem_elem._A_matrix.ravel())
                g["cb"].append(fem_elem._D_matrix.ravel())
                g["cs"].append(fem_elem._Cs_matrix.ravel())
                h = fem_elem.thickness
                g["thickness"].append(h)
                a_trace = np.trace(fem_elem._A_matrix)
                g["e_equiv"].append(a_trace / (3.0 * h) if h > 0 else 0.0)
                g["mass_per_area"].append(fem_elem._mass_per_area())
                g["rotational_inertia"].append(fem_elem._rotational_inertia())
                continue

            # Isotropic elements
            if isinstance(fem_elem, MITC3):
                etype = "MITC3"
            elif isinstance(fem_elem, MITC4):
                etype = "MITC4"
            else:
                continue

            mat = fem_elem.material
            sc = getattr(fem_elem, "_shear_correction_factor", 5.0 / 6.0)
            key = (etype, mat.E, mat.nu, mat.rho, fem_elem.thickness, sc)

            if key not in groups:
                groups[key] = {
                    "indices": [],
                    "coords": [],
                    "etype": etype,
                    "E": mat.E,
                    "nu": mat.nu,
                    "rho": mat.rho,
                    "thickness": fem_elem.thickness,
                    "shear_correction": sc,
                }
            groups[key]["indices"].append(e)
            groups[key]["coords"].append(fem_elem._initial_coords.ravel())

        # Build isotropic groups
        for g in groups.values():
            indices = np.array(g["indices"], dtype=np.intp)
            ndof = 18 if g["etype"] == "MITC3" else 24

            if self._is_mixed_mesh:
                dofs = np.array(
                    [self._dofs_list[i] for i in indices], dtype=np.int64
                )
            else:
                dofs = self._dofs_array[indices].copy()

            self._rust_groups.append(
                {
                    "etype": g["etype"],
                    "coords": np.array(g["coords"], dtype=np.float64),
                    "dofs": dofs,
                    "ndof": ndof,
                    "E": g["E"],
                    "nu": g["nu"],
                    "rho": g["rho"],
                    "thickness": g["thickness"],
                    "shear_correction": g["shear_correction"],
                    "n_elem": len(indices),
                    "orig_indices": indices.tolist(),
                }
            )

        # Build composite groups
        for g in composite_groups.values():
            indices = np.array(g["indices"], dtype=np.intp)
            ndof = 18 if g["etype"] == "MITC3_composite" else 24

            if self._is_mixed_mesh:
                dofs = np.array(
                    [self._dofs_list[i] for i in indices], dtype=np.int64
                )
            else:
                dofs = self._dofs_array[indices].copy()

            self._rust_composite_groups.append(
                {
                    "etype": g["etype"],
                    "coords": np.array(g["coords"], dtype=np.float64),
                    "dofs": dofs,
                    "ndof": ndof,
                    "cm": np.array(g["cm"], dtype=np.float64),
                    "cb": np.array(g["cb"], dtype=np.float64),
                    "cs": np.array(g["cs"], dtype=np.float64),
                    "thickness": np.array(g["thickness"], dtype=np.float64),
                    "e_equiv": np.array(g["e_equiv"], dtype=np.float64),
                    "mass_per_area": np.array(g["mass_per_area"], dtype=np.float64),
                    "rotational_inertia": np.array(g["rotational_inertia"], dtype=np.float64),
                    "n_elem": len(indices),
                    "orig_indices": indices.tolist(),
                }
            )

        n_rust = sum(g["n_elem"] for g in self._rust_groups)
        n_rust += sum(g["n_elem"] for g in self._rust_composite_groups)
        self._all_elements_rust = n_rust == len(elem_list)

    def _rust_batch_overwrite_ke_me(self):
        """Replace Python-computed ke/me arrays with Rust batch results.

        When the Rust backend covers all elements, this overwrites the per-element
        stiffness and mass matrices that ``_precompute_elements()`` computed in
        Python.  The result is identical (validated by tests) but much faster for
        large meshes.
        """
        if not self._has_rust or not self._all_elements_rust:
            return
        if self._is_mixed_mesh:
            # Mixed mesh: overwrite individual lists
            self._rust_batch_overwrite_lists()
            return

        ndof = self._ke_array.shape[1]
        n_elem = self._ke_array.shape[0]

        # Overwrite with isotropic Rust batch results
        for g in self._rust_groups:
            indices = np.array(
                [i for i, _ in enumerate(range(g["n_elem"]))],
                dtype=np.intp,
            )
            # Find the original element indices from the dofs mapping
            # The group stores the original element index from the element list
            # We need to reconstruct which rows of _ke_array correspond to this group
            pass  # Will be populated below

        # For now: compute Rust ke/me and write directly into the arrays
        # using the element ordering from the groups
        for g in self._rust_groups:
            if g["etype"] == "MITC3":
                ke_flat = self._fsc.batch_ke_mitc3(
                    g["coords"], g["E"], g["nu"], g["thickness"],
                    g["shear_correction"],
                )
                me_flat = self._fsc.batch_me_mitc3(
                    g["coords"], g["E"], g["nu"], g["rho"],
                    g["thickness"], g["shear_correction"],
                )
                elem_ndof = 18
            else:
                ke_flat = self._fsc.batch_ke_mitc4(
                    g["coords"], g["E"], g["nu"], g["thickness"],
                    g["shear_correction"],
                )
                me_flat = self._fsc.batch_me_mitc4(
                    g["coords"], g["E"], g["nu"], g["rho"],
                    g["thickness"], g["shear_correction"],
                )
                elem_ndof = 24

            ke_arr = np.asarray(ke_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)
            me_arr = np.asarray(me_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)

            # Map group-local indices back to the global element array
            # _rust_groups stores original indices from _element_map iteration
            orig_indices = self._rust_group_indices(g)
            for local_i, global_i in enumerate(orig_indices):
                self._ke_array[global_i] = ke_arr[local_i]
                self._me_array[global_i] = me_arr[local_i]

        for g in self._rust_composite_groups:
            if g["etype"] == "MITC3_composite":
                ke_flat = self._fsc.batch_ke_mitc3_composite(
                    g["coords"], g["cm"], g["cb"], g["cs"],
                    g["thickness"], g["e_equiv"],
                )
                me_flat = self._fsc.batch_me_mitc3_composite(
                    g["coords"], g["mass_per_area"], g["rotational_inertia"],
                )
                elem_ndof = 18
            else:
                ke_flat = self._fsc.batch_ke_mitc4_composite(
                    g["coords"], g["cm"], g["cb"], g["cs"],
                    g["thickness"], g["e_equiv"],
                )
                me_flat = self._fsc.batch_me_mitc4_composite(
                    g["coords"], g["mass_per_area"], g["rotational_inertia"],
                )
                elem_ndof = 24

            ke_arr = np.asarray(ke_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)
            me_arr = np.asarray(me_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)

            orig_indices = self._rust_group_indices(g)
            for local_i, global_i in enumerate(orig_indices):
                self._ke_array[global_i] = ke_arr[local_i]
                self._me_array[global_i] = me_arr[local_i]

    def _rust_batch_overwrite_lists(self):
        """Overwrite ke/me lists for mixed meshes with Rust batch results."""
        for g in self._rust_groups:
            if g["etype"] == "MITC3":
                ke_flat = self._fsc.batch_ke_mitc3(
                    g["coords"], g["E"], g["nu"], g["thickness"],
                    g["shear_correction"],
                )
                me_flat = self._fsc.batch_me_mitc3(
                    g["coords"], g["E"], g["nu"], g["rho"],
                    g["thickness"], g["shear_correction"],
                )
                elem_ndof = 18
            else:
                ke_flat = self._fsc.batch_ke_mitc4(
                    g["coords"], g["E"], g["nu"], g["thickness"],
                    g["shear_correction"],
                )
                me_flat = self._fsc.batch_me_mitc4(
                    g["coords"], g["E"], g["nu"], g["rho"],
                    g["thickness"], g["shear_correction"],
                )
                elem_ndof = 24

            ke_arr = np.asarray(ke_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)
            me_arr = np.asarray(me_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)

            orig_indices = self._rust_group_indices(g)
            for local_i, global_i in enumerate(orig_indices):
                self._ke_list[global_i] = ke_arr[local_i]
                self._me_list[global_i] = me_arr[local_i]

        for g in self._rust_composite_groups:
            if g["etype"] == "MITC3_composite":
                ke_flat = self._fsc.batch_ke_mitc3_composite(
                    g["coords"], g["cm"], g["cb"], g["cs"],
                    g["thickness"], g["e_equiv"],
                )
                me_flat = self._fsc.batch_me_mitc3_composite(
                    g["coords"], g["mass_per_area"], g["rotational_inertia"],
                )
                elem_ndof = 18
            else:
                ke_flat = self._fsc.batch_ke_mitc4_composite(
                    g["coords"], g["cm"], g["cb"], g["cs"],
                    g["thickness"], g["e_equiv"],
                )
                me_flat = self._fsc.batch_me_mitc4_composite(
                    g["coords"], g["mass_per_area"], g["rotational_inertia"],
                )
                elem_ndof = 24

            ke_arr = np.asarray(ke_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)
            me_arr = np.asarray(me_flat).reshape(g["n_elem"], elem_ndof, elem_ndof)

            orig_indices = self._rust_group_indices(g)
            for local_i, global_i in enumerate(orig_indices):
                self._ke_list[global_i] = ke_arr[local_i]
                self._me_list[global_i] = me_arr[local_i]

    def _build_py_mesh_assembler(self):
        """Build a PyMeshAssembler (Rust) from the precomputed element data.

        This provides a single unified Rust assembler as an alternative to the
        group-based batch approach. The ``self._rust`` attribute is set when
        construction succeeds; otherwise it is ``None`` and fallback paths are used.
        """
        self._rust = None
        if not self._has_rust:
            return

        try:
            from fem_shell_core import PyMeshAssembler  # noqa: PLC0415
        except ImportError:
            return

        from fem_shell.elements.MITC3 import MITC3  # noqa: PLC0415
        from fem_shell.elements.MITC3_composite import MITC3Composite  # noqa: PLC0415
        from fem_shell.elements.MITC4 import MITC4  # noqa: PLC0415
        from fem_shell.elements.MITC4_composite import MITC4Composite  # noqa: PLC0415
        from fem_shell.elements.QUAD import QUAD4, QUAD8, QUAD9  # noqa: PLC0415
        from fem_shell.elements.SOLID import (  # noqa: PLC0415
            HEXA8,
            HEXA20,
            PYRAMID5,
            PYRAMID13,
            TETRA4,
            TETRA10,
            WEDGE6,
            WEDGE15,
        )

        # Build node_coords array (n_nodes × 3)
        node_id_to_index = self.mesh.node_id_to_index
        n_nodes = self.mesh.node_count
        node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for node in self.mesh.nodes:
            idx = node_id_to_index[node.id]
            node_coords[idx, :] = node.coords[:3]

        # Build connectivity and material lists ordered by element_map
        connectivity = []
        elem_types = []
        materials_list = []

        _QUAD_TYPE_CODE = {QUAD4: 104, QUAD8: 108, QUAD9: 109}
        _SOLID_TYPE_CODE = {
            HEXA8: 208, HEXA20: 220,
            TETRA4: 304, TETRA10: 310,
            WEDGE6: 306, WEDGE15: 315,
            PYRAMID5: 305, PYRAMID13: 313,
        }

        for fem_elem in self._element_map.values():
            # 0-based node indices
            conn = [node_id_to_index[nid] for nid in fem_elem.node_ids]
            connectivity.append(conn)

            # Determine element type code
            elem_cls = type(fem_elem)
            if isinstance(fem_elem, (MITC3, MITC3Composite)):
                code = 33 if isinstance(fem_elem, MITC3Composite) else 3
            elif isinstance(fem_elem, (MITC4, MITC4Composite)):
                code = 44 if isinstance(fem_elem, MITC4Composite) else 4
            elif elem_cls in _QUAD_TYPE_CODE:
                code = _QUAD_TYPE_CODE[elem_cls]
            elif elem_cls in _SOLID_TYPE_CODE:
                code = _SOLID_TYPE_CODE[elem_cls]
            else:
                # Unknown element type — fall back to None assembler
                self._rust = None
                return
            elem_types.append(code)

            # Build material dict
            if isinstance(fem_elem, MITC3Composite) or isinstance(fem_elem, MITC4Composite):
                h = fem_elem.thickness
                a_trace = np.trace(fem_elem._A_matrix)
                e_equiv = a_trace / (3.0 * h) if h > 0 else 0.0
                mat_dict = {
                    "type": "composite",
                    "cm": fem_elem._A_matrix.ravel().tolist(),
                    "cb": fem_elem._D_matrix.ravel().tolist(),
                    "cs": fem_elem._Cs_matrix.ravel().tolist(),
                    "thickness": h,
                    "e_equiv": e_equiv,
                    "mass_per_area": fem_elem._mass_per_area(),
                    "rotational_inertia": fem_elem._rotational_inertia(),
                }
            elif isinstance(fem_elem, (QUAD4, QUAD8, QUAD9)):
                mat = fem_elem.material
                mat_dict = {
                    "type": "plane_stress",
                    "e": mat.E,
                    "nu": mat.nu,
                    "rho": mat.rho,
                    "thickness": getattr(fem_elem, "thickness", 1.0),
                }
            elif isinstance(fem_elem, (HEXA8, HEXA20, TETRA4, TETRA10, WEDGE6, WEDGE15, PYRAMID5, PYRAMID13)):
                mat = fem_elem.material
                mat_dict = {
                    "type": "solid_3d",
                    "e": mat.E,
                    "nu": mat.nu,
                    "rho": mat.rho,
                }
            else:
                mat = fem_elem.material
                sc = getattr(fem_elem, "_shear_correction_factor", 5.0 / 6.0)
                mat_dict = {
                    "type": "isotropic",
                    "e": mat.E,
                    "nu": mat.nu,
                    "rho": mat.rho,
                    "thickness": fem_elem.thickness,
                    "shear_correction": sc,
                }
            materials_list.append(mat_dict)

        try:
            self._rust = PyMeshAssembler(
                node_coords,
                connectivity,
                elem_types,
                materials_list,
            )
        except Exception:  # noqa: BLE001
            self._rust = None

    @staticmethod
    def _rust_group_indices(g):
        """Extract original element indices stored during group preparation."""
        # The group preparation stores dofs array; we need the original element
        # indices. We store them during _prepare_rust_batch_data as 'orig_indices'.
        return g.get("orig_indices", list(range(g["n_elem"])))

    def _coo_to_petsc(
        self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> PETSc.Mat:
        """Convert COO triplets to a PETSc sparse matrix via scipy CSR."""
        from scipy.sparse import coo_matrix

        n = self.dofs_count
        csr = coo_matrix(
            (vals, (rows.astype(np.int64), cols.astype(np.int64))),
            shape=(n, n),
        ).tocsr()

        mat = PETSc.Mat().create(self.comm)
        mat.createAIJ(
            size=(n, n),
            csr=(
                csr.indptr.astype(PETSc.IntType),
                csr.indices.astype(PETSc.IntType),
                csr.data.astype(PETSc.ScalarType),
            ),
        )
        mat.assemble()
        return mat

    def _create_petsc_matrix(self) -> PETSc.Mat:
        """
        Create a PETSc sparse matrix with optimized memory preallocation.

        Returns
        -------
        PETSc.Mat
            A sparse matrix configured for efficient assembly

        Notes
        -----
        Uses AIJ format (Compressed Sparse Row) by default. For better
        GPU performance consider setting type to 'seqaijcusparse'
        """
        mat = PETSc.Mat().create(self.comm)
        mat.setType("aij")
        mat.setSizes([self.dofs_count, self.dofs_count])

        d_nnz = self._row_nnz.astype(PETSc.IntType)
        o_nnz = np.zeros_like(d_nnz)  # Ajustar según particionado paralelo

        mat.setPreallocationNNZ((d_nnz, o_nnz))
        mat.setUp()
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        return mat

    def assemble_stiffness_matrix(self) -> PETSc.Mat:
        """
        Assemble the global stiffness matrix using PETSc.

        Returns
        -------
        PETSc.Mat
            Distributed sparse stiffness matrix

        Notes
        -----
        Performs parallel assembly using local element contributions.
        Matrix entries are accumulated using ADD_VALUES mode.
        Supports both uniform and mixed-element meshes.

        When the Rust backend (``fem_shell_core``) is available and all
        elements are covered, assembly uses COO triplets from Rust for
        significantly faster sparse-matrix construction.
        """
        # PyMeshAssembler fast-path (unified Rust assembler)
        if self._rust is not None:
            rows, cols, vals = self._rust.assemble_k()
            return self._coo_to_petsc(rows, cols, vals)

        # Rust fast-path: COO assembly from precomputed element matrices
        if self._has_rust and self._all_elements_rust and not self._is_mixed_mesh:
            ndof = self._dofs_array.shape[1]
            ke_flat = np.ascontiguousarray(self._ke_array, dtype=np.float64).ravel()
            rows, cols, vals = self._fsc.coo_assembly(self._dofs_array, ke_flat, ndof)
            return self._coo_to_petsc(rows, cols, vals)

        K = self._create_petsc_matrix()
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        if self._is_mixed_mesh:
            # Mixed mesh: use lists
            for dofs, ke in zip(self._dofs_list, self._ke_list):
                dofs_int = dofs.astype(PETSc.IntType)
                ke_flat = ke.flatten(order="C")
                K.setValuesLocal(dofs_int, dofs_int, ke_flat, addv=PETSc.InsertMode.ADD_VALUES)
        else:
            # Uniform mesh: use arrays (more efficient)
            for e in range(self._dofs_array.shape[0]):
                dofs = self._dofs_array[e].astype(PETSc.IntType)
                ke = self._ke_array[e].flatten(order="C")  # Row-major flattening

                # Use block insertion for better performance
                K.setValuesLocal(dofs, dofs, ke, addv=PETSc.InsertMode.ADD_VALUES)

        K.assemble()
        return K

    def assemble_mass_matrix(self) -> PETSc.Mat:
        """
        Assemble the global mass matrix using PETSc.

        Returns
        -------
        PETSc.Mat
            Distributed sparse mass matrix

        Notes
        -----
        Supports both uniform and mixed-element meshes.

        When the Rust backend is available the COO fast-path is used,
        mirroring :meth:`assemble_stiffness_matrix`.
        """
        # PyMeshAssembler fast-path
        if self._rust is not None:
            rows, cols, vals = self._rust.assemble_m()
            return self._coo_to_petsc(rows, cols, vals)

        # Rust fast-path: COO assembly from precomputed element matrices
        if self._has_rust and self._all_elements_rust and not self._is_mixed_mesh:
            ndof = self._dofs_array.shape[1]
            me_flat = np.ascontiguousarray(self._me_array, dtype=np.float64).ravel()
            rows, cols, vals = self._fsc.coo_assembly(self._dofs_array, me_flat, ndof)
            return self._coo_to_petsc(rows, cols, vals)

        M = self._create_petsc_matrix()
        M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        if self._is_mixed_mesh:
            # Mixed mesh: use lists
            for dofs, me in zip(self._dofs_list, self._me_list):
                dofs_int = dofs.astype(PETSc.IntType)
                me_flat = me.flatten(order="C")
                M.setValuesLocal(dofs_int, dofs_int, me_flat, addv=PETSc.InsertMode.ADD_VALUES)
        else:
            # Uniform mesh: use arrays
            for e in range(self._dofs_array.shape[0]):
                dofs = self._dofs_array[e].astype(PETSc.IntType)
                me = self._me_array[e].flatten(order="C")

                M.setValuesLocal(dofs, dofs, me, addv=PETSc.InsertMode.ADD_VALUES)

        M.assemble()
        return M

    def assemble_load_vector(self, load_condition) -> PETSc.Vec:
        """
        Assemble the global load vector using PETSc.

        Parameters
        ----------
        load_condition : LoadCondition
            The loading condition to apply

        Returns
        -------
        PETSc.Vec
            Distributed load vector

        Notes
        -----
        Supports both nodal and distributed loading conditions.
        Supports both uniform and mixed-element meshes.
        """
        f = PETSc.Vec().create(self.comm)
        f.setSizes(self.dofs_count)
        f.setUp()
        f.zeroEntries()

        # Compute element load vectors
        fe_list = [
            self._element_map[eid].body_load(load_condition.value) for eid in self._element_map
        ]

        if self._is_mixed_mesh:
            # Mixed mesh: iterate directly over lists
            for dofs, fe in zip(self._dofs_list, fe_list):
                dofs_int = dofs.astype(PETSc.IntType)
                f.setValuesLocal(dofs_int, fe, addv=PETSc.InsertMode.ADD_VALUES)
        else:
            # Uniform mesh: use arrays
            fe_array = np.array(fe_list, dtype=PETSc.ScalarType)

            for e in range(fe_array.shape[0]):
                dofs = self._dofs_array[e].astype(PETSc.IntType)
                fe = fe_array[e]

                # Use local-to-global mapping if using mesh partitioning
                f.setValuesLocal(dofs, fe, addv=PETSc.InsertMode.ADD_VALUES)

        f.assemble()
        return f

    def assemble_geometric_stiffness(
        self,
        stress_field: Optional[Dict[int, np.ndarray]] = None,
        omega: Optional[float] = None,
        rotation_axis: Optional[np.ndarray] = None,
        rotation_center: Optional[np.ndarray] = None,
    ) -> PETSc.Mat:
        """
        Assemble the global geometric stiffness matrix for stress stiffening effects.

        The geometric stiffness matrix K_G captures the effect of membrane prestress
        on structural stiffness. This is essential for:
        - Rotating structures (centrifugal stiffening of wind turbine blades)
        - Prestressed structures
        - Buckling analysis
        - Geometric nonlinear analysis

        The method supports two modes of operation:
        1. Direct stress field: Provide membrane stress for each element
        2. Centrifugal loading: Automatically compute prestress from rotation parameters

        Parameters
        ----------
        stress_field : Dict[int, np.ndarray], optional
            Dictionary mapping element IDs to membrane stress tensors.
            Each stress tensor should be shape (3,) in Voigt notation [σ_xx, σ_yy, σ_xy]
            or shape (2, 2) as full tensor.
        omega : float, optional
            Angular velocity (rad/s) for centrifugal loading calculation.
            Required if stress_field is not provided.
        rotation_axis : np.ndarray, optional
            Unit vector defining rotation axis (3,). Default [0, 0, 1] (z-axis).
            Required if omega is provided.
        rotation_center : np.ndarray, optional
            Point on rotation axis (3,). Default [0, 0, 0] (origin).

        Returns
        -------
        PETSc.Mat
            Distributed sparse geometric stiffness matrix K_G

        Raises
        ------
        ValueError
            If neither stress_field nor omega is provided

        Notes
        -----
        The geometric stiffness matrix is computed as:
        K_G = Σ_e K_G^(e) where K_G^(e) = ∫_A B_G^T · S_m · B_G · dA

        For rotating blades, the centrifugal stress creates tensile membrane
        stresses that stiffen the structure, raising natural frequencies.

        The total effective stiffness matrix is: K_eff = K + K_G

        References
        ----------
        - Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in
          geometric nonlinear analysis." Computers & Structures, 185, 1-14.

        Examples
        --------
        >>> # Using centrifugal loading for wind turbine blade
        >>> K_G = assembler.assemble_geometric_stiffness(
        ...     omega=1.5,  # rad/s
        ...     rotation_axis=np.array([0, 0, 1]),
        ...     rotation_center=np.array([0, 0, 0])
        ... )
        >>> K_eff = K + K_G  # Total stiffness with stress stiffening

        >>> # Using direct stress field
        >>> stress_field = {elem_id: np.array([1e6, 0, 0]) for elem_id in element_ids}
        >>> K_G = assembler.assemble_geometric_stiffness(stress_field=stress_field)
        """
        # Validate inputs
        if stress_field is None and omega is None:
            raise ValueError(
                "Either 'stress_field' or 'omega' must be provided for geometric stiffness"
            )

        # Set default rotation parameters
        if rotation_axis is None:
            rotation_axis = np.array([0.0, 0.0, 1.0])
        if rotation_center is None:
            rotation_center = np.array([0.0, 0.0, 0.0])

        # Normalize rotation axis
        rotation_axis = np.asarray(rotation_axis, dtype=float)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_center = np.asarray(rotation_center, dtype=float)

        # Create PETSc matrix
        K_G = self._create_petsc_matrix()
        K_G.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        # Compute and assemble element geometric stiffness matrices
        for e, (elem_id, fem_element) in enumerate(self._element_map.items()):
            # Get membrane stress for this element
            if stress_field is not None and elem_id in stress_field:
                sigma_membrane = stress_field[elem_id]
            elif omega is not None:
                # Compute centrifugal prestress
                sigma_membrane = fem_element.compute_centrifugal_prestress(
                    omega=omega,
                    rotation_axis=rotation_axis,
                    rotation_center=rotation_center,
                )
            else:
                # Skip elements without stress data
                continue

            # Skip if stress is negligible
            if np.max(np.abs(sigma_membrane)) < 1e-20:
                continue

            # Compute element geometric stiffness
            kg_e = fem_element.compute_geometric_stiffness(
                sigma_membrane=sigma_membrane,
                transform_to_global=True,
            )

            # Get DOFs for this element
            if self._is_mixed_mesh:
                dofs = self._dofs_list[e].astype(PETSc.IntType)
            else:
                dofs = self._dofs_array[e].astype(PETSc.IntType)
            kg_flat = kg_e.flatten(order="C")

            # Assemble into global matrix
            K_G.setValuesLocal(dofs, dofs, kg_flat, addv=PETSc.InsertMode.ADD_VALUES)

        K_G.assemble()
        return K_G

    # ------------------------------------------------------------------
    # Nonlinear assembly (Rust-accelerated)
    # ------------------------------------------------------------------

    def assemble_tangent_stiffness(self, u: np.ndarray) -> PETSc.Mat:
        """Assemble the global tangent stiffness matrix K_T(u).

        Uses Rust batch computation (``fem_shell_core``) for parallel element
        processing and COO sparse assembly.  This is the hot-path operation
        in Newton-Raphson iterations.

        Parameters
        ----------
        u : np.ndarray
            Global displacement vector, shape ``(dofs_count,)``.

        Returns
        -------
        PETSc.Mat
            Assembled tangent stiffness matrix.

        Raises
        ------
        RuntimeError
            If the Rust backend is not available.
        """
        # PyMeshAssembler fast-path
        if self._rust is not None:
            u_c = np.ascontiguousarray(u, dtype=np.float64)
            rows, cols, vals = self._rust.assemble_kt(u_c)
            return self._coo_to_petsc(rows, cols, vals)

        if not self._has_rust or not self._rust_groups:
            raise RuntimeError(
                "Rust backend (fem_shell_core) required for tangent stiffness assembly. "
                "Install with: cd crates/fem_shell_core && maturin develop --release"
            )

        all_rows, all_cols, all_vals = [], [], []

        for g in self._rust_groups:
            # Gather per-element displacements from the global vector
            u_local = np.ascontiguousarray(u[g["dofs"]], dtype=np.float64)

            if g["etype"] == "MITC3":
                kt_flat = self._fsc.batch_kt_mitc3(
                    g["coords"],
                    u_local,
                    g["E"],
                    g["nu"],
                    g["thickness"],
                    g["shear_correction"],
                )
            else:
                kt_flat = self._fsc.batch_kt_mitc4(
                    g["coords"],
                    u_local,
                    g["E"],
                    g["nu"],
                    g["thickness"],
                    g["shear_correction"],
                )

            rows, cols, vals = self._fsc.coo_assembly(
                g["dofs"], kt_flat, g["ndof"]
            )
            all_rows.append(rows)
            all_cols.append(cols)
            all_vals.append(vals)

        return self._coo_to_petsc(
            np.concatenate(all_rows),
            np.concatenate(all_cols),
            np.concatenate(all_vals),
        )

    def assemble_internal_forces(
        self, u: np.ndarray, nonlinear: bool = True
    ) -> PETSc.Vec:
        """Assemble the global internal force vector f_int(u).

        Uses Rust batch computation (``fem_shell_core``) for parallel element
        processing.

        Parameters
        ----------
        u : np.ndarray
            Global displacement vector, shape ``(dofs_count,)``.
        nonlinear : bool
            If ``True``, include geometric-nonlinear (Green-Lagrange) strain
            contributions.  Set to ``False`` for a linear f_int = K·u
            equivalent.

        Returns
        -------
        PETSc.Vec
            Assembled internal force vector.

        Raises
        ------
        RuntimeError
            If the Rust backend is not available.
        """
        # PyMeshAssembler fast-path
        if self._rust is not None:
            u_c = np.ascontiguousarray(u, dtype=np.float64)
            fint = np.asarray(self._rust.assemble_fint(u_c, nonlinear), dtype=np.float64)
            vec = PETSc.Vec().create(self.comm)
            vec.setSizes(self.dofs_count)
            vec.setUp()
            vec.setArray(fint)
            return vec

        if not self._has_rust or not self._rust_groups:
            raise RuntimeError(
                "Rust backend (fem_shell_core) required for internal force assembly. "
                "Install with: cd crates/fem_shell_core && maturin develop --release"
            )

        fint = np.zeros(self.dofs_count, dtype=np.float64)

        for g in self._rust_groups:
            u_local = np.ascontiguousarray(u[g["dofs"]], dtype=np.float64)

            if g["etype"] == "MITC3":
                fint_flat = self._fsc.batch_fint_mitc3(
                    g["coords"],
                    u_local,
                    g["E"],
                    g["nu"],
                    g["thickness"],
                    g["shear_correction"],
                    nonlinear,
                )
                fint_local = fint_flat.reshape(-1, 18)
            else:
                fint_flat = self._fsc.batch_fint_mitc4(
                    g["coords"],
                    u_local,
                    g["E"],
                    g["nu"],
                    g["thickness"],
                    g["shear_correction"],
                    nonlinear,
                )
                fint_local = fint_flat.reshape(-1, 24)

            # Scatter into global vector (handles shared DOFs correctly)
            np.add.at(fint, g["dofs"], fint_local)

        vec = PETSc.Vec().create(self.comm)
        vec.setSizes(self.dofs_count)
        vec.setUp()
        vec.setArray(fint)
        return vec
