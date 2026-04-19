import logging
import time
from typing import Dict, Iterable, Optional

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.mesh import MeshModel
from fem_shell.core.properties import ShellPropertyType
from fem_shell.elements import ElementFactory, ElementFamily, FemElement

logger = logging.getLogger(__name__)


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
        """
        self.mesh = mesh
        self.model = model["elements"]
        self.comm = MPI.COMM_WORLD
        self._element_map: Dict[int, FemElement] = {}
        self._dofs_array: np.ndarray = None
        self._node_dofs_map: Dict[int, Iterable] = {}
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self._row_nnz: Optional[np.ndarray] = None

        t0 = time.perf_counter()
        logger.info("[assembler] START __init__ — nodes=%d", mesh.node_count)

        t1 = time.perf_counter()
        self._precompute_elements()
        logger.info("[assembler] _precompute_elements done in %.2fs — elements=%d", time.perf_counter() - t1, len(self._element_map))

        t2 = time.perf_counter()
        self._compute_sparsity_pattern()
        logger.info("[assembler] _compute_sparsity_pattern done in %.2fs", time.perf_counter() - t2)

        t3 = time.perf_counter()
        self._prepare_rust_batch_data()
        logger.info("[assembler] _prepare_rust_batch_data done in %.2fs", time.perf_counter() - t3)

        t4 = time.perf_counter()
        self._build_py_mesh_assembler()
        logger.info("[assembler] _build_py_mesh_assembler done in %.2fs — rust=%s", time.perf_counter() - t4, self._rust is not None)

        logger.info("[assembler] TOTAL __init__ done in %.2fs — dofs=%d", time.perf_counter() - t0, self.dofs_count)

    # Mapping from ElementFamily to (dofs_per_node, spatial_dim).
    # This avoids instantiating every element twice just to query these constants.
    _FAMILY_PROPERTIES = {
        ElementFamily.SHELL: (6, 3),
        ElementFamily.PLANE: (2, 2),
        ElementFamily.SOLID: (3, 3),
    }

    def _precompute_elements(self):
        """Precompute element DOF connectivity arrays.

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
        _t_loop_start = time.perf_counter()
        _t_factory = 0.0
        _t_dofs = 0.0
        _t_km = 0.0

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
                elapsed = time.perf_counter() - _t_loop_start
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (n_elements - idx) / rate if rate > 0 else 0
                logger.info(
                    "[assembler._precompute] %d/%d (%.0f%%) | elapsed=%.1fs rate=%.0f/s ETA=%.1fs"
                    " | t_factory=%.2fs t_dofs=%.2fs t_km=%.2fs",
                    idx, n_elements, 100 * idx / n_elements,
                    elapsed, rate, eta,
                    _t_factory, _t_dofs, _t_km,
                )
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

            _t0 = time.perf_counter()
            fem_element = ElementFactory.get_element(
                mesh_element=element, shell_property=shell_property, **element_model
            )
            _t_factory += time.perf_counter() - _t0

            if not fem_element:
                continue

            # Remap DOFs using the GLOBAL stride to avoid aliasing
            _t0 = time.perf_counter()
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
            _t_dofs += time.perf_counter() - _t0

            self._element_map[element.id] = fem_element
            dofs_list.append(dofs)

            # NOTE: K and M are NOT computed here — assembly is done entirely
            # by the Rust PyMeshAssembler. Computing them in Python would be
            # dead code and waste ~165s on 85k composite elements.
            _t_km += 0.0  # kept for timing instrumentation consistency

        print(
            f"\r  Precomputing element matrices... {n_elements}/{n_elements} (100%)",
            flush=True,
        )
        logger.info(
            "[assembler._precompute] DONE total=%.2fs | t_factory=%.2fs t_dofs=%.2fs t_km=%.2fs",
            time.perf_counter() - _t_loop_start, _t_factory, _t_dofs, _t_km,
        )

        # Check if all elements have the same DOF count (uniform mesh)
        self._is_mixed_mesh = len(dof_sizes) > 1

        if self._is_mixed_mesh:
            self._dofs_list = dofs_list
            self._dofs_array = None
        else:
            self._dofs_array = np.array(dofs_list, dtype=np.int64)
            self._dofs_list = None

        # K/M arrays are NOT stored — assembly is done by the Rust assembler.
        self._ke_array = None
        self._me_array = None
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
        for batch processing metadata. This populates ``_rust_groups``,
        ``_rust_composite_groups``, ``_has_rust``, and ``_all_elements_rust``
        which are used by tests and introspection code.

        NOTE: The actual K/M assembly now goes through ``self._rust``
        (PyMeshAssembler) exclusively. These groups are kept for compatibility.
        """
        self._rust_groups = []
        self._rust_composite_groups = []
        self._has_rust = False
        self._all_elements_rust = False

        try:
            import fem_shell_core  # noqa: F401

            self._has_rust = True
        except ImportError:
            return

        from fem_shell.elements.MITC3 import MITC3
        from fem_shell.elements.MITC3_composite import MITC3Composite
        from fem_shell.elements.MITC4 import MITC4
        from fem_shell.elements.MITC4_composite import MITC4Composite

        groups: dict = {}
        composite_groups: dict = {}
        elem_list = list(self._element_map.values())

        for e, fem_elem in enumerate(elem_list):
            if isinstance(fem_elem, MITC3Composite):
                key = "MITC3_composite"
                if key not in composite_groups:
                    composite_groups[key] = {"indices": [], "coords": [], "etype": key}
                composite_groups[key]["indices"].append(e)
                composite_groups[key]["coords"].append(fem_elem._initial_coords.ravel())
                continue

            if isinstance(fem_elem, MITC4Composite):
                key = "MITC4_composite"
                if key not in composite_groups:
                    composite_groups[key] = {"indices": [], "coords": [], "etype": key}
                composite_groups[key]["indices"].append(e)
                composite_groups[key]["coords"].append(fem_elem._initial_coords.ravel())
                continue

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
                    "n_elem": len(indices),
                    "orig_indices": indices.tolist(),
                }
            )

        n_rust = sum(g["n_elem"] for g in self._rust_groups)
        n_rust += sum(g["n_elem"] for g in self._rust_composite_groups)
        self._all_elements_rust = n_rust == len(elem_list)

    def _build_py_mesh_assembler(self):
        """Build a PyMeshAssembler (Rust) from the precomputed element data.

        This provides a single unified Rust assembler. The ``self._rust`` attribute
        is set when construction succeeds; otherwise it is ``None``.
        """
        self._rust = None

        try:
            from fem_shell_core import PyMeshAssembler  # noqa: PLC0415
        except ImportError:
            logger.warning("[assembler._build_py_mesh_assembler] fem_shell_core not available — Rust assembler disabled")
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

        logger.info("[assembler._build_py_mesh_assembler] building node_coords for %d nodes", self.mesh.node_count)

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

        n_elems = len(self._element_map)
        logger.info("[assembler._build_py_mesh_assembler] iterating %d elements for connectivity/materials", n_elems)
        _t_iter = time.perf_counter()
        _unknown_types = []

        for i, fem_elem in enumerate(self._element_map.values()):
            if i % max(n_elems // 10, 1) == 0:
                logger.info("[assembler._build_py_mesh_assembler] connectivity %d/%d (%.0f%%) elapsed=%.2fs",
                            i, n_elems, 100 * i / n_elems, time.perf_counter() - _t_iter)

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
                _unknown_types.append(elem_cls.__name__)
                logger.error("[assembler._build_py_mesh_assembler] unknown element type: %s at index %d", elem_cls.__name__, i)
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

        logger.info("[assembler._build_py_mesh_assembler] connectivity loop done in %.2fs, calling PyMeshAssembler()", time.perf_counter() - _t_iter)

        try:
            _t_rust = time.perf_counter()
            self._rust = PyMeshAssembler(
                node_coords,
                connectivity,
                elem_types,
                materials_list,
            )
            logger.info("[assembler._build_py_mesh_assembler] PyMeshAssembler() constructed in %.2fs", time.perf_counter() - _t_rust)
        except Exception as exc:  # noqa: BLE001
            logger.error("[assembler._build_py_mesh_assembler] PyMeshAssembler() FAILED: %s", exc)
            self._rust = None

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
        Uses the PyMeshAssembler (Rust) unified assembler exclusively.
        Raises RuntimeError if the Rust assembler is not available.
        """
        if self._rust is not None:
            rows, cols, vals = self._rust.assemble_k()
            return self._coo_to_petsc(rows, cols, vals)

        raise RuntimeError("Rust assembler not available — cannot assemble K")

    def assemble_mass_matrix(self) -> PETSc.Mat:
        """
        Assemble the global mass matrix using PETSc.

        Returns
        -------
        PETSc.Mat
            Distributed sparse mass matrix

        Notes
        -----
        Uses the PyMeshAssembler (Rust) unified assembler exclusively.
        Raises RuntimeError if the Rust assembler is not available.
        """
        if self._rust is not None:
            rows, cols, vals = self._rust.assemble_m()
            return self._coo_to_petsc(rows, cols, vals)

        raise RuntimeError("Rust assembler not available — cannot assemble M")

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

        raise RuntimeError(
            "Rust backend (fem_shell_core) required for tangent stiffness assembly. "
            "Install with: cd crates/fem_shell_core && maturin develop --release"
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

        raise RuntimeError(
            "Rust backend (fem_shell_core) required for internal force assembly. "
            "Install with: cd crates/fem_shell_core && maturin develop --release"
        )
