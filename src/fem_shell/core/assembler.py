import logging
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.mesh import MeshModel
from fem_shell.core.properties import CompositeShellProperty, ShellProperty, ShellPropertyType
from fem_shell.elements import ElementFactory, ElementFamily, FemElement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (no FemElement instantiation)
# ---------------------------------------------------------------------------

def _shell_local_axes(node_coords: np.ndarray, conn: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local orthonormal axes (e1, e2, e3) for a shell element.

    Replicates the logic in MITC4._compute_local_coordinates / MITC3 without
    instantiating any Python element object.  Works for 3-node and 4-node elements.

    Parameters
    ----------
    node_coords : (n_nodes_total, 3) float array — global node coordinates.
    conn        : list of 0-based indices into node_coords for this element.

    Returns
    -------
    e1, e2, e3 : unit vectors defining the local coordinate system.
    """
    pts = [node_coords[c] for c in conn]

    normals = []
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    n1 = np.cross(v1, v2)
    if np.linalg.norm(n1) > 1e-12:
        normals.append(n1 / np.linalg.norm(n1))

    if len(pts) >= 4:
        v1b = pts[2] - pts[0]
        v2b = pts[3] - pts[0]
        n2 = np.cross(v1b, v2b)
        if np.linalg.norm(n2) > 1e-12:
            normals.append(n2 / np.linalg.norm(n2))

    if normals:
        e3 = np.mean(normals, axis=0)
        e3 /= np.linalg.norm(e3)
    else:
        e3 = np.array([0.0, 0.0, 1.0])

    e1 = pts[1] - pts[0]
    e1 = e1 - np.dot(e1, e3) * e3
    if np.linalg.norm(e1) < 1e-12:
        e1 = pts[2] - pts[0]
        e1 = e1 - np.dot(e1, e3) * e3
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, e3


def _apply_span_direction(
    node_coords: np.ndarray,
    conn: list,
    span_dir: np.ndarray,
    lam,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return corrected (A, D, Cs) matrices for a laminate given a span direction.

    Replicates MITC4Composite._recompute_abd_for_span without instantiating
    the element object.  If the angle offset is negligible (< 0.01°) returns
    the original laminate matrices.

    Parameters
    ----------
    node_coords : global node coordinate array (n_nodes_total, 3).
    conn        : 0-based connectivity indices for this element.
    span_dir    : 3-vector span direction in global coordinates.
    lam         : Laminate object with .plies, .A, .D, .Cs.

    Returns
    -------
    A, D, Cs : (3,3), (3,3), (2,2) corrected stiffness matrices.
    """
    from fem_shell.core.laminate import Laminate as Lam, Ply as Pl  # noqa: PLC0415

    e1, e2, e3 = _shell_local_axes(node_coords, conn)

    sd = span_dir - float(np.dot(span_dir, e3)) * e3
    sd_len = float(np.linalg.norm(sd))
    if sd_len < 1e-10:
        return lam.A.copy(), lam.D.copy(), lam.Cs.copy()

    sd_hat = sd / sd_len
    cos_a = float(np.dot(sd_hat, e1))
    sin_a = float(np.dot(sd_hat, e2))
    angle_offset_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))

    if abs(angle_offset_deg) < 0.01:
        return lam.A.copy(), lam.D.copy(), lam.Cs.copy()

    corrected_plies = [
        Pl(material=ply.material, thickness=ply.thickness, angle=ply.angle + angle_offset_deg)
        for ply in lam.plies
    ]
    corrected_lam = Lam(plies=corrected_plies)
    return corrected_lam.A.copy(), corrected_lam.D.copy(), corrected_lam.Cs.copy()


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
        self._element_map_built: bool = False  # lazy flag
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
        logger.info("[assembler] _precompute_elements done in %.2fs — elements=%d", time.perf_counter() - t1, len(self.mesh.elements))

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
        """Compute element DOF connectivity without instantiating FemElement objects.

        Builds DOF arrays directly from mesh topology and the known dofs_per_node
        derived from the element family declaration.  ElementFactory is no longer
        called here — element Python objects are created lazily on first demand.
        """
        elements = self.mesh.elements
        if not elements:
            return

        # --- Determine global maximum stride from element family ---
        element_family = self.model.get("element_family")
        if element_family is not None and element_family in self._FAMILY_PROPERTIES:
            self.dofs_per_node, self.spatial_dim = self._FAMILY_PROPERTIES[element_family]
        else:
            # Mixed / unknown family: probe with one element per distinct node count.
            # This is O(distinct_node_counts) — typically 1-3 probes, not 85k.
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

        # --- Build DOF arrays directly from mesh topology ---
        node_id_to_index = self.mesh.node_id_to_index
        dpn = self.dofs_per_node
        dofs_list = []
        dof_sizes = set()
        _t0 = time.perf_counter()

        for element in elements:
            node_ids = element.node_ids
            dofs = np.array(
                [node_id_to_index[nid] * dpn + d for nid in node_ids for d in range(dpn)],
                dtype=np.int64,
            )
            dof_sizes.add(len(dofs))
            dofs_list.append(dofs)
            # Populate _node_dofs_map for legacy callers
            for nid in node_ids:
                if nid not in self._node_dofs_map:
                    start = node_id_to_index[nid] * dpn
                    self._node_dofs_map[nid] = tuple(range(start, start + dpn))

        logger.info(
            "[assembler._precompute] DOF arrays built in %.2fs — %d elements, dofs_per_node=%d",
            time.perf_counter() - _t0, len(elements), dpn,
        )

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
        """Stub kept for backward compatibility. No-op since batch groups are
        no longer used — all K/M assembly goes through ``self._rust`` directly."""
        self._rust_groups = []
        self._rust_composite_groups = []
        self._has_rust = False
        self._all_elements_rust = False
        try:
            import fem_shell_core  # noqa: F401
            self._has_rust = True
            self._all_elements_rust = True
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Lazy element map: only built when needed for body loads / stress stiffening
    # ------------------------------------------------------------------

    def _ensure_element_map(self) -> None:
        """Build ``_element_map`` on first demand.

        The element map is expensive to construct (~89s for 85k elements) and is
        only needed for body-load assembly and geometric stiffness.  All K/M
        assembly goes through ``self._rust`` (PyMeshAssembler) and does not
        require Python element objects.
        """
        if self._element_map_built:
            return

        logger.info("[assembler._ensure_element_map] building element map lazily (%d elements)…", len(self.mesh.elements))
        _t0 = time.perf_counter()

        elements = self.mesh.elements
        node_id_to_index = self.mesh.node_id_to_index

        # Build property lookup
        properties_map: Optional[Dict[str, ShellPropertyType]] = self.model.get("properties")
        element_property_lookup: Dict[int, ShellPropertyType] = {}
        if properties_map is not None:
            for set_name, prop in properties_map.items():
                if set_name in self.mesh.element_sets:
                    for elem in self.mesh.element_sets[set_name].elements:
                        element_property_lookup[elem.id] = prop

        for element in elements:
            shell_property = element_property_lookup.get(element.id)
            element_model = {k: v for k, v in self.model.items() if k != "properties"}

            if shell_property is not None:
                element_model = {k: v for k, v in element_model.items()
                                 if k not in ("material", "thickness", "laminate")}
            elif (element.thickness is not None
                  and element_model.get("element_family") == ElementFamily.SHELL):
                import warnings
                warnings.warn(
                    "Per-element thickness via MeshElement.thickness is deprecated. "
                    "Use a 'properties' dict mapping element-set names to "
                    "ShellProperty / CompositeShellProperty instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                element_model = {**element_model, "thickness": element.thickness}

            fem_element = ElementFactory.get_element(
                mesh_element=element, shell_property=shell_property, **element_model
            )
            if fem_element:
                self._element_map[element.id] = fem_element

        self._element_map_built = True
        logger.info("[assembler._ensure_element_map] built %d elements in %.2fs",
                    len(self._element_map), time.perf_counter() - _t0)

    def _build_py_mesh_assembler(self):
        """Build a PyMeshAssembler (Rust) directly from mesh topology and properties.

        Iterates over ``mesh.elements`` and the properties map WITHOUT
        instantiating Python FemElement objects, eliminating the ~89s
        ElementFactory bottleneck for the 85k-element blade case.
        """
        self._rust = None

        try:
            from fem_shell_core import PyMeshAssembler  # noqa: PLC0415
        except ImportError:
            logger.warning("[assembler._build_py_mesh_assembler] fem_shell_core not available — Rust assembler disabled")
            return

        from fem_shell.core.laminate import Laminate as Lam, Ply as Pl  # noqa: PLC0415

        logger.info("[assembler._build_py_mesh_assembler] building node_coords for %d nodes", self.mesh.node_count)

        # Build node_coords array (n_nodes × 3)
        node_id_to_index = self.mesh.node_id_to_index
        n_nodes = self.mesh.node_count
        node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for node in self.mesh.nodes:
            idx = node_id_to_index[node.id]
            node_coords[idx, :] = node.coords[:3]

        # Build per-element property lookup
        properties_map: Optional[Dict[str, ShellPropertyType]] = self.model.get("properties")
        element_property_lookup: Dict[int, ShellPropertyType] = {}
        if properties_map is not None:
            for set_name, prop in properties_map.items():
                if set_name in self.mesh.element_sets:
                    for elem in self.mesh.element_sets[set_name].elements:
                        element_property_lookup[elem.id] = prop

        # span_direction for fibre orientation correction
        span_dir_raw = self.model.get("span_direction")
        span_dir: Optional[np.ndarray] = (
            np.asarray(span_dir_raw, dtype=np.float64) if span_dir_raw is not None else None
        )

        # Fallback for non-composite / non-property elements (isotropic path)
        fallback_material = self.model.get("material")
        fallback_thickness = self.model.get("thickness", 1.0)
        fallback_sc = self.model.get("shear_correction", 5.0 / 6.0)
        fallback_family = self.model.get("element_family")

        connectivity = []
        elem_types = []
        materials_list = []

        n_elems = len(self.mesh.elements)
        progress_interval = max(n_elems // 10, 1)
        _t_iter = time.perf_counter()

        for i, element in enumerate(self.mesh.elements):
            if i % progress_interval == 0:
                logger.info(
                    "[assembler._build_py_mesh_assembler] %d/%d (%.0f%%) elapsed=%.2fs",
                    i, n_elems, 100.0 * i / n_elems, time.perf_counter() - _t_iter,
                )

            conn = [node_id_to_index[nid] for nid in element.node_ids]
            connectivity.append(conn)

            prop = element_property_lookup.get(element.id)
            n_nodes_elem = element.node_count

            if isinstance(prop, CompositeShellProperty):
                # Composite shell (MITC3Composite or MITC4Composite)
                code = 33 if n_nodes_elem == 3 else 44
                lam = prop.laminate
                A, D, Cs = lam.A.copy(), lam.D.copy(), lam.Cs.copy()
                h = lam.total_thickness

                if span_dir is not None:
                    A, D, Cs = _apply_span_direction(
                        node_coords, conn, span_dir, lam
                    )

                a_trace = float(np.trace(A))
                e_equiv = a_trace / (3.0 * h) if h > 0 else 0.0
                mpa = sum(ply.material.rho * ply.thickness for ply in lam.plies)
                rot_inertia = sum(
                    ply.material.rho * (ply.z_top**3 - ply.z_bottom**3) / 3.0
                    for ply in lam.plies
                )
                mat_dict = {
                    "type": "composite",
                    "cm": A.ravel().tolist(),
                    "cb": D.ravel().tolist(),
                    "cs": Cs.ravel().tolist(),
                    "thickness": h,
                    "e_equiv": e_equiv,
                    "mass_per_area": mpa,
                    "rotational_inertia": rot_inertia,
                }

            elif isinstance(prop, ShellProperty):
                # Isotropic / single-layer shell (MITC3 or MITC4)
                code = 3 if n_nodes_elem == 3 else 4
                mat = prop.material
                sc = getattr(prop, "shear_correction", fallback_sc)
                mat_dict = {
                    "type": "isotropic",
                    "e": float(mat.E),
                    "nu": float(mat.nu),
                    "rho": float(mat.rho),
                    "thickness": float(prop.thickness),
                    "shear_correction": float(sc),
                }

            else:
                # No property in lookup — use legacy model fields or skip
                if fallback_family == ElementFamily.SHELL and fallback_material is not None:
                    code = 3 if n_nodes_elem == 3 else 4
                    mat = fallback_material
                    mat_dict = {
                        "type": "isotropic",
                        "e": float(mat.E),
                        "nu": float(mat.nu),
                        "rho": float(mat.rho),
                        "thickness": float(fallback_thickness),
                        "shear_correction": float(fallback_sc),
                    }
                else:
                    # Cannot determine element type — abort Rust assembler
                    logger.error(
                        "[assembler._build_py_mesh_assembler] no property for element %d (index %d) — aborting Rust assembler",
                        element.id, i,
                    )
                    self._rust = None
                    return

            elem_types.append(code)
            materials_list.append(mat_dict)

        logger.info(
            "[assembler._build_py_mesh_assembler] loop done in %.2fs (%d elements), calling PyMeshAssembler()",
            time.perf_counter() - _t_iter, n_elems,
        )

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

        # Ensure element map is built (lazy — only constructed on first body-load call)
        self._ensure_element_map()

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

        # Ensure element map is built (lazy)
        self._ensure_element_map()

        # Build elem_id → DOF-array index (mesh.elements order)
        elem_id_to_dof_idx = {elem.id: i for i, elem in enumerate(self.mesh.elements)}

        # Compute and assemble element geometric stiffness matrices
        for elem_id, fem_element in self._element_map.items():
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

            # Get DOFs for this element (index in mesh.elements order)
            e = elem_id_to_dof_idx[elem_id]
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
