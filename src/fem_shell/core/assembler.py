import logging
import time
from typing import Dict, Iterable, Optional

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.mesh import MeshModel
from fem_shell.elements import ElementFamily

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
        self._dofs_array: np.ndarray = None
        self._node_dofs_map: Dict[int, Iterable] = {}
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self._row_nnz: Optional[np.ndarray] = None
        self._rho_per_elem: Optional[np.ndarray] = None  # (n_elems,) — set in _build_py_mesh_assembler

        t0 = time.perf_counter()
        logger.info("[assembler] START __init__ — nodes=%d", mesh.node_count)

        t1 = time.perf_counter()
        self._precompute_elements()
        logger.info("[assembler] _precompute_elements done in %.2fs — elements=%d", time.perf_counter() - t1, len(self.mesh.elements))

        t3 = time.perf_counter()
        self._build_py_mesh_assembler()
        logger.info("[assembler] _build_py_mesh_assembler done in %.2fs — rust=%s", time.perf_counter() - t3, self._rust is not None)

        logger.info("[assembler] TOTAL __init__ done in %.2fs — dofs=%d", time.perf_counter() - t0, self.dofs_count)

    # Mapping from ElementFamily to (dofs_per_node, spatial_dim).
    # This avoids instantiating every element twice just to query these constants.
    _FAMILY_PROPERTIES = {
        ElementFamily.SHELL: (6, 3),
        ElementFamily.PLANE: (2, 2),
        ElementFamily.SOLID: (3, 3),
    }

    # Static vector_form per element family — no element instantiation needed.
    _FAMILY_VECTOR_FORM: Dict = {
        ElementFamily.SHELL: {"U": ("Ux", "Uy", "Uz"), "θ": ("θx", "θy", "θz")},
        ElementFamily.SOLID: {"U": ("Ux", "Uy", "Uz")},
        ElementFamily.PLANE: {"U": ("Ux", "Uy")},
    }

    @property
    def vector_form(self) -> Dict:
        """DOF vector layout for VTK output and solvers.

        Derived from the element family declared in the model.
        """
        family = self.model.get("element_family")
        if family in self._FAMILY_VECTOR_FORM:
            return self._FAMILY_VECTOR_FORM[family]
        # Mixed / unknown: infer from dofs_per_node
        if self.dofs_per_node == 6:
            return self._FAMILY_VECTOR_FORM[ElementFamily.SHELL]
        if self.dofs_per_node == 3:
            return self._FAMILY_VECTOR_FORM[ElementFamily.SOLID]
        return self._FAMILY_VECTOR_FORM[ElementFamily.PLANE]

    @property
    def element_family(self) -> Optional["ElementFamily"]:
        """Primary element family of this mesh (SHELL / SOLID / PLANE)."""
        family = self.model.get("element_family")
        if family is not None:
            return family
        # Infer from dofs_per_node
        if self.dofs_per_node == 6:
            return ElementFamily.SHELL
        if self.dofs_per_node == 3:
            return ElementFamily.SOLID
        return ElementFamily.PLANE

    def _precompute_elements(self):
        """Compute element DOF connectivity from mesh topology.

        Builds DOF arrays directly from the known dofs_per_node
        derived from the element family declaration.
        """
        elements = self.mesh.elements
        if not elements:
            return

        # --- Determine global maximum stride from element family ---
        element_family = self.model.get("element_family")
        if element_family is not None and element_family in self._FAMILY_PROPERTIES:
            self.dofs_per_node, self.spatial_dim = self._FAMILY_PROPERTIES[element_family]
        else:
            raise ValueError(
                f"element_family must be one of {list(self._FAMILY_PROPERTIES.keys())} "
                f"(got {element_family!r}). Mixed meshes without explicit element_family "
                "are not supported."
            )

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

    # ------------------------------------------------------------------
    # Lazy element map: only built when needed for body loads / stress stiffening
    # ------------------------------------------------------------------

    def _mesh_to_rust(self, properties_map):
        """Convert Python MeshModel + properties_map to a Rust MeshModel.

        Element type codes are set to composite (33/44) for elements in composite sets,
        or isotropic (3/4) otherwise — matching what from_model() expects.

        Vectorised with numpy to avoid per-element Python overhead (~86k elements).
        """
        from fem_shell_core import MeshModel as RustMeshModel  # noqa: PLC0415

        mesh = self.mesh

        # ── Nodes ─────────────────────────────────────────────────────────────
        nodes = mesh.nodes
        node_ids_list = [n.id for n in nodes]
        # Stack (N, 3) then ravel to flat [x0,y0,z0, x1,y1,z1, ...]
        coords_flat = np.stack([n.coords for n in nodes], axis=0).ravel().tolist()

        # ── Elements ──────────────────────────────────────────────────────────
        elements = mesh.elements
        element_ids_arr = np.fromiter((e.id for e in elements), dtype=np.int64, count=len(elements))

        # node_count per element — fast fromiter
        node_counts_arr = np.fromiter((e.node_count for e in elements), dtype=np.int8, count=len(elements))

        # Per-element composite flag: build set of composite element ids.
        # A property is composite if it is a RustLaminate (fem_shell_core.Laminate)
        # or a legacy Python CompositeShellProperty — anything that is NOT an
        # isotropic dict.
        try:
            from fem_shell_core import Laminate as _RustLaminate  # noqa: PLC0415
        except ImportError:
            _RustLaminate = None
        try:
            from fem_shell.core.properties import CompositeShellProperty as _CSP  # noqa: PLC0415
        except ImportError:
            _CSP = None

        composite_elem_ids: set = set()
        for set_name, prop in properties_map.items():
            is_composite = (
                (_RustLaminate is not None and isinstance(prop, _RustLaminate))
                or (_CSP is not None and isinstance(prop, _CSP))
            )
            if is_composite and set_name in mesh.element_sets:
                composite_elem_ids.update(e.id for e in mesh.element_sets[set_name].elements)

        if composite_elem_ids:
            composite_arr = np.isin(element_ids_arr, np.fromiter(composite_elem_ids, dtype=np.int64))
        else:
            composite_arr = np.zeros(len(elements), dtype=bool)

        # type code: tri composite=33, quad composite=44, tri iso=3, quad iso=4
        is_tri = node_counts_arr == 3
        type_codes = np.where(
            composite_arr,
            np.where(is_tri, 33, 44),
            np.where(is_tri, 3, 4),
        ).tolist()

        element_ids_list = element_ids_arr.tolist()
        element_node_ids_list = [list(e.node_ids) for e in elements]

        # ── Sets ──────────────────────────────────────────────────────────────
        rust_esets = {
            name: [e.id for e in eset.elements]
            for name, eset in mesh.element_sets.items()
        }
        rust_nsets = {
            name: list(nset.node_ids)
            for name, nset in mesh.node_sets.items()
        }

        return RustMeshModel.from_raw_data(
            node_ids_list,
            coords_flat,
            element_ids_list,
            element_node_ids_list,
            type_codes,
            rust_esets,
            rust_nsets,
        )

    def _properties_to_rust(self, properties_map, RustLaminate, RustPly, RustMat):
        """Convert Python properties_map to a dict suitable for PyMeshAssembler.from_model().

        If the map already contains Rust-native types (``fem_shell_core.Laminate``
        or isotropic dicts) this is a no-op pass-through.  Python
        ``CompositeShellProperty`` / ``ShellProperty`` objects are still
        supported for backwards compatibility with callers that haven't
        migrated yet.

        Returns dict[str, RustLaminate | dict].
        """
        try:
            from fem_shell.core.properties import CompositeShellProperty, ShellProperty  # noqa: PLC0415
            _has_py_props = True
        except ImportError:
            _has_py_props = False

        result = {}
        for set_name, prop in properties_map.items():
            # Already Rust-native — pass through unchanged
            if isinstance(prop, (RustLaminate, dict)):
                result[set_name] = prop
            elif _has_py_props and isinstance(prop, CompositeShellProperty):
                lam = prop.laminate
                rust_plies = []
                for ply in lam.plies:
                    m = ply.material
                    E = m.E
                    G = m.G
                    nu = m.nu
                    rust_mat = RustMat(
                        float(E[0]), float(E[1]), float(E[2]),
                        float(G[0]), float(G[1]), float(G[2]),
                        float(nu[0]), float(nu[1]), float(nu[2]),
                        float(m.rho),
                    )
                    rust_plies.append(RustPly(rust_mat, float(ply.thickness), float(ply.angle)))
                scf = float(getattr(lam, 'shear_correction_factor', 0.75))
                result[set_name] = RustLaminate(rust_plies, scf)
            elif _has_py_props and isinstance(prop, ShellProperty):
                m = prop.material
                result[set_name] = {
                    "type": "isotropic",
                    "e": float(m.E),
                    "nu": float(m.nu),
                    "rho": float(m.rho),
                    "thickness": float(prop.thickness),
                    "shear_correction": float(getattr(prop, 'shear_correction', 5.0 / 6.0)),
                }
            else:
                result[set_name] = prop
        return result

    def _build_py_mesh_assembler(self):
        """Build a PyMeshAssembler (Rust) from mesh topology and properties.

        When a ``properties`` map is present in the model config the Rust
        ``PyMeshAssembler.from_model()`` fast-path is used exclusively —
        no Python element loop.

        For the legacy isotropic path (model has ``material`` / ``thickness``
        instead of a properties map) the direct ``PyMeshAssembler()``
        constructor is used, which still requires a Python loop to build the
        materials list from model fields.
        """
        self._rust = None

        try:
            from fem_shell_core import PyMeshAssembler, MeshModel as RustMeshModel, Laminate as RustLaminate, Ply as RustPly, OrthotropicMaterial as RustMat  # noqa: PLC0415
        except ImportError:
            logger.warning("[assembler._build_py_mesh_assembler] fem_shell_core not available — Rust assembler disabled")
            return

        properties_map: Optional[Dict] = self.model.get("properties")

        # ── COMPOSITE / MULTI-PROPERTY PATH: PyMeshAssembler.from_model() ─────
        if properties_map is not None:
            _t_fast = time.perf_counter()
            rust_mesh = self._mesh_to_rust(properties_map)
            rust_properties = self._properties_to_rust(properties_map, RustLaminate, RustPly, RustMat)
            span_dir_raw = self.model.get("span_direction")
            span_dir_list = list(span_dir_raw) if span_dir_raw is not None else None
            self._rust = PyMeshAssembler.from_model(
                rust_mesh,
                rust_properties,
                span_dir_list,
                None,  # fallback_material — handled via properties_map
            )
            self._row_nnz = np.asarray(self._rust.nnz_per_row(), dtype=PETSc.IntType)
            # _rho_per_elem: extracted from Rust MaterialSpec — no Python loop needed
            self._rho_per_elem = np.asarray(self._rust.rho_per_elem(), dtype=np.float64)
            logger.info("[assembler._build_py_mesh_assembler] from_model() done in %.2fs", time.perf_counter() - _t_fast)
            return

        # ── LEGACY ISOTROPIC PATH: direct PyMeshAssembler() constructor ───────
        logger.info("[assembler._build_py_mesh_assembler] isotropic path — %d nodes", self.mesh.node_count)
        _t0 = time.perf_counter()

        node_id_to_index = self.mesh.node_id_to_index
        n_nodes = self.mesh.node_count
        node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for node in self.mesh.nodes:
            idx = node_id_to_index[node.id]
            node_coords[idx, :] = node.coords[:3]

        fallback_material = self.model.get("material")
        fallback_thickness = self.model.get("thickness", 1.0)
        fallback_sc = self.model.get("shear_correction", 5.0 / 6.0)
        fallback_family = self.model.get("element_family")

        elements = self.mesh.elements
        n_elems = len(elements)
        connectivity = []
        elem_types = []
        materials_list = []

        for i, element in enumerate(elements):
            conn = [node_id_to_index[nid] for nid in element.node_ids]
            connectivity.append(conn)
            n_nodes_elem = element.node_count

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
                logger.error(
                    "[assembler._build_py_mesh_assembler] no property for element %d (index %d) — aborting Rust assembler",
                    element.id, i,
                )
                self._rust = None
                return

            elem_types.append(code)
            materials_list.append(mat_dict)

        self._rho_per_elem = np.array(
            [m.get("rho", m.get("mass_per_area", 0.0)) for m in materials_list],
            dtype=np.float64,
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
            self._row_nnz = np.asarray(self._rust.nnz_per_row(), dtype=PETSc.IntType)
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
        Uses Rust batch assembly when available (fast-path via PyMeshAssembler).
        Falls back to Python per-element loop for non-body loads or when Rust
        backend is not available.
        """
        load_value = load_condition.value

        # ------------------------------------------------------------------
        # Rust fast-path: body load (gravity-like [fx, fy, fz] body force)
        # ------------------------------------------------------------------
        if self._rust is not None and np.ndim(load_value) == 1 and len(load_value) == 3:
           gravity = np.asarray(load_value, dtype=np.float64)
           f_dense = self._rust.assemble_f_body(gravity)  # (dofs_count,) numpy array
           f = PETSc.Vec().create(self.comm)
           f.setSizes(self.dofs_count)
           f.setUp()
           f.zeroEntries()
           dofs_all = np.arange(self.dofs_count, dtype=PETSc.IntType)
           f.setValuesLocal(dofs_all, f_dense.astype(PETSc.ScalarType), addv=PETSc.InsertMode.ADD_VALUES)
           f.assemble()
           return f

        # ------------------------------------------------------------------
        # Non-body loads require Rust assembler (nodal/non-uniform not yet implemented)
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "assemble_load_vector: non-body-force loads (non-uniform or nodal) "
            "require load_value to be a 3-element body-force vector [fx, fy, fz] "
            "and a live Rust assembler. Got load_value with shape/type: "
            f"{np.shape(load_value)} / {type(load_value).__name__}. "
            "The Python per-element fallback has been removed."
        )

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
        if stress_field is None and omega is None:
            raise ValueError(
                "Either 'stress_field' or 'omega' must be provided for geometric stiffness"
            )

        # Set default rotation parameters
        if rotation_axis is None:
            rotation_axis = np.array([0.0, 0.0, 1.0])
        if rotation_center is None:
            rotation_center = np.array([0.0, 0.0, 0.0])

        rotation_axis = np.asarray(rotation_axis, dtype=float)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_center = np.asarray(rotation_center, dtype=float)

        # ------------------------------------------------------------------
        # Rust fast-path (requires PyMeshAssembler + cached arrays)
        # ------------------------------------------------------------------
        if (
            self._rust is not None
            and self._rho_per_elem is not None
        ):
            elements = self.mesh.elements
            n_elems = len(elements)
            sigma_array = np.zeros((n_elems, 3), dtype=np.float64)

            if stress_field is not None:
                # Map elem_id → index, then fill sigma_array from stress_field dict
                for i, element in enumerate(elements):
                    if element.id in stress_field:
                        sv = stress_field[element.id]
                        sv = np.asarray(sv, dtype=np.float64).ravel()
                        sigma_array[i, :3] = sv[:3]
                rows, cols, vals = self._rust.assemble_geometric_k(sigma_array)
            else:
                # Rust-accelerated centrifugal prestress
                rows, cols, vals = self._rust.assemble_centrifugal_k(
                    float(omega),
                    list(map(float, rotation_axis)),
                    list(map(float, rotation_center)),
                    self._rho_per_elem,
                )
            return self._coo_to_petsc(rows, cols, vals)

        # ------------------------------------------------------------------
        # Python fallback removed — Rust assembler required
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "assemble_geometric_stiffness: the Python per-element fallback has been removed. "
            "A live Rust assembler (self._rust) is required. "
            "Ensure fem_shell_core is built and PyMeshAssembler initialised."
        )

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
