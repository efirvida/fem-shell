from __future__ import annotations

import logging
import time
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from aeroelast.elements import ElementFamily

if TYPE_CHECKING:
    from aeroelast.core.mesh import MeshModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight proxy types that duck-type Python MeshModel nodes/elements
# so that solvers can use domain.nodes / domain.elements without
# holding a reference to the Python MeshModel.
# ---------------------------------------------------------------------------

class _NodeProxy:
    """Minimal node proxy backed by a flat coords array."""
    __slots__ = ("id", "x", "y", "z")

    def __init__(self, node_id: int, x: float, y: float, z: float):
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class _NodeProxyList:
    """Sequence of _NodeProxy objects built from Rust node_ids / coords_flat."""

    def __init__(self, ids, coords_flat):
        self._ids = ids
        self._coords = np.asarray(coords_flat, dtype=np.float64).reshape(-1, 3)

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        for i, nid in enumerate(self._ids):
            c = self._coords[i]
            yield _NodeProxy(nid, float(c[0]), float(c[1]), float(c[2]))

    def __getitem__(self, idx):
        nid = self._ids[idx]
        c = self._coords[idx]
        return _NodeProxy(nid, float(c[0]), float(c[1]), float(c[2]))


class _ElemProxy:
    """Minimal element proxy with .id and .node_ids."""
    __slots__ = ("id", "node_ids")

    def __init__(self, elem_id: int, node_ids):
        self.id = elem_id
        self.node_ids = node_ids


class _ElemProxyList:
    """Sequence of _ElemProxy objects built from Rust element_ids (and optionally node_ids)."""

    def __init__(self, ids, node_ids_per_elem=None):
        self._ids = ids
        self._nids = node_ids_per_elem  # list[list[int]] or None

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        if self._nids is not None:
            for eid, nids in zip(self._ids, self._nids):
                yield _ElemProxy(eid, nids)
        else:
            for eid in self._ids:
                yield _ElemProxy(eid, [])

    def __getitem__(self, idx):
        nids = self._nids[idx] if self._nids is not None else []
        return _ElemProxy(self._ids[idx], nids)


class MeshAssembler:
    def __init__(self, mesh: MeshModel, model: Dict):
        """
        Finite Element assembler using PETSc for distributed sparse matrices.

        Parameters
        ----------
        mesh : MeshModel
            The computational mesh containing nodes and elements.
            Kept only for the `_mesh_to_rust()` conversion; not stored after init.
        model : Dict
            Material and element configuration dictionary

        Attributes
        ----------
        dofs_count : int
            Total number of degrees of freedom in the system
        """
        self.model = model["elements"]
        self.comm = MPI.COMM_WORLD
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self._row_nnz: Optional[np.ndarray] = None
        self._rho_per_elem: Optional[np.ndarray] = None
        self._rust_mesh = None   # RustMeshModel — set in _build_py_mesh_assembler
        self._rust = None        # PyMeshAssembler — set in _build_py_mesh_assembler
        # Lazy caches for Rust-backed mesh proxies
        self._node_id_to_index_cache: Optional[Dict] = None
        self._node_dofs_map_cache: Optional[Dict] = None

        t0 = time.perf_counter()

        # Resolve dofs_per_node / spatial_dim from element_family declaration
        element_family = self.model.get("element_family")
        if element_family is not None and element_family in self._FAMILY_PROPERTIES:
            self.dofs_per_node, self.spatial_dim = self._FAMILY_PROPERTIES[element_family]
        else:
            raise ValueError(
                f"element_family must be one of {list(self._FAMILY_PROPERTIES.keys())} "
                f"(got {element_family!r})."
            )

        logger.info("[assembler] START __init__ — nodes=%d", mesh.node_count)

        t3 = time.perf_counter()
        self._build_py_mesh_assembler(mesh)
        logger.info("[assembler] _build_py_mesh_assembler done in %.2fs — rust=%s", time.perf_counter() - t3, self._rust is not None)

        # dofs_count: prefer Rust authoritative value, fall back to geometry
        if self._rust is not None:
            self.dofs_count = self._rust.dofs_count
        else:
            self.dofs_count = mesh.node_count * self.dofs_per_node

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

    # ------------------------------------------------------------------
    # Rust-backed mesh property delegates
    # These replace direct access to self.mesh (Python MeshModel) from solvers.
    # All data comes from self._rust_mesh (RustMeshModel) when available,
    # falling back to self.mesh for legacy callers.
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Total number of nodes in the mesh."""
        if self._rust_mesh is not None:
            return self._rust_mesh.node_count
        return self.mesh.node_count

    @property
    def node_id_to_index(self) -> Dict:
        """Dict mapping node id → 0-based index (lazy, cached)."""
        if self._node_id_to_index_cache is None:
            if self._rust_mesh is not None:
                ids = self._rust_mesh.node_ids()
                self._node_id_to_index_cache = {nid: i for i, nid in enumerate(ids)}
            else:
                self._node_id_to_index_cache = self.mesh.node_id_to_index
        return self._node_id_to_index_cache

    @property
    def nodes(self):
        """Iterable of node proxies with .id, .x, .y, .z, .coords attributes.

        When _rust_mesh is available the data comes from Rust (no Python MeshModel needed).
        """
        if self._rust_mesh is not None:
            ids = self._rust_mesh.node_ids()
            coords_flat = self._rust_mesh.node_coords_flat()
            return _NodeProxyList(ids, coords_flat)
        return self.mesh.nodes

    @property
    def coords_array(self) -> np.ndarray:
        """Node coordinates as (n_nodes, 3) numpy array (row = node index order)."""
        if self._rust_mesh is not None:
            flat = np.asarray(self._rust_mesh.node_coords_flat(), dtype=np.float64)
            return flat.reshape(-1, 3)
        return self.mesh.coords_array

    @property
    def elements(self):
        """Iterable of element proxies with .id and .node_ids attributes."""
        if self._rust_mesh is not None:
            return _ElemProxyList(
                self._rust_mesh.element_ids(),
                self._rust_mesh.element_node_ids() if hasattr(self._rust_mesh, "element_node_ids") else None,
            )
        return self.mesh.elements

    @property
    def _node_dofs_map(self) -> Dict:
        """Dict mapping node id → tuple of global DOF indices (lazy, cached)."""
        if self._node_dofs_map_cache is None:
            dpn = self.dofs_per_node
            if dpn == 0:
                return {}
            if self._rust_mesh is not None:
                ids = self._rust_mesh.node_ids()
                self._node_dofs_map_cache = {
                    nid: tuple(range(i * dpn, i * dpn + dpn))
                    for i, nid in enumerate(ids)
                }
        return self._node_dofs_map_cache

    def _build_py_mesh_assembler(self, mesh: MeshModel):
        """Convert mesh + model config to PyMeshAssembler (Rust).

        Uses `PyMeshAssembler.from_model()` for the composite / multi-property
        path (the normal production path).  Falls back to the direct
        `PyMeshAssembler()` constructor for the legacy single-material isotropic
        path.

        The Python MeshModel `mesh` is consumed here; it is NOT stored on `self`.
        """
        try:
            from _aeroelast import PyMeshAssembler, MeshModel as RustMeshModel, Laminate as RustLaminate, Ply as RustPly, OrthotropicMaterial as RustMat  # noqa: PLC0415
        except ImportError:
            logger.warning("[assembler] _aeroelast not available — Rust assembler disabled")
            return

        properties_map: Optional[Dict] = self.model.get("properties")

        # ── COMPOSITE / MULTI-PROPERTY PATH ───────────────────────────────────
        if properties_map is not None:
            _t = time.perf_counter()

            # ── Convert Python MeshModel → RustMeshModel ──────────────────────
            nodes = mesh.nodes
            node_ids_list = [n.id for n in nodes]
            coords_flat = np.stack([n.coords for n in nodes], axis=0).ravel().tolist()

            elements = mesh.elements
            element_ids_arr = np.fromiter((e.id for e in elements), dtype=np.int64, count=len(elements))
            node_counts_arr = np.fromiter((e.node_count for e in elements), dtype=np.int8, count=len(elements))

            try:
                _RustLaminate = RustLaminate
            except NameError:
                _RustLaminate = None
            try:
                from aeroelast.core.properties import CompositeShellProperty as _CSP  # noqa: PLC0415
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

            is_tri = node_counts_arr == 3
            type_codes = np.where(
                composite_arr,
                np.where(is_tri, 33, 44),
                np.where(is_tri, 3, 4),
            ).tolist()

            rust_esets = {name: [e.id for e in eset.elements] for name, eset in mesh.element_sets.items()}
            rust_nsets = {name: list(nset.node_ids) for name, nset in mesh.node_sets.items()}

            rust_mesh = RustMeshModel.from_raw_data(
                node_ids_list,
                coords_flat,
                element_ids_arr.tolist(),
                [list(e.node_ids) for e in elements],
                type_codes,
                rust_esets,
                rust_nsets,
            )
            self._rust_mesh = rust_mesh

            # ── Convert properties_map to Rust-native types ───────────────────
            try:
                from aeroelast.core.properties import CompositeShellProperty, ShellProperty  # noqa: PLC0415
                _has_py_props = True
            except ImportError:
                _has_py_props = False

            rust_properties = {}
            for set_name, prop in properties_map.items():
                if isinstance(prop, (RustLaminate, dict)):
                    rust_properties[set_name] = prop
                elif _has_py_props and isinstance(prop, CompositeShellProperty):
                    lam = prop.laminate
                    rust_plies = []
                    for ply in lam.plies:
                        m = ply.material
                        rust_mat = RustMat(
                            float(m.E[0]), float(m.E[1]), float(m.E[2]),
                            float(m.G[0]), float(m.G[1]), float(m.G[2]),
                            float(m.nu[0]), float(m.nu[1]), float(m.nu[2]),
                            float(m.rho),
                        )
                        rust_plies.append(RustPly(rust_mat, float(ply.thickness), float(ply.angle)))
                    rust_properties[set_name] = RustLaminate(rust_plies, float(getattr(lam, 'shear_correction_factor', 0.75)))
                elif _has_py_props and isinstance(prop, ShellProperty):
                    m = prop.material
                    rust_properties[set_name] = {
                        "type": "isotropic",
                        "e": float(m.E), "nu": float(m.nu), "rho": float(m.rho),
                        "thickness": float(prop.thickness),
                        "shear_correction": float(getattr(prop, 'shear_correction', 5.0 / 6.0)),
                        "drilling_scale": float(getattr(prop, 'drilling_scale', 1.0)),
                    }
                else:
                    rust_properties[set_name] = prop

            span_dir_raw = self.model.get("span_direction")
            span_dir_list = list(span_dir_raw) if span_dir_raw is not None else None
            self._rust = PyMeshAssembler.from_model(rust_mesh, rust_properties, span_dir_list, None)
            self._row_nnz = np.asarray(self._rust.nnz_per_row(), dtype=PETSc.IntType)
            self._rho_per_elem = np.asarray(self._rust.rho_per_elem(), dtype=np.float64)
            logger.info("[assembler] from_model() done in %.2fs", time.perf_counter() - _t)
            return

        # ── LEGACY ISOTROPIC PATH ──────────────────────────────────────────────
        logger.info("[assembler] isotropic path — %d nodes", mesh.node_count)
        _t0 = time.perf_counter()

        node_id_to_index = mesh.node_id_to_index
        n_nodes = mesh.node_count
        node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for node in mesh.nodes:
            idx = node_id_to_index[node.id]
            node_coords[idx, :] = node.coords[:3]

        fallback_material = self.model.get("material")
        fallback_thickness = self.model.get("thickness", 1.0)
        fallback_sc = self.model.get("shear_correction", 5.0 / 6.0)
        fallback_drill = self.model.get("drilling_scale", 1.0)
        fallback_family = self.model.get("element_family")

        elements = mesh.elements
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
                    "e": float(mat.E), "nu": float(mat.nu), "rho": float(mat.rho),
                    "thickness": float(fallback_thickness),
                    "shear_correction": float(fallback_sc),
                    "drilling_scale": float(fallback_drill),
                }
            else:
                logger.error("[assembler] no property for element %d (index %d) — aborting", element.id, i)
                self._rust = None
                return

            elem_types.append(code)
            materials_list.append(mat_dict)

        self._rho_per_elem = np.array(
            [m.get("rho", m.get("mass_per_area", 0.0)) for m in materials_list], dtype=np.float64
        )

        try:
            _t_rust = time.perf_counter()
            self._rust = PyMeshAssembler(node_coords, connectivity, elem_types, materials_list)
            logger.info("[assembler] PyMeshAssembler() constructed in %.2fs", time.perf_counter() - _t_rust)
            self._row_nnz = np.asarray(self._rust.nnz_per_row(), dtype=PETSc.IntType)
        except Exception as exc:  # noqa: BLE001
            logger.error("[assembler] PyMeshAssembler() FAILED: %s", exc)
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
            elements = self.elements
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
            "Ensure _aeroelast is built and PyMeshAssembler initialised."
        )

    # ------------------------------------------------------------------
    # Nonlinear assembly (Rust-accelerated)
    # ------------------------------------------------------------------

    def assemble_tangent_stiffness(self, u: np.ndarray) -> PETSc.Mat:
        """Assemble the global tangent stiffness matrix K_T(u).

        Uses Rust batch computation (``_aeroelast``) for parallel element
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
            "Rust backend (_aeroelast) required for tangent stiffness assembly. "
            "Install with: cd crates/_aeroelast && maturin develop --release"
        )

    def assemble_internal_forces(
        self, u: np.ndarray, nonlinear: bool = True
    ) -> PETSc.Vec:
        """Assemble the global internal force vector f_int(u).

        Uses Rust batch computation (``_aeroelast``) for parallel element
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
            "Rust backend (_aeroelast) required for internal force assembly. "
            "Install with: cd crates/_aeroelast && maturin develop --release"
        )
