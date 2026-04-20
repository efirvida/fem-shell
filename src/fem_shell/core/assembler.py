import logging
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.mesh import MeshModel
from fem_shell.core.properties import CompositeShellProperty, ShellProperty, ShellPropertyType
from fem_shell.elements import ElementFamily

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _batch_angle_offsets(
    node_coords: np.ndarray,
    conn_array: np.ndarray,
    span_dir: np.ndarray,
) -> np.ndarray:
    """Compute span-direction angle offset (degrees) for a batch of 4-node elements.

    Fully vectorized: no Python loop over elements.

    Parameters
    ----------
    node_coords : (n_nodes, 3) global node coordinate array.
    conn_array  : (n_elem, 4) int array of 0-based node indices.
    span_dir    : (3,) span direction unit vector.

    Returns
    -------
    angle_offsets : (n_elem,) float array in degrees.
    """
    # pts[i] = node_coords[conn_array[:, i]]  — shape (n_elem, 3)
    p0 = node_coords[conn_array[:, 0]]
    p1 = node_coords[conn_array[:, 1]]
    p2 = node_coords[conn_array[:, 2]]
    p3 = node_coords[conn_array[:, 3]]

    # Normal from triangle 012
    v1 = p1 - p0
    v2 = p2 - p0
    n1 = np.cross(v1, v2)  # (n_elem, 3)
    n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)

    # Normal from triangle 023
    v1b = p2 - p0
    v2b = p3 - p0
    n2 = np.cross(v1b, v2b)  # (n_elem, 3)
    n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)

    # Average normals where valid; fall back to z-axis
    valid1 = (n1_norm > 1e-12).squeeze()
    valid2 = (n2_norm > 1e-12).squeeze()

    e3 = np.zeros_like(n1)
    e3[:, 2] = 1.0  # default z-axis

    n1_safe = np.where(n1_norm > 1e-12, n1 / np.maximum(n1_norm, 1e-30), 0.0)
    n2_safe = np.where(n2_norm > 1e-12, n2 / np.maximum(n2_norm, 1e-30), 0.0)

    both = valid1 & valid2
    only1 = valid1 & ~valid2
    only2 = ~valid1 & valid2

    e3[both] = (n1_safe[both] + n2_safe[both]) / 2.0
    e3[only1] = n1_safe[only1]
    e3[only2] = n2_safe[only2]

    norms_e3 = np.linalg.norm(e3, axis=1, keepdims=True)
    e3 = e3 / np.maximum(norms_e3, 1e-30)

    # e1 = edge 01 projected onto plane perpendicular to e3
    e1 = v1 - (np.sum(v1 * e3, axis=1, keepdims=True)) * e3
    e1_norm = np.linalg.norm(e1, axis=1, keepdims=True)
    # Fall back to edge 02 for degenerate cases
    fallback = (e1_norm < 1e-12).squeeze()
    if fallback.any():
        e1_fb = v2[fallback] - (np.sum(v2[fallback] * e3[fallback], axis=1, keepdims=True)) * e3[fallback]
        e1[fallback] = e1_fb
        e1_norm[fallback] = np.linalg.norm(e1_fb, axis=1, keepdims=True)
    e1 = e1 / np.maximum(e1_norm, 1e-30)

    e2 = np.cross(e3, e1)
    e2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)

    # Project span_dir onto element plane
    sd = span_dir[np.newaxis, :] - np.sum(span_dir * e3, axis=1, keepdims=True) * e3  # (n_elem, 3)
    sd_norm = np.linalg.norm(sd, axis=1)  # (n_elem,)

    cos_a = np.sum(sd * e1, axis=1) / np.maximum(sd_norm, 1e-30)
    sin_a = np.sum(sd * e2, axis=1) / np.maximum(sd_norm, 1e-30)
    angle_offsets = np.degrees(np.arctan2(sin_a, cos_a))

    # Elements where span_dir is parallel to normal — offset = 0
    angle_offsets[sd_norm < 1e-10] = 0.0

    return angle_offsets


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

    # Compute local orthonormal axes (e1, e2, e3) for this element
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


def _corrected_abd_cached(
    lam,
    angle_offset_deg: float,
    cache: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, D, Cs) for lam rotated by angle_offset_deg, using cache.

    Cache key is (id(lam), round(angle_offset_deg, 1)) — 0.1° resolution
    is sufficient given the blade's negligible variation (std ≈ 0.03°).
    """
    from fem_shell.core.laminate import Laminate as Lam, Ply as Pl  # noqa: PLC0415

    if abs(angle_offset_deg) < 0.01:
        key = (id(lam), 0.0)
        if key not in cache:
            cache[key] = (lam.A.copy(), lam.D.copy(), lam.Cs.copy())
        return cache[key]

    bucket = round(angle_offset_deg, 1)
    key = (id(lam), bucket)
    if key not in cache:
        corrected_plies = [
            Pl(material=ply.material, thickness=ply.thickness, angle=ply.angle + bucket)
            for ply in lam.plies
        ]
        corrected_lam = Lam(plies=corrected_plies)
        cache[key] = (corrected_lam.A.copy(), corrected_lam.D.copy(), corrected_lam.Cs.copy())
    return cache[key]


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

    def _build_py_mesh_assembler(self):
        """Build a PyMeshAssembler (Rust) directly from mesh topology and properties.

        Uses a two-phase strategy to eliminate the per-element Python overhead:

        Phase 1 — Vectorized angle offsets (numpy, no loop):
            For meshes with a span_direction and composite elements, compute all
            e1/e2/e3 local axes and angle offsets in a single batch numpy operation
            using ``_batch_angle_offsets``.

        Phase 2 — Cached laminate ABD matrices:
            Cache key is ``(id(lam), round(angle_offset, 1))`` — 0.1° resolution
            is sufficient (blade variation is std ≈ 0.03°).  In practice, the
            number of unique (lam, angle) pairs equals the number of unique laminates
            (~696 for the IEA-15 blade), not the number of elements (84889).
        """
        self._rust = None

        try:
            from fem_shell_core import PyMeshAssembler  # noqa: PLC0415
        except ImportError:
            logger.warning("[assembler._build_py_mesh_assembler] fem_shell_core not available — Rust assembler disabled")
            return

        logger.info("[assembler._build_py_mesh_assembler] building node_coords for %d nodes", self.mesh.node_count)
        _t0 = time.perf_counter()

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

        elements = self.mesh.elements
        n_elems = len(elements)

        # ------------------------------------------------------------------
        # Phase 1: Vectorized angle offsets for 4-node composite elements
        # ------------------------------------------------------------------
        # Build connectivity array (n_elems, 4) for 4-node elements only
        # (3-node elements fall back to per-element scalar path — rare in blade)
        angle_offsets_map: Dict[int, float] = {}  # elem_id → angle_offset_deg

        if span_dir is not None:
            # Separate 4-node composite elements for vectorized batch
            quad_indices = []  # index in elements list
            quad_conn_rows = []  # (4,) connectivity for each

            for i, element in enumerate(elements):
                prop = element_property_lookup.get(element.id)
                if isinstance(prop, CompositeShellProperty) and element.node_count == 4:
                    quad_indices.append(i)
                    conn = [node_id_to_index[nid] for nid in element.node_ids]
                    quad_conn_rows.append(conn)

            if quad_indices:
                _t_batch = time.perf_counter()
                conn_arr = np.array(quad_conn_rows, dtype=np.int64)  # (n_quad, 4)
                offsets = _batch_angle_offsets(node_coords, conn_arr, span_dir)  # (n_quad,)
                logger.info(
                    "[assembler._build_py_mesh_assembler] batch angle offsets: %d elements in %.3fs",
                    len(quad_indices), time.perf_counter() - _t_batch,
                )
                for i_elem, angle in zip(quad_indices, offsets):
                    angle_offsets_map[elements[i_elem].id] = float(angle)

        logger.info(
            "[assembler._build_py_mesh_assembler] phase1 done in %.2fs — %d angle offsets computed",
            time.perf_counter() - _t0, len(angle_offsets_map),
        )

        # ------------------------------------------------------------------
        # Phase 2: Build materials_list with cached ABD matrices
        # ------------------------------------------------------------------
        abd_cache: dict = {}  # (id(lam), angle_bucket) → (A, D, Cs)
        connectivity = []
        elem_types = []
        materials_list = []

        _t_loop = time.perf_counter()

        for i, element in enumerate(elements):
            conn = [node_id_to_index[nid] for nid in element.node_ids]
            connectivity.append(conn)

            prop = element_property_lookup.get(element.id)
            n_nodes_elem = element.node_count

            if isinstance(prop, CompositeShellProperty):
                code = 33 if n_nodes_elem == 3 else 44
                lam = prop.laminate
                h = lam.total_thickness

                if span_dir is not None:
                    if element.id in angle_offsets_map:
                        # Fast path: use pre-computed vectorized offset + cache
                        angle_offset = angle_offsets_map[element.id]
                        A, D, Cs = _corrected_abd_cached(lam, angle_offset, abd_cache)
                    else:
                        # Slow path: 3-node or not in batch (scalar fallback)
                        A, D, Cs = _apply_span_direction(node_coords, conn, span_dir, lam)
                else:
                    key = (id(lam), 0.0)
                    if key not in abd_cache:
                        abd_cache[key] = (lam.A.copy(), lam.D.copy(), lam.Cs.copy())
                    A, D, Cs = abd_cache[key]

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
                    logger.error(
                        "[assembler._build_py_mesh_assembler] no property for element %d (index %d) — aborting Rust assembler",
                        element.id, i,
                    )
                    self._rust = None
                    return

            elem_types.append(code)
            materials_list.append(mat_dict)

        logger.info(
            "[assembler._build_py_mesh_assembler] phase2 loop done in %.2fs (%d elements, %d ABD cache hits avoided)",
            time.perf_counter() - _t_loop, n_elems, n_elems - len(abd_cache),
        )

        # Cache per-element rho and node connectivity for vectorized centrifugal prestress
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
            # Compute sparsity pattern from Rust (replaces scipy _compute_sparsity_pattern)
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
