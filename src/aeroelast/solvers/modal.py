import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, ElementTree

import meshio
import numpy as np

from aeroelast.core.bc import BoundaryConditionManager
from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.solver import Solver

logger = logging.getLogger(__name__)


class ModalSolver(Solver):
    """Eigenvalue solver for natural frequencies and mode shapes.

    Solves the generalized eigenvalue problem  K φ = ω² M φ  for the lowest
    modes using Rust/SLEPc via _aeroelast.modal_solve_coo.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        self.num_modes = fem_model_properties["solver"].get("num_modes", 2)
        super().__init__(mesh, fem_model_properties)
        self.participation_factors: Optional[np.ndarray] = None

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        from _aeroelast import modal_solve_coo  # Rust/PyO3 binding

        _t0 = time.perf_counter()
        logger.info("[modal] START solve() — num_modes=%d", self.num_modes)

        # ── Assemble K and M as COO triplets directly from the Rust assembler ──
        logger.info("[modal] assembling K (COO)...")
        _t = time.perf_counter()
        k_rows, k_cols, k_vals = self.domain._rust.assemble_k()
        n_dof_total = self.domain.dofs_count
        logger.info("[modal] K assembled in %.2fs — nnz=%d, n_dof=%d",
                    time.perf_counter() - _t, len(k_vals), n_dof_total)

        logger.info("[modal] assembling M (COO)...")
        _t = time.perf_counter()
        m_rows, m_cols, m_vals = self.domain._rust.assemble_m()
        logger.info("[modal] M assembled in %.2fs — nnz=%d", time.perf_counter() - _t, len(m_vals))

        # Mass sanity check using COO diagonal
        try:
            diag_mask = k_rows == k_cols
            m_trace = m_vals[m_rows == m_cols].sum()
            dofs_per_node = self.domain.dofs_per_node
            trans_fraction = 3.0 / dofs_per_node if dofs_per_node > 0 else 0.5
            total_mass_estimate = m_trace * trans_fraction * (9.0 / 4.0) / 3.0
            logger.info(
                "[modal] mass sanity check: tr(M)=%.1f, estimated total mass=%.1f kg",
                m_trace, total_mass_estimate,
            )
        except Exception as exc:
            logger.warning("[modal] mass sanity check failed: %s", exc)

        # ── Compute free DOFs via BoundaryConditionManager ────────────────────
        # We still use BoundaryConditionManager to get free_dofs — it only needs
        # to apply Dirichlet BCs conceptually; we pass dummy PETSc objects for
        # the K/M needed by its constructor.
        logger.info("[modal] computing free DOFs from BCs...")
        _t = time.perf_counter()
        from petsc4py import PETSc
        K_petsc = self.domain.assemble_stiffness_matrix()
        F_petsc = K_petsc.createVecRight()
        F_petsc.set(0.0)
        self.bc_applier = BoundaryConditionManager(
            K_petsc, F_petsc, None, self.domain.dofs_per_node
        )
        self.bc_applier.apply_dirichlet(self.dirichlet_conditions)
        free_dofs = np.array(sorted(self.bc_applier.free_dofs), dtype=np.int64)
        K_petsc.destroy()
        F_petsc.destroy()
        logger.info("[modal] free DOFs: %d of %d (%.2fs)",
                    len(free_dofs), n_dof_total, time.perf_counter() - _t)

        # ── Rust eigenvalue solve ─────────────────────────────────────────────
        logger.info("[modal] calling Rust modal_solve_coo — n_modes=%d", self.num_modes)
        _t = time.perf_counter()
        frequencies_hz, modes_flat = modal_solve_coo(
            k_rows.astype(np.int64), k_cols.astype(np.int64), k_vals.astype(np.float64),
            m_rows.astype(np.int64), m_cols.astype(np.int64), m_vals.astype(np.float64),
            n_dof_total,
            free_dofs,
            self.num_modes,
        )
        logger.info("[modal] Rust solve done in %.2fs — %d modes converged",
                    time.perf_counter() - _t, len(frequencies_hz))

        if len(frequencies_hz) < self.num_modes:
            raise RuntimeError(
                f"Convergence insufficient: {len(frequencies_hz)}/{self.num_modes} modes"
            )

        # modes_flat is (n_conv * n_free,) row-major — reshape to (n_free, n_conv)
        n_conv = len(frequencies_hz)
        n_free = len(free_dofs)
        eigvecs_red = modes_flat.reshape(n_conv, n_free).T  # (n_free, n_conv)

        # eigenvalues ω² = (2π·f)²
        eigvals = (2.0 * np.pi * frequencies_hz) ** 2

        # ── Participation factors (NumPy, uses M COO → dense diagonal approx) ─
        self.participation_factors = self._compute_participation_factors_coo(
            eigvals, eigvecs_red, free_dofs,
            m_rows, m_cols, m_vals, n_dof_total,
        )

        logger.info("[modal] total solve time: %.2fs", time.perf_counter() - _t0)

        # ── Expand mode shapes to full DOF space ─────────────────────────────
        full_modes = np.zeros((n_dof_total, n_conv))
        full_modes[free_dofs, :] = eigvecs_red

        return frequencies_hz[: self.num_modes], full_modes[:, : self.num_modes]

    def _compute_participation_factors_coo(
        self,
        eigvals: np.ndarray,
        eigvecs_red: np.ndarray,
        free_dofs: np.ndarray,
        m_rows: np.ndarray,
        m_cols: np.ndarray,
        m_vals: np.ndarray,
        n_dof_total: int,
    ) -> np.ndarray:
        """Compute effective mass participation factors using NumPy sparse M.

        Identical physics to the PETSc version, but uses scipy COO→CSR
        for sparse M·v products — no PETSc objects required.

        Returns array of shape ``(n_modes, n_rb_dirs)`` with percentage values (0–100).
        """
        from scipy.sparse import coo_matrix

        n_free = len(free_dofs)

        # Build free-DOF index map for restricting M to the reduced space
        free_map = {int(g): i for i, g in enumerate(free_dofs)}
        mask = np.array(
            [(r in free_map and c in free_map) for r, c in zip(m_rows, m_cols)],
            dtype=bool,
        )
        rr = np.array([free_map[int(r)] for r in m_rows[mask]], dtype=np.int32)
        cc = np.array([free_map[int(c)] for c in m_cols[mask]], dtype=np.int32)
        vv = m_vals[mask]
        M_red_sp = coo_matrix((vv, (rr, cc)), shape=(n_free, n_free)).tocsr()

        dpn = self.domain.dofs_per_node

        # Rigid-body vectors in full space, then restricted to free DOFs
        r_vectors = []
        n_trans = min(dpn, 3)
        for d in range(n_trans):
            r_full = np.zeros(n_dof_total)
            r_full[d::dpn] = 1.0
            r_vectors.append(r_full[free_dofs])

        if dpn >= 6:
            coords = self.mesh_obj.coords_array
            for d in range(3):
                r_full = np.zeros(n_dof_total)
                if d == 0:
                    r_full[1::dpn] = coords[:, 2]
                    r_full[2::dpn] = -coords[:, 1]
                    r_full[3::dpn] = 1.0
                elif d == 1:
                    r_full[0::dpn] = -coords[:, 2]
                    r_full[2::dpn] = coords[:, 0]
                    r_full[4::dpn] = 1.0
                else:
                    r_full[0::dpn] = -coords[:, 1]
                    r_full[1::dpn] = coords[:, 0]
                    r_full[5::dpn] = 1.0
                r_vectors.append(r_full[free_dofs])

        n_rb = len(r_vectors)
        m_total = np.array([r @ M_red_sp @ r for r in r_vectors])

        valid = eigvals > 1e-8
        vecs = eigvecs_red[:, valid]
        idx = np.argsort(eigvals[valid])
        vecs = vecs[:, idx]

        n_modes = min(vecs.shape[1], self.num_modes)
        factors = np.zeros((n_modes, n_rb))

        for i in range(n_modes):
            phi = vecs[:, i]
            Mphi = M_red_sp @ phi
            gen_mass = phi @ Mphi
            if gen_mass < 1e-30:
                continue
            for d in range(n_rb):
                gamma = np.dot(Mphi, r_vectors[d])
                m_eff = gamma**2 / gen_mass
                if m_total[d] > 1e-30:
                    factors[i, d] = m_eff / m_total[d] * 100.0

        return factors

    def write_modal_results(
        self,
        output_dir: str,
        frequencies: np.ndarray,
        mode_shapes: np.ndarray,
    ) -> None:
        """Write mode shapes as VTU files with a PVD collection.

        Creates one VTU per mode with displacement fields and a PVD
        index file for sequential browsing in ParaView.

        Parameters
        ----------
        output_dir : str
            Directory for output files.  Created if it does not exist.
        frequencies : np.ndarray
            Natural frequencies in Hz, shape ``(n_modes,)``.
        mode_shapes : np.ndarray
            Mode shape vectors, shape ``(n_dofs, n_modes)``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        vector_form = self.vector_form
        vector_components = [c for v in vector_form.values() for c in v]
        n_components = len(vector_components)
        dpn = self.domain.dofs_per_node

        points = self.mesh_obj.coords_array

        # Build meshio cells list
        cells = []
        for element in self.mesh_obj.element_map.values():
            if element:
                cells.append((element.element_type.name, [element.node_ids]))

        vtu_names = []
        for mode_idx in range(len(frequencies)):
            U = mode_shapes[:, mode_idx].reshape(-1, dpn)

            # Normalize so max translational displacement = 1.0
            trans = U[:, :min(dpn, 3)]
            mag = np.linalg.norm(trans, axis=1)
            max_mag = mag.max()
            if max_mag > 1e-30:
                U = U / max_mag

            point_data: Dict[str, np.ndarray] = {}

            for i, comp in enumerate(vector_components):
                if i < dpn:
                    point_data[comp] = U[:, i]

            for vec_name, components in vector_form.items():
                arr = np.column_stack(
                    [point_data[c] for c in components if c in point_data]
                )
                if arr.shape[1] == 2:
                    arr = np.column_stack([arr, np.zeros(arr.shape[0])])
                point_data[vec_name] = arr

            # Displacement magnitude scalar
            trans_norm = U[:, :min(dpn, 3)]
            point_data["UMAG"] = np.linalg.norm(trans_norm, axis=1)

            fname = f"mode_{mode_idx + 1:02d}.vtu"
            vtu_names.append(fname)
            mesh_obj = meshio.Mesh(points, cells=cells, point_data=point_data)
            mesh_obj.write(str(out / fname))

        # Write PVD collection file
        self._write_pvd(out / "modal_results.pvd", vtu_names, frequencies)

    @staticmethod
    def _write_pvd(
        pvd_path: Path,
        vtu_names: list,
        frequencies: np.ndarray,
    ) -> None:
        """Write a PVD (ParaView Data) collection file."""
        root = Element("VTKFile")
        root.set("type", "Collection")
        root.set("version", "0.1")
        collection = SubElement(root, "Collection")
        for i, (name, freq) in enumerate(zip(vtu_names, frequencies)):
            ds = SubElement(collection, "DataSet")
            ds.set("timestep", f"{freq:.6f}")
            ds.set("group", "")
            ds.set("part", "0")
            ds.set("file", name)

        tree = ElementTree(root)
        with open(pvd_path, "wb") as fh:
            tree.write(fh, xml_declaration=True)
