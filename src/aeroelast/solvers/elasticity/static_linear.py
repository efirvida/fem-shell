"""Linear static solver — PETSc KSP CG + ICC via Rust assembler."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.solver import Solver

_log = logging.getLogger(__name__)


class StaticLinearSolver(Solver):
    """
    Linear static solver using PETSc KSP (CG + ICC) via Rust.

    Solves K·u = F where K is the stiffness matrix and F is the load vector.
    Assembly and the KSP solve run entirely in Rust/PETSc.

    YAML solver parameters (all optional)
    --------------------------------------
    (no extra parameters; KSP tolerances are fixed: rtol=1e-8, atol=1e-12,
    max_it=1000.  Override at runtime via -ksp_* PETSc options.)
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)
        self.u: Optional[np.ndarray] = None
        self._extra_forces: List[Tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_force_on_dofs(self, dofs: List[int], value) -> None:
        """Add concentrated forces to specific DOFs.

        Parameters
        ----------
        dofs : List[int]
            DOF indices. Can be a flat list for individual assignment or a
            grouped list used together with a repeating ``value`` pattern.
        value : float | List[float]
            Values to apply. If a sequence, it is tiled to match ``len(dofs)``.
        """
        dofs_np = np.asarray(dofs, dtype=np.int64)
        if hasattr(value, "__len__"):
            vals_np = np.tile(value, len(dofs) // len(value)).astype(np.float64)
        else:
            vals_np = np.full(len(dofs), float(value), dtype=np.float64)
        self._extra_forces.append((dofs_np, vals_np))

    def solve(self) -> np.ndarray:
        """
        Run the linear static solve via KSP.

        Returns
        -------
        np.ndarray
            Full displacement vector (n_dofs,).

        Raises
        ------
        RuntimeError
            If the Rust assembler is not available or KSP does not converge.
        """
        import _aeroelast  # noqa: PLC0415 — optional Rust extension

        if self.domain._rust is None:
            raise RuntimeError(
                "StaticLinearSolver requires the Rust assembler. "
                "Make sure _aeroelast is built and the mesh is assembled."
            )

        k_rows, k_cols, k_vals = self.domain._rust.assemble_k()
        n_dof = self.domain.dofs_count

        f = self._build_f_ext()

        free_dofs = self._compute_free_dofs(n_dof)

        u_arr = _aeroelast.linear_static_solve_coo(
            k_rows.astype(np.int64),
            k_cols.astype(np.int64),
            k_vals.astype(np.float64),
            f,
            n_dof,
            free_dofs,
        )

        self.u = np.asarray(u_arr, dtype=np.float64)
        _log.info("KSP converged — n_free=%d", len(free_dofs))

        return self.u

    def print_solver_info(self, plot_residuals: bool = True) -> None:
        """Print linear solver status to stdout."""
        print("\n--- Linear Solver (KSP / CG + ICC) ---")
        if self.u is not None:
            print(f"  Solution DOFs  : {self.u.shape[0]}")
            print(f"  Max |u|        : {np.abs(self.u).max():.3e}")
        else:
            print("  (not solved yet)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_f_ext(self) -> np.ndarray:
        """Assemble the external force vector from body + nodal + extra loads."""
        n = self.domain.dofs_count
        f = np.zeros(n, dtype=np.float64)

        for force in self.body_forces:
            fe = self.domain.assemble_load_vector(force)
            f += np.asarray(fe.getArray(), dtype=np.float64)

        for load in self.nodal_loads:
            dofs = np.asarray(load.dofs, dtype=np.int64)
            vals = np.asarray(load.force, dtype=np.float64)
            if len(dofs) != len(vals):
                raise ValueError(
                    "Nodal load dof/value length mismatch in linear solver: "
                    f"{len(dofs)} != {len(vals)}"
                )
            f[dofs] += vals

        for dofs, vals in self._extra_forces:
            f[dofs] += vals

        return f

    def _compute_free_dofs(self, n_dof: int) -> np.ndarray:
        """Return sorted array of unconstrained DOF indices."""
        constrained: set = set()
        for bc in self.dirichlet_conditions:
            constrained.update(bc.dofs)
        return np.array(sorted(set(range(n_dof)) - constrained), dtype=np.int64)
