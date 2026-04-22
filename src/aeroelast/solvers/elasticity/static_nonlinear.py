"""Nonlinear static solver — PETSc SNES Newton-Raphson via Rust assembler."""

import logging
from typing import List, Optional

import numpy as np

from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.solver import Solver

_log = logging.getLogger(__name__)


class StaticNonlinearSolver(Solver):
    """
    Nonlinear static solver using PETSc SNES (Newton-Raphson) via Rust.

    Solves R(u) = F_int(u) - F_ext = 0 through geometric nonlinearity
    using the tangent stiffness K_T(u) and internal forces F_int(u).

    Assembly and the SNES loop run entirely in Rust/PETSc.
    The Python layer orchestrates BCs, post-processing and convergence
    parameters forwarded from the YAML ``solver`` block.

    YAML solver parameters (all optional)
    --------------------------------------
    atol : float
        Absolute residual tolerance for SNES (default: 1e-10).
    rtol : float
        Relative residual tolerance for SNES (default: 1e-8).
    stol : float
        Step-length tolerance for SNES (default: 1e-8).
    max_it : int
        Maximum Newton iterations (default: 50).
    """

    _DEFAULT_ATOL: float = 1e-10
    _DEFAULT_RTOL: float = 1e-8
    _DEFAULT_STOL: float = 1e-8
    _DEFAULT_MAX_IT: int = 50

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)
        self.u: Optional[np.ndarray] = None
        self._iterations: int = 0
        self._residual_norm: float = 0.0
        self._converged_reason: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> np.ndarray:
        """
        Run the nonlinear static solve via SNES.

        Returns
        -------
        np.ndarray
            Full displacement vector (n_dofs,).

        Raises
        ------
        RuntimeError
            If the Rust assembler is not available or SNES diverges.
        """
        import _aeroelast  # noqa: PLC0415 — optional Rust extension

        if self.domain._rust is None:
            raise RuntimeError(
                "StaticNonlinearSolver requires the Rust assembler. "
                "Make sure _aeroelast is built and the mesh is assembled."
            )

        f_ext = self._build_f_ext()
        dirichlet_dofs = self._collect_dirichlet_dofs()
        params = self.solver_params if isinstance(self.solver_params, dict) else {}

        atol = float(params.get("atol", self._DEFAULT_ATOL))
        rtol = float(params.get("rtol", self._DEFAULT_RTOL))
        stol = float(params.get("stol", self._DEFAULT_STOL))
        max_it = int(params.get("max_it", self._DEFAULT_MAX_IT))

        u_arr, iters, res_norm, conv_reason = _aeroelast.nonlinear_static_solve_coo(
            self.domain._rust,
            f_ext,
            dirichlet_dofs,
            atol,
            rtol,
            stol,
            max_it,
        )

        self.u = np.asarray(u_arr, dtype=np.float64)
        self._iterations = iters
        self._residual_norm = res_norm
        self._converged_reason = conv_reason

        _log.info(
            "SNES converged in %d iterations, |R|=%.3e, reason=%d",
            iters,
            res_norm,
            conv_reason,
        )

        return self.u

    def print_solver_info(self) -> None:
        """Print nonlinear solver statistics to stdout."""
        print("\n--- Nonlinear Solver (SNES / Newton-Raphson) ---")
        print(f"  Converged reason : {self._converged_reason}")
        print(f"  Iterations       : {self._iterations}")
        print(f"  Final |R|        : {self._residual_norm:.3e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_f_ext(self) -> np.ndarray:
        """Assemble the external force vector from body + nodal loads."""
        n = self.domain.dofs_count
        f = np.zeros(n, dtype=np.float64)

        # Distributed/body loads
        for force in self.body_forces:
            fe = self.domain.assemble_load_vector(force)
            f += np.asarray(fe.getArray(), dtype=np.float64)

        # Concentrated nodal loads
        for load in self.nodal_loads:
            dofs = np.asarray(load.dofs, dtype=np.int64)
            vals = np.asarray(load.force, dtype=np.float64)
            if len(dofs) != len(vals):
                raise ValueError(
                    "Nodal load dof/value length mismatch in nonlinear solver: "
                    f"{len(dofs)} != {len(vals)}"
                )
            f[dofs] += vals

        return f

    def _collect_dirichlet_dofs(self) -> np.ndarray:
        """Collect all constrained DOF indices from Dirichlet BCs."""
        dofs: List[int] = []
        for bc in self.dirichlet_conditions:
            dofs.extend(bc.dofs)
        return np.array(dofs, dtype=np.int64)
