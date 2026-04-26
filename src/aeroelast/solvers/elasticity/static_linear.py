"""Linear static solver — KSP CG + preconditioner (Rust/PETSc or petsc4py)."""

import logging
import time
from typing import List, Optional

import numpy as np
from petsc4py import PETSc

from aeroelast.core.bc import BoundaryConditionManager
from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.solver import Solver

logger = logging.getLogger(__name__)


class StaticLinearSolver(Solver):
    """
    Linear static solver using Rust/PETSc via _aeroelast.linear_static_solve_coo.

    Solves K·u = F with CG + ICC for symmetric positive definite stiffness matrices.
    Falls back to petsc4py CG + GAMG if the Rust solver is unavailable or diverges.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)

    def solve(self) -> np.ndarray:
        """
        Solve the linear static FEM problem. Tries Rust first, falls back to petsc4py.

        Returns
        -------
        np.ndarray
            Displacement vector at all DOFs, shape (n_dof,).
        """
        try:
            import importlib
            _aeroelast = importlib.import_module("_aeroelast")
            linear_static_solve_coo = getattr(_aeroelast, "linear_static_solve_coo")
            logger.info("[static_linear] Using Rust linear_static_solve_coo")
            return self._solve_rust(linear_static_solve_coo)
        except Exception as e:
            logger.warning(
                "[static_linear] Rust unavailable or failed: %s, using petsc4py fallback", e
            )
            return self._solve_petsc4py()

    def _solve_rust(self, linear_static_solve_coo) -> np.ndarray:
        """Solve using Rust extension."""
        _t0 = time.perf_counter()
        logger.info("[static_linear] START solve()")

        # Override the hardcoded ICC with GAMG — same as the petsc4py fallback.
        # The Rust KSP calls KSPSetFromOptions, so these are picked up automatically.
        opts = PETSc.Options()
        opts["pc_type"] = "gamg"
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 1
        opts["pc_gamg_threshold"] = 0.05
        opts["pc_gamg_coarse_eq_limit"] = 1000
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "sor"
        opts["mg_levels_ksp_max_it"] = 2

        n_dof_total = self.domain.dofs_count

        k_rows, k_cols, k_vals = self.domain._rust.assemble_k()

        F_full = np.zeros(n_dof_total, dtype=np.float64)
        for force in self.body_forces:
            f_vec = self.domain.assemble_load_vector(force)
            F_full += np.asarray(f_vec.getArray(), dtype=np.float64)
        for load in self.nodal_loads:
            for dof, val in zip(load.dofs, load.force):
                F_full[dof] += val

        K_petsc = self.domain.assemble_stiffness_matrix()
        F_dummy = K_petsc.createVecRight()
        F_dummy.set(0.0)
        bc_manager = BoundaryConditionManager(
            K_petsc, F_dummy, None, self.domain.dofs_per_node
        )
        bc_manager.apply_dirichlet(self.dirichlet_conditions)
        free_dofs = np.array(sorted(bc_manager.free_dofs), dtype=np.int64)
        K_petsc.destroy()
        F_dummy.destroy()

        displacements = linear_static_solve_coo(
            k_rows.astype(np.int64),
            k_cols.astype(np.int64),
            k_vals.astype(np.float64),
            F_full,
            n_dof_total,
            free_dofs,
        )
        logger.info("[static_linear] Rust solve done in %.2fs", time.perf_counter() - _t0)
        return displacements

    def _solve_petsc4py(self) -> np.ndarray:
        """Fallback solve using pure petsc4py (CG + GAMG)."""
        from mpi4py import MPI

        _t0 = time.perf_counter()
        logger.info("[static_linear] START petsc4py solve()")

        comm = MPI.COMM_WORLD
        n_dof_total = self.domain.dofs_count

        K = self.domain.assemble_stiffness_matrix()

        F = PETSc.Vec().create(comm)
        F.setSizes(n_dof_total)
        F.setUp()
        F.set(0.0)

        for force in self.body_forces:
            f_vec = self.domain.assemble_load_vector(force)
            F.axpy(1.0, f_vec)
        if self.nodal_loads:
            all_dofs = np.concatenate(
                [np.asarray(load.dofs, dtype=PETSc.IntType) for load in self.nodal_loads]
            )
            all_vals = np.concatenate(
                [np.asarray(load.force, dtype=PETSc.ScalarType) for load in self.nodal_loads]
            )
            F.setValues(all_dofs, all_vals, addv=PETSc.InsertMode.ADD_VALUES)
        F.assemble()

        bc_manager = BoundaryConditionManager(K, F, dof_per_node=self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)

        K_red, F_red, _ = bc_manager.reduced_system

        solver = PETSc.KSP().create(comm)
        solver.setType("cg")
        pc = solver.getPC()
        pc.setType("gamg")
        solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=5000)
        solver.setFromOptions()
        solver.setOperators(K_red)

        u_red = K_red.createVecRight()
        solver.solve(F_red, u_red)

        u = bc_manager.expand_solution(u_red)

        K_red.destroy()
        F_red.destroy()
        u_red.destroy()
        solver.destroy()

        logger.info("[static_linear] petsc4py solve done in %.2fs", time.perf_counter() - _t0)
        return np.asarray(u.getArray(), dtype=np.float64)

    def print_solver_info(self, plot_residuals: bool = True) -> None:
        """Print solver statistics."""
        logger.info("[static_linear] solve() completed — see logs above")
