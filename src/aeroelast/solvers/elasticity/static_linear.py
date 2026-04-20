"""Linear static solver — KSP CG + BoomerAMG preconditioner (PETSc)."""

import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import meshio
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from aeroelast.core.bc import BodyForce, BoundaryConditionManager
from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.solver import Solver


class StaticLinearSolver(Solver):
    """
    High-performance linear static solver using PETSc KSP.

    Solves K·u = F with CG + BoomerAMG for distributed systems.

    Parameters
    ----------
    mesh : MeshModel
        The computational mesh model.
    fem_model_properties : dict
        Dictionary with material and element properties.

    Attributes
    ----------
    comm : MPI.Comm
        MPI communicator for parallel processing.
    K : PETSc.Mat
        Distributed stiffness matrix.
    F : PETSc.Vec
        Distributed load vector.
    u : PETSc.Vec
        Solution displacement vector.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)
        self.comm = MPI.COMM_WORLD
        self.K: PETSc.Mat = None
        self.F: PETSc.Vec = None
        self.u: PETSc.Vec = None
        self._solver: PETSc.KSP = None
        self._prepared = False
        self._applyed_forces = False
        self._residual_history = []

    def add_force_on_dofs(self, dofs: List[int], value: List[float]):
        """Add concentrated forces to specific DOFs in distributed system.

        Parameters
        ----------
        dofs : List[int]
            Lista de grados de libertad a modificar. Puede ser:
            - Lista plana para asignación individual
            - Lista agrupada para asignación vectorial
        value : List[float]
            Valores a aplicar. La longitud determina el agrupamiento:
            - len(value) = 1: asigna el mismo valor a todos los dofs
            - len(value) > 1: agrupa los dofs en bloques de este tamaño
        """
        if self.F is None:
            self._initialize_vectors()
        dofs_np = np.asarray(dofs, dtype=PETSc.IntType)
        if isinstance(value, Iterable):
            values_np = np.tile(value, len(dofs) // len(value)).astype(PETSc.ScalarType)
        else:
            values_np = np.tile(value, len(dofs)).astype(PETSc.ScalarType)

        self.F.setValues(dofs_np, values_np, addv=PETSc.InsertMode.ADD_VALUES)
        self.F.assemble()
        self._applyed_forces = True

    def _initialize_vectors(self):
        """Initialize PETSc vectors with proper parallel layout."""
        self.F = PETSc.Vec().create(self.comm)
        self.F.setSizes(self.domain.dofs_count)
        self.F.setUp()
        self.F.zeroEntries()

    def _setup_solver(self):
        """Configure PETSc KSP with CG + BoomerAMG."""
        self._solver = PETSc.KSP().create(self.comm)
        self._residual_history = []
        self._solver.setType("cg")

        pc = self._solver.getPC()
        pc.setType("hypre")

        opts = PETSc.Options()
        opts["pc_hypre_type"] = "boomeramg"
        opts["pc_hypre_boomeramg_coarsen_type"] = "HMIS"
        opts["pc_hypre_boomeramg_interp_type"] = "classical"
        opts["pc_hypre_boomeramg_relax_type_all"] = "symmetric-sor/jacobi"
        opts["pc_hypre_boomeramg_strong_threshold"] = 0.5
        opts["pc_hypre_boomeramg_max_levels"] = 5
        opts["pc_hypre_boomeramg_print_statistics"] = 0

        self._solver.setMonitor(self._residual_monitor)
        self._solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
        self._solver.setFromOptions()

    def solve(self) -> PETSc.Vec:
        """
        Solve the linear static FEM problem.

        Returns
        -------
        PETSc.Vec
            Distributed solution vector.
        """
        self.K = self.domain.assemble_stiffness_matrix()

        if not self._applyed_forces:
            self._initialize_vectors()
            for force in self.body_forces:
                fe_vector = self.domain.assemble_load_vector(force)
                self.F.axpy(1.0, fe_vector)

        bc_manager = BoundaryConditionManager(
            self.K, self.F, dof_per_node=self.domain.dofs_per_node
        )
        bc_manager.apply_dirichlet(self.dirichlet_conditions)

        K_red, F_red, _ = bc_manager.reduced_system

        if not self._prepared:
            self._setup_solver()
            self._solver.setOperators(K_red)
            self._prepared = True

        u_red = K_red.createVecRight()
        self._solver.solve(F_red, u_red)

        self.u = bc_manager.expand_solution(u_red)

        K_red.destroy()
        F_red.destroy()
        u_red.destroy()

        return self.u

    def _residual_monitor(self, ksp, iteration, residual):
        """PETSc callback function for residual monitoring."""
        if iteration == 0:
            self._residual_history = []
        self._residual_history.append(residual)

    def print_solver_info(self, plot_residuals=True):
        """Print solver statistics and optionally plot residuals."""
        if self.comm.rank == 0 and self._solver:
            print("\n--- Solver Performance ---")
            print(f"Converged Reason: {self._solver.getConvergedReason()}")
            print(f"Iterations: {self._solver.getIterationNumber()}")
            print(f"Final Residual: {self._solver.getResidualNorm():.3e}")

            if plot_residuals and self._residual_history:
                try:
                    plt.figure(figsize=(10, 5))
                    plt.semilogy(self._residual_history, "bo-")
                    plt.title("Residual Convergence History")
                    plt.xlabel("Iteration")
                    plt.ylabel("Residual Norm (log scale)")
                    plt.grid(True, which="both")
                    plt.show()
                except ImportError:
                    print("\nResidual History:")
                    print(
                        "\n".join(
                            f"Iter {i:3d}: {r:.3e}"
                            for i, r in enumerate(self._residual_history)
                        )
                    )
