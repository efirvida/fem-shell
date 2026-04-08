from typing import Tuple

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.solver import Solver


class ModalSolver(Solver):
    """Eigenvalue solver for natural frequencies and mode shapes.

    Uses SLEPc shift-and-invert spectral transformation to solve the
    generalized eigenvalue problem  K φ = ω² M φ  for the lowest modes.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        self.num_modes = fem_model_properties["solver"].get("num_modes", 2)
        super().__init__(mesh, fem_model_properties)
        self.comm = PETSc.COMM_WORLD

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        K = self.domain.assemble_stiffness_matrix()
        M = self.domain.assemble_mass_matrix()

        F = K.createVecRight()
        F.set(0.0)

        self.bc_applier = BoundaryConditionManager(
            K, F, M, self.domain.dofs_per_node
        )
        self.bc_applier.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = self.bc_applier.reduced_system

        n_red = K_red.getSize()[0]
        eff_num_modes = min(self.num_modes + 5, n_red)

        # Configure SLEPc eigensolver
        eps = SLEPc.EPS().create(self.comm)
        eps.setOperators(K_red, M_red)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(0.0)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)

        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("petsc")

        eps.setDimensions(eff_num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-10, max_it=1000)
        eps.setFromOptions()

        try:
            eps.solve()
        except Exception as exc:
            self._cleanup(eps, K_red, M_red, F_red, F)
            raise RuntimeError(
                f"SLEPc eigenvalue solve failed: {exc}"
            ) from exc

        nconv = eps.getConverged()
        if nconv < self.num_modes:
            self._cleanup(eps, K_red, M_red, F_red, F)
            raise RuntimeError(
                f"Convergence insufficient: {nconv}/{self.num_modes} modes"
            )

        eigvals, eigvecs = self._extract_eigenpairs(eps, nconv, K_red)

        self._cleanup(eps, K_red, M_red, F_red, F)
        return self._postprocess_results(eigvals, eigvecs)

    def _extract_eigenpairs(
        self, eps: SLEPc.EPS, nconv: int, K_red: PETSc.Mat
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_red = K_red.getSize()[0]
        vr = K_red.createVecRight()
        vi = K_red.createVecRight()

        eigvals = []
        eigvecs = []
        for i in range(nconv):
            eigval = eps.getEigenpair(i, vr, vi)
            eigvals.append(eigval.real)
            eigvecs.append(vr.array.copy())

        vr.destroy()
        vi.destroy()

        return np.array(eigvals), np.column_stack(eigvecs)

    def _postprocess_results(
        self, eigvals: np.ndarray, eigvecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid = eigvals > 1e-8
        eigvals = eigvals[valid]
        eigvecs = eigvecs[:, valid]

        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        full_modes = np.zeros((self.domain.dofs_count, eigvecs.shape[1]))
        full_modes[self.bc_applier.free_dofs, :] = eigvecs

        frequencies = np.sqrt(eigvals) / (2 * np.pi)

        return frequencies[: self.num_modes], full_modes[:, : self.num_modes]

    @staticmethod
    def _cleanup(eps, *mats_and_vecs):
        eps.destroy()
        for obj in mats_and_vecs:
            if obj is not None:
                obj.destroy()
