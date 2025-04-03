from typing import Tuple

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.solver import Solver


class ModalSolver(Solver):
    """
    Clase para análisis modal usando SLEPc/PETSc de manera eficiente
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        self.num_modes = fem_model_properties["solver"].get("num_modes", 2)
        super().__init__(mesh, fem_model_properties)

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        # Ensamblar matrices directamente en formato PETSc
        K = self.domain.assemble_stiffness_matrix()
        M = self.domain.assemble_mass_matrix()

        F = PETSc.Vec().create()
        F.setSizes(self.domain.dofs_count)
        F.setUp()
        F.zeroEntries()

        # Aplicar condiciones de contorno manteniendo las matrices en PETSc
        self.bc_applier = BoundaryConditionManager(K, F, M)
        self.bc_applier.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = self.bc_applier.reduced_system

        eff_num_modes = min(self.num_modes + 5, self.domain.dofs_count)

        eps = SLEPc.EPS().create()
        eps.setOperators(K_red, M_red)  # K primero, M segundo
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # <-- Usar TARGET_MAGNITUDE
        eps.setTarget(0.0)  # <-- Necesario para el shift-and-invert

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)  # Shift en 0 para buscar valores propios pequeños

        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")

        eps.setDimensions(eff_num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-10, max_it=1000)
        eps.setFromOptions()

        eps.solve()

        nconv = eps.getConverged()
        if nconv < self.num_modes:
            raise RuntimeError(f"Solo {nconv} modos convergieron de {num_modes} solicitados.")

        # Verificar convergencia
        nconv = eps.getConverged()
        if nconv < self.num_modes:
            raise RuntimeError(f"Convergencia insuficiente: {nconv}/{self.num_modes} modos")

        # Extraer resultados
        eigvals, eigvecs = self._extract_eigenpairs(eps, nconv, K_red)
        return self._postprocess_results(eigvals, eigvecs)

    def _extract_eigenpairs(
        self, eps: SLEPc.EPS, nconv: int, K_red
    ) -> Tuple[np.ndarray, np.ndarray]:
        eigvals = []
        eigvecs = []
        for i in range(nconv):
            eigval = eps.getEigenpair(i)
            eigvals.append(eigval.real)

            # Obtener vector propio en espacio reducido
            vec = PETSc.Vec().createWithArray(np.zeros(K_red.getSize()[0]))
            eps.getEigenvector(i, vec)
            eigvecs.append(vec.array.copy())

        return np.array(eigvals), np.column_stack(eigvecs)

    def _postprocess_results(
        self, eigvals: np.ndarray, eigvecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Filtrado y ordenamiento
        valid = eigvals > 1e-8
        eigvals = eigvals[valid]
        eigvecs = eigvecs[:, valid]

        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Expandir vectores al espacio completo
        full_modes = np.zeros((self.domain.dofs_count, eigvecs.shape[1]))
        full_modes[self.bc_applier.free_dofs, :] = eigvecs

        # Calcular frecuencias
        frequencies = np.sqrt(eigvals) / (2 * np.pi)

        return frequencies[: self.num_modes], full_modes[:, : self.num_modes]
