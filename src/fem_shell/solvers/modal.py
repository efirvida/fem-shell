from typing import Tuple

import numpy as np
import scipy
import scipy.linalg

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.solver import Solver


class ModalSolver(Solver):
    """
    A class to perform modal analysis on a finite element model.

    Parameters
    ----------
    mesh : MeshModel
        The mesh model containing nodes, elements, and node sets.
    fem_model_properties : dict
        Dictionary containing the properties of the finite element model.
        Required keys:
        - "material": Material properties.
        - "element_family": Type of element family (e.g., SHELL, BEAM).
        - "thickness": Thickness of the element (required for SHELL elements).
        - "num_modes": Number of modes to compute (optional, default is 5).
    bcs : List[BoundaryCondition]
        List of boundary conditions to apply to the model.

    Raises
    ------
    KeyError
        If any of the required keys are missing in `fem_model_properties`.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        self.num_modes = fem_model_properties["solver"].get("num_modes", 2)
        super().__init__(mesh, fem_model_properties)

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform modal analysis with improved numerical stability.
        """

        K = self.domain.assemble_stiffness_matrix()
        M = self.domain.assemble_mass_matrix()

        F = np.zeros(self.domain.dofs_count)
        self.bc_applier = BoundaryConditionManager(K, F, M)
        self.bc_applier.apply_dirichlet(self.dirichlet_conditions)
        K_reduced, _, M_reduced = self.bc_applier.get_reduced_system()

        eigvals, eigvecs = scipy.linalg.eigh(
            K_reduced, M_reduced, subset_by_index=(0, self.num_modes - 1)
        )

        frequencies, mode_shapes = self._postprocess_results(eigvals, eigvecs)
        return frequencies, mode_shapes

    def _postprocess_results(
        self, eigvals: np.ndarray, eigvecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process eigenvalues and eigenvectors into physical quantities.
        """
        valid = eigvals > 1e-8
        eigvals = eigvals[valid]
        eigvecs = eigvecs[:, valid]

        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        frequencies = np.sqrt(eigvals) / (2 * np.pi)

        # Expandir los modos al espacio completo de DOFs
        full_mode_shapes = np.zeros((self.domain.dofs_count, eigvecs.shape[1]))
        full_mode_shapes[self.bc_applier.free_dofs, :] = eigvecs

        return frequencies[: self.num_modes], full_mode_shapes[:, : self.num_modes]
