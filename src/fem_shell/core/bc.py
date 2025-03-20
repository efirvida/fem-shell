from typing import Iterable, List, Tuple

import numpy as np


class DirichletCondition:
    """Represents a Dirichlet boundary condition (fixed DOFs) for a finite element system.

    Parameters
    ----------
    dofs : Iterable[int]
        Global degree of freedom indices where the condition is applied.
    value : float
        Fixed displacement value imposed on the specified DOFs.

    Attributes
    ----------
    dofs : tuple[int]
        Global DOF indices where the condition is applied.
    value : float
        Fixed displacement value at the specified DOFs.
    """

    def __init__(self, dofs: Iterable[int], value: float):
        self.dofs = tuple(sorted(set(dofs)))
        self.value = value


class BodyForce:
    """Represents a distributed body force (Neumann condition) for a finite element system.

    Parameters
    ----------
    value : Iterable[float]
        Force values applied to element DOFs. The length should match the number
        of element DOFs.

    Attributes
    ----------
    value : ndarray
        Force vector applied to element DOFs.
    """

    def __init__(self, value: Iterable[float]):
        self.value = np.asarray(value)


class BoundaryConditionApplier:
    """Handles application of boundary conditions to FEM system components.

    Parameters
    ----------
    stiffness_matrix : np.ndarray
        Global stiffness matrix of shape (n_dof, n_dof)
    load_vector : np.ndarray
        Global load vector of shape (n_dof,)
    mass_matrix : Optional[np.ndarray], optional
        Global mass matrix of shape (n_dof, n_dof), by default None

    Attributes
    ----------
    fixed_dofs : Set[int]
        Indices of constrained degrees of freedom
    free_dofs : np.ndarray
        Indices of unconstrained degrees of freedom
    """

    def __init__(
        self,
        stiffness_matrix: np.ndarray,
        load_vector: np.ndarray,
        mass_matrix: np.ndarray | None = None,
    ):
        self.stiffness_matrix = stiffness_matrix.copy()
        self.load_vector = load_vector.copy()
        self.mass_matrix = (
            mass_matrix.copy() if mass_matrix is not None else np.zeros_like(stiffness_matrix)
        )
        self.fixed_dofs: set[int] = set()
        self.free_dofs: np.ndarray = np.array([], dtype=int)

    def apply_dirichlet(self, dirichlet_conditions: List["DirichletCondition"]):
        """Apply Dirichlet boundary conditions to system components.

        Parameters
        ----------
        dirichlet_conditions : List[DirichletCondition]
            List of Dirichlet boundary conditions to apply
        """
        # Collect all fixed DOFs
        self.fixed_dofs = set()
        for bc in dirichlet_conditions:
            self.fixed_dofs.update(bc.dofs)

        # Update free DOFs
        all_dofs = set(range(self.stiffness_matrix.shape[0]))
        self.free_dofs = np.array(sorted(all_dofs - self.fixed_dofs), dtype=int)

        # Apply BCs to each system component
        for dof in self.fixed_dofs:
            # Stiffness matrix modifications
            self.stiffness_matrix[dof, :] = 0.0
            self.stiffness_matrix[:, dof] = 0.0
            self.stiffness_matrix[dof, dof] = 1.0

            # Mass matrix modifications
            self.mass_matrix[dof, :] = 0.0
            self.mass_matrix[:, dof] = 0.0
            self.mass_matrix[dof, dof] = 1.0

            # Load vector modifications
            self.load_vector[dof] = 0.0

        return self.stiffness_matrix, self.load_vector, self.mass_matrix

    def get_reduced_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get reduced system matrices eliminating fixed DOFs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Reduced stiffness matrix, load vector, and mass matrix
        """
        K_red = self.stiffness_matrix[np.ix_(self.free_dofs, self.free_dofs)]
        F_red = self.load_vector[self.free_dofs]
        M_red = self.mass_matrix[np.ix_(self.free_dofs, self.free_dofs)]
        return K_red, F_red, M_red

    def reduce_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Reduce arbitrary matrix using current free DOFs.

        Parameters
        ----------
        matrix : np.ndarray
            Full system matrix to reduce

        Returns
        -------
        np.ndarray
            Reduced matrix
        """
        return matrix[np.ix_(self.free_dofs, self.free_dofs)]

    def reduce_vector(self, vector: np.ndarray) -> np.ndarray:
        """Reduce arbitrary vector using current free DOFs.

        Parameters
        ----------
        vector : np.ndarray
            Full system vector to reduce

        Returns
        -------
        np.ndarray
            Reduced vector
        """
        return vector[self.free_dofs]
