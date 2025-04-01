"""
Finite Element Method Boundary Condition Manager for 2D/3D Elasticity Problems.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


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


class DirichletCondition:
    """Represents a Dirichlet boundary condition (fixed DOFs) in a FEM system.

    Parameters
    ----------
    dofs : Iterable[int]
        Global degree of freedom indices (0-based) where the condition is applied.
    value : float
        Fixed displacement value imposed on the specified DOFs.

    Attributes
    ----------
    dofs : tuple[int]
        Sorted unique global DOF indices where the condition is applied.
    value : float
        Prescribed displacement value at the specified DOFs.

    Examples
    --------
    >>> bc = DirichletCondition([0, 1], 0.0)  # Fix DOFs 0 and 1 at 0 displacement
    """

    def __init__(self, dofs: Iterable[int], value: float):
        self.dofs = tuple(sorted(set(dofs)))
        self.value = value


class BoundaryConditionManager:
    """Handles boundary conditions and system reduction for FEM elasticity problems.

    Features:
    - Handles both 2D (plane strain/stress) and 3D problems
    - Validates DOF organization and boundary conditions
    - Efficient system reduction and solution expansion
    - Non-homogeneous Dirichlet condition support

    Parameters
    ----------
    stiffness : np.ndarray
        Global stiffness matrix (n_dof, n_dof)
    load : np.ndarray
        Global load vector (n_dof,)
    mass : Optional[np.ndarray], optional
        Global mass matrix (n_dof, n_dof), by default None
    dof_per_node : int
        Degrees of freedom per node (2 for 2D, 3 for 3D)

    Attributes
    ----------
    n_dof : int
        Total number of degrees of freedom in the system
    free_dofs : np.ndarray
        Indices of unconstrained degrees of freedom
    fixed_dofs : Dict[int, float]
        Constrained DOFs with their prescribed values

    Raises
    ------
    ValueError
        If input matrices have inconsistent dimensions
        If invalid DOFs are specified in boundary conditions

    Examples
    --------
    >>> K = np.eye(4)
    >>> F = np.zeros(4)
    >>> bcm = BoundaryConditionManager(K, F, dof_per_node=2)
    >>> bcm.apply_dirichlet([DirichletCondition([0, 1], 0.0)])
    >>> K_red, F_red, _ = bcm.reduced_system
    """

    def __init__(
        self,
        stiffness: np.ndarray,
        load: np.ndarray,
        mass: Optional[np.ndarray] = None,
        dof_per_node: int = 2,
    ):
        self._validate_inputs(stiffness, load, mass)

        self._K = stiffness.copy()
        self._F = load.copy()
        self._M = mass.copy() if mass is not None else None
        self.dof_per_node = dof_per_node
        self.n_dof = stiffness.shape[0]

        self._fixed_dofs: Dict[int, float] = {}
        self._free_dofs = np.arange(self.n_dof)
        self._dirichlet_applied = False

    @staticmethod
    def _validate_inputs(
        stiffness: np.ndarray, load: np.ndarray, mass: Optional[np.ndarray]
    ) -> None:
        """Validate matrix dimensions and properties."""
        if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
            raise ValueError("Stiffness matrix must be square")

        if load.ndim != 1:
            raise ValueError("Load vector must be 1-dimensional")

        if stiffness.shape[0] != load.shape[0]:
            raise ValueError("Stiffness and load dimensions mismatch")

        if mass is not None and mass.shape != stiffness.shape:
            raise ValueError("Mass matrix must match stiffness dimensions")

    def apply_dirichlet(self, conditions: Iterable[DirichletCondition]) -> None:
        """Apply Dirichlet boundary conditions to the system.

        Parameters
        ----------
        conditions : Iterable[DirichletCondition]
            Boundary conditions to apply

        Raises
        ------
        ValueError
            If invalid DOFs are specified or conflicting values are provided
        """
        fixed_dofs = {}
        for bc in conditions:
            for dof in bc.dofs:
                self._validate_dof(dof)
                if dof in fixed_dofs and not np.isclose(fixed_dofs[dof], bc.value):
                    raise ValueError(
                        f"Conflicting values for DOF {dof}: {fixed_dofs[dof]} vs {bc.value}"
                    )
                fixed_dofs[dof] = bc.value

        self._fixed_dofs = dict(sorted(fixed_dofs.items()))
        self._free_dofs = np.setdiff1d(np.arange(self.n_dof), list(self._fixed_dofs))
        self._validate_2d_constraints()

    def _validate_dof(self, dof: int) -> None:
        """Validate DOF index."""
        if not 0 <= dof < self.n_dof:
            raise ValueError(f"DOF {dof} out of range [0, {self.n_dof - 1}]")

    def _validate_2d_constraints(self) -> None:
        """Ensure no Z-axis constraints in 2D problems."""
        if self.dof_per_node == 2:
            invalid_dofs = [dof for dof in self._fixed_dofs if dof % 2 == 2]
            if invalid_dofs:
                raise ValueError(f"Attempted to constrain Z-axis DOFs {invalid_dofs} in 2D problem")

    @property
    def reduced_system(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get reduced system matrices (K_red, F_red, M_red)."""
        K_red = self._K[np.ix_(self._free_dofs, self._free_dofs)]
        F_red = self._adjusted_load_vector()
        M_red = (
            self._M[np.ix_(self._free_dofs, self._free_dofs)]
            if (self._M is not None and self._M.size > 0)
            else None
        )
        return K_red, F_red, M_red

    def _adjusted_load_vector(self) -> np.ndarray:
        """Calculate adjusted load vector accounting for fixed displacements."""
        if not self._fixed_dofs:
            return self._F[self._free_dofs]

        fixed_indices = list(self._fixed_dofs.keys())
        U_fixed = np.array([self._fixed_dofs[dof] for dof in fixed_indices])
        K_free_fixed = self._K[np.ix_(self._free_dofs, fixed_indices)]

        return self._F[self._free_dofs] - K_free_fixed @ U_fixed

    def expand_solution(self, u_red: np.ndarray) -> np.ndarray:
        """Expand reduced solution vector to full system DOFs.

        Parameters
        ----------
        u_red : np.ndarray
            Solution vector from reduced system

        Returns
        -------
        np.ndarray
            Full displacement vector with constrained DOFs inserted

        Raises
        ------
        ValueError
            If solution dimensions mismatch or invalid displacements in 2D
        """
        if u_red.shape[0] != self._free_dofs.size:
            raise ValueError("Solution vector size doesn't match free DOFs count")

        u_full = np.zeros(self.n_dof)
        u_full[self._free_dofs] = u_red
        for dof, val in self._fixed_dofs.items():
            u_full[dof] = val

        return u_full

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

    @property
    def free_dofs(self) -> np.ndarray:
        """Indices of unconstrained degrees of freedom."""
        return self._free_dofs.copy()

    @property
    def fixed_dofs(self) -> Dict[int, float]:
        """Dictionary of constrained DOFs with prescribed values."""
        return self._fixed_dofs.copy()
