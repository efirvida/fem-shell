"""
Finite Element Method Boundary Condition Manager for 2D/3D Elasticity Problems.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from petsc4py import PETSc


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
    """Handles boundary conditions and system reduction for FEM problems using PETSc.

    Parameters
    ----------
    stiffness : PETSc.Mat
        Global stiffness matrix in sparse format
    load : PETSc.Vec
        Global load vector
    mass : Optional[PETSc.Mat], optional
        Global mass matrix, by default None
    dof_per_node : int
        Degrees of freedom per node (2 for 2D, 3 for 3D)

    Attributes
    ----------
    comm : PETSc.Comm
        MPI communicator
    n_dof : int
        Total number of degrees of freedom
    free_dofs : PETSc.IS
        Index set of unconstrained degrees of freedom
    fixed_dofs : Dict[int, float]
        Constrained DOFs with prescribed values
    """

    def __init__(
        self,
        stiffness: PETSc.Mat,
        load: PETSc.Vec,
        mass: Optional[PETSc.Mat] = None,
        dof_per_node: int = 2,
    ):
        self._validate_inputs(stiffness, load, mass)

        self.K = stiffness
        self.F = load.duplicate()
        self.F.array = load.array.copy()
        self.M = mass
        self.comm = stiffness.getComm()
        self.dof_per_node = dof_per_node
        self.n_dof = stiffness.getSize()[0]

        self._fixed_dofs: Dict[int, float] = {}
        self._free_is: Optional[PETSc.IS] = None
        self._fixed_is: Optional[PETSc.IS] = None

    @staticmethod
    def _validate_inputs(stiffness: PETSc.Mat, load: PETSc.Vec, mass: Optional[PETSc.Mat]) -> None:
        """Validate matrix dimensions and properties."""
        if stiffness.getSize()[0] != stiffness.getSize()[1]:
            raise ValueError("Stiffness matrix must be square")

        if stiffness.getSize()[0] != load.getSize():
            raise ValueError("Stiffness and load dimensions mismatch")

        if mass and mass.getSize() != stiffness.getSize():
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

        self._fixed_dofs = fixed_dofs
        self._create_index_sets()
        self._apply_bc_to_system()

    def _validate_dof(self, dof: int) -> None:
        """Validate DOF index."""
        if not 0 <= dof < self.n_dof:
            raise ValueError(f"DOF {dof} out of range [0, {self.n_dof - 1}]")

    def _create_index_sets(self) -> None:
        """Create PETSc index sets for free and fixed DOFs."""
        all_dofs = np.arange(self.n_dof, dtype=PETSc.IntType)
        fixed_dofs_array = np.array(list(self._fixed_dofs.keys()), dtype=PETSc.IntType)
        free_dofs_array = np.setdiff1d(all_dofs, fixed_dofs_array)

        self._free_is = PETSc.IS().createGeneral(free_dofs_array, comm=self.comm)
        self._fixed_is = PETSc.IS().createGeneral(fixed_dofs_array, comm=self.comm)

    def _apply_bc_to_system(self) -> None:
        """Apply BCs to matrices and vectors using PETSc operations."""
        # Apply to stiffness matrix
        fixed_indices = np.array(list(self._fixed_dofs.keys()), dtype=PETSc.IntType)
        fixed_values = np.array(list(self._fixed_dofs.values()), dtype=PETSc.ScalarType)

        U_fixed = self.K.createVecRight()
        U_fixed.setValues(fixed_indices, fixed_values)
        U_fixed.assemble()

        self.K.zeroRows(fixed_indices, 1.0, U_fixed, self.F)
        self.K.assemble()

        # Apply to mass matrix if present
        if self.M:
            self.M.zeroRows(fixed_indices, 0.0)
            self.M.assemble()

        U_fixed.destroy()

    @property
    def reduced_system(self) -> Tuple[PETSc.Mat, PETSc.Vec, Optional[PETSc.Mat]]:
        """Get reduced system matrices and vectors.

        Returns
        -------
        Tuple[PETSc.Mat, PETSc.Vec, Optional[PETSc.Mat]]
            (K_red, F_red, M_red) Reduced system components
        """
        if not self._free_is:
            raise RuntimeError("Boundary conditions not applied")

        K_red = self.K.createSubMatrix(self._free_is, self._free_is)
        F_red = self._adjusted_load_vector()
        M_red = self.M.createSubMatrix(self._free_is, self._free_is) if self.M else None

        K_red.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        K_red.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)

        return K_red, F_red, M_red

    def _adjusted_load_vector(self) -> PETSc.Vec:
        """Calculate adjusted load vector accounting for fixed displacements."""
        F_red = self.F.getSubVector(self._free_is)

        if not self._fixed_dofs:
            return F_red

        # 1. Obtener índices y valores fijos en el orden correcto
        fixed_indices = self._fixed_is.getIndices()  # Orden real de los DOFs fijos
        fixed_size = len(fixed_indices)
        fixed_values = np.array(
            [self._fixed_dofs[d] for d in fixed_indices], dtype=PETSc.ScalarType
        )

        # 2. Crear vector U_fixed con tamaño correcto
        U_fixed = PETSc.Vec().create(self.comm)
        U_fixed.setSizes(fixed_size)  # Tamaño = número de DOFs fijos
        U_fixed.setUp()

        # 3. Asignar valores usando índices locales (0-based)
        local_indices = np.arange(fixed_size, dtype=PETSc.IntType)
        U_fixed.setValues(local_indices, fixed_values)
        U_fixed.assemble()

        # 4. Crear matriz de acoplamiento y validar dimensiones
        K_free_fixed = self.K.createSubMatrix(self._free_is, self._fixed_is)

        if K_free_fixed.getSize()[1] != fixed_size:
            raise ValueError(
                f"Dimensiones inconsistentes: "
                f"Matriz {K_free_fixed.getSize()[1]} vs Vector {fixed_size}"
            )

        # 5. Calcular término de corrección
        correction = F_red.duplicate()
        K_free_fixed.mult(U_fixed, correction)
        F_red.axpy(-1.0, correction)

        # 6. Liberar recursos
        K_free_fixed.destroy()
        U_fixed.destroy()
        return F_red

    def expand_solution(self, u_red: PETSc.Vec) -> PETSc.Vec:
        """Expand reduced solution vector to full system DOFs.

        Parameters
        ----------
        u_red : PETSc.Vec
            Solution vector from reduced system

        Returns
        -------
        PETSc.Vec
            Full solution vector with fixed DOFs inserted

        Notes
        -----
        Handles parallel distribution of vectors
        """
        # Crear vector completo distribuido
        u_full = self.K.createVecRight()
        u_full.set(0.0)  # Inicializar a cero

        # 1. Copiar solución reducida usando el index set
        free_is = self._free_is
        local_free_size = free_is.getLocalSize()

        # Obtener indices libres locales
        free_indices = free_is.getIndices()

        # Obtener array local de la solución reducida
        u_red_local = u_red.getArray(readonly=True)

        # Copiar valores a las posiciones libres
        u_full_local = u_full.getArray()
        u_full_local[free_indices - u_full.getOwnershipRange()[0]] = u_red_local[:local_free_size]

        # 2. Aplicar valores fijos en la porción local
        local_start, local_end = u_full.getOwnershipRange()
        for dof, val in self._fixed_dofs.items():
            if local_start <= dof < local_end:
                u_full_local[dof - local_start] = val

        # 3. Sincronizar entre procesos
        u_full.assemble()

        return u_full

    def reduce_vector(self, vector: PETSc.Vec) -> PETSc.Vec:
        """Reduce arbitrary vector using current free DOFs.

        Parameters
        ----------
        vector : PETSc.Vec
            Full system vector to reduce

        Returns
        -------
        PETSc.Vec
            Reduced vector
        """
        return vector.getSubVector(self._free_is)

    @property
    def free_dofs(self) -> np.ndarray:
        """Indices of unconstrained degrees of freedom."""
        return self._free_is.getIndices() if self._free_is else np.array([])

    @property
    def fixed_dofs(self) -> Dict[int, float]:
        """Dictionary of constrained DOFs with prescribed values."""
        return self._fixed_dofs.copy()
