from typing import Dict, Optional

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.mesh import MeshModel
from fem_shell.elements import ElementFactory, FemElement


class MeshAssembler:
    def __init__(self, mesh: MeshModel, model: Dict):
        """
        Finite Element assembler using PETSc for distributed sparse matrices.

        Parameters
        ----------
        mesh : MeshModel
            The computational mesh containing nodes and elements
        model : Dict
            Material and element configuration dictionary

        Attributes
        ----------
        dofs_count : int
            Total number of degrees of freedom in the system
        _dofs_array : np.ndarray
            Element-to-DOF connectivity array
        _ke_array : np.ndarray
            Precomputed local stiffness matrices
        _me_array : np.ndarray
            Precomputed local mass matrices
        """
        self.mesh = mesh
        self.model = model["elements"]
        self.comm = MPI.COMM_WORLD
        self._element_map: Dict[int, FemElement] = {}
        self._dofs_array: np.ndarray = None
        self._ke_array: np.ndarray = None
        self._me_array: np.ndarray = None
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self._row_nnz: Optional[np.ndarray] = None
        self._precompute_elements()
        self._compute_sparsity_pattern()

    def _precompute_elements(self):
        """Precompute element matrices and DOF connectivity arrays."""
        elements = self.mesh.elements
        if not elements:
            return

        dofs_list = []
        ke_list = []
        me_list = []

        for element in elements:
            fem_element = ElementFactory.get_element(mesh_element=element, **self.model)
            if not fem_element:
                continue

            dofs = np.array(
                [dof for node in fem_element.global_dof_indices.values() for dof in node],
                dtype=np.int64,
            )

            self._element_map[element.id] = fem_element
            dofs_list.append(dofs)
            ke_list.append(fem_element.K)
            me_list.append(fem_element.M)

            if not self.dofs_per_node:
                self.dofs_per_node = fem_element.dofs_per_node
                self.spatial_dim = fem_element.spatial_dimmension

        self._dofs_array = np.array(dofs_list, dtype=np.int64)
        self._ke_array = np.array(ke_list, dtype=np.float64)
        self._me_array = np.array(me_list, dtype=np.float64)
        self.dofs_count = self.mesh.node_count * self.dofs_per_node

    def _compute_sparsity_pattern(self):
        """
        Compute the sparse matrix non-zero pattern for efficient preallocation.

        Notes
        -----
        Determines the number of non-zeros per matrix row using element
        connectivity information. Critical for PETSc matrix performance.
        """
        nnz = [set() for _ in range(self.dofs_count)]
        for elem_dofs in self._dofs_array:
            for dof_i in elem_dofs:
                nnz[dof_i].update(dof_j for dof_j in elem_dofs)

        self._row_nnz = np.array([len(s) for s in nnz], dtype=PETSc.IntType)

    def _create_petsc_matrix(self) -> PETSc.Mat:
        """
        Create a PETSc sparse matrix with optimized memory preallocation.

        Returns
        -------
        PETSc.Mat
            A sparse matrix configured for efficient assembly

        Notes
        -----
        Uses AIJ format (Compressed Sparse Row) by default. For better
        GPU performance consider setting type to 'seqaijcusparse'
        """
        mat = PETSc.Mat().create(self.comm)
        mat.setType("aij")
        mat.setSizes([self.dofs_count, self.dofs_count])

        d_nnz = self._row_nnz.astype(PETSc.IntType)
        o_nnz = np.zeros_like(d_nnz)  # Ajustar según particionado paralelo

        mat.setPreallocationNNZ((d_nnz, o_nnz))
        mat.setUp()
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        return mat

    def assemble_stiffness_matrix(self) -> PETSc.Mat:
        """
        Assemble the global stiffness matrix using PETSc.

        Returns
        -------
        PETSc.Mat
            Distributed sparse stiffness matrix

        Notes
        -----
        Performs parallel assembly using local element contributions.
        Matrix entries are accumulated using ADD_VALUES mode.
        """
        K = self._create_petsc_matrix()
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        for e in range(self._dofs_array.shape[0]):
            dofs = self._dofs_array[e].astype(PETSc.IntType)
            ke = self._ke_array[e].flatten(order="C")  # Row-major flattening

            # Use block insertion for better performance
            K.setValuesLocal(dofs, dofs, ke, addv=PETSc.InsertMode.ADD_VALUES)

        K.assemble()
        return K

    def assemble_mass_matrix(self) -> PETSc.Mat:
        """
        Assemble the global mass matrix using PETSc.

        Returns
        -------
        PETSc.Mat
            Distributed sparse mass matrix
        """
        M = self._create_petsc_matrix()
        M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        for e in range(self._dofs_array.shape[0]):
            dofs = self._dofs_array[e].astype(PETSc.IntType)
            me = self._me_array[e].flatten(order="C")

            M.setValuesLocal(dofs, dofs, me, addv=PETSc.InsertMode.ADD_VALUES)

        M.assemble()
        return M

    def assemble_load_vector(self, load_condition) -> PETSc.Vec:
        """
        Assemble the global load vector using PETSc.

        Parameters
        ----------
        load_condition : LoadCondition
            The loading condition to apply

        Returns
        -------
        PETSc.Vec
            Distributed load vector

        Notes
        -----
        Supports both nodal and distributed loading conditions.
        """
        f = PETSc.Vec().create(self.comm)
        f.setSizes(self.dofs_count)
        f.setUp()
        f.zeroEntries()

        fe_list = [
            self._element_map[eid].body_load(load_condition.value) for eid in self._element_map
        ]
        fe_array = np.array(fe_list, dtype=PETSc.ScalarType)

        for e in range(fe_array.shape[0]):
            dofs = self._dofs_array[e].astype(PETSc.IntType)
            fe = fe_array[e]

            # Use local-to-global mapping if using mesh partitioning
            f.setValuesLocal(dofs, fe, addv=PETSc.InsertMode.ADD_VALUES)

        f.assemble()
        return f
