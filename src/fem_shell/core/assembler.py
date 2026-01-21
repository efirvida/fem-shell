from typing import Dict, Iterable, Optional

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
        self._node_dofs_map: Dict[int, Iterable] = {}
        self._ke_array: np.ndarray = None
        self._me_array: np.ndarray = None
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self._row_nnz: Optional[np.ndarray] = None
        self._precompute_elements()
        self._compute_sparsity_pattern()

    def _precompute_elements(self):
        """Precompute element matrices and DOF connectivity arrays.

        Uses mesh.node_id_to_index mapping to ensure DOF indices are consecutive
        starting from 0, regardless of the original node IDs in the mesh.
        This is critical for merged meshes where node IDs may not be consecutive.
        
        Supports mixed-element meshes where elements may have different numbers
        of nodes (e.g., pyramid + tetrahedron meshes). In this case, stores
        lists instead of uniform numpy arrays.
        """
        elements = self.mesh.elements
        if not elements:
            return

        dofs_list = []
        ke_list = []
        me_list = []
        dof_sizes = set()  # Track if we have variable-size elements

        # Get the node ID to index mapping from the mesh
        node_id_to_index = self.mesh.node_id_to_index

        for element in elements:
            fem_element = ElementFactory.get_element(mesh_element=element, **self.model)
            if not fem_element:
                continue

            # Remap DOFs using mesh's node_id_to_index to ensure consecutive indices
            # Original global_dof_indices uses node_id directly, which fails for
            # merged meshes with non-consecutive node IDs
            remapped_dof_indices = {}
            for node_id in fem_element.node_ids:
                node_index = node_id_to_index[node_id]
                start_dof = node_index * fem_element.dofs_per_node
                end_dof = start_dof + fem_element.dofs_per_node
                remapped_dof_indices[node_id] = tuple(range(start_dof, end_dof))

            self._node_dofs_map.update(remapped_dof_indices)

            dofs = np.array(
                [dof for node_id in fem_element.node_ids for dof in remapped_dof_indices[node_id]],
                dtype=np.int64,
            )
            dof_sizes.add(len(dofs))

            self._element_map[element.id] = fem_element
            dofs_list.append(dofs)
            ke_list.append(fem_element.K)
            me_list.append(fem_element.M)

            if not self.dofs_per_node:
                self.dofs_per_node = fem_element.dofs_per_node
                self.spatial_dim = fem_element.spatial_dimmension

        # Check if all elements have the same DOF count (uniform mesh)
        self._is_mixed_mesh = len(dof_sizes) > 1
        
        if self._is_mixed_mesh:
            # Mixed mesh: store as lists (variable-size arrays not supported by numpy)
            self._dofs_list = dofs_list
            self._ke_list = ke_list
            self._me_list = me_list
            self._dofs_array = None
            self._ke_array = None
            self._me_array = None
        else:
            # Uniform mesh: store as numpy arrays for efficiency
            self._dofs_array = np.array(dofs_list, dtype=np.int64)
            self._ke_array = np.array(ke_list, dtype=np.float64)
            self._me_array = np.array(me_list, dtype=np.float64)
            self._dofs_list = None
            self._ke_list = None
            self._me_list = None
            
        self.dofs_count = self.mesh.node_count * self.dofs_per_node

    def _compute_sparsity_pattern(self):
        """
        Compute the sparse matrix non-zero pattern for efficient preallocation.

        Notes
        -----
        Determines the number of non-zeros per matrix row using element
        connectivity information. Critical for PETSc matrix performance.
        Supports both uniform and mixed-element meshes.
        """
        nnz = [set() for _ in range(self.dofs_count)]
        
        # Get the appropriate DOF data (list for mixed, array for uniform)
        dofs_data = self._dofs_list if self._is_mixed_mesh else self._dofs_array
        
        for elem_dofs in dofs_data:
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
        Supports both uniform and mixed-element meshes.
        """
        K = self._create_petsc_matrix()
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        if self._is_mixed_mesh:
            # Mixed mesh: use lists
            for dofs, ke in zip(self._dofs_list, self._ke_list):
                dofs_int = dofs.astype(PETSc.IntType)
                ke_flat = ke.flatten(order="C")
                K.setValuesLocal(dofs_int, dofs_int, ke_flat, addv=PETSc.InsertMode.ADD_VALUES)
        else:
            # Uniform mesh: use arrays (more efficient)
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
            
        Notes
        -----
        Supports both uniform and mixed-element meshes.
        """
        M = self._create_petsc_matrix()
        M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        if self._is_mixed_mesh:
            # Mixed mesh: use lists
            for dofs, me in zip(self._dofs_list, self._me_list):
                dofs_int = dofs.astype(PETSc.IntType)
                me_flat = me.flatten(order="C")
                M.setValuesLocal(dofs_int, dofs_int, me_flat, addv=PETSc.InsertMode.ADD_VALUES)
        else:
            # Uniform mesh: use arrays
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
        Supports both uniform and mixed-element meshes.
        """
        f = PETSc.Vec().create(self.comm)
        f.setSizes(self.dofs_count)
        f.setUp()
        f.zeroEntries()

        # Compute element load vectors
        fe_list = [
            self._element_map[eid].body_load(load_condition.value) for eid in self._element_map
        ]

        if self._is_mixed_mesh:
            # Mixed mesh: iterate directly over lists
            for dofs, fe in zip(self._dofs_list, fe_list):
                dofs_int = dofs.astype(PETSc.IntType)
                f.setValuesLocal(dofs_int, fe, addv=PETSc.InsertMode.ADD_VALUES)
        else:
            # Uniform mesh: use arrays
            fe_array = np.array(fe_list, dtype=PETSc.ScalarType)

            for e in range(fe_array.shape[0]):
                dofs = self._dofs_array[e].astype(PETSc.IntType)
                fe = fe_array[e]

                # Use local-to-global mapping if using mesh partitioning
                f.setValuesLocal(dofs, fe, addv=PETSc.InsertMode.ADD_VALUES)

        f.assemble()
        return f

    def assemble_geometric_stiffness(
        self,
        stress_field: Optional[Dict[int, np.ndarray]] = None,
        omega: Optional[float] = None,
        rotation_axis: Optional[np.ndarray] = None,
        rotation_center: Optional[np.ndarray] = None,
    ) -> PETSc.Mat:
        """
        Assemble the global geometric stiffness matrix for stress stiffening effects.

        The geometric stiffness matrix K_G captures the effect of membrane prestress
        on structural stiffness. This is essential for:
        - Rotating structures (centrifugal stiffening of wind turbine blades)
        - Prestressed structures
        - Buckling analysis
        - Geometric nonlinear analysis

        The method supports two modes of operation:
        1. Direct stress field: Provide membrane stress for each element
        2. Centrifugal loading: Automatically compute prestress from rotation parameters

        Parameters
        ----------
        stress_field : Dict[int, np.ndarray], optional
            Dictionary mapping element IDs to membrane stress tensors.
            Each stress tensor should be shape (3,) in Voigt notation [σ_xx, σ_yy, σ_xy]
            or shape (2, 2) as full tensor.
        omega : float, optional
            Angular velocity (rad/s) for centrifugal loading calculation.
            Required if stress_field is not provided.
        rotation_axis : np.ndarray, optional
            Unit vector defining rotation axis (3,). Default [0, 0, 1] (z-axis).
            Required if omega is provided.
        rotation_center : np.ndarray, optional
            Point on rotation axis (3,). Default [0, 0, 0] (origin).

        Returns
        -------
        PETSc.Mat
            Distributed sparse geometric stiffness matrix K_G

        Raises
        ------
        ValueError
            If neither stress_field nor omega is provided

        Notes
        -----
        The geometric stiffness matrix is computed as:
        K_G = Σ_e K_G^(e) where K_G^(e) = ∫_A B_G^T · S_m · B_G · dA

        For rotating blades, the centrifugal stress creates tensile membrane
        stresses that stiffen the structure, raising natural frequencies.

        The total effective stiffness matrix is: K_eff = K + K_G

        References
        ----------
        - Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in
          geometric nonlinear analysis." Computers & Structures, 185, 1-14.

        Examples
        --------
        >>> # Using centrifugal loading for wind turbine blade
        >>> K_G = assembler.assemble_geometric_stiffness(
        ...     omega=1.5,  # rad/s
        ...     rotation_axis=np.array([0, 0, 1]),
        ...     rotation_center=np.array([0, 0, 0])
        ... )
        >>> K_eff = K + K_G  # Total stiffness with stress stiffening

        >>> # Using direct stress field
        >>> stress_field = {elem_id: np.array([1e6, 0, 0]) for elem_id in element_ids}
        >>> K_G = assembler.assemble_geometric_stiffness(stress_field=stress_field)
        """
        # Validate inputs
        if stress_field is None and omega is None:
            raise ValueError(
                "Either 'stress_field' or 'omega' must be provided for geometric stiffness"
            )

        # Set default rotation parameters
        if rotation_axis is None:
            rotation_axis = np.array([0.0, 0.0, 1.0])
        if rotation_center is None:
            rotation_center = np.array([0.0, 0.0, 0.0])

        # Normalize rotation axis
        rotation_axis = np.asarray(rotation_axis, dtype=float)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_center = np.asarray(rotation_center, dtype=float)

        # Create PETSc matrix
        K_G = self._create_petsc_matrix()
        K_G.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        # Compute and assemble element geometric stiffness matrices
        for e, (elem_id, fem_element) in enumerate(self._element_map.items()):
            # Get membrane stress for this element
            if stress_field is not None and elem_id in stress_field:
                sigma_membrane = stress_field[elem_id]
            elif omega is not None:
                # Compute centrifugal prestress
                sigma_membrane = fem_element.compute_centrifugal_prestress(
                    omega=omega,
                    rotation_axis=rotation_axis,
                    rotation_center=rotation_center,
                )
            else:
                # Skip elements without stress data
                continue

            # Skip if stress is negligible
            if np.max(np.abs(sigma_membrane)) < 1e-20:
                continue

            # Compute element geometric stiffness
            kg_e = fem_element.compute_geometric_stiffness(
                sigma_membrane=sigma_membrane,
                transform_to_global=True,
            )

            # Get DOFs for this element
            dofs = self._dofs_array[e].astype(PETSc.IntType)
            kg_flat = kg_e.flatten(order="C")

            # Assemble into global matrix
            K_G.setValuesLocal(dofs, dofs, kg_flat, addv=PETSc.InsertMode.ADD_VALUES)

        K_G.assemble()
        return K_G


