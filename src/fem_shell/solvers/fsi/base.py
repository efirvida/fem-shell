"""
Base classes and utilities for FSI solvers.

This module contains shared components used by all FSI solvers:
- ForceClipper: Conservative force clipping
- SolverState: Checkpoint state storage
- Adapter: preCICE adapter for FSI coupling
"""

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import precice
from petsc4py import PETSc

from fem_shell.core.assembler import MeshAssembler


class ForceClipper:
    """
    Conservative force clipping to prevent pathological spikes in FSI coupling.

    Only clips force magnitudes that exceed a specified threshold. Does NOT smooth
    or average forces—preserves the actual CFD solution in the normal range.

    Strategy: Detect when nodal force magnitude is excessive (e.g., > 10x typical),
    and scale it down to a reasonable cap. All other forces pass through unchanged.

    Parameters
    ----------
    force_max_cap : Optional[float]
        Hard cap on per-node force magnitude. If None, no clipping is applied.
        Recommended: estimate from steady-state or pre-simulation (e.g., 500 kN for blades).
    """

    def __init__(self, force_max_cap: Optional[float] = None):
        self.force_max_cap = force_max_cap
        self._clipped_count = 0
        self._total_count = 0

    def apply(self, force_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply conservative clipping to force data.

        Parameters
        ----------
        force_data : np.ndarray
            Raw force data from preCICE. Shape: (n_nodes, n_dims) or (n_components,).

        Returns
        -------
        clipped_force : np.ndarray
            Force data with excessive magnitudes clipped, same shape as input.
        diagnostics : Dict[str, float]
            Statistics: mean, max, n_clipped (count of clipped nodes).
        """
        if self.force_max_cap is None:
            # No clipping
            force_mags = (
                np.linalg.norm(force_data, axis=1) if force_data.ndim == 2 else np.abs(force_data)
            )
            return force_data.copy(), {
                "mean": float(np.mean(force_mags)),
                "max": float(np.max(force_mags)),
                "n_clipped": 0,
                "cap": None,
            }

        # Ensure 2D for consistent processing
        if force_data.ndim == 1:
            n_dims = force_data.shape[0] if len(force_data.shape) == 1 else 1
            force_2d = force_data.reshape(-1, max(n_dims, 1))
            reshape_1d = True
        else:
            force_2d = force_data
            reshape_1d = False

        force_clipped = force_2d.copy()

        # Compute per-node magnitude
        force_mags = np.linalg.norm(force_clipped, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-12)  # Avoid division by zero

        # Identify and clip excessive magnitudes
        clip_mask = force_mags[:, 0] > self.force_max_cap
        n_clipped = np.sum(clip_mask)

        if n_clipped > 0:
            scale_factors = self.force_max_cap / force_mags[clip_mask, 0]
            force_clipped[clip_mask] *= scale_factors[:, np.newaxis]

        self._clipped_count += n_clipped
        self._total_count += force_2d.shape[0]

        # Diagnostics
        force_mags_final = np.linalg.norm(force_clipped, axis=1)
        diagnostics = {
            "mean": float(np.mean(force_mags_final)),
            "max": float(np.max(force_mags_final)),
            "n_clipped": int(n_clipped),
            "cap": self.force_max_cap,
        }

        # Reshape to match input if needed
        if reshape_1d:
            return force_clipped.flatten(), diagnostics
        return force_clipped, diagnostics

    def get_statistics(self) -> Dict[str, float]:
        """Return overall clipping statistics."""
        if self._total_count == 0:
            return {"clipped_fraction": 0.0, "total_nodes_processed": 0}
        return {
            "clipped_fraction": self._clipped_count / self._total_count,
            "total_nodes_processed": self._total_count,
        }


class SolverState:
    """Stores solver states (PETSc vectors) for efficient checkpointing."""

    def __init__(self, states: Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec, float]):
        """
        Store states (PETSc vectors and scalars) for checkpointing.

        Parameters
        ----------
        states : tuple
            Tuple of PETSc vectors and/or scalar values to checkpoint.
        """
        self._state = []
        for state in states:
            if isinstance(state, PETSc.Vec):
                self._state.append(state.copy())
            else:
                # For scalars (like time 't') or other types
                self._state.append(copy.deepcopy(state))

    def get_state(self):
        """Return cloned vectors."""
        return self._state

    def __del__(self):
        """Free PETSc vector memory when object is destroyed."""
        for vec in self._state:
            if isinstance(vec, PETSc.Vec):
                vec.destroy()


class Adapter:
    """preCICE adapter for FSI coupling with structural solvers.

    This adapter class provides an interface to the preCICE coupling library
    for setting up FSI simulations. All configuration is passed directly as
    parameters - no external configuration file is needed.

    Parameters
    ----------
    participant : str
        Name of this participant in the preCICE configuration.
    config_file : str
        Path to the preCICE configuration XML file.
    coupling_mesh : str
        Name of the coupling mesh in preCICE.
    write_data : list of str
        Names of data fields to write to preCICE.
    read_data : list of str
        Names of data fields to read from preCICE.
    """

    def __init__(
        self,
        participant: str,
        config_file: str,
        coupling_mesh: str,
        write_data: List[str],
        read_data: List[str],
    ):
        self._participant = str(participant)
        self._config_file = str(config_file)
        self._coupling_mesh = str(coupling_mesh)
        self._write_data = write_data if isinstance(write_data, list) else [write_data]
        self._read_data = read_data if isinstance(read_data, list) else [read_data]

        self._interface = precice.Participant(self._participant, self._config_file, 0, 1)

        # coupling mesh related quantities
        self._solver_vertices = None
        self._precice_vertex_ids = None
        self._mesh_vertex_ids = {}

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint: SolverState | None = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

    @property
    def mesh_dimensions(self) -> int:
        """Get dimensions of the coupling mesh."""
        return self._interface.get_mesh_dimensions(self._coupling_mesh)

    def read_data(self, data_name: Optional[str] = None) -> np.ndarray:
        """Read data from preCICE.

        Parameters
        ----------
        data_name : str, optional
            Name of the data field to read. If None, reads the first field
            in read_data list.

        Returns
        -------
        np.ndarray
            The incoming data containing nodal data.
        """
        if data_name is None:
            data_name = self._read_data[0]

        read_data = self._interface.read_data(
            self._coupling_mesh, data_name, self._precice_vertex_ids, self.dt
        )
        return copy.deepcopy(read_data.astype(PETSc.ScalarType))

    def read_all_data(self) -> Dict[str, np.ndarray]:
        """Read all configured data fields from preCICE.

        Returns
        -------
        dict
            Dictionary mapping data names to their values.
        """
        return {name: self.read_data(name) for name in self._read_data}

    def write_data(self, write_function, data_name: Optional[str] = None) -> None:
        """Write data to preCICE.

        Parameters
        ----------
        write_function : array-like
            Data to write, indexed by interface DOFs.
        data_name : str, optional
            Name of the data field to write. If None, writes to the first
            field in write_data list.
        """
        if data_name is None:
            data_name = self._write_data[0]

        write_data = write_function[self.interface_dofs]
        self._interface.write_data(
            self._coupling_mesh,
            data_name,
            self._precice_vertex_ids,
            write_data,
        )

    def write_interface_data(
        self, data: np.ndarray, data_name: str, mesh_name: Optional[str] = None
    ) -> None:
        """Write data already mapped to interface vertices directly to preCICE.

        This bypasses the global-to-local slicing of write_data(), allowing
        writing of custom fields that exist only on the interface (like locally
        computed coupling variables).

        Parameters
        ----------
        data : np.ndarray
            Data array aligned with interface vertices.
        data_name : str
            Name of the data field.
        mesh_name : str, optional
            Name of the mesh to write to. If None, uses the default coupling mesh.
        """
        target_mesh = mesh_name if mesh_name else self._coupling_mesh

        vertex_ids = self._precice_vertex_ids
        if target_mesh != self._coupling_mesh:
            vertex_ids = self._mesh_vertex_ids.get(target_mesh)
            if vertex_ids is None:
                raise ValueError(f"Mesh '{target_mesh}' not registered via register_mesh()")

        self._interface.write_data(
            target_mesh,
            data_name,
            vertex_ids,
            data,
        )

    def write_all_data(self, data_dict: Dict[str, np.ndarray]) -> None:
        """Write multiple data fields to preCICE.

        Parameters
        ----------
        data_dict : dict
            Dictionary mapping data names to their values.
        """
        for name, data in data_dict.items():
            self.write_data(data, name)

    def register_mesh(self, mesh_name: str, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Register an additional mesh with preCICE.

        Parameters
        ----------
        mesh_name : str
            Name of the mesh to register.
        coordinates : np.ndarray, optional
            Coordinates of the mesh vertices. If None, uses the main interface coordinates.

        Returns
        -------
        np.ndarray
            Vertex IDs of the registered mesh.
        """
        if coordinates is None:
            if not hasattr(self, "_interface_coords"):
                raise RuntimeError("Cannot register mesh without coordinates before initialization")
            coordinates = self._interface_coords

        vertex_ids = self._interface.set_mesh_vertices(mesh_name, coordinates)
        self._mesh_vertex_ids[mesh_name] = vertex_ids
        return vertex_ids

    def set_mesh_vertices(self, coordinates: np.ndarray, mesh_name: Optional[str] = None) -> np.ndarray:
        """Update mesh vertices in preCICE (e.g. for rotating reference frames).
        
        Parameters
        ----------
        coordinates : np.ndarray
             New coordinates for the mesh vertices.
        mesh_name : str, optional
             Name of the mesh to update. If None, uses the default coupling mesh.
             
        Returns
        -------
        np.ndarray
             Vertex IDs (should match existing ones).
        """
        target_mesh = mesh_name if mesh_name else self._coupling_mesh
        
        # Determine vertex IDs (optional for set_mesh_vertices in some language bindings, but good to ensure consistency)
        # In python binding, set_mesh_vertices returns user-defined IDs or internal IDs
        vertex_ids = self._interface.set_mesh_vertices(target_mesh, coordinates)
        
        # Update our tracking of coords
        if target_mesh == self._coupling_mesh:
            self._interface_coords = coordinates
            
        return vertex_ids

    def initialize(
        self,
        domain: MeshAssembler,
        coupling_boundaries: Sequence[str],
        fixed_dofs: Tuple[int],
        extra_meshes: Optional[List[str]] = None,
        custom_mesh_coords: Optional[Dict[str, np.ndarray]] = None,
        initial_data_values: Optional[Dict[str, float]] = None,
    ) -> float:
        """Initialize the coupling and set up the mesh in preCICE.

        Parameters
        ----------
        domain : MeshAssembler
            The mesh assembler containing the FEM mesh.
        coupling_boundaries : list of str
            Names of node sets representing the coupling interface.
        fixed_dofs : tuple
            Indices of fixed degrees of freedom.

        Returns
        -------
        float
            Recommended time step value from preCICE.
        """
        self._domain = domain
        self._mesh = domain.mesh
        self._node_sets = []
        for n_set_name in coupling_boundaries:
            self._node_sets.append(self._mesh.node_sets[n_set_name])

        # Collect interface nodes and sort by node ID for consistent ordering
        # This ensures the order matches between preCICE and our internal structures
        nodes = {node.id: node.coords for _set in self._node_sets for node in _set.nodes.values()}
        sorted_node_ids = sorted(nodes.keys())

        # Store node IDs in sorted order for proper mapping
        self._interface_node_ids = np.array(sorted_node_ids, dtype=np.int64)

        # Use full available coordinates (e.g. 3D embedding) regardless of element topology dim
        self._interface_coords = np.array([nodes[nid] for nid in sorted_node_ids])

        # If the simulation is 2D, ensure we pass 2D coordinates to preCICE
        if self._domain.spatial_dim == 2 and self._interface_coords.shape[1] > 2:
            self._interface_coords = self._interface_coords[:, :2]

        self._interface_dofs = np.array([
            self._domain._node_dofs_map[nid] for nid in sorted_node_ids
        ])

        self._precice_vertex_ids = self._interface.set_mesh_vertices(
            self._coupling_mesh, self._interface_coords
        )
        self._mesh_vertex_ids[self._coupling_mesh] = self._precice_vertex_ids

        # Register extra meshes found in configuration using the same interface coordinates
        if extra_meshes:
            for mesh_name in extra_meshes:
                try:
                    v_ids = self._interface.set_mesh_vertices(mesh_name, self._interface_coords)
                    self._mesh_vertex_ids[mesh_name] = v_ids
                except Exception as e:
                    logging.debug(f"Could not register extra mesh '{mesh_name}': {e}")

        # Register custom meshes with specific coordinates
        if custom_mesh_coords:
            for mesh_name, coords in custom_mesh_coords.items():
                try:
                    v_ids = self._interface.set_mesh_vertices(mesh_name, coords)
                    self._mesh_vertex_ids[mesh_name] = v_ids
                except Exception as e:
                    logging.warning(f"Could not register custom mesh '{mesh_name}': {e}")

        np.savetxt(
            "interface_coords.csv",
            self.interface_coordinates,
            header="X,Y,Z" if self._domain.spatial_dim == 3 else "X,Y",
            delimiter=",",
        )

        if self._interface.requires_initial_data():
            # Write initial zero data for all write fields on main coupling mesh
            for data_name in self._write_data:
                self._interface.write_data(
                    self._coupling_mesh,
                    data_name,
                    self._precice_vertex_ids,
                    np.zeros(self.interface_dofs.shape),
                )
            # Write initial zero data for custom meshes (e.g., GlobalSolidMesh)
            if custom_mesh_coords:
                for mesh_name, coords in custom_mesh_coords.items():
                    if mesh_name in self._mesh_vertex_ids:
                        n_verts = coords.shape[0]
                        vertex_ids = self._mesh_vertex_ids[mesh_name]
                        # Try to write AngularVelocity if this mesh has it
                        try:
                            # Use provided initial value or default to 0
                            init_val = 0.0
                            if initial_data_values and "AngularVelocity" in initial_data_values:
                                init_val = initial_data_values["AngularVelocity"]
                            self._interface.write_data(
                                mesh_name,
                                "AngularVelocity",
                                vertex_ids,
                                np.full(n_verts, init_val),
                            )
                        except Exception:
                            pass
        self._interface.initialize()
        return self._interface.get_max_time_step_size()

    def store_checkpoint(self, states: Sequence) -> None:
        """Store current solver state for checkpointing."""
        if self._first_advance_done:
            assert self.is_time_window_complete
        logging.debug("Store checkpoint")
        self._checkpoint = SolverState(states)

    def retrieve_checkpoint(self) -> List:
        """Retrieve stored checkpoint state.

        Returns
        -------
        list
            The stored checkpoint state.
        """
        assert not self.is_time_window_complete
        logging.debug("Restore solver state")
        if self._checkpoint:
            return self._checkpoint.get_state()

    def advance(self, dt: float) -> float:
        """Advance coupling in preCICE.

        Parameters
        ----------
        dt : float
            Length of timestep used by the solver.

        Returns
        -------
        float
            Maximum length of timestep for next iteration.
        """
        self._first_advance_done = True
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self) -> None:
        """Finalize the coupling via preCICE."""
        self._interface.finalize()

    @property
    def is_coupling_ongoing(self) -> bool:
        """Check if the coupled simulation is still ongoing."""
        return self._interface.is_coupling_ongoing()

    @property
    def is_time_window_complete(self) -> bool:
        """Check if implicit iteration has converged."""
        return self._interface.is_time_window_complete()

    @property
    def requires_reading_checkpoint(self) -> bool:
        """Check if reading a checkpoint is required."""
        return self._interface.requires_reading_checkpoint()

    @property
    def requires_writing_checkpoint(self) -> bool:
        """Check if writing a checkpoint is required."""
        return self._interface.requires_writing_checkpoint()

    @property
    def interface_dofs(self):
        """Return the interface degrees of freedom (translational DOFs only).

        For shell elements with 6 DOFs per node, returns only the first 3
        (translational) DOFs. For solid elements with 3 DOFs, returns all DOFs.
        """
        if self._interface_dofs.ndim == 2 and self._interface_dofs.shape[1] > 3:
            return self._interface_dofs[:, :3].astype(PETSc.IntType)
        return self._interface_dofs.astype(PETSc.IntType)

    @property
    def interface_coordinates(self):
        """Return the interface coordinates."""
        return self._interface_coords

    @property
    def interface_node_ids(self) -> np.ndarray:
        """Return the node IDs of interface nodes in order."""
        return self._interface_node_ids

    @property
    def interface_node_indices(self) -> np.ndarray:
        """Return the mesh node indices for interface nodes.

        These indices correspond to positions in mesh.nodes and coords_array,
        suitable for indexing point_data arrays in VTU output.
        """
        node_id_to_index = self._mesh.node_id_to_index
        return np.array([node_id_to_index[nid] for nid in self._interface_node_ids], dtype=np.int64)

    @property
    def precice(self):
        """Return the preCICE interface object."""
        return self._interface

    @property
    def dt(self):
        """Return the maximum time step size allowed by preCICE."""
        return self._interface.get_max_time_step_size()

    @property
    def read_data_names(self) -> List[str]:
        """Return list of read data field names."""
        return self._read_data

    @property
    def write_data_names(self) -> List[str]:
        """Return list of write data field names."""
        return self._write_data
