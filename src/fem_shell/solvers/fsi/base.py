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
        self._participant = participant
        self._config_file = config_file
        self._coupling_mesh = coupling_mesh
        self._write_data = write_data if isinstance(write_data, list) else [write_data]
        self._read_data = read_data if isinstance(read_data, list) else [read_data]

        self._interface = precice.Participant(participant, config_file, 0, 1)

        # coupling mesh related quantities
        self._solver_vertices = None
        self._precice_vertex_ids = None

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint: SolverState | None = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

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

    def write_all_data(self, data_dict: Dict[str, np.ndarray]) -> None:
        """Write multiple data fields to preCICE.

        Parameters
        ----------
        data_dict : dict
            Dictionary mapping data names to their values.
        """
        for name, data in data_dict.items():
            self.write_data(data, name)

    def initialize(
        self, domain: MeshAssembler, coupling_boundaries: Sequence[str], fixed_dofs: Tuple[int]
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

        nodes = {node.id: node.coords for _set in self._node_sets for node in _set.nodes.values()}

        self._interface_coords = np.array(list(nodes.values()))[:, : self._domain.spatial_dim]
        self._interface_dofs = np.array([self._domain._node_dofs_map[n] for n in nodes])

        self._precice_vertex_ids = self._interface.set_mesh_vertices(
            self._coupling_mesh, self._interface_coords
        )
        np.savetxt(
            "interface_coords.csv",
            self.interface_coordinates,
            header="X,Y,Z" if self._domain.spatial_dim == 3 else "X,Y",
            delimiter=",",
        )

        if self._interface.requires_initial_data():
            # Write initial zero data for all write fields
            for data_name in self._write_data:
                self._interface.write_data(
                    self._coupling_mesh,
                    data_name,
                    self._precice_vertex_ids,
                    np.zeros(self.interface_dofs.shape),
                )
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
        """Return the interface degrees of freedom."""
        if self._interface_dofs.shape[0] > 3:
            return self._interface_dofs[:, :3].astype(PETSc.IntType)
        return self._interface_dofs.astype(PETSc.IntType)

    @property
    def interface_coordinates(self):
        """Return the interface coordinates."""
        return self._interface_coords

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
