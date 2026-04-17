"""
Base classes and utilities for FSI solvers.

This module contains shared components used by all FSI solvers:
- SolverState: Checkpoint state storage
- Adapter: preCICE adapter for FSI coupling

ForceClipper and NewmarkCoefficients are re-exported from their own modules
for backward compatibility.
"""

import copy
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import precice
from petsc4py import PETSc

# Re-export from dedicated modules for backward compatibility
from .force_clipper import ForceClipper
from .time_integration import NewmarkCoefficients

__all__ = ["Adapter", "ForceClipper", "NewmarkCoefficients", "SolverState"]


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
    """General-purpose preCICE adapter for coupling simulations.

    This adapter wraps the preCICE ``Participant`` API for mesh registration,
    data exchange, checkpointing, and coupling control.  It is **solver-agnostic**:
    callers are responsible for preparing vertex coordinates, choosing which
    mesh to read/write, and handling data types.

    Parameters
    ----------
    participant : str
        Name of this participant in the preCICE configuration.
    config_file : str
        Path to the preCICE configuration XML file.
    coupling_meshes : dict
        Mapping of mesh name → vertex coordinates (``ndarray``).
    """

    def __init__(
        self,
        participant: str,
        config_file: str,
        coupling_meshes: Dict[str, np.ndarray],
    ):
        self._participant = str(participant)
        self._config_file = str(config_file)
        self._coupling_meshes = {k: v.copy() for k, v in coupling_meshes.items()}

        self._interface = precice.Participant(self._participant, self._config_file, 0, 1)

        # Populated during initialize()
        self._mesh_vertex_ids: Dict[str, np.ndarray] = {}

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint: SolverState | None = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> float:
        """Register all coupling meshes and initialize preCICE.

        If preCICE requires initial data, the caller must write it
        *before* calling this method using :meth:`write_data`.

        Returns
        -------
        float
            Maximum time step size from preCICE.
        """
        for mesh_name, coords in self._coupling_meshes.items():
            vertex_ids = self._interface.set_mesh_vertices(mesh_name, coords)
            self._mesh_vertex_ids[mesh_name] = vertex_ids

        self._interface.initialize()
        return self._interface.get_max_time_step_size()

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def read_data(self, mesh_name: str, data_name: str) -> np.ndarray:
        """Read data from preCICE.

        Parameters
        ----------
        mesh_name : str
            Name of the mesh to read from.
        data_name : str
            Name of the data field to read.

        Returns
        -------
        np.ndarray
            Raw numpy array from preCICE (no type cast applied).
        """
        return self._interface.read_data(
            mesh_name, data_name, self._mesh_vertex_ids[mesh_name], self.dt
        )

    def write_data(self, mesh_name: str, data_name: str, data: np.ndarray) -> None:
        """Write data to preCICE.

        Parameters
        ----------
        mesh_name : str
            Name of the mesh to write to.
        data_name : str
            Name of the data field to write.
        data : np.ndarray
            Data array aligned with the mesh vertices.
        """
        vertex_ids = self._mesh_vertex_ids.get(mesh_name)
        if vertex_ids is None:
            raise ValueError(f"Mesh '{mesh_name}' not found in coupling_meshes")

        self._interface.write_data(mesh_name, data_name, vertex_ids, data)

    # ------------------------------------------------------------------
    # Mesh queries
    # ------------------------------------------------------------------

    def mesh_dimensions(self, mesh_name: str) -> int:
        """Get spatial dimensions of a coupling mesh."""
        return self._interface.get_mesh_dimensions(mesh_name)

    @property
    def requires_initial_data(self) -> bool:
        """Check if preCICE requires data before initialization."""
        return self._interface.requires_initial_data()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Coupling control
    # ------------------------------------------------------------------

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
        return self._interface.advance(dt)

    def finalize(self) -> None:
        """Finalize the coupling via preCICE."""
        self._interface.finalize()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    def precice(self):
        """Return the preCICE Participant object."""
        return self._interface

    @property
    def dt(self) -> float:
        """Return the maximum time step size allowed by preCICE."""
        return self._interface.get_max_time_step_size()
