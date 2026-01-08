"""
FSI Rotor Solver for Wind Turbine Applications.
================================================

This module implements a Fluid-Structure Interaction (FSI) solver for wind turbine
rotors, coupling structural dynamics with aerodynamic loads via the preCICE library.

The module is organized into submodules for maintainability:

- **config**: Configuration dataclasses and enums (RotorConfig, LoadTorqueMode, etc.)
- **generator**: Generator/gearbox load torque controller (GeneratorController)
- **checkpoint_state**: Checkpoint state for implicit coupling (CheckpointState)
- **rotor_logger**: Rotor dynamics CSV logging (RotorLogger)
- **transforms**: Coordinate frame transformations (rotation matrices)
- **inertial_forces**: Inertial force calculations (centrifugal, Coriolis, etc.)
- **aerodynamics**: Torque and aerodynamic coefficient calculations
- **solver**: Main FSIRotorSolver class

Example Usage
-------------
Basic setup with dynamic rotation::

    from fem_shell.solvers.fsi_rotor import FSIRotorSolver, RotorConfig

    config = {
        "omega_mode": "dynamic",
        "inertia_mode": "total",
        "rotor_inertia": 1.5e7,
        "initial_omega": 1.57,
    }

    solver = FSIRotorSolver(mesh, fem_model_properties)
    solver.solve()

See Also
--------
fem_shell.solvers.linear : Base linear dynamic solver
fem_shell.solvers.fsi : preCICE adapter
"""

from .aerodynamics import AerodynamicsCalculator
from .checkpoint_state import CheckpointState
from .config import InertiaMode, LoadTorqueMode, OmegaMode, RotorConfig
from .generator import GeneratorController
from .inertial_forces import InertialForcesCalculator
from .rotor_logger import RotorLogger
from .solver import FSIRotorSolver
from .transforms import CoordinateTransforms

__all__ = [
    # Main solver
    "FSIRotorSolver",
    # Configuration
    "RotorConfig",
    "LoadTorqueMode",
    "OmegaMode",
    "InertiaMode",
    # Components
    "GeneratorController",
    "CheckpointState",
    "RotorLogger",
    "CoordinateTransforms",
    "InertialForcesCalculator",
    "AerodynamicsCalculator",
]
