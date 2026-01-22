"""
Core module for fem-shell.

Provides mesh handling, materials, boundary conditions, and configuration.
"""

from .config import FSISimulationConfig
from .laminate import (
    Ply,
    Laminate,
    StrengthProperties,
    compute_Q,
    compute_Qbar,
    create_symmetric_laminate,
    create_laminate_from_angles,
    quasi_isotropic_layup,
    cross_ply_layup,
    angle_ply_layup,
)

__all__ = [
    "FSISimulationConfig",
    "Ply",
    "Laminate",
    "StrengthProperties",
    "compute_Q",
    "compute_Qbar",
    "create_symmetric_laminate",
    "create_laminate_from_angles",
    "quasi_isotropic_layup",
    "cross_ply_layup",
    "angle_ply_layup",
]
