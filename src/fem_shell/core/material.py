from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class IsotropicMaterial:
    """
    Class representing an isotropic material with uniform properties in all directions.

    Parameters
    ----------
    name : str
        The name of the material.
    E : float
        Young's Modulus of the material.
    nu : float
        Poisson's ratio of the material.
    rho : float
        Density of the material.
    """

    name: str
    E: float
    nu: float
    rho: float


@dataclass
class OrthotropicMaterial:
    """
    Class representing an orthotropic material with different properties in three orthogonal directions.

    Parameters
    ----------
    name : str
        The name of the material.
    E : Tuple[float, float, float]
        Young's Modulus in three directions (E1, E2, E3).
    G : Tuple[float, float, float]
        Shear Modulus in three planes (G12, G23, G31).
    nu : Tuple[float, float, float]
        Poisson's ratio in three planes (nu12, nu23, nu31).
    rho : float
        Density of the material.
    """

    name: str
    E: Tuple[float, float, float]
    G: Tuple[float, float, float]
    nu: Tuple[float, float, float]
    rho: float


Material = Union[IsotropicMaterial, OrthotropicMaterial]
