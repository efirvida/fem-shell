from dataclasses import dataclass
from typing import Optional, Tuple, Union


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
    shear_correction_factor : Optional[float]
        Shear correction factor for Reissner-Mindlin shells.
        Default is None (uses 5/6). For sandwich panels use ~0.25.
    """

    name: str
    E: float
    nu: float
    rho: float
    shear_correction_factor: Optional[float] = None


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
    shear_correction_factor : Optional[float]
        Shear correction factor for Reissner-Mindlin shells.
        Default is None (uses 5/6). For laminates typically 0.70-0.85,
        for sandwich panels with soft core use 0.15-0.40.
    """

    name: str
    E: Tuple[float, float, float]
    G: Tuple[float, float, float]
    nu: Tuple[float, float, float]
    rho: float
    shear_correction_factor: Optional[float] = None

    @property
    def E1(self) -> float:
        return self.E[0]

    @property
    def E2(self) -> float:
        return self.E[1]

    @property
    def E3(self) -> float:
        return self.E[2]

    @property
    def G12(self) -> float:
        return self.G[0]

    @property
    def G23(self) -> float:
        return self.G[1]

    @property
    def G31(self) -> float:
        return self.G[2]

    @property
    def nu12(self) -> float:
        return self.nu[0]

    @property
    def nu23(self) -> float:
        return self.nu[1]

    @property
    def nu31(self) -> float:
        return self.nu[2]


MaterialType = Union[IsotropicMaterial, OrthotropicMaterial]


def Material(
    *,
    name: str = "Material",
    E,
    nu,
    rho: float,
    G=None,
) -> MaterialType:
    """Factory that returns an isotropic or orthotropic material instance.

    Parameters are inspected to decide which concrete material class to
    create. Scalars for ``E`` and ``nu`` produce an ``IsotropicMaterial``;
    length-3 iterables for ``E``, ``G``, and ``nu`` produce an
    ``OrthotropicMaterial``. This keeps the public API backward compatible
    with callers that import ``Material`` and expect to construct objects
    directly.
    """

    def _as_tuple3(value, label: str):
        try:
            items = tuple(value)
        except TypeError as exc:  # not iterable
            raise ValueError(f"{label} must be an iterable with 3 entries") from exc
        if len(items) != 3:
            raise ValueError(f"{label} must have exactly 3 components")
        return items

    # Orthotropic path: any iterable provided for E or nu triggers it
    is_iterable_E = hasattr(E, "__iter__") and not isinstance(E, (str, bytes))
    is_iterable_nu = hasattr(nu, "__iter__") and not isinstance(nu, (str, bytes))
    if is_iterable_E or is_iterable_nu:
        E_vals = _as_tuple3(E, "E")
        nu_vals = _as_tuple3(nu, "nu")
        if G is None:
            raise ValueError("Orthotropic materials require shear moduli G")
        G_vals = _as_tuple3(G, "G")
        return OrthotropicMaterial(name=name, E=E_vals, G=G_vals, nu=nu_vals, rho=rho)

    # Default to isotropic if inputs are scalar-like
    return IsotropicMaterial(name=name, E=float(E), nu=float(nu), rho=float(rho))
