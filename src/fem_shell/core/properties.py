from dataclasses import dataclass
from typing import Union

from fem_shell.core.laminate import Laminate
from fem_shell.core.material import MaterialType


@dataclass
class ShellProperty:
    """Isotropic or single-layer orthotropic shell property.

    Wraps a material and a uniform thickness.  Used by the assembler and
    element factory to build standard (non-composite) shell elements.

    Parameters
    ----------
    material : MaterialType
        Isotropic or orthotropic material.
    thickness : float
        Shell thickness in meters.
    """

    material: MaterialType
    thickness: float

    @property
    def total_thickness(self) -> float:
        return self.thickness

    def to_element_kwargs(self) -> dict:
        """Return kwargs suitable for ``ElementFactory.get_element``."""
        return {"material": self.material, "thickness": self.thickness}


@dataclass
class CompositeShellProperty:
    """Composite (laminated) shell property.

    Wraps a :class:`Laminate` instance that already contains the full ply
    stack. Used by the assembler and element factory to build composite
    shell elements (MITC3Composite / MITC4Composite).

    Parameters
    ----------
    laminate : Laminate
        Laminate definition with plies, ABD matrices, etc.
    """

    laminate: Laminate

    @property
    def total_thickness(self) -> float:
        return self.laminate.total_thickness

    def to_element_kwargs(self) -> dict:
        """Return kwargs suitable for ``ElementFactory.get_element``."""
        return {"laminate": self.laminate}


ShellPropertyType = Union[ShellProperty, CompositeShellProperty]
