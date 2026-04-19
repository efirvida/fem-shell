import warnings

from .elements import ElementFactory, ElementFamily, FemElement
from .MITC3 import MITC3
from .MITC3_composite import MITC3Composite
from .MITC4 import MITC4
from .MITC4_composite import MITC4Composite

# Suppress the DeprecationWarning that QUAD.py and SOLID.py emit — those warnings
# are intended for *external* consumers that import the submodules directly,
# not for this public re-export layer.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from .QUAD import QUAD4, QUAD8, QUAD9
    from .SOLID import (
        SolidElement,
        HEXA8,
        HEXA20,
        PYRAMID5,
        PYRAMID13,
        TETRA4,
        TETRA10,
        WEDGE6,
        WEDGE15,
    )

__all__ = [
    "ElementFactory",
    "ElementFamily",
    "FemElement",
    "MITC3",
    "MITC3Composite",
    "MITC4",
    "MITC4Composite",
    "QUAD4",
    "QUAD8",
    "QUAD9",
    "SolidElement",
    "HEXA8",
    "HEXA20",
    "PYRAMID5",
    "PYRAMID13",
    "TETRA4",
    "TETRA10",
    "WEDGE6",
    "WEDGE15",
]
