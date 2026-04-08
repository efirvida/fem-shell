from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from fem_shell.core.laminate import Laminate
from fem_shell.core.material import IsotropicMaterial, MaterialType


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


def build_element_data(
    properties: Dict[str, ShellPropertyType],
    mesh,
) -> Dict[str, Dict[int, float]]:
    """Build per-element scalar fields from a properties map for visualization.

    Returns a dictionary suitable for ``mesh.view(element_data=...)``.
    Each key is a field name and each value maps element IDs to scalars.

    Fields produced
    ---------------
    Thickness : float
        Total laminate / shell thickness [m].
    N_Plies : float
        Number of plies (1 for ``ShellProperty``).
    Ex_membrane : float
        Equivalent membrane Young's modulus in the 1-direction [Pa].
    Ey_membrane : float
        Equivalent membrane Young's modulus in the 2-direction [Pa].
    Gxy_membrane : float
        Equivalent membrane shear modulus [Pa].
    nuxy_membrane : float
        Equivalent membrane Poisson's ratio.
    Ex_bending : float
        Equivalent bending modulus in the 1-direction [Pa].
    Ey_bending : float
        Equivalent bending modulus in the 2-direction [Pa].
    Mass_per_area : float
        Laminate areal density [kg/m²].
    A11 : float
        Extensional stiffness component A₁₁ [N/m].
    D11 : float
        Bending stiffness component D₁₁ [N·m].

    Parameters
    ----------
    properties : dict[str, ShellPropertyType]
        Mapping of element-set name to property (as returned by
        ``Blade.get_element_properties()``).
    mesh : MeshModel
        Mesh with ``element_sets`` used to resolve set membership.

    Returns
    -------
    dict[str, dict[int, float]]
    """
    fields: Dict[str, Dict[int, float]] = {
        "Thickness": {},
        "N_Plies": {},
        "Ex_membrane": {},
        "Ey_membrane": {},
        "Gxy_membrane": {},
        "nuxy_membrane": {},
        "Ex_bending": {},
        "Ey_bending": {},
        "Mass_per_area": {},
        "A11": {},
        "D11": {},
    }

    for set_name, prop in properties.items():
        if set_name not in mesh.element_sets:
            continue

        elem_ids = [e.id for e in mesh.element_sets[set_name].elements]

        thickness = prop.total_thickness

        if isinstance(prop, CompositeShellProperty):
            lam = prop.laminate
            eq = lam.get_equivalent_properties()
            n_plies = float(lam.n_plies)
            ex_m = eq["Ex_membrane"]
            ey_m = eq["Ey_membrane"]
            gxy_m = eq["Gxy_membrane"]
            nuxy_m = eq["nuxy_membrane"]
            ex_b = eq["Ex_bending"]
            ey_b = eq["Ey_bending"]
            mass_area = sum(p.material.rho * p.thickness for p in lam.plies)
            a11 = float(lam.A[0, 0])
            d11 = float(lam.D[0, 0])
        else:
            # ShellProperty (isotropic single layer)
            mat = prop.material
            n_plies = 1.0
            if isinstance(mat, IsotropicMaterial):
                ex_m = ey_m = mat.E
                gxy_m = mat.E / (2.0 * (1.0 + mat.nu))
                nuxy_m = mat.nu
                ex_b = ey_b = mat.E
            else:
                ex_m = mat.E1
                ey_m = mat.E2
                gxy_m = mat.G12
                nuxy_m = mat.nu12
                ex_b = mat.E1
                ey_b = mat.E2
            mass_area = mat.rho * thickness
            # Thin plate: A11 ≈ E·h/(1−ν²), D11 ≈ E·h³/12(1−ν²)
            a11 = ex_m * thickness / (1.0 - nuxy_m**2)
            d11 = ex_m * thickness**3 / (12.0 * (1.0 - nuxy_m**2))

        for eid in elem_ids:
            fields["Thickness"][eid] = thickness
            fields["N_Plies"][eid] = n_plies
            fields["Ex_membrane"][eid] = ex_m
            fields["Ey_membrane"][eid] = ey_m
            fields["Gxy_membrane"][eid] = gxy_m
            fields["nuxy_membrane"][eid] = nuxy_m
            fields["Ex_bending"][eid] = ex_b
            fields["Ey_bending"][eid] = ey_b
            fields["Mass_per_area"][eid] = mass_area
            fields["A11"][eid] = a11
            fields["D11"][eid] = d11

    return fields
