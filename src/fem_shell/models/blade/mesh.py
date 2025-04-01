from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.core.mesh import ElementSet, ElementType, MeshElement, MeshModel, Node, NodeSet
from fem_shell.elements import ElementFamily
from fem_shell.models.blade.numad import Blade
from fem_shell.models.blade.numad.mesh_gen import get_shell_mesh


class BladeMesh:
    def __init__(self, blade_yaml: str, element_size: float = 0.1, n_samples: int = 300) -> None:
        self.yaml_file = blade_yaml
        self.blade_definition = Blade()
        self.blade_definition.read_yaml(self.yaml_file)
        self.mesh = MeshModel()

        self.element_size = element_size
        self.n_samples = n_samples
        self.numad_mesh_obj: Dict[str, Iterable[Union[List, Set, np.ndarray]]] = {}

        for stat in self.blade_definition.definition.stations:
            stat.airfoil.resample(n_samples=n_samples)

    def _generate_numad_shell_mesh(self) -> None:
        self.blade_definition.update_blade()
        n_stations = self.blade_definition.geometry.coordinates.shape[2]
        min_TE_lengths = 0.001 * np.ones(n_stations)
        self.blade_definition.expand_blade_geometry_te(min_TE_lengths)
        return get_shell_mesh(self.blade_definition, 0, self.element_size)

    def generate(self):
        self.numad_mesh_obj = self._generate_numad_shell_mesh()
        for node in self.numad_mesh_obj["nodes"]:
            self.mesh.add_node(Node(node))

        for node_ids in self.numad_mesh_obj["elements"]:
            if node_ids[3] == -1:
                element_type = ElementType.triangle
                node_ids = node_ids[:3]
            else:
                element_type = ElementType.quad
            node_objs = [self.mesh.get_node_by_id(n_id) for n_id in node_ids]
            self.mesh.add_element(MeshElement(nodes=node_objs, element_type=element_type))

        for element_set in self.numad_mesh_obj["sets"]["element"]:
            name = element_set["name"]
            elements = {self.mesh.get_element_by_id(i) for i in element_set["labels"]}
            self.mesh.add_element_set(ElementSet(name=name, elements=elements))

        for node_set in self.numad_mesh_obj["sets"]["node"]:
            name = node_set["name"]
            nodes = {self.mesh.get_node_by_id(i) for i in node_set["labels"]}
            self.mesh.add_node_set(NodeSet(name=name, nodes=nodes))

        for element_set in [
            eset for eset in self.mesh.element_sets.values() if "all" in eset.name.lower()
        ]:
            name = element_set.name.replace("Els", "Nods")
            nodes = {node for element in element_set.elements for node in element.nodes}
            self.mesh.add_node_set(NodeSet(name=name, nodes=nodes))

    def view(self) -> None:
        self.mesh.view()


def material_factory(
    material_data: Dict[str, Union[Dict[str, float], Dict[str, Tuple[float, float, float]]]],
) -> Union[Material, OrthotropicMaterial]:
    """
    Factory function to create a material object (either Isotropic or Orthotropic)
    based on the provided material data.

    Parameters
    ----------
    material_data : dict
        A dictionary containing material properties. It should include the keys:
        - 'name' : str
            The name of the material.
        - 'density' : float
            The density of the material (kg/m^3).
        - 'elastic' : dict
            A dictionary containing the elastic properties:
            - 'E' : float or list of floats
                Young's Modulus (single value for isotropic, list for orthotropic).
            - 'nu' : float or list of floats
                Poisson's ratio (single value for isotropic, list for orthotropic).
            - 'G' : list of floats
                Shear Modulus (required for orthotropic materials only).

    Returns
    -------
    Union[IsotropicMaterial, OrthotropicMaterial]
        - An instance of `IsotropicMaterial` if the elastic properties are provided as single values or
        if the orthotropic material has equal values for E, G, and nu.
        - An instance of `OrthotropicMaterial` if the elastic properties are provided as lists and they are not identical.
    """
    material_name = material_data["name"]
    density = material_data["density"]
    elastic_props = material_data["elastic"]

    # Check if elastic properties are provided as lists for E, G, and nu (orthotropic material)
    if isinstance(elastic_props["E"], Iterable):
        # Check if all components are identical (i.e., isotropic material)
        if (
            all(val == elastic_props["E"][0] for val in elastic_props["E"])
            and all(val == elastic_props["nu"][0] for val in elastic_props["nu"])
            and all(val == elastic_props["G"][0] for val in elastic_props["G"])
        ):
            # It's an isotropic material, even if it was defined as orthotropic
            E = elastic_props["E"][0]
            nu = elastic_props["nu"][0]
            return Material(name=material_name, E=E, nu=nu, rho=density)
        else:
            # It's a true orthotropic material
            E = tuple(elastic_props["E"])
            G = tuple(elastic_props["G"])
            nu = tuple(elastic_props["nu"])
            return OrthotropicMaterial(name=material_name, E=E, G=G, nu=nu, rho=density)
    else:
        # It's an isotropic material
        E = elastic_props["E"]
        nu = elastic_props["nu"]
        return Material(name=material_name, E=E, nu=nu, rho=density)


@dataclass
class LayerDef:
    material: str
    thickens: float
    angle: float


@dataclass
class BladeSection:
    set_name: str
    x_dir: np.ndarray
    xy_dir: np.ndarray
    layers: List[LayerDef]


class BladeModel:
    def __init__(self, blade_yaml: str, element_size: float = 0.1, n_samples: int = 300) -> None:
        self.blade_mesh_obj = BladeMesh(blade_yaml, element_size, n_samples)

        self.generage_mesh()
        print(self.get_element_material(1))

    def generage_mesh(self):
        self._materials_db = {}
        self._sections_db = {}
        self.blade_mesh_obj.generate()
        self.mesh = self.blade_mesh_obj.mesh
        self.numad_mesh_obj = self.blade_mesh_obj.numad_mesh_obj

    def view(self):
        self.blade_mesh_obj.view()

    def write_mesh(self, filename, **kwargs):
        self.blade_mesh_obj.mesh.write_mesh(filename, **kwargs)

    @property
    def materials(self) -> Dict[str, Union[Material, OrthotropicMaterial]]:
        if self._materials_db:
            return self._materials_db
        else:
            for material in self.numad_mesh_obj["materials"]:
                material = material_factory(material)
                self._materials_db[material.name] = material
        return self._materials_db

    @property
    def sections(self) -> Dict[str, BladeSection]:
        if self._sections_db:
            return self._sections_db
        else:
            for section in self.numad_mesh_obj["sections"]:
                layers = [LayerDef(*l) for l in section["layup"]]
                section = BladeSection(
                    set_name=section["elementSet"],
                    x_dir=section["xDir"],
                    xy_dir=section["xyDir"],
                    layers=layers,
                )
                self._sections_db[section.set_name] = section
            return self._sections_db

    def get_element_material(self, element_id: int) -> BladeSection:
        element = self.mesh.get_element_by_id(element_id)
        element_sets = self.mesh.get_element_associated_set(element)

        # use the first asociate element set
        valid_set = element_sets[0]
        return self.sections[valid_set.name]

    def to_model_dict(self):
        model = {}
        for element in self.mesh.elements:
            section = self.get_element_material(element_id=element.id)
            model[element.id] = {
                "element_family": ElementFamily.SHELL,
                "layers": section.layers,
                "x_dir": section.x_dir,
                "xy_dir": section.xy_dir,
            }

            return model
