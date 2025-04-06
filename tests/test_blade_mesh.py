import os

import pytest

from fem_shell.models.blade.mesh import Blade, MeshElement, Node

# Path to the reference directory
blades_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "examples", "reference_turbines", "yamls"
)

# Find all .yaml files in the directory
yaml_files = [os.path.join(blades_path, f) for f in os.listdir(blades_path) if f.endswith(".yaml")]

# Extract just the filenames for test IDs
yaml_file_ids = [os.path.basename(f) for f in yaml_files]


@pytest.fixture(autouse=True)
def reset_counters():
    """Fixture to reset ID counters before each test"""
    Node._id_counter = 0
    MeshElement._id_counter = 0


@pytest.mark.parametrize("blade_file", yaml_files, ids=yaml_file_ids)
def test_blade_mesh_generation(blade_file):
    """Test blade mesh generation for all reference turbine YAML files"""
    blade = Blade(blade_file, element_size=0.5)
    blade.generate()

    # Assertions
    assert blade.mesh.node_count > 0, f"{blade_file} has no nodes"
    assert blade.mesh.elements_count > 0, f"{blade_file} has no elements"
    assert "RootNodes" in blade.mesh.node_sets, f"{blade_file} missing 'RootNodes' node set"
    assert "allOuterShellNods" in blade.mesh.node_sets, (
        f"{blade_file} missing 'allOuterShellNods' node set"
    )
    assert "allShearWebNods" in blade.mesh.node_sets, (
        f"{blade_file} missing 'allShearWebNods' node set"
    )
