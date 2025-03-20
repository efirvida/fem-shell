import numpy as np
import pytest

from fem_shell.models import MeshElement, MeshModel, Node, NodeSet, SquareShapeMesh


@pytest.fixture
def empty_mesh():
    return MeshModel()


@pytest.fixture
def sample_mesh():
    mesh = MeshModel()
    mesh.add_node(Node([0.0, 0.0]))
    mesh.add_node(Node([1.0, 0.0]))
    mesh.add_node(Node([1.0, 1.0]))
    return mesh


@pytest.fixture
def generated_mesh():
    return SquareShapeMesh.create_rectangle(width=2.0, height=1.0, nx=2, ny=1, triangular=False)


class TestMeshModel:
    def test_initialization(self, empty_mesh):
        assert len(empty_mesh.nodes) == 0
        assert len(empty_mesh.elements) == 0
        assert len(empty_mesh.node_sets) == 0

    def test_add_node(self, empty_mesh):
        empty_mesh.add_node(Node(id=1, x=0.0, y=0.0))
        assert len(empty_mesh.nodes) == 1
        assert 1 in empty_mesh.node_map

        with pytest.raises(ValueError):
            empty_mesh.add_node(Node(id=1, x=1.0, y=1.0))

    def test_add_element(self, sample_mesh):
        assert len(sample_mesh.elements) == 1
        assert sample_mesh.elements[0].id == 1

        with pytest.raises(ValueError):
            sample_mesh.add_element(MeshElement(id=1, nodes=[]))

    def test_node_sets(self, sample_mesh):
        assert "test_set" in sample_mesh.node_sets
        assert len(sample_mesh.get_node_set("test_set").node_ids) == 2

        with pytest.raises(ValueError):
            sample_mesh.add_node_set(NodeSet(name="test_set", node_ids=[]))

        with pytest.raises(ValueError):
            sample_mesh.get_node_set("non_existent")

    def test_element_connectivity(self, sample_mesh):
        element = sample_mesh.elements[0]
        assert all(isinstance(n, Node) for n in element.nodes)
        assert len(element.nodes) == 3


class TestSquareMeshGeneration:
    def test_mesh_dimensions(self, generated_mesh):
        assert len(generated_mesh.nodes) > 0
        assert len(generated_mesh.elements) > 0
        assert all(n.x <= 1.0 and n.x >= -1.0 for n in generated_mesh.nodes)
        assert all(n.y <= 1.0 and n.y >= 0.0 for n in generated_mesh.nodes)

    def test_node_sets_creation(self, generated_mesh):
        required_sets = ["all", "top", "bottom", "left", "right", "surface"]
        for s in required_sets:
            assert generated_mesh.get_node_set(s) is not None

        top_set = generated_mesh.get_node_set("top")
        assert len(top_set.node_ids) > 0
        assert all(generated_mesh.get_node_by_id(nid).y == 1.0 for nid in top_set.node_ids)

    def test_element_types(self):
        tri_mesh = SquareShapeMesh.create_rectangle(
            width=2.0, height=1.0, nx=2, ny=1, triangular=True
        )
        quad_mesh = SquareShapeMesh.create_rectangle(
            width=2.0, height=1.0, nx=2, ny=1, triangular=False
        )

        assert len(tri_mesh.elements[0].nodes) == 3
        assert len(quad_mesh.elements[0].nodes) == 4

    def test_connectivity_integrity(self, generated_mesh):
        for element in generated_mesh.elements:
            assert len(element.nodes) >= 3
            for node in element.nodes:
                assert node.id in generated_mesh.node_map

    def test_edge_cases(self):
        # Test mínima malla
        tiny_mesh = SquareShapeMesh.create_rectangle(width=0.1, height=0.1, nx=1, ny=1)
        assert len(tiny_mesh.elements) == 1

        # Test elementos cuadráticos
        quad_mesh = SquareShapeMesh.create_rectangle(
            width=2.0, height=1.0, nx=2, ny=1, quadratic=True
        )
        assert len(quad_mesh.nodes) > 12  # Más nodos que versión lineal


class TestNodeOperations:
    def test_node_creation(self):
        node = Node(id=1, x=1.0, y=2.0)
        assert node.id == 1
        assert np.isclose(node.x, 1.0)
        assert np.isclose(node.y, 2.0)

    def test_node_fixity(self):
        node = Node(id=1, x=0.0, y=0.0)
        node.apply_fixity(dof_x=True, dof_y=False)
        assert node.constraints == {"x": True, "y": False}


if __name__ == "__main__":
    pytest.main(["-v", "--tb=line"])
