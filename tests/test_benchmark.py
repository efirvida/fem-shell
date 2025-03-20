import pytest

from fem_shell.core.mesh import MeshModel, Node, SquareShapeMesh


@pytest.fixture(scope="module")
def large_mesh():
    """Fixture para tests de rendimiento con malla grande"""
    return SquareShapeMesh.create_rectangle(width=10.0, height=5.0, nx=100, ny=50, triangular=False)


@pytest.fixture(scope="function")
def empty_mesh():
    """Fixture fresco para cada test de operaciones básicas"""
    return MeshModel()


class TestMeshPerformance:
    def test_mesh_generation_quad(self, benchmark):
        """Benchmark para generación de malla cuadrilátera"""
        result = benchmark(
            SquareShapeMesh.create_rectangle,
            width=5.0,
            height=2.0,
            nx=200,
            ny=80,
            triangular=False,
        )
        assert len(result.nodes) > 0

    def test_mesh_generation_tri(self, benchmark):
        """Benchmark para generación de malla triangular"""
        result = benchmark(
            SquareShapeMesh.create_rectangle,
            width=5.0,
            height=2.0,
            nx=200,
            ny=80,
            triangular=True,
        )
        assert len(result.elements) > 0

    def test_node_addition(self, benchmark, empty_mesh):
        """Rendimiento de adición masiva de nodos"""

        def setup():
            return [Node(id=i, x=i * 0.1, y=i * 0.2) for i in range(10000)], empty_mesh

        nodes, mesh = benchmark.pedantic(
            setup=setup, args=lambda: (nodes, mesh), rounds=10, iterations=3
        )

        @benchmark
        def add_nodes():
            for node in nodes:
                mesh.add_node(node)

        assert len(mesh.nodes) == 10000

    def test_element_retrieval(self, benchmark, large_mesh):
        """Benchmark para acceso a elementos"""
        element_ids = [e.id for e in large_mesh.elements[:1000]]

        def test_fn():
            for eid in element_ids:
                large_mesh.get_element_by_id(eid)

        benchmark(test_fn)

    def test_node_set_creation(self, benchmark, large_mesh):
        """Rendimiento en creación de conjuntos de nodos"""
        node_ids = [n.id for n in large_mesh.nodes[::10]]  # 10% de los nodos

        @benchmark
        def create_set():
            node_set = NodeSet(name="bench_set", node_ids=node_ids)
            large_mesh.add_node_set(node_set)

        assert "bench_set" in large_mesh.node_sets

    def test_spatial_queries(self, benchmark, large_mesh):
        """Benchmark para consultas espaciales básicas"""

        @benchmark
        def find_nodes():
            return [n for n in large_mesh.nodes if 4.5 <= n.x <= 5.5 and 2.0 <= n.y <= 3.0]

        result = find_nodes()
        assert len(result) > 0

    @pytest.mark.parametrize("mesh_size", [50, 200, 500], ids=["small", "medium", "large"])
    def test_scalability(self, benchmark, mesh_size):
        """Test de escalabilidad con diferentes tamaños de malla"""
        result = benchmark(
            SquareShapeMesh.create_rectangle,
            width=10.0,
            height=10.0,
            nx=mesh_size,
            ny=mesh_size,
            triangular=False,
        )
        assert len(result.elements) == mesh_size**2


class TestMemoryPerformance:
    def test_mesh_memory_footprint(self, benchmark):
        """Benchmark de uso de memoria para diferentes tamaños"""

        def setup(size):
            return SquareShapeMesh.create_rectangle(width=size, height=size, nx=size, ny=size)

        result = benchmark.weave(setup, [10, 50, 100], name="Memory growth vs mesh size")

        # Análisis de crecimiento de memoria
        for size, memory in result.items():
            assert memory < size**2 * 1000  # Aprox 1KB por elemento


if __name__ == "__main__":
    pytest.main(["-v", "--benchmark-enable", "--benchmark-autosave"])
