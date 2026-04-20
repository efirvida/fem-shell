"""Unit and integration tests for the boundary-layer volumetric mesh pipeline.

Tests cover:
- get_vertex_normals: output shape, unit length, outward orientation
- create_offset_layers: output structure, hex8 shape, layer counts, thickness/growth
- get_vol_mesh: integration with real blade YAML; webs exclusion; set names
"""

import numpy as np
import pytest

from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals
from aeroelast.models.blade.numad.mesh_gen.mesh3d import create_offset_layers

# ---------------------------------------------------------------------------
# Helpers / synthetic geometry
# ---------------------------------------------------------------------------

def _unit_square_mesh():
    """A 2x2 grid of quad4 elements forming a flat XY plane (Z=0).
    9 nodes, 4 quads.  Expected normals: +Z everywhere.
    """
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0],
    ], dtype=float)
    elements = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
    ], dtype=int)
    return nodes, elements


def _cylinder_ring_mesh(n=12, r=1.0, z=0.0):
    """Single ring of n quad4 elements around a cylinder at height z.
    Two Z levels (z and z+1), quads connecting them.
    Normals should point radially outward (+X/+Y direction).
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(n, z)])
    top = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(n, z + 1.0)])
    nodes = np.vstack([bot, top])  # shape (2n, 3)
    # Quads: [i, (i+1)%n, (i+1)%n + n, i + n]
    elements = np.array([
        [i, (i + 1) % n, (i + 1) % n + n, i + n]
        for i in range(n)
    ], dtype=int)
    return nodes, elements


# ---------------------------------------------------------------------------
# TestGetVertexNormals
# ---------------------------------------------------------------------------

class TestGetVertexNormals:
    def test_output_shape(self):
        nodes, elements = _unit_square_mesh()
        normals = get_vertex_normals(nodes, elements)
        assert normals.shape == (len(nodes), 3)

    def test_unit_length(self):
        nodes, elements = _cylinder_ring_mesh(n=16)
        normals = get_vertex_normals(nodes, elements)
        mags = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-10)

    def test_flat_mesh_points_up(self):
        """Flat XY-plane mesh centred on origin; normals should be +Z."""
        nodes, elements = _unit_square_mesh()
        # Translate so centroid is at (1,1,0)
        normals = get_vertex_normals(nodes, elements)
        # All normals should have dominant Z component
        assert np.all(np.abs(normals[:, 2]) > 0.9), (
            f"Expected +Z normals, got: {normals}"
        )

    def test_cylinder_normals_point_outward(self):
        """Cylinder ring normals must point radially outward (away from Z axis)."""
        n = 16
        nodes, elements = _cylinder_ring_mesh(n=n, r=2.0)
        normals = get_vertex_normals(nodes, elements)
        # For each node: dot(normal_XY, position_XY) > 0
        pos_xy = nodes[:, :2]
        normal_xy = normals[:, :2]
        dots = np.sum(pos_xy * normal_xy, axis=1)
        assert np.all(dots > 0), "Some cylinder normals point inward"

    def test_skips_degenerate_triangles(self):
        """Elements with el[3]==-1 should be skipped without errors."""
        nodes, elements = _unit_square_mesh()
        elements_with_tri = elements.copy()
        elements_with_tri[0, 3] = -1  # mark first element as degenerate
        normals = get_vertex_normals(nodes, elements_with_tri)
        assert normals.shape == (len(nodes), 3)


# ---------------------------------------------------------------------------
# TestCreateOffsetLayers
# ---------------------------------------------------------------------------

class TestCreateOffsetLayers:
    @pytest.fixture
    def cylinder_mesh(self):
        return _cylinder_ring_mesh(n=12, r=1.0)

    def test_return_keys(self, cylinder_mesh):
        nodes, elements = cylinder_mesh
        result = create_offset_layers(nodes, elements, n_layers=3, first_thickness=0.01, growth_rate=1.2)
        assert "nodes" in result
        assert "elements" in result
        assert "sets" in result

    def test_node_count(self, cylinder_mesh):
        nodes, elements = cylinder_mesh
        N = len(nodes)
        n_layers = 4
        result = create_offset_layers(nodes, elements, n_layers=n_layers, first_thickness=0.01, growth_rate=1.2)
        assert len(result["nodes"]) == N * (n_layers + 1)

    def test_element_count(self, cylinder_mesh):
        nodes, elements = cylinder_mesh
        M = len(elements)  # all quads in this mesh
        n_layers = 4
        result = create_offset_layers(nodes, elements, n_layers=n_layers, first_thickness=0.01, growth_rate=1.2)
        assert len(result["elements"]) == M * n_layers

    def test_hex8_shape(self, cylinder_mesh):
        nodes, elements = cylinder_mesh
        result = create_offset_layers(nodes, elements, n_layers=3, first_thickness=0.01, growth_rate=1.2)
        assert result["elements"].shape[1] == 8

    def test_first_layer_thickness(self, cylinder_mesh):
        """Average distance between layer 0 and layer 1 nodes ≈ first_thickness."""
        nodes, elements = cylinder_mesh
        N = len(nodes)
        first_thickness = 0.05
        result = create_offset_layers(nodes, elements, n_layers=5, first_thickness=first_thickness, growth_rate=1.0)
        all_nodes = result["nodes"]
        layer0 = all_nodes[:N]
        layer1 = all_nodes[N : 2 * N]
        distances = np.linalg.norm(layer1 - layer0, axis=1)
        np.testing.assert_allclose(distances.mean(), first_thickness, rtol=0.05)

    def test_growth_rate(self, cylinder_mesh):
        """Ratio of layer2/layer1 thickness ≈ growth_rate."""
        nodes, elements = cylinder_mesh
        N = len(nodes)
        growth_rate = 1.3
        result = create_offset_layers(nodes, elements, n_layers=5, first_thickness=0.01, growth_rate=growth_rate)
        all_nodes = result["nodes"]
        t1 = np.linalg.norm(all_nodes[N : 2 * N] - all_nodes[:N], axis=1).mean()
        t2 = np.linalg.norm(all_nodes[2 * N : 3 * N] - all_nodes[N : 2 * N], axis=1).mean()
        np.testing.assert_allclose(t2 / t1, growth_rate, rtol=0.05)

    def test_blade_wall_set(self, cylinder_mesh):
        nodes, elements = cylinder_mesh
        N = len(nodes)
        result = create_offset_layers(nodes, elements, n_layers=3, first_thickness=0.01, growth_rate=1.2)
        set_names = {s["name"] for s in result["sets"]["node"]}
        assert "bladeWallNodes" in set_names
        wall = next(s for s in result["sets"]["node"] if s["name"] == "bladeWallNodes")
        assert len(wall["labels"]) == N

    def test_outer_boundary_set(self, cylinder_mesh):
        nodes, elements = cylinder_mesh
        N = len(nodes)
        n_layers = 3
        result = create_offset_layers(nodes, elements, n_layers=n_layers, first_thickness=0.01, growth_rate=1.2)
        outer = next(s for s in result["sets"]["node"] if s["name"] == "outerBoundaryNodes")
        assert len(outer["labels"]) == N
        assert min(outer["labels"]) == n_layers * N

    def test_positive_jacobian(self, cylinder_mesh):
        """Every hex element should have positive Jacobian at its first corner."""
        nodes, elements = cylinder_mesh
        n_layers = 3
        result = create_offset_layers(nodes, elements, n_layers=n_layers, first_thickness=0.01, growth_rate=1.2)
        all_nodes = result["nodes"]
        neg = 0
        for el in result["elements"]:
            n0, n1, n3, n4 = el[0], el[1], el[3], el[4]
            v1 = all_nodes[n1] - all_nodes[n0]
            v2 = all_nodes[n3] - all_nodes[n0]
            v3 = all_nodes[n4] - all_nodes[n0]
            if np.linalg.det(np.array([v1, v2, v3])) <= 0:
                neg += 1
        assert neg == 0, f"{neg} elements have non-positive Jacobian"

    def test_degenerate_elements_excluded(self):
        """Degenerate (tri) elements in input must not produce hex elements."""
        nodes, elements = _unit_square_mesh()
        mixed = elements.copy()
        mixed[0, 3] = -1  # make one element a triangle
        result = create_offset_layers(nodes, mixed, n_layers=2, first_thickness=0.01, growth_rate=1.1)
        # Only 3 quad elements should be extruded (one was a tri)
        assert len(result["elements"]) == 3 * 2


# ---------------------------------------------------------------------------
# TestGetVolMesh — integration with real blade
# ---------------------------------------------------------------------------

class TestGetVolMesh:
    """Integration tests that require the IEA-15-240-RWT.yaml fixture."""

    N_LAYERS = 5
    FIRST_THICKNESS = 0.005
    GROWTH_RATE = 1.2
    ELEMENT_SIZE = 0.5

    @pytest.fixture(scope="class")
    def vol_mesh(self, iea_numad_blade):
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        return get_vol_mesh(
            iea_numad_blade,
            elementSize=self.ELEMENT_SIZE,
            overset_layers=self.N_LAYERS,
            first_thickness=self.FIRST_THICKNESS,
            growth_rate=self.GROWTH_RATE,
        )

    def test_return_keys(self, vol_mesh):
        assert "nodes" in vol_mesh
        assert "elements" in vol_mesh
        assert "sets" in vol_mesh

    def test_hex8_elements(self, vol_mesh):
        assert vol_mesh["elements"].shape[1] == 8

    def test_node_count_consistent(self, vol_mesh):
        """(n_layers+1) equal-sized layers of nodes."""
        total = len(vol_mesh["nodes"])
        # Must be divisible by (n_layers+1)
        assert total % (self.N_LAYERS + 1) == 0

    def test_element_count_consistent(self, vol_mesh):
        """n_layers equal-sized layers of elements."""
        total = len(vol_mesh["elements"])
        assert total % self.N_LAYERS == 0

    def test_blade_wall_set_exists(self, vol_mesh):
        names = {s["name"] for s in vol_mesh["sets"]["node"]}
        assert "bladeWallNodes" in names

    def test_outer_boundary_set_exists(self, vol_mesh):
        names = {s["name"] for s in vol_mesh["sets"]["node"]}
        assert "outerBoundaryNodes" in names

    def test_blade_wall_is_layer0(self, vol_mesh):
        """bladeWallNodes must coincide with the first N node indices."""
        N_surf = len(vol_mesh["nodes"]) // (self.N_LAYERS + 1)
        wall = next(s for s in vol_mesh["sets"]["node"] if s["name"] == "bladeWallNodes")
        expected = set(range(N_surf))
        assert set(wall["labels"]) == expected

    def test_no_negative_jacobians(self, vol_mesh):
        all_nodes = vol_mesh["nodes"]
        neg = 0
        for el in vol_mesh["elements"]:
            n0, n1, n3, n4 = el[0], el[1], el[3], el[4]
            v1 = all_nodes[n1] - all_nodes[n0]
            v2 = all_nodes[n3] - all_nodes[n0]
            v3 = all_nodes[n4] - all_nodes[n0]
            if np.linalg.det(np.array([v1, v2, v3])) <= 0:
                neg += 1
        assert neg == 0, f"{neg} negative-Jacobian elements in blade BL mesh"
"""
Unit tests for the volumetric mesh pipeline (cap closure + hex extrusion).

Covers:
  - cap_mesh: _tfi_block, generate_cap_quads, build_cap_mesh_data
  - element_utils: get_vertex_normals
  - mesh3d: create_offset_layers
  - mesh_quality: non_orthogonality, quality_report, laplacian_smooth
  - mesh_io: export_vtk, export_surface_vtk
  - mesh_gen: get_vol_mesh with overset_layers > 0 using the IEA-15-240-RWT blade
"""

import os
import pathlib
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IEA_YAML = pathlib.Path(__file__).parent.parent / "examples" / "blade" / "bem" / "IEA-15-240-RWT.yaml"


@pytest.fixture(scope="module")
def iea_blade():
    """Load the IEA-15-240-RWT numad Blade object once per module."""
    from aeroelast.models.blade.numad import Blade as numadBlade
    blade = numadBlade()
    blade.read_yaml(str(IEA_YAML))
    n_stations = blade.geometry.coordinates.shape[2]
    min_TE = 0.001 * np.ones(n_stations)
    blade.expand_blade_geometry_te(min_TE)
    blade.update_blade()
    return blade


@pytest.fixture(scope="module")
def airfoil_loop():
    """Small synthetic airfoil loop (cls=37) for unit tests."""
    t = np.linspace(0, 2 * np.pi, 37, endpoint=False)
    x = 0.5 * np.cos(t)
    y = 0.15 * np.sin(t)
    z = np.zeros(37)
    return np.column_stack([x, y, z])


@pytest.fixture(scope="module")
def simple_hex_mesh():
    """Two-cell aligned hex mesh for quality tests."""
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],
        ],
        dtype=float,
    )
    elements = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10, 11],
        ],
        dtype=int,
    )
    return nodes, elements


# ---------------------------------------------------------------------------
# cap_mesh
# ---------------------------------------------------------------------------

class TestTfiBlock:
    def test_shape(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import _tfi_block

        N, M = 5, 3
        lower = np.linspace([0, 0, 0], [1, 0, 0], N)
        upper = np.linspace([0, 1, 0], [1, 1, 0], N)
        left  = np.linspace(lower[0], upper[0], M)
        right = np.linspace(lower[-1], upper[-1], M)

        nodes, quads = _tfi_block(lower, upper, left, right)

        assert nodes.shape == (N * M, 3)
        assert len(quads) == (N - 1) * (M - 1)

    def test_corners_preserved(self):
        """Corner nodes of the TFI patch must match the boundary inputs exactly."""
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import _tfi_block

        N, M = 4, 3
        lower = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], dtype=float)
        upper = np.array([[0,2,0],[1,2,0],[2,2,0],[3,2,0]], dtype=float)
        left  = np.linspace(lower[0], upper[0], M)
        right = np.linspace(lower[-1], upper[-1], M)

        nodes, _ = _tfi_block(lower, upper, left, right)
        nodes_2d = nodes.reshape(M, N, 3)

        np.testing.assert_allclose(nodes_2d[0, 0],  lower[0],  atol=1e-12)
        np.testing.assert_allclose(nodes_2d[0, -1], lower[-1], atol=1e-12)
        np.testing.assert_allclose(nodes_2d[-1, 0], upper[0],  atol=1e-12)
        np.testing.assert_allclose(nodes_2d[-1,-1], upper[-1], atol=1e-12)


class TestGenerateCapQuads:
    def test_output_shapes(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads

        nodes, quads = generate_cap_quads(airfoil_loop, le_idx=18, n_rows=1)

        assert nodes.ndim == 2 and nodes.shape[1] == 3
        assert quads.ndim == 2 and quads.shape[1] == 4

    def test_no_duplicate_index_in_quads(self, airfoil_loop):
        """Every quad must reference 4 distinct node indices."""
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads

        nodes, quads = generate_cap_quads(airfoil_loop, le_idx=18, n_rows=2)
        for q in quads:
            assert len(set(q.tolist())) == 4, f"Degenerate quad: {q}"

    def test_node_indices_in_range(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads

        nodes, quads = generate_cap_quads(airfoil_loop, le_idx=18, n_rows=2)
        assert quads.min() >= 0
        assert quads.max() < len(nodes)

    def test_outward_normal_enforced(self, airfoil_loop):
        """All quad normals must have a positive component along outward_normal."""
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads

        out_n = np.array([0.0, 0.0, -1.0])
        nodes, quads = generate_cap_quads(airfoil_loop, le_idx=18, n_rows=1,
                                           outward_normal=out_n)
        for q in quads:
            v0 = nodes[q[1]] - nodes[q[0]]
            v1 = nodes[q[3]] - nodes[q[0]]
            fn = np.cross(v0, v1)
            assert np.dot(fn, out_n) >= 0.0, "Quad normal points inward"

    def test_n_rows_increases_node_count(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads

        n1, _ = generate_cap_quads(airfoil_loop, 18, n_rows=1)
        n2, _ = generate_cap_quads(airfoil_loop, 18, n_rows=3)
        assert len(n2) > len(n1)


class TestBuildCapMeshData:
    def test_node_offset_applied(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import build_cap_mesh_data

        offset = 500
        nodes, els = build_cap_mesh_data(airfoil_loop, 18, 2, [0, 0, -1], offset)
        assert els.min() >= offset

    def test_returns_correct_shapes(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import build_cap_mesh_data

        nodes, els = build_cap_mesh_data(airfoil_loop, 18, 1, [0, 0, 1], 0)
        assert nodes.shape[1] == 3
        assert els.shape[1] == 4


# ---------------------------------------------------------------------------
# element_utils — get_vertex_normals
# ---------------------------------------------------------------------------

class TestGetVertexNormals:
    def _make_flat_quad_mesh(self):
        nodes = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [2, 0, 0], [2, 1, 0],
        ], dtype=float)
        elements = np.array([
            [0, 1, 2, 3, -1, -1, -1, -1],
            [1, 4, 5, 2, -1, -1, -1, -1],
        ], dtype=int)
        return nodes, elements

    def test_unit_normals(self):
        from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals

        nodes, elements = self._make_flat_quad_mesh()
        normals = get_vertex_normals(nodes, elements)

        assert normals.shape == (len(nodes), 3)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_flat_xy_plane_points_z(self):
        """Normals of a flat XY-plane quad must point along ±Z."""
        from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals

        nodes, elements = self._make_flat_quad_mesh()
        normals = get_vertex_normals(nodes, elements)

        for n in normals:
            assert abs(n[2]) > 0.99, f"Normal not along Z: {n}"


# ---------------------------------------------------------------------------
# mesh3d — create_offset_layers
# ---------------------------------------------------------------------------

class TestCreateOffsetLayers:
    def _tube_shell(self, n_circ=8, n_span=3):
        angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
        zs = np.linspace(0.0, 1.0, n_span)
        surf_nodes = np.array(
            [[np.cos(a), np.sin(a), z] for z in zs for a in angles], dtype=float
        )
        quads = []
        for j in range(n_span - 1):
            for i in range(n_circ):
                n0 = j * n_circ + i
                n1 = j * n_circ + (i + 1) % n_circ
                n2 = (j + 1) * n_circ + (i + 1) % n_circ
                n3 = (j + 1) * n_circ + i
                quads.append([n0, n1, n2, n3, -1, -1, -1, -1])
        return {"nodes": surf_nodes, "elements": np.array(quads, dtype=int)}

    def test_output_shapes(self):
        from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals
        from aeroelast.models.blade.numad.mesh_gen.mesh3d import create_offset_layers

        shell = self._tube_shell()
        v_n = get_vertex_normals(shell["nodes"], shell["elements"])
        n_layers = 4
        vol = create_offset_layers(shell, n_layers, 0.05, 1.2, v_n)

        n_surf = len(shell["nodes"])
        n_quads = len(shell["elements"])
        assert vol["nodes"].shape == (n_surf * (n_layers + 1), 3)
        assert vol["elements"].shape == (n_quads * n_layers, 8)

    def test_geometric_growth(self):
        """Outer node must be farther from surface than inner node."""
        from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals
        from aeroelast.models.blade.numad.mesh_gen.mesh3d import create_offset_layers

        shell = self._tube_shell(n_circ=4, n_span=2)
        v_n = get_vertex_normals(shell["nodes"], shell["elements"])
        n_layers = 3
        first_t = 0.1
        growth  = 1.5
        vol = create_offset_layers(shell, n_layers, first_t, growth, v_n)

        n_surf = len(shell["nodes"])
        # radii of the original surface vs last offset layer
        r_surf  = np.linalg.norm(shell["nodes"][:, :2], axis=1).mean()
        r_outer = np.linalg.norm(vol["nodes"][n_layers * n_surf:, :2], axis=1).mean()
        assert r_outer > r_surf

    def test_required_sets_present(self):
        from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals
        from aeroelast.models.blade.numad.mesh_gen.mesh3d import create_offset_layers

        shell = self._tube_shell()
        v_n = get_vertex_normals(shell["nodes"], shell["elements"])
        vol = create_offset_layers(shell, 2, 0.05, 1.2, v_n)

        names = {s["name"] for s in vol["sets"]["element"]}
        assert "blade_wall" in names
        assert "overset_boundary" in names

    def test_cyl_blend_outer_displacement_is_radial(self):
        """With cyl_blend_exp=1.0, the XY displacement of OML nodes at the outer
        layer must be aligned with the XY radial direction (dot product ≈ 1.0).

        This verifies the core property: t=1 → transverse component of
        blended normal is exactly radial → displacement direction is radial.
        """
        from aeroelast.models.blade.numad.mesh_gen.element_utils import get_vertex_normals
        from aeroelast.models.blade.numad.mesh_gen.mesh3d import create_offset_layers

        # Ellipse: a=2, b=1 — vertex normals differ from radial direction
        n_circ, n_span = 12, 3
        a, b = 2.0, 1.0
        angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
        zs = np.linspace(0.0, 1.0, n_span)
        surf_nodes = np.array(
            [[a * np.cos(ang), b * np.sin(ang), z] for z in zs for ang in angles],
            dtype=float,
        )
        quads = []
        for j in range(n_span - 1):
            for i in range(n_circ):
                n0 = j * n_circ + i
                n1 = j * n_circ + (i + 1) % n_circ
                n2 = (j + 1) * n_circ + (i + 1) % n_circ
                n3 = (j + 1) * n_circ + i
                quads.append([n0, n1, n2, n3, -1, -1, -1, -1])
        shell = {"nodes": surf_nodes, "elements": np.array(quads, dtype=int)}

        v_n = get_vertex_normals(shell["nodes"], shell["elements"])
        n_layers = 6

        vol = create_offset_layers(shell, n_layers, 0.2, 1.0, v_n, cyl_blend_exp=1.0)

        n_surf = len(surf_nodes)
        outer_nodes = vol["nodes"][n_layers * n_surf:]  # outermost layer

        # XY displacement from surface
        disp_xy = outer_nodes[:, :2] - surf_nodes[:, :2]
        disp_xy_mag = np.linalg.norm(disp_xy, axis=1)

        # Radial direction for each surface node
        surf_xy = surf_nodes[:, :2]
        r_surf = np.linalg.norm(surf_xy, axis=1)

        # Only check nodes away from the axis
        mask = (disp_xy_mag > 1e-10) & (r_surf > 1e-10)
        disp_hat = disp_xy[mask] / disp_xy_mag[mask, None]
        radial_hat = surf_xy[mask] / r_surf[mask, None]

        dots = np.sum(disp_hat * radial_hat, axis=1)
        np.testing.assert_allclose(
            dots, 1.0, atol=1e-10,
            err_msg="At cyl_blend_exp=1.0 outer layer, XY displacement must be radial",
        )

    def test_cyl_blend_flat_caps_unchanged(self):
        """Flat end-caps (normals ∥ Z) must not be displaced in XY by the blend.

        Build a short cylinder with explicit top/bottom flat cap nodes whose
        vertex normals point purely in ±Z.  After blending, their XY positions
        must remain identical to the pure-normal-offset result.
        """
        from aeroelast.models.blade.numad.mesh_gen.mesh3d import create_offset_layers

        n_circ, n_layers, t0 = 8, 4, 0.1

        # OML ring: z=0 and z=1, normals pointing radially outward (XY)
        angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
        oml_nodes = np.array(
            [[np.cos(a), np.sin(a), z] for z in (0.0, 1.0) for a in angles],
            dtype=float,
        )
        oml_quads = np.array(
            [
                [i, (i + 1) % n_circ, n_circ + (i + 1) % n_circ, n_circ + i, -1, -1, -1, -1]
                for i in range(n_circ)
            ],
            dtype=int,
        )
        # Cap nodes at z=0 (normal -Z) and z=1 (normal +Z)
        # Use a single quad cap per face for simplicity
        cap_bot = np.array([[0.0, 0.0, 0.0]])  # center
        cap_top = np.array([[0.0, 0.0, 1.0]])  # center

        all_nodes = np.vstack([oml_nodes, cap_bot, cap_top])
        n_oml = len(oml_nodes)
        cap_bot_idx = n_oml
        cap_top_idx = n_oml + 1

        # Build normals manually: OML = radial (XY), caps = ±Z
        v_n = np.zeros((len(all_nodes), 3))
        for i, nd in enumerate(oml_nodes):
            r = np.array([nd[0], nd[1], 0.0])
            v_n[i] = r / np.linalg.norm(r)
        v_n[cap_bot_idx] = [0.0, 0.0, -1.0]
        v_n[cap_top_idx] = [0.0, 0.0,  1.0]

        # Minimal shell (oml quads only; cap quad would need 4 nodes — skip)
        shell = {"nodes": all_nodes, "elements": oml_quads}

        vol_normal = create_offset_layers(shell, n_layers, t0, 1.0, v_n)
        vol_blend  = create_offset_layers(shell, n_layers, t0, 1.0, v_n, cyl_blend_exp=2.0)

        n_surf = len(all_nodes)
        # Cap nodes are the last two in all_nodes / each offset layer
        for k in range(1, n_layers + 1):
            off = k * n_surf
            xy_normal = vol_normal["nodes"][off + cap_bot_idx : off + cap_top_idx + 1, :2]
            xy_blend  = vol_blend["nodes"][off + cap_bot_idx : off + cap_top_idx + 1, :2]
            np.testing.assert_allclose(
                xy_blend, xy_normal, atol=1e-12,
                err_msg=f"Cap XY positions changed at layer {k} with cyl_blend",
            )


# ---------------------------------------------------------------------------
# mesh_quality
# ---------------------------------------------------------------------------

class TestMeshQuality:
    def test_perfect_alignment_zero_nonortho(self, simple_hex_mesh):
        """Aligned hex cells must report 0° non-orthogonality."""
        from aeroelast.models.blade.numad.mesh_gen.mesh_quality import non_orthogonality

        nodes, elements = simple_hex_mesh
        angles, _ = non_orthogonality(nodes, elements)
        assert len(angles) > 0
        np.testing.assert_allclose(angles, 0.0, atol=1e-8)

    def test_quality_report_returns_bool(self, simple_hex_mesh):
        from aeroelast.models.blade.numad.mesh_gen.mesh_quality import quality_report

        nodes, elements = simple_hex_mesh
        angles, passed = quality_report(nodes, elements)
        assert isinstance(passed, (bool, np.bool_))
        assert passed  # aligned mesh must pass

    def test_skewed_mesh_nonzero_nonortho(self, simple_hex_mesh):
        """Perturbing a non-shared node shifts only one cell's centroid,
        making the centroid-centroid vector diverge from the face normal."""
        from aeroelast.models.blade.numad.mesh_gen.mesh_quality import non_orthogonality

        nodes, elements = simple_hex_mesh
        nodes_skewed = nodes.copy()
        # Node 2 belongs only to cell 0 (bottom face), so moving it tilts d
        # relative to the shared face normal without touching the shared face.
        nodes_skewed[2] += np.array([0.4, 0.3, 0.0])

        angles_orig,   _ = non_orthogonality(nodes, elements)
        angles_skewed, _ = non_orthogonality(nodes_skewed, elements)
        assert angles_skewed.max() > angles_orig.max()


class TestLaplacianSmooth:
    def test_free_nodes_move(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads
        from aeroelast.models.blade.numad.mesh_gen.mesh_quality import laplacian_smooth

        nodes, quads = generate_cap_quads(airfoil_loop, 18, n_rows=3)
        fixed = list(range(len(airfoil_loop)))           # keep boundary loop fixed
        smoothed = laplacian_smooth(nodes, quads, fixed_nodes=fixed, n_iter=5)

        assert smoothed.shape == nodes.shape
        # At least some interior nodes must have moved
        diff = np.linalg.norm(smoothed - nodes, axis=1)
        assert diff.max() > 1e-10, "No nodes moved after smoothing"

    def test_fixed_nodes_unchanged(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads
        from aeroelast.models.blade.numad.mesh_gen.mesh_quality import laplacian_smooth

        nodes, quads = generate_cap_quads(airfoil_loop, 18, n_rows=2)
        fixed = list(range(len(airfoil_loop)))
        smoothed = laplacian_smooth(nodes, quads, fixed_nodes=fixed, n_iter=10)

        np.testing.assert_allclose(smoothed[fixed], nodes[fixed], atol=1e-12)

    def test_zero_iters_unchanged(self, airfoil_loop):
        from aeroelast.models.blade.numad.mesh_gen.cap_mesh import generate_cap_quads
        from aeroelast.models.blade.numad.mesh_gen.mesh_quality import laplacian_smooth

        nodes, quads = generate_cap_quads(airfoil_loop, 18, n_rows=2)
        smoothed = laplacian_smooth(nodes, quads, fixed_nodes=[], n_iter=0)
        np.testing.assert_array_equal(smoothed, nodes)


# ---------------------------------------------------------------------------
# mesh_io
# ---------------------------------------------------------------------------

class TestExportVtk:
    def _minimal_meshdata(self, simple_hex_mesh):
        nodes, elements = simple_hex_mesh
        return {
            "nodes": nodes,
            "elements": elements,
            "sets": {
                "element": [{"name": "all", "labels": [0, 1]}],
                "node":    [{"name": "bottom", "labels": [0, 1, 2, 3]}],
            },
        }

    def test_vtu_file_created(self, simple_hex_mesh):
        from aeroelast.models.blade.numad.mesh_gen.mesh_io import export_vtk

        meshdata = self._minimal_meshdata(simple_hex_mesh)
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "out.vtu"
            export_vtk(meshdata, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_vtk_legacy_file_created(self, simple_hex_mesh):
        from aeroelast.models.blade.numad.mesh_gen.mesh_io import export_vtk

        meshdata = self._minimal_meshdata(simple_hex_mesh)
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "out.vtk"
            export_vtk(meshdata, out)
            assert out.exists()

    def test_directory_created_if_missing(self, simple_hex_mesh):
        from aeroelast.models.blade.numad.mesh_gen.mesh_io import export_vtk

        meshdata = self._minimal_meshdata(simple_hex_mesh)
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "subdir" / "deep" / "out.vtu"
            export_vtk(meshdata, out)
            assert out.exists()

    def test_surface_vtk(self):
        from aeroelast.models.blade.numad.mesh_gen.mesh_io import export_surface_vtk

        # Build a tiny quad surface
        nodes = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
        elements = np.array([[0,1,2,3,-1,-1,-1,-1]], dtype=int)
        meshdata = {"nodes": nodes, "elements": elements, "sets": {}}

        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "surf.vtu"
            export_surface_vtk(meshdata, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# Integration — get_vol_mesh with IEA-15-240-RWT  (coarse, fast)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not IEA_YAML.exists(), reason="IEA-15-240-RWT YAML not found")
class TestGetVolMeshIEA:
    """End-to-end tests using the IEA-15-240-RWT blade."""

    @pytest.fixture(scope="class")
    def shell_only(self, iea_blade):
        """Shell mesh without overset (default behaviour, backward-compatible)."""
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        return get_vol_mesh(iea_blade, elementSize=1.0)

    @pytest.fixture(scope="class")
    def vol_mesh(self, iea_blade):
        """Closed volumetric mesh with 3 hex layers."""
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        return get_vol_mesh(
            iea_blade,
            elementSize=1.0,
            overset_layers=3,
            overset_first_thickness=0.05,
            overset_growth=1.2,
            overset_smooth_iters=5,
        )

    # -- shell-only (overset_layers=0) backward compatibility --

    def test_shell_has_nodes(self, shell_only):
        assert len(shell_only["nodes"]) > 0

    def test_shell_has_elements(self, shell_only):
        assert len(shell_only["elements"]) > 0

    def test_shell_root_nodes_set(self, shell_only):
        names = {s["name"] for s in shell_only["sets"]["node"]}
        assert "RootNodes" in names

    def test_shell_tip_nodes_set(self, shell_only):
        names = {s["name"] for s in shell_only["sets"]["node"]}
        assert "TipNodes" in names

    # -- volumetric mesh --

    def test_vol_has_nodes(self, vol_mesh):
        assert len(vol_mesh["nodes"]) > 0

    def test_vol_has_hex_elements(self, vol_mesh):
        els = vol_mesh["elements"]
        # All elements must be brick8 (8 valid node ids, no -1 in any column)
        assert els.shape[1] == 8
        assert np.all(els >= 0), "Hex mesh contains -1 entries (degenerate elements)"

    def test_vol_element_count(self, vol_mesh, iea_blade):
        """Vol mesh must have strictly more elements than a shell mesh."""
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        shell = get_vol_mesh(iea_blade, elementSize=1.0)
        assert len(vol_mesh["elements"]) > len(shell["elements"])

    def test_vol_blade_wall_set(self, vol_mesh):
        names = {s["name"] for s in vol_mesh["sets"]["element"]}
        assert "blade_wall" in names

    def test_vol_overset_boundary_set(self, vol_mesh):
        names = {s["name"] for s in vol_mesh["sets"]["element"]}
        assert "overset_boundary" in names

    def test_vol_quality_ok_key(self, vol_mesh):
        assert "quality_ok" in vol_mesh

    def test_vol_spline_metadata_preserved(self, vol_mesh):
        for key in ("splineXi", "splineYi", "splineZi"):
            assert key in vol_mesh, f"Missing metadata key: {key}"

    def test_vol_vtk_export(self, vol_mesh):
        from aeroelast.models.blade.numad.mesh_gen.mesh_io import export_vtk

        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "iea15mw_overset.vtu"
            export_vtk(vol_mesh, out)
            assert out.exists()
            assert out.stat().st_size > 1000  # non-trivial file

    # -- webs exclusion --

    def test_shell_includes_webs(self, shell_only):
        """FEM shell mesh must contain shear-web element sets."""
        names = {s["name"] for s in shell_only["sets"]["element"]}
        assert "allShearWebEls" in names, (
            "Shell mesh should include shear-web elements for FEM analysis"
        )

    def test_vol_excludes_webs(self, vol_mesh):
        """Overset body mesh must NOT contain shear-web surfaces.

        Webs are internal to the blade structure and would break the
        closed-body assumption required for overset CFD extrusion.
        """
        names = {s["name"] for s in vol_mesh["sets"]["element"]}
        assert "allShearWebEls" not in names, (
            "Overset mesh must not contain shear-web elements"
        )
        # Also verify no web node count bleed: elements must only come from OML + caps
        for s in vol_mesh["sets"]["element"]:
            assert "web" not in s["name"].lower() and "sw" not in s["name"].lower(), (
                f"Web-related element set found in overset mesh: {s['name']}"
            )

    # -- total_thickness API --

    def test_total_thickness_matches_layers_equivalent(self, iea_blade):
        """Specifying total_thickness must yield the same n_layers as manual calculation."""
        import math
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh

        t0, g = 0.05, 1.2
        total_t = 0.5
        # Expected n_layers from geometric series
        n_expected = math.ceil(math.log(1 + total_t * (g - 1) / t0) / math.log(g))

        vol = get_vol_mesh(
            iea_blade,
            elementSize=1.0,
            overset_total_thickness=total_t,
            overset_first_thickness=t0,
            overset_growth=g,
            overset_smooth_iters=0,
        )
        assert vol["n_layers"] == n_expected

    def test_total_thickness_and_layers_raises(self, iea_blade):
        """Supplying both overset_layers and overset_total_thickness must raise."""
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh

        with pytest.raises(ValueError, match="not both"):
            get_vol_mesh(
                iea_blade,
                elementSize=1.0,
                overset_layers=3,
                overset_total_thickness=0.5,
                overset_first_thickness=0.05,
                overset_smooth_iters=0,
            )

    def test_total_thickness_uniform_growth(self, iea_blade):
        """With growth=1.0, n_layers = ceil(total_thickness / first_thickness)."""
        import math
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh

        t0, total_t = 0.1, 0.5
        n_expected = math.ceil(total_t / t0)

        vol = get_vol_mesh(
            iea_blade,
            elementSize=1.0,
            overset_total_thickness=total_t,
            overset_first_thickness=t0,
            overset_growth=1.0,
            overset_smooth_iters=0,
        )
        assert vol["n_layers"] == n_expected


# ---------------------------------------------------------------------------
# cyl_blend_exp: cylinder convergence
# ---------------------------------------------------------------------------

class TestCylBlendExp:
    """Tests for cyl_blend_exp=2.0 (restored default) with TE zone exclusion."""

    @pytest.fixture(scope="class")
    def blended_mesh(self, iea_blade):
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        return get_vol_mesh(
            iea_blade,
            elementSize=1.0,
            overset_layers=5,
            overset_first_thickness=0.05,
            overset_growth=1.2,
            overset_smooth_iters=0,
            overset_cyl_blend_exp=2.0,
        )

    def test_cyl_blend_default_is_2(self, iea_blade):
        """get_vol_mesh must use cyl_blend_exp=2.0 when not specified."""
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        import inspect
        sig = inspect.signature(get_vol_mesh)
        assert sig.parameters["overset_cyl_blend_exp"].default == 2.0

    def test_cyl_blend_produces_valid_mesh(self, blended_mesh):
        """cyl_blend_exp=2.0 must produce a mesh with no degenerate hexes."""
        els = blended_mesh["elements"]
        assert els.shape[1] == 8
        assert np.all(els >= 0)

    def test_cyl_blend_no_negative_jacobians(self, iea_blade):
        """cyl_blend_exp=2.0 must not introduce additional negative Jacobians
        compared to pure normal extrusion (any pre-existing degenerate elements
        from the surface mesh itself are irrelevant to the blend behaviour).
        """
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh

        kwargs = dict(
            elementSize=1.0,
            overset_layers=5,
            overset_first_thickness=0.05,
            overset_growth=1.2,
            overset_smooth_iters=0,
        )

        def count_neg_jac(vol):
            nodes = vol["nodes"]; els = vol["elements"]
            return sum(
                1 for e in els
                if np.dot(
                    np.cross(nodes[e[1]] - nodes[e[0]], nodes[e[3]] - nodes[e[0]]),
                    nodes[e[4]] - nodes[e[0]],
                ) <= 0
            )

        negj_none  = count_neg_jac(get_vol_mesh(iea_blade, **kwargs, overset_cyl_blend_exp=None))
        negj_blend = count_neg_jac(get_vol_mesh(iea_blade, **kwargs, overset_cyl_blend_exp=2.0))

        assert negj_blend <= negj_none, (
            f"cyl_blend_exp=2.0 introduced {negj_blend - negj_none} extra "
            f"negative Jacobians vs pure normal extrusion"
        )

    def test_cyl_blend_excluded_from_webs(self, blended_mesh):
        """Blended mesh must still exclude shear-web elements."""
        names = {s["name"] for s in blended_mesh["sets"]["element"]}
        assert "allShearWebEls" not in names
        for s in blended_mesh["sets"]["element"]:
            assert "web" not in s["name"].lower() and "sw" not in s["name"].lower()


# ---------------------------------------------------------------------------
# Two-zone mesh: BL zone + outer uniform zone
# ---------------------------------------------------------------------------

class TestOuterZone:
    """Tests for overset_outer_layers (uniform outer zone)."""

    @pytest.fixture(scope="class")
    def two_zone_mesh(self, iea_blade):
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        return get_vol_mesh(
            iea_blade,
            elementSize=1.0,
            overset_layers=5,
            overset_first_thickness=0.05,
            overset_growth=1.2,
            overset_smooth_iters=0,
            overset_outer_layers=3,
        )

    @pytest.fixture(scope="class")
    def bl_only_mesh(self, iea_blade):
        from aeroelast.models.blade.numad.mesh_gen import get_vol_mesh
        return get_vol_mesh(
            iea_blade,
            elementSize=1.0,
            overset_layers=5,
            overset_first_thickness=0.05,
            overset_growth=1.2,
            overset_smooth_iters=0,
            overset_outer_layers=0,
        )

    def test_outer_zone_more_elements(self, two_zone_mesh, bl_only_mesh):
        """Two-zone mesh must have more elements than BL-only mesh."""
        assert len(two_zone_mesh["elements"]) > len(bl_only_mesh["elements"])

    def test_outer_zone_element_count(self, two_zone_mesh, bl_only_mesh):
        """Total elements = BL_elements + outer_layers * n_surface_quads."""
        n_bl    = len(bl_only_mesh["elements"])
        n_total = len(two_zone_mesh["elements"])
        n_surf_quads = n_bl // 5  # bl_layers=5
        assert n_total == n_bl + 3 * n_surf_quads

    def test_outer_zone_bl_boundary_set(self, two_zone_mesh):
        """bl_boundary element set must exist in two-zone mesh."""
        names = {s["name"] for s in two_zone_mesh["sets"]["element"]}
        assert "bl_boundary" in names

    def test_outer_zone_overset_boundary_set(self, two_zone_mesh):
        """overset_boundary must still exist in two-zone mesh."""
        names = {s["name"] for s in two_zone_mesh["sets"]["element"]}
        assert "overset_boundary" in names

    def test_bl_only_has_no_bl_boundary_set(self, bl_only_mesh):
        """bl_boundary set must NOT appear when outer_layers=0."""
        names = {s["name"] for s in bl_only_mesh["sets"]["element"]}
        assert "bl_boundary" not in names

    def test_outer_zone_valid_hexes(self, two_zone_mesh):
        """Two-zone mesh must contain only valid brick-8 elements."""
        els = two_zone_mesh["elements"]
        assert els.shape[1] == 8
        assert np.all(els >= 0)

    def test_outer_zone_excludes_webs(self, two_zone_mesh):
        """Two-zone mesh must NOT contain shear-web element sets."""
        names = {s["name"] for s in two_zone_mesh["sets"]["element"]}
        assert "allShearWebEls" not in names
        for s in two_zone_mesh["sets"]["element"]:
            assert "web" not in s["name"].lower() and "sw" not in s["name"].lower()
