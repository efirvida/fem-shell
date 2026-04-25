"""Comprehensive Shell Element Validation Tests - Fixed Version.

Uses exact same approach as working CCX parity tests.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from aeroelast.core.bc import DirichletCondition, NodalLoad
from aeroelast.core.material import IsotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.properties import ShellProperty
from aeroelast.elements import ElementFamily
from aeroelast.solvers.elasticity.static_linear import StaticLinearSolver
from aeroelast.solvers.elasticity.static_nonlinear import StaticNonlinearSolver
from aeroelast.solvers.modal import ModalSolver


# =============================================================================
# REFERENCE DATA
# =============================================================================

L, b, h = 1.0, 0.1, 0.001
E, nu, rho = 2.1e11, 0.3, 7800.0
STEEL = IsotropicMaterial(name="Steel", E=E, nu=nu, rho=rho)

# Expected reference values from validation tests
EXPECTED = {
    "ux": 4.835e-3,
    "uy": 7.147e-3,
    "uz": 7.145e-3,
    "modal_1": 0.553,
}


# =============================================================================
# EXACT COPY OF WORKING HELPERS FROM CCX PARITY TEST
# =============================================================================

def _build_cantilever_mesh(nx=8, ny=4):
    """Build cantilever mesh - EXACT copy."""
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, b, ny + 1)

    grid = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            node = Node([float(x), float(y), 0.0], geometric_node=False)
            mesh.add_node(node)
            grid[(i, j)] = node

    for j in range(ny):
        for i in range(nx):
            mesh.add_element(
                MeshElement(
                    nodes=[grid[(i, j)], grid[(i+1, j)], 
                           grid[(i+1, j+1)], grid[(i, j+1)]],
                    element_type=ElementType.quad,
                )
            )

    clamped_nodes = {n for n in mesh.nodes if np.isclose(n.x, 0.0, atol=1e-12)}
    free_nodes = {n for n in mesh.nodes if np.isclose(n.x, L, atol=1e-12)}

    mesh.add_node_set(NodeSet("clamped", clamped_nodes))
    mesh.add_node_set(NodeSet("free_edge", free_nodes))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))
    return mesh


def _center_free_edge_node(mesh):
    """Get center node of free edge - EXACT copy."""
    target_y = 0.5 * b
    candidates = [n for n in mesh.nodes if np.isclose(n.x, L, atol=1e-12)]
    return min(candidates, key=lambda n: abs(float(n.y) - target_y))


def _clamped_dofs(mesh, dofs_per_node):
    """Get clamped DOFs - EXACT copy."""
    dofs = []
    m = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = m[node.id] * dofs_per_node
        dofs.extend(range(i0, i0 + dofs_per_node))
    return sorted(set(dofs))


def _load_as_nodal(mesh, dofs_per_node, load6):
    """Create nodal loads - EXACT copy."""
    nodes = sorted(mesh.get_node_set("free_edge").nodes.values(), key=lambda n: n.id)
    n = max(len(nodes), 1)
    per_node = np.asarray(load6, dtype=float) / float(n)
    idx = mesh.node_id_to_index
    out = []
    for node in nodes:
        i0 = idx[node.id] * dofs_per_node
        dofs = list(range(i0, i0 + dofs_per_node))
        out.append(NodalLoad(dofs, per_node.tolist()))
    return out


# =============================================================================
# TEST CASES
# =============================================================================

class TestLinearStatic:
    """Linear static analysis tests."""
    
    def test_fx(self):
        """FX in-plane loading."""
        mesh = _build_cantilever_mesh()
        prop = ShellProperty(material=STEEL, thickness=h)
        
        cfg = {
            "solver": {},
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        solver = StaticLinearSolver(mesh, cfg)
        dpn = solver.domain.dofs_per_node
        
        solver.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver.add_nodal_loads(_load_as_nodal(mesh, dpn, (600.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        
        u = solver.solve()
        center = _center_free_edge_node(mesh)
        idx = mesh.node_id_to_index[center.id]
        ux = abs(u.array[idx * dpn + 0])
        
        print(f"\nFX: {ux*1000:.3f} mm (ref: {EXPECTED['ux']*1000:.3f} mm)")
        
        error = abs(ux - EXPECTED["ux"]) / EXPECTED["ux"] * 100
        assert error < 5.0
    
    def test_fy(self):
        """FY in-plane loading."""
        mesh = _build_cantilever_mesh()
        prop = ShellProperty(material=STEEL, thickness=h)
        
        cfg = {
            "solver": {},
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        solver = StaticLinearSolver(mesh, cfg)
        dpn = solver.domain.dofs_per_node
        
        solver.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver.add_nodal_loads(_load_as_nodal(mesh, dpn, (0.0, 600.0, 0.0, 0.0, 0.0, 0.0)))
        
        u = solver.solve()
        center = _center_free_edge_node(mesh)
        idx = mesh.node_id_to_index[center.id]
        uy = abs(u.array[idx * dpn + 1])
        
        print(f"\nFY: {uy*1000:.3f} mm (ref: {EXPECTED['uy']*1000:.3f} mm)")
        
        error = abs(uy - EXPECTED["uy"]) / EXPECTED["uy"] * 100
        assert error < 5.0
    
    def test_fz(self):
        """FZ out-of-plane loading."""
        mesh = _build_cantilever_mesh()
        prop = ShellProperty(material=STEEL, thickness=h)
        
        cfg = {
            "solver": {},
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        solver = StaticLinearSolver(mesh, cfg)
        dpn = solver.domain.dofs_per_node
        
        solver.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver.add_nodal_loads(_load_as_nodal(mesh, dpn, (0.0, 0.0, 600.0, 0.0, 0.0, 0.0)))
        
        u = solver.solve()
        center = _center_free_edge_node(mesh)
        idx = mesh.node_id_to_index[center.id]
        uz = abs(u.array[idx * dpn + 2])
        
        print(f"\nFZ: {uz*1000:.3f} mm (ref: {EXPECTED['uz']*1000:.3f} mm)")
        
        error = abs(uz - EXPECTED["uz"]) / EXPECTED["uz"] * 100
        assert error < 5.0
    
    def test_ratio_physical(self):
        """KEY: Physical ratio uY/uX."""
        prop = ShellProperty(material=STEEL, thickness=h)
        
        cfg = {
            "solver": {},
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        # UX
        mesh = _build_cantilever_mesh()
        solver = StaticLinearSolver(mesh, cfg)
        dpn = solver.domain.dofs_per_node
        solver.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver.add_nodal_loads(_load_as_nodal(mesh, dpn, (600.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        u = solver.solve()
        center = _center_free_edge_node(mesh)
        idx = mesh.node_id_to_index[center.id]
        ux = abs(u.array[idx * dpn + 0])
        
        # UY
        mesh = _build_cantilever_mesh()
        solver = StaticLinearSolver(mesh, cfg)
        solver.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver.add_nodal_loads(_load_as_nodal(mesh, dpn, (0.0, 600.0, 0.0, 0.0, 0.0, 0.0)))
        u = solver.solve()
        idx = mesh.node_id_to_index[center.id]
        uy = abs(u.array[idx * dpn + 1])
        
        ratio = uy / ux
        
        print(f"\nPhysical ratio: uY/uX = {ratio:.2f}")
        print(f"Expected: 1.3 - 1.8")
        
        # This is the KEY validation
        assert 1.3 <= ratio <= 1.8, f"Ratio {ratio:.2f} outside physical range"


class TestNonlinearStatic:
    """Nonlinear static tests."""
    
    def test_geometric_nonlinearity(self):
        """Check geometric nonlinearity effect."""
        mesh = _build_cantilever_mesh()
        prop = ShellProperty(material=STEEL, thickness=h)
        
        # Linear solution
        cfg_lin = {
            "solver": {},
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        solver_lin = StaticLinearSolver(mesh, cfg_lin)
        dpn = solver_lin.domain.dofs_per_node
        solver_lin.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver_lin.add_nodal_loads(_load_as_nodal(mesh, dpn, (0.0, 0.0, 600.0, 0.0, 0.0, 0.0)))
        
        u_lin = solver_lin.solve()
        center = _center_free_edge_node(mesh)
        idx = mesh.node_id_to_index[center.id]
        dz_lin = abs(u_lin.array[idx * dpn + 2])
        
        # Nonlinear
        cfg_nl = {
            "solver": {
                "nl_initial_increment": 0.1,
                "nl_max_increments": 50,
            },
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        mesh = _build_cantilever_mesh()
        solver_nl = StaticNonlinearSolver(mesh, cfg_nl)
        solver_nl.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        solver_nl.add_nodal_loads(_load_as_nodal(mesh, dpn, (0.0, 0.0, 600.0, 0.0, 0.0, 0.0)))
        
        u_nl = solver_nl.solve()
        idx = mesh.node_id_to_index[center.id]
        dz_nl = abs(u_nl[idx * dpn + 2])
        
        ratio = dz_nl / dz_lin
        
        print(f"\nNonlinear ratio: {ratio:.3f} (expected ~1.0-1.3)")
        
        assert 0.9 <= ratio <= 1.3


class TestModal:
    """Modal tests."""
    
    def test_first_mode(self):
        """First modal frequency."""
        mesh = _build_cantilever_mesh()
        prop = ShellProperty(material=STEEL, thickness=h)
        
        cfg = {
            "solver": {"num_modes": 3},
            "elements": {
                "element_family": ElementFamily.SHELL,
                "properties": {"plate": prop},
                "span_direction": (1.0, 0.0, 0.0),
            },
        }
        
        solver = ModalSolver(mesh, cfg)
        dpn = solver.domain.dofs_per_node
        solver.add_dirichlet_conditions([
            DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)
        ])
        
        freqs, _ = solver.solve()
        f1 = freqs[0]
        
        print(f"\nModal mode 1: {f1:.3f} Hz (ref: {EXPECTED['modal_1']:.3f} Hz)")
        
        error = abs(f1 - EXPECTED["modal_1"]) / EXPECTED["modal_1"] * 100
        assert error < 2.0


if __name__ == "__main__":
    print("="*60)
    print("SHELL VALIDATION TESTS")
    print("="*60)
    
    for cls in [TestLinearStatic, TestNonlinearStatic, TestModal]:
        print(f"\n{cls.__name__}")
        print("-"*40)
        
        instance = cls()
        for m in dir(instance):
            if m.startswith("test_"):
                try:
                    getattr(instance, m)()
                except AssertionError as e:
                    print(f"  FAILED: {e}")
                except Exception as e:
                    print(f"  ERROR: {e}")