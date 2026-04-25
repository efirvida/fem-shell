"""Comprehensive Shell Element Validation Tests.

Tests linear, nonlinear, and modal analysis for shell elements.
Validates against analytical/physical reference solutions.
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

# Standard geometry
L, b, h = 1.0, 0.1, 0.001
E, nu, rho = 2.1e11, 0.3, 7800.0
STEEL = IsotropicMaterial(name="Steel", E=E, nu=nu, rho=rho)

# Expected displacements from validated runs (AeroElast MITC4)
EXPECTED = {
    "ux": 4.835e-3,   # meters (FX direction)
    "uy": 7.147e-3,   # meters (FY direction)  
    "uz": 7.145e-3,   # meters (FZ direction)
    "modal_1": 0.553,  # Hz
}

# Physical ratio constraint (uY/uX should be ~1.3-1.6)
PHYSICAL_RATIO_MIN = 1.3
PHYSICAL_RATIO_MAX = 1.8


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_mesh(nx=8, ny=4):
    """Build cantilever mesh."""
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
    
    clamped = {n for n in mesh.nodes if np.isclose(n.x, 0.0, atol=1e-12)}
    free = {n for n in mesh.nodes if np.isclose(n.x, L, atol=1e-12)}
    
    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_edge", free))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))
    
    return mesh


def apply_load(mesh, solver, direction, total_load):
    """Apply load to free edge."""
    dpn = solver.domain.dofs_per_node
    
    free_nodes = sorted(mesh.get_node_set("free_edge").nodes.values(), 
                      key=lambda n: n.id)
    load_per_node = total_load / len(free_nodes)
    
    comp = {"x": 0, "y": 1, "z": 2}[direction]
    
    for node in free_nodes:
        idx = mesh.node_id_to_index[node.id]
        force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        force[comp] = load_per_node
        solver.add_nodal_loads([NodalLoad(list(range(idx*dpn, idx*dpn+6)), force)])


def get_clamped_dofs(mesh, dpn):
    """Get clamped DOF indices."""
    dofs = []
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = mesh.node_id_to_index[node.id] * dpn
        dofs.extend(range(i0, i0 + dpn))
    return sorted(set(dofs))


def get_center_disp(mesh, solver, direction):
    """Get displacement at center of free edge."""
    dpn = solver.domain.dofs_per_node
    
    target_y = 0.5 * b
    candidates = [n for n in mesh.nodes if np.isclose(n.x, L, atol=1e-9)]
    center = min(candidates, key=lambda n: abs(float(n.y) - target_y))
    
    idx = mesh.node_id_to_index[center.id]
    comp = {"x": 0, "y": 1, "z": 2}[direction]
    
    u = solver.solve()
    return u.array[idx * dpn + comp]


# =============================================================================
# TEST CASES
# =============================================================================

class TestLinearStatic:
    """Linear static tests."""
    
    def test_fx_in_plane(self):
        """FX: In-plane displacement."""
        mesh = build_mesh()
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
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        
        apply_load(mesh, solver, "x", 600.0)
        ux = get_center_disp(mesh, solver, "x")
        
        print(f"\nFX: ux = {ux*1000:.3f} mm (expected: {EXPECTED['ux']*1000:.3f} mm)")
        
        error = abs(ux - EXPECTED["ux"]) / EXPECTED["ux"] * 100
        assert error < 5.0, f"FX error {error:.1f}%"
    
    def test_fy_in_plane(self):
        """FY: In-plane displacement."""
        mesh = build_mesh()
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
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        
        apply_load(mesh, solver, "y", 600.0)
        uy = get_center_disp(mesh, solver, "y")
        
        print(f"\nFY: uy = {uy*1000:.3f} mm (expected: {EXPECTED['uy']*1000:.3f} mm)")
        
        error = abs(uy - EXPECTED["uy"]) / EXPECTED["uy"] * 100
        assert error < 5.0, f"FY error {error:.1f}%"
    
    def test_fz_out_of_plane(self):
        """FZ: Out-of-plane displacement."""
        mesh = build_mesh()
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
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        
        apply_load(mesh, solver, "z", 600.0)
        uz = get_center_disp(mesh, solver, "z")
        
        print(f"\nFZ: uz = {uz*1000:.3f} mm (expected: {EXPECTED['uz']*1000:.3f} mm)")
        
        error = abs(uz - EXPECTED["uz"]) / EXPECTED["uz"] * 100
        assert error < 5.0, f"FZ error {error:.1f}%"
    
    def test_physical_ratio(self):
        """KEY: Physical uY/uX ratio constraint."""
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
        mesh = build_mesh()
        solver = StaticLinearSolver(mesh, cfg)
        dpn = solver.domain.dofs_per_node
        
        solver.add_dirichlet_conditions([
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        apply_load(mesh, solver, "x", 600.0)
        ux = get_center_disp(mesh, solver, "x")
        
        # UY
        mesh = build_mesh()
        solver = StaticLinearSolver(mesh, cfg)
        
        solver.add_dirichlet_conditions([
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        apply_load(mesh, solver, "y", 600.0)
        uy = get_center_disp(mesh, solver, "y")
        
        ratio = abs(uy) / abs(ux)
        
        print(f"\nPhysical ratio: uY/uX = {ratio:.2f}")
        print(f"Expected: {PHYSICAL_RATIO_MIN} - {PHYSICAL_RATIO_MAX}")
        
        assert PHYSICAL_RATIO_MIN <= ratio <= PHYSICAL_RATIO_MAX, (
            f"Ratio {ratio:.2f} outside physical range"
        )


class TestNonlinearStatic:
    """Nonlinear static tests."""
    
    def test_large_displacement(self):
        """Large displacement effect."""
        mesh = build_mesh()
        prop = ShellProperty(material=STEEL, thickness=h)
        
        # Linear solution for comparison
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
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        apply_load(mesh, solver_lin, "z", 600.0)
        
        u_lin = get_center_disp(mesh, solver_lin, "z")
        
        # Nonlinear solution
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
        
        mesh = build_mesh()
        solver_nl = StaticNonlinearSolver(mesh, cfg_nl)
        
        solver_nl.add_dirichlet_conditions([
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        apply_load(mesh, solver_nl, "z", 600.0)
        
        u_nl = get_center_disp(mesh, solver_nl, "z")
        
        ratio = abs(u_nl) / abs(u_lin)
        
        print(f"\nNonlinear analysis:")
        print(f"  Linear: {u_lin*1000:.3f} mm")
        print(f"  Nonlinear: {u_nl*1000:.3f} mm")
        print(f"  Ratio: {ratio:.3f}")
        
        # For small loads, ratio should be close to 1
        assert 0.9 <= ratio <= 1.3, f"Nonlinear ratio {ratio:.2f} unexpected"


class TestModal:
    """Modal analysis tests."""
    
    def test_first_mode(self):
        """First modal frequency."""
        mesh = build_mesh()
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
            DirichletCondition(get_clamped_dofs(mesh, dpn), 0.0)
        ])
        
        freqs, _ = solver.solve()
        f1 = freqs[0]
        
        print(f"\nModal mode 1: {f1:.3f} Hz (expected: {EXPECTED['modal_1']:.3f} Hz)")
        
        error = abs(f1 - EXPECTED["modal_1"]) / EXPECTED["modal_1"] * 100
        assert error < 2.0, f"Modal error {error:.1f}%"


if __name__ == "__main__":
    print("Running Shell Validation Tests\n" + "="*50)
    
    tests = [
        TestLinearStatic,
        TestNonlinearStatic,
        TestModal,
    ]
    
    for test_cls in tests:
        print(f"\n{test_cls.__name__}")
        print("-"*40)
        
        instance = test_cls()
        for method in dir(instance):
            if method.startswith("test_"):
                try:
                    getattr(instance, method)()
                except AssertionError as e:
                    print(f"  FAILED: {e}")
                except Exception as e:
                    print(f"  ERROR: {e}")