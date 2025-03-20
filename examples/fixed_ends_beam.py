"""
Test Case: 2D Beam Analysis with Plane Stress and Shell Elements

Description:
-------------
This test case compares two formulations for a rectangular beam (10x1 units):
1. Plane stress elements (2D linear elasticity)
2. MITC4 shell elements (3D shell theory in 2D plane)

The beam is clamped at both ends and subjected to gravitational body force.

The implementation follows the test problem published in:
https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html

Parameters and Setup:
---------------------
- Geometry:
  - Dimensions: 10 units (length) x 1 unit (height)
  - Thickness (shell only): 0.01 units

- Mesh:
  - Structured quadrilateral mesh
  - Resolution: 80x8 cells (8 cells/unit)
  - Linear shape functions

- Material (Steel properties):
  - Young's Modulus (E) = 210e3 MPa
  - Poisson's Ratio (ν) = 0.3
  - Density (ρ) = 2e-3

- Loading:
  - Body force: Gravity load (9.81 m/s²) applied as:
    * Plane stress: Force per unit area = ρ * g
    * Shell: Force per unit volume = ρ * g * thickness

"""

from fem_shell.core.bc import BodyForce, DirichletCondition
from fem_shell.core.material import Material
from fem_shell.core.mesh import SquareShapeMesh
from fem_shell.elements import ElementFamily
from fem_shell.solvers.linear import LinearStaticSolver

# material definition
E = 210e3
NU = 0.3
RHO = 2e-3
THICKNESS = 0.01
GRAVITY = 9.810

material = Material(name="Steel", E=E, nu=NU, rho=RHO)

# Geometry and discretization
WIDTH, HEIGHT = 10, 1
CELLS_PER_UNIT = 8
NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT

# Force calculations
plane_load = [0, -material.rho * GRAVITY, 0]  # Plane stress (force/area)
shell_load = [0, -material.rho * GRAVITY * THICKNESS, 0]  # Shell (force/length)

# Select model type
model_configs = {
    "plane": {
        "solver": {},
        "elements": {
            "material": material,
            "element_family": ElementFamily.PLANE,
        },
    },
    "shell": {
        "solver": {},
        "elements": {
            "material": material,
            "element_family": ElementFamily.SHELL,
            "thickness": THICKNESS,
        },
    },
}
# Initialize and solve
mesh = SquareShapeMesh.create_rectangle(WIDTH, HEIGHT, NX, NY)

for model_type, model in model_configs.items():
    print(f"================\n {model_type} model...\n================")

    problem = LinearStaticSolver(mesh=mesh, fem_model_properties=model)

    load = plane_load if model_type == "plane" else shell_load

    # Apply boundary conditions
    left = problem.get_dofs_by_nodeset_name("left")
    right = problem.get_dofs_by_nodeset_name("right")
    problem.add_dirichlet_conditions(
        [DirichletCondition(left, 0.0), DirichletCondition(right, 0.0)]
    )

    problem.add_body_forces([BodyForce(load)])

    # Solve and output
    U = problem.solve()
    print(f"  Max displacement ({model_type}): {U.max():.4e} m\n")
    # problem.view_results()
    problem.write_results(f"results_{model_type}.vtk")
