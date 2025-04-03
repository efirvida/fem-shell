import cProfile
import pstats

from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.core.mesh import SquareShapeMesh
from fem_shell.elements import ElementFamily
from fem_shell.solvers import LinearStaticSolver

pr = cProfile.Profile()
pr.enable()
# Material definition
E = 210e9  # Young's modulus (Pa)
NU = 0.3  # Poisson's ratio
RHO = 7850  # Density (kg/m³)
material = Material(name="Steel", E=E, nu=NU, rho=RHO)

# Domain geometry (beam-like structure)
WIDTH, HEIGHT = 1, 20  # Dimensions (m)
CELLS_PER_UNIT = 10  # Mesh density
NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT
THICKNESS = 0.01  # Beam thickness (m)

model_configs = {
    "solver": {},
    "elements": {
        "material": material,
        "thickness": THICKNESS,
        "element_family": ElementFamily.SHELL,  # Shell elements (MITC4)
    },
}

# Create mesh and problem
mesh = SquareShapeMesh.create_rectangle(WIDTH, HEIGHT, NX, NY, quadratic=False)
problem = LinearStaticSolver(mesh=mesh, fem_model_properties=model_configs)

# ------------------------------------------------------------------
# DOF sets for boundary conditions (6 DOFs per node: [ux, uy, uz, θx, θy, θz])
clamped_dofs = list(problem.get_dofs_by_nodeset_name("bottom"))  # Base nodes
top_dofs = list(problem.get_dofs_by_nodeset_name("top"))  # Top nodes

# Extract DOF groups by type (for clarity)
# Base DOFs
clamped_x_dofs = clamped_dofs[0::6]  # ux
clamped_y_dofs = clamped_dofs[1::6]  # uy
clamped_z_dofs = clamped_dofs[2::6]  # uz
clamped_theta_x_dofs = clamped_dofs[3::6]  # θx
clamped_theta_y_dofs = clamped_dofs[4::6]  # θy
clamped_theta_z_dofs = clamped_dofs[5::6]  # θz

# Top DOFs
top_x_dofs = top_dofs[0::6]
top_y_dofs = top_dofs[1::6]
top_z_dofs = top_dofs[2::6]
top_theta_x_dofs = top_dofs[3::6]
top_theta_y_dofs = top_dofs[4::6]
top_theta_z_dofs = top_dofs[5::6]

# ------------------------------------------------------------------
# Boundary Case Setup
# Case 0: Pinned-Pinned      - Base and top with pinned conditions (fixed translations)
# Case 1: Fixed-Pinned       - Fixed base, pinned top
# Case 2: Fixed-Fixed        - Fixed base and top (restrict all DOFs except axial displacement)
# Case 3: Fixed-Free         - Fixed base, free top (cantilever)
case = 1  # Change to 0, 1, 2, or 3

# ------------------------------------------------------------------
# Boundary Conditions Definitions
# Base conditions
fixed_bottom = [DirichletCondition(clamped_dofs, 0.0)]  # Fully clamped (all DOFs = 0)
pinned_bottom = [  # Pinned: fix translations, free rotations
    DirichletCondition(clamped_x_dofs, 0.0),
    DirichletCondition(clamped_y_dofs, 0.0),
    DirichletCondition(clamped_z_dofs, 0.0),
]

# Top conditions
fixed_top = [  # Clamped top (restrict all except uy for loading)
    DirichletCondition(top_x_dofs, 0.0),
    DirichletCondition(top_z_dofs, 0.0),
    DirichletCondition(top_theta_x_dofs, 0.0),
    DirichletCondition(top_theta_y_dofs, 0.0),
    DirichletCondition(top_theta_z_dofs, 0.0),
]
pinned_top = [  # Pinned top: fix translations, free rotations
    DirichletCondition(top_x_dofs, 0.0),
    DirichletCondition(top_z_dofs, 0.0),
]

# Apply boundary conditions
if case == 0:  # Pinned-Pinned
    problem.add_dirichlet_conditions(pinned_bottom)
    problem.add_dirichlet_conditions(pinned_top)
elif case == 1:  # Fixed-Pinned
    problem.add_dirichlet_conditions(fixed_bottom)
    problem.add_dirichlet_conditions(pinned_top)
elif case == 2:  # Fixed-Fixed
    problem.add_dirichlet_conditions(fixed_bottom)
    problem.add_dirichlet_conditions(fixed_top)
elif case == 3:  # Fixed-Free
    problem.add_dirichlet_conditions(fixed_bottom)

# Apply compressive load (axial direction = Y)
problem.add_force_on_dofs(dofs=top_y_dofs, value=-20)

# Solve and save results
problem.solve()
problem.write_results(f"beam_case_{case}.vtk")
pr.disable()
stats = pstats.Stats(pr).sort_stats("cumulative")
stats.print_stats(10)
problem.print_solver_info()
