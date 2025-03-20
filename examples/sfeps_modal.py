import numpy as np
from sfepy.discrete import (
    Conditions,
    Equation,
    Equations,
    EssentialBC,
    FieldVariable,
    Material,
    Problem,
)
from sfepy.discrete.fem import FEDomain, Field
from sfepy.mechanics.matcoefs import elasticity_from_youngpoisson
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers.eigen import EigenvalueSolver
from sfepy.terms import Term

solve_eigen = EigenvalueSolver

# Parameters
E = 210e3  # Young's modulus (MPa)
nu = 0.3  # Poisson's ratio
rho = 2e-3  # Density (tonne/mm³)
thickness = 0.1
width = 1
height = 10
cells_per_unit = 8
nx = cells_per_unit * width
ny = cells_per_unit * height
num_modes = 10

# Plane Stress Model
print("================\nPlane model...\n================")

# Generate 2D mesh
mesh_plane = gen_block_mesh((0.0, 0.0), (width, height), nx, ny, name="shell10x", verbose=False)
domain_plane = FEDomain("domain_plane", mesh_plane)
region_all_plane = domain_plane.create_region("All", "all")

# Define field (2D vector for displacements)
field_plane = Field.from_args("fu", np.float64, "vector", region_all_plane, approx_order=1)

# Material properties for plane stress
D_plane = elasticity_from_youngpoisson(2, E, nu, plane="stress")
material_plane = Material(
    "SteelPlane",
    values={
        "D": D_plane,
        "density": rho,
    },
)

# Variables
u_plane = FieldVariable("u", "unknown", field_plane)
v_plane = FieldVariable("v", "test", field_plane, primary_var_name="u")

# Equations
integral_plane = Integral("i", order=2)
term1_plane = Term.new(
    "dw_lin_elastic(m.D, v, u)",
    integral_plane,
    region_all_plane,
    m=material_plane,
    v=v_plane,
    u=u_plane,
)
term2_plane = Term.new(
    "dw_volume_dot(m.density, v, u)",
    integral_plane,
    region_all_plane,
    m=material_plane,
    v=v_plane,
    u=u_plane,
)
equation_plane = Equation("balance", term1_plane - term2_plane)
equations_plane = Equations([equation_plane])

# Boundary conditions: fix bottom edge
min_y_plane = np.min(mesh_plane.coors[:, 1])
bottom_nodes_plane = np.where(mesh_plane.coors[:, 1] <= min_y_plane + 1e-6)[0]
bottom_region_plane = domain_plane.create_region("Bottom", "vertices by vrt", bottom_nodes_plane)
ebc_plane = EssentialBC("fixed_plane", bottom_region_plane, {"u.all": 0.0})

# Solve eigenproblem
problem_plane = Problem("PlaneModel", equations=equations_plane)
problem_plane.time_update(ebcs=Conditions([ebc_plane]))
eigs_plane, _ = solve_eigen(
    problem_plane, np.asarray((0.0, 0.0)), num_modes, solver_kind="eig.sgscipy", eigenvectors=True
)
frequencies_plane = np.sqrt(np.abs(eigs_plane)) / (2 * np.pi)

for i, freq in enumerate(frequencies_plane):
    print(f"Mode {i + 1}: {freq:.2f} Hz")

# Shell Model (simulated with 3D solid elements due to SfePy shell complexity)
print("================\nShell model...\n================")

# Generate 3D block mesh (approximation)
mesh_shell = gen_block_mesh(
    (0.0, 0.0, 0.0), (width, height, thickness), nx, ny, name="shell10x", verbose=False
)
domain_shell = FEDomain("domain_shell", mesh_shell)
region_all_shell = domain_shell.create_region("All", "all")

# Define field (3D vector for displacements)
field_shell = Field.from_args("fu", np.float64, "vector", region_all_shell, approx_order=1)

# Material properties for 3D
D_shell = elasticity_from_youngpoisson(3, E, nu)
material_shell = Material(
    "SteelShell",
    values={
        "D": D_shell,
        "density": rho,
    },
)

# Variables
u_shell = FieldVariable("u", "unknown", field_shell)
v_shell = FieldVariable("v", "test", field_shell, primary_var_name="u")

# Equations
integral_shell = Integral("i", order=2)
term1_shell = Term.new(
    "dw_lin_elastic(m.D, v, u)",
    integral_shell,
    region_all_shell,
    m=material_shell,
    v=v_shell,
    u=u_shell,
)
term2_shell = Term.new(
    "dw_volume_dot(m.density, v, u)",
    integral_shell,
    region_all_shell,
    m=material_shell,
    v=v_shell,
    u=u_shell,
)
equation_shell = Equation("balance", term1_shell - term2_shell)
equations_shell = Equations([equation_shell])

# Boundary conditions: fix bottom face
min_z_shell = np.min(mesh_shell.coors[:, 2])
bottom_nodes_shell = np.where(mesh_shell.coors[:, 2] <= min_z_shell + 1e-6)[0]
bottom_region_shell = domain_shell.create_region("Bottom", "vertices by vrt", bottom_nodes_shell)
ebc_shell = EssentialBC("fixed_shell", bottom_region_shell, {"u.all": 0.0})

# Solve eigenproblem
problem_shell = Problem("ShellModel", equations=equations_shell)
problem_shell.time_update(ebcs=Conditions([ebc_shell]))
eigs_shell, _ = solve_eigen(
    problem_shell, np.asarray((0.0, 0.0)), num_modes, solver_kind="eig.sgscipy", eigenvectors=True
)
frequencies_shell = np.sqrt(np.abs(eigs_shell)) / (2 * np.pi)

for i, freq in enumerate(frequencies_shell):
    print(f"Mode {i + 1}: {freq:.2f} Hz")
