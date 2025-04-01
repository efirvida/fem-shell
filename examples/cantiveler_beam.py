from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.core.mesh import SquareShapeMesh
from fem_shell.elements import ElementFamily
from fem_shell.solvers import LinearStaticSolver

# material definition
E = 210e9
NU = 0.3
RHO = 7850
material = Material(name="Steel", E=E, nu=NU, rho=RHO)

# Domain geometry
WIDTH, HEIGHT = 1, 20
CELLS_PER_UNIT = 10
NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT

# Select model type
THICKNESS = 0.1

model_configs = {
    "solver": {},
    "elements": {
        "material": material,
        "thickness": 1,
        "element_family": ElementFamily.SHELL,
    },
}


mesh = SquareShapeMesh.create_rectangle(WIDTH, HEIGHT, NX, NY, quadratic=False)

problem = LinearStaticSolver(mesh=mesh, fem_model_properties=model_configs)

# Apply boundary conditions
clamped = list(problem.get_dofs_by_nodeset_name("bottom"))[::6]
top_dofs = list(problem.get_dofs_by_nodeset_name("top"))
top_x_dofs = list(problem.get_dofs_by_nodeset_name("top"))[::6]

problem.add_dirichlet_conditions([
    DirichletCondition(clamped, 0.0),
    DirichletCondition(top_x_dofs, 0.0),
])

problem.add_force_on_dofs(dofs=top_dofs, value=[0, -20])

problem.solve()
problem.write_results("clamped4.vtk")
