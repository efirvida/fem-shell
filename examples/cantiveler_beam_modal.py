from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.core.mesh import SquareShapeMesh
from fem_shell.elements import ElementFamily
from fem_shell.solvers import ModalSolver

# material definition
E = 210e9
NU = 0.3
RHO = 7850
material = Material(name="Steel", E=E, nu=NU, rho=RHO)

# Domain geometry
WIDTH, HEIGHT = 1, 10
CELLS_PER_UNIT = 8
NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT

# Select model type
THICKNESS = 0.1

NUM_MODES = 10
model_configs = {
    "plane": {
        "solver": {"num_modes": NUM_MODES},
        "elements": {
            "material": material,
            "element_family": ElementFamily.PLANE,
        },
    },
    "shell": {
        "solver": {"num_modes": NUM_MODES},
        "elements": {
            "material": material,
            "element_family": ElementFamily.SHELL,
            "thickness": THICKNESS,
        },
    },
}

mesh = SquareShapeMesh.create_rectangle(WIDTH, HEIGHT, NX, NY)

for model_type, model in model_configs.items():
    print(f"================\n {model_type} model...\n================")

    problem = ModalSolver(mesh=mesh, fem_model_properties=model)

    # Apply boundary conditions
    clamped = problem.get_dofs_by_nodeset_name("bottom")
    problem.add_dirichlet_conditions([DirichletCondition(clamped, 0.0)])

    # Solve and output
    frequencies, _ = problem.solve()
    for mode, frec in enumerate(frequencies, start=1):
        print(f"  Mode {mode}: freq {frec:2f} Hz")
