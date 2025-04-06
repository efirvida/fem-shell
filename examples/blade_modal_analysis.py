import os

from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.elements import ElementFamily
from fem_shell.models.blade.mesh import Blade
from fem_shell.solvers import ModalSolver

# material definition
E = 210e9
NU = 0.3
RHO = 7850
material = Material(name="Steel", E=E, nu=NU, rho=RHO)

blade_yaml = os.path.join(
    os.getcwd(), "examples", "reference_turbines", "yamls", "IEA-15-240-RWT_VolturnUS-S.yaml"
)
blade = Blade(blade_yaml=blade_yaml, element_size=0.1)
blade.generate()
mesh = blade.mesh
mesh.view()

# Select model type
THICKNESS = 0.1

NUM_MODES = 10
model_configs = {
    "solver": {"num_modes": NUM_MODES},
    "elements": {
        "material": material,
        "element_family": ElementFamily.SHELL,
        "thickness": THICKNESS,
    },
}


print("================================")
problem = ModalSolver(mesh=mesh, fem_model_properties=model_configs)

# Apply boundary conditions
clamped = problem.get_dofs_by_nodeset_name("RootNodes")
problem.add_dirichlet_conditions([DirichletCondition(clamped, 0.0)])

# Solve and output
frequencies, _ = problem.solve()
for mode, frec in enumerate(frequencies, start=1):
    print(f"  Mode {mode}: freq {frec:2f} Hz")
print("================================")
