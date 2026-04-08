import os

from fem_shell.core.bc import DirichletCondition
from fem_shell.core.properties import build_element_data
from fem_shell.elements import ElementFamily
from fem_shell.models.blade.model import Blade
from fem_shell.postprocess.precice import FSIDataVisualizer
from fem_shell.solvers.fsi import LinearDynamicFSISolver

BLADE_YAML_PATH = os.path.join(os.path.dirname(__file__), "IEA-15-240-RWT.yaml")
ELEMENT_SIZE = 0.2

blade = Blade(blade_yaml=BLADE_YAML_PATH, element_size=ELEMENT_SIZE)
blade.generate_mesh()
mesh = blade.mesh

# Build per-element-set material properties from WindIO YAML sections
properties = blade.get_element_properties()

model_config = {
    "solver": {
        "total_time": 15,
        "time_step": 0.0001,
        "adapter_cfg": "precice-adapter.yaml",
        "coupling_boundaries": ["allOuterShellNods"],
    },
    "elements": {
        "properties": properties,
        "element_family": ElementFamily.SHELL,
    },
}

# Visualize element properties (thickness, stiffness, plies, etc.)
element_data = build_element_data(properties, mesh)
mesh.view(element_data=element_data)

exit()

problem = LinearDynamicFSISolver(mesh, model_config)
bottom_node_set = problem.get_dofs_by_nodeset_name("RootNodes")
problem.add_dirichlet_conditions([DirichletCondition(bottom_node_set, 0.0)])
problem.solve()

# Crear visualizador
visualizer = FSIDataVisualizer("precice-Solid-watchpoint-Flap-Tip.log")
visualizer.plot_displacement(save_path="desplazamientos.png")
visualizer.plot_force(save_path="fuerzas.png")
