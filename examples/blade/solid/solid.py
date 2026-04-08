from pathlib import Path

from fem_shell.core.bc import DirichletCondition
from fem_shell.elements import ElementFamily
from fem_shell.models.blade.model import Blade
from fem_shell.postprocess.precice import FSIDataVisualizer
from fem_shell.solvers.fsi import LinearDynamicFSISolver

SCRIPT_DIR = Path(__file__).parent
BLADE_YAML_PATH = SCRIPT_DIR / "IEA-15-240-RWT.yaml"
ELEMENT_SIZE = 0.15

blade = Blade(blade_yaml=str(BLADE_YAML_PATH), element_size=ELEMENT_SIZE)
blade.generate_mesh(renumber="rcm")
mesh = blade.mesh
mesh.write_mesh(str(SCRIPT_DIR / "blade_mesh.stl"), close_tip=True)

# Build per-element-set material properties from WindIO YAML sections
properties = blade.get_element_properties()

model_config = {
    "solver": {
        "total_time": 15,
        "time_step": 0.0001,
        "coupling_boundaries": ["allOuterShellNods"],
        "coupling": {
            "participant": "Solid",
            "config_file": SCRIPT_DIR / "../precice-config.xml",
            "coupling_mesh": "Solid-Mesh",
            "write_data": "Displacement",
            "read_data": "Force",
        },
        # Rayleigh damping
        "damping": {
            "enabled": True,
            "eta_m": 0.02,
            "eta_k": 0.02,
        },
        # Newmark-beta parameters
        "beta": 0.25,
        "gamma": 0.5,
        # Solver type: "auto", "direct", or "iterative"
        "solver_type": "auto",
    },
    "elements": {
        "properties": properties,
        "element_family": ElementFamily.SHELL,
    },
}

# # Visualize element properties (thickness, stiffness, plies, etc.)
# element_data = build_element_data(properties, mesh)
# mesh.view(element_data=element_data)

# exit()

problem = LinearDynamicFSISolver(mesh, model_config)
bottom_node_set = problem.get_dofs_by_nodeset_name("RootNodes")
problem.add_dirichlet_conditions([DirichletCondition(bottom_node_set, 0.0)])
problem.solve()

# Crear visualizador
visualizer = FSIDataVisualizer("precice-Solid-watchpoint-Flap-Tip.log")
visualizer.plot_displacement(save_path="desplazamientos.png")
visualizer.plot_force(save_path="fuerzas.png")
