from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.core.mesh import BoxSurfaceMesh
from fem_shell.elements import ElementFamily
from fem_shell.postprocess.precice import FSIDataVisualizer
from fem_shell.solvers.fsi import LinearDynamicFSISolver

# material definition

E = 4000000
NU = 0.3
RHO = 3000
material = Material(name="Steel", E=E, nu=NU, rho=RHO)

# Domain geometry
WIDTH, HEIGHT = 1.0, 10.0
NX, NY, NZ = 5, 20, 5

model_config = {
    "solver": {
        "total_time": 1,
        "time_step": 1e-3,
        "adapter_cfg": "precice-adapter.yaml",
        "coupling_boundaries": ["left", "right", "top", "front", "back"],
    },
    "elements": {
        "material": material,
        "thickness": 0.1,
        "element_family": ElementFamily.SHELL,
    },
}

mesh = BoxSurfaceMesh.create_box(
    center=(0, HEIGHT / 2, 0), dims=(WIDTH, HEIGHT, WIDTH), nx=NX, ny=NY, nz=NZ
)
mesh.write_mesh("mesh.vtk")

problem = LinearDynamicFSISolver(mesh, model_config)
bottom_node_set = problem.get_dofs_by_nodeset_name("bottom")
problem.add_dirichlet_conditions([DirichletCondition(bottom_node_set, 0.0)])
problem.solve()

# Crear visualizador
visualizer = FSIDataVisualizer("precice-Solid-watchpoint-Flap-Tip.log")
visualizer.plot_displacement(save_path="desplazamientos.png")
visualizer.plot_force(save_path="fuerzas.png")
