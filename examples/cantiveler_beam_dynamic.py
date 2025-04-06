import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.core.mesh import SquareShapeMesh
from fem_shell.elements import ElementFamily
from fem_shell.solvers import LinearDynamicSolver

# ======================================================================================
# Configuración de la simulación
# ======================================================================================
# Material
E = 210e3  # Módulo de Young (MPa = N/mm²) -> Correcto
NU = 0.3  # Coeficiente de Poisson -> Sin unidades
RHO = 7.8e-3  # Densidad (ton/mm³) -> Corregido
material = Material(name="Acero", E=E, nu=NU, rho=RHO)

# Geometría
LENGTH = 8.0  # Longitud de la viga (mm)
HEIGHT = 0.2  # Altura de la sección (mm)
THICKNESS = 0.1  # Espesor (mm)

NX, NY = 20, 5  # Divisiones de la malla

# Configuración del solver
model_configs = {
    "solver": {
        "total_time": 2.0,
        "time_step": 1e-3,
        "use_critical_dt": False,
        "save_history": True,
        "output_folder": "results",
    },
    "elements": {
        "element_family": ElementFamily.SHELL,
        "material": material,
        "thickness": THICKNESS,
    },
}


# ======================================================================================
# Simulación principal
# ======================================================================================
# Crear malla
mesh = SquareShapeMesh.create_rectangle(LENGTH, HEIGHT, NX, NY)

# Configurar y resolver problema
problem = LinearDynamicSolver(mesh=mesh, fem_model_properties=model_configs)
clamped_dofs = problem.get_dofs_by_nodeset_name("left")
problem.add_dirichlet_conditions([DirichletCondition(clamped_dofs, 0.0)])
problem.solve()

# Exportar resultados
problem.write_results()

# ======================================================================================
# Cálculo de frecuencias naturales
# ======================================================================================
times = np.array(list(problem.time_history.keys()))

cmap = plt.get_cmap("plasma")
colors = cmap(times / max(times))

B = THICKNESS
H = HEIGHT
L = LENGTH

tip_dofs = list(problem.get_dofs_by_nodeset_name("right"))[:3]
tip_displacement = np.array([u[tip_dofs] for u in problem.time_history.values()])

cmap = plt.get_cmap("plasma")
colors = cmap(times / max(times))

I_y = B * H**3 / 12
I_z = H * B**3 / 12

omega_y = 1.875**2 * np.sqrt(E * I_y / (RHO * B * H * L**4))
omega_z = 1.875**2 * np.sqrt(E * I_z / (RHO * B * H * L**4))

fig = plt.figure()
ax = fig.gca()
y = max(tip_displacement[:, 1]) * np.sin(omega_z * times)
z = max(tip_displacement[:, 2]) * np.sin(omega_y * times)

lines = ax.plot(y, z, "--k", alpha=0.7)
markers = []


def draw_frame(n):
    markers.append(ax.plot(y[n], z[n], "o", color=colors[n])[0])
    return markers


anim = animation.FuncAnimation(fig, draw_frame, frames=len(times), interval=20, blit=True)
plt.show()
