"""
Exportar fuerzas BEM sobre la superficie de la pala a VTU
=========================================================

Genera el mesh de la pala, corre BEM y proyecta las fuerzas aerodinámicas
sobre los nodos de la malla de superficie. Exporta archivos VTU para
cada caso de carga, listos para visualizar en Paraview o ParaFOAM.

Campos exportados por nodo:
  - Force          : vector de fuerza nodal (Fx, Fy, Fz) [N]
  - Force_Mag      : módulo de la fuerza nodal [N]
  - Np_per_m       : fuerza normal por unidad de longitud (flapwise) [N/m]
  - Tp_per_m       : fuerza tangencial por unidad de longitud (edgewise) [N/m]
  - alpha_deg      : ángulo de ataque de la estación BEM más cercana [°]
  - Cl             : coeficiente de sustentación [-]
  - Cd             : coeficiente de resistencia [-]

Dos casos:
  1. parked  — V = 7.15 m/s,  ω = 5.10 RPM
  2. rated   — V = 10.59 m/s, ω = 7.56 RPM
"""

from collections import defaultdict
from pathlib import Path

import meshio
import numpy as np
from scipy.interpolate import interp1d

from fem_shell.models.blade.aerodynamics import load_blade_aero
from fem_shell.models.blade.model import Blade
from fem_shell.solvers.bem.engine import BEMSolver
from fem_shell.solvers.bem.force_projection import ForceProjector

# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
YAML_FILE = str(SCRIPT_DIR  / "IEA-15-240-RWT.yaml")

# Tamaño de elemento: aumentar para meshes más rápidos de generar
ELEMENT_SIZE = 0.15

# ── 1. Datos aerodinámicos ────────────────────────────────────────────
print("Cargando datos aerodinámicos...")
blade_aero = load_blade_aero(YAML_FILE)

# ── 2. Mesh de la pala ───────────────────────────────────────────────
print(f"Generando mesh (element_size={ELEMENT_SIZE} m)...")
blade = Blade(blade_yaml=YAML_FILE, element_size=ELEMENT_SIZE)
blade.generate_mesh()
mesh = blade.mesh
print(f"  Nodos: {len(mesh.nodes):,}   Elementos: {len(mesh.elements):,}")

# ── 3. BEM solver ────────────────────────────────────────────────────
solver = BEMSolver(
    blade_aero,
    rho=1.225,
    mu=1.81206e-5,
    hub_height=150.0,
    shear_exp=0.2,
)

CASES = {
    "parked": dict(v_inf=7.15,  omega=5.09850149029421, pitch=0.0),
    "rated":  dict(v_inf=10.59, omega=7.56,             pitch=0.0),
}

# ── 4. Proyector de fuerzas ──────────────────────────────────────────
# La pala del IEA-15 tiene el span en Z, flapwise en X, edgewise en Y
projector = ForceProjector(
    mesh=mesh,
    blade_aero=blade_aero,
    span_direction=[0.0, 0.0, 1.0],
    normal_direction=[1.0, 0.0, 0.0],
    tangential_direction=[0.0, 1.0, 0.0],
)

# ── 5. Preparar geometría meshio (igual para todos los casos) ─────────
coords = mesh.coords_array  # (N, 3) — indices coinciden con ForceProjector
cells_dict: dict = defaultdict(list)
for el in mesh.elements:
    indices = tuple(mesh.node_id_to_index[nid] for nid in el.node_ids)
    cells_dict[el.element_type.name].append(indices)
cells = [(et, np.array(el_list)) for et, el_list in cells_dict.items()]

# Coordenada span de cada nodo (proyectada sobre Z)
span_node = coords[:, 2]


def _bem_scalars_to_nodes(bem_result) -> dict:
    """Interpola campos escalares BEM desde estaciones radiales a nodos.

    Usa interpolación lineal; las extrapolaciones en el raíz y la punta
    mantienen el valor del extremo más cercano.
    """
    r = bem_result.r
    scalar_fields = {
        "Np_per_m":  bem_result.Np,
        "Tp_per_m":  bem_result.Tp,
        "alpha_deg": bem_result.alpha,
        "Cl":        bem_result.cl,
        "Cd":        bem_result.cd,
    }
    node_fields = {}
    for name, values in scalar_fields.items():
        f = interp1d(
            r, values,
            kind="linear",
            bounds_error=False,
            fill_value=(values[0], values[-1]),
        )
        node_fields[name] = f(span_node).astype(np.float64)
    return node_fields


# ── 6. Por cada caso: BEM → proyección → VTU ─────────────────────────
for case_name, params in CASES.items():
    print(f"\nCaso: {case_name}  (V={params['v_inf']} m/s, ω={params['omega']:.2f} RPM)")

    result = solver.compute(**params)
    forces = projector.project(result)          # (N, 3)  [N]
    force_mag = np.linalg.norm(forces, axis=1)  # (N,)    [N]

    scalar_fields = _bem_scalars_to_nodes(result)

    # Verificación rápida de conservación
    chk = projector.verify(result, forces)
    print(f"  Error fuerza total: {chk['force_error']:.4f} N  "
          f"(BEM={np.linalg.norm(chk['force_bem']):.1f} N, "
          f"mesh={np.linalg.norm(chk['force_mesh']):.1f} N)")

    point_data = {
        "Force":     forces.astype(np.float64),
        "Force_Mag": force_mag.astype(np.float64),
        **scalar_fields,
    }

    out_path = SCRIPT_DIR / f"blade_bem_forces_{case_name}.vtu"
    m = meshio.Mesh(points=coords, cells=cells, point_data=point_data)
    meshio.write(str(out_path), m)
    print(f"  Exportado: {out_path}")

print("\nListo. Abrí los .vtu en Paraview para visualizar.")
