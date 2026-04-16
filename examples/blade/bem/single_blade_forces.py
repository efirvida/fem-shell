"""
Ejemplo: Cálculo de fuerzas aerodinámicas en una pala IEA 15 MW
================================================================

Carga la definición de la pala desde el YAML WindIO, ejecuta BEM con CCBlade
y muestra las distribuciones seccionales y los resultados globales.

Dos casos:
  1. Pala estacionada (parked) a 45 m/s — condición extrema de carga
  2. Operación cerca de potencia nominal a 10.59 m/s, 7.56 RPM
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem_shell.models.blade.aerodynamics import load_blade_aero
from fem_shell.solvers.bem.engine import BEMSolver

# ── 1. Cargar datos aerodinámicos de la pala ─────────────────────────
YAML_FILE = str(Path(__file__).resolve().parent.parent / "solid" / "IEA-15-240-RWT.yaml")

blade_aero = load_blade_aero(YAML_FILE)

print(f"Pala: longitud = {blade_aero.blade_length:.1f} m")
print(f"       R_hub   = {blade_aero.hub_radius:.2f} m")
print(f"       R_tip   = {blade_aero.rotor_radius:.2f} m")
print(f"       N_palas = {blade_aero.n_blades}")
print(f"       Estaciones BEM: {len(blade_aero.stations)}")
print()

# ── 2. Crear el solver BEM ───────────────────────────────────────────
solver = BEMSolver(
    blade_aero,
    rho=1.225,  # kg/m³
    mu=1.81206e-5,  # Pa·s
    hub_height=150.0,  # m
    shear_exp=0.2,
)

V_INF = 7.15
OMEGA = 5.09850149029421
PITCH = 0

# ── 3. Caso 1: Pala estacionada (parked) ────────────────────────────
result_parked = solver.compute(
    v_inf=V_INF,  # m/s (viento extremo)
    omega=OMEGA,  # RPM (estacionada)
    pitch=PITCH,  # grados
)

print("=" * 60)
print(f"CASO 1: Pala estacionada — V = {V_INF:.2f} m/s, ω = {OMEGA:.2f} RPM")
print("=" * 60)
print(f"  Thrust (rotor)  = {result_parked.thrust / 1e3:10.1f} kN")
print(f"  Torque (rotor)  = {result_parked.torque / 1e3:10.1f} kN·m")
print(f"  Power           = {result_parked.power / 1e6:10.3f} MW")
print()
print(
    f"  {'r [m]':>8s}  {'Np [N/m]':>10s}  {'Tp [N/m]':>10s}  {'α [°]':>8s}  {'Cl':>8s}  {'Cd':>8s}"
)
print("  " + "-" * 58)
for i in range(0, len(result_parked.r), 5):  # cada 5 estaciones
    print(
        f"  {result_parked.r[i]:8.1f}"
        f"  {result_parked.Np[i]:10.1f}"
        f"  {result_parked.Tp[i]:10.1f}"
        f"  {result_parked.alpha[i]:8.2f}"
        f"  {result_parked.cl[i]:8.4f}"
        f"  {result_parked.cd[i]:8.4f}"
    )

# ── 4. Caso 2: Operación nominal ────────────────────────────────────
result_rated = solver.compute(
    v_inf=10.59,  # m/s (cerca de nominal)
    omega=7.56,  # RPM
    pitch=0.0,  # grados
)

print()
print("=" * 60)
print("CASO 2: Operación nominal — V = 10.59 m/s, ω = 7.56 RPM")
print("=" * 60)
print(f"  Thrust (rotor)  = {result_rated.thrust / 1e3:10.1f} kN")
print(f"  Torque (rotor)  = {result_rated.torque / 1e3:10.1f} kN·m")
print(f"  Power           = {result_rated.power / 1e6:10.3f} MW")
print()


# ── 5. Fuerza total en UNA pala (integración trapezoidal) ───────────
def blade_total_force(result):
    """Integra Np y Tp a lo largo del span para una sola pala."""
    F_normal = np.trapezoid(result.Np, result.r)  # N
    F_tangential = np.trapezoid(result.Tp, result.r)  # N
    return F_normal, F_tangential


Fn_parked, Ft_parked = blade_total_force(result_parked)
Fn_rated, Ft_rated = blade_total_force(result_rated)

print("\nFuerza total en UNA pala:")
print(f"  {'':20s}  {'Normal [kN]':>12s}  {'Tangencial [kN]':>15s}  {'Resultante [kN]':>15s}")
print(
    f"  {'Estacionada':20s}  {Fn_parked / 1e3:12.1f}  {Ft_parked / 1e3:15.1f}  {np.hypot(Fn_parked, Ft_parked) / 1e3:15.1f}"
)
print(
    f"  {'Operación nominal':20s}  {Fn_rated / 1e3:12.1f}  {Ft_rated / 1e3:15.1f}  {np.hypot(Fn_rated, Ft_rated) / 1e3:15.1f}"
)

# ── 6. Gráficas ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Np distribution
ax = axes[0, 0]
ax.plot(result_parked.r, result_parked.Np / 1e3, "r-o", ms=2, label="Parked 45 m/s")
ax.plot(result_rated.r, result_rated.Np / 1e3, "b-s", ms=2, label="Rated 10.59 m/s")
ax.set_xlabel("r [m]")
ax.set_ylabel("Np [kN/m]")
ax.set_title("Fuerza normal (flapwise)")
ax.legend()
ax.grid(True, alpha=0.3)

# Tp distribution
ax = axes[0, 1]
ax.plot(result_parked.r, result_parked.Tp / 1e3, "r-o", ms=2, label="Parked")
ax.plot(result_rated.r, result_rated.Tp / 1e3, "b-s", ms=2, label="Rated")
ax.set_xlabel("r [m]")
ax.set_ylabel("Tp [kN/m]")
ax.set_title("Fuerza tangencial (edgewise)")
ax.legend()
ax.grid(True, alpha=0.3)

# Alpha
ax = axes[1, 0]
ax.plot(result_parked.r, result_parked.alpha, "r-o", ms=2, label="Parked")
ax.plot(result_rated.r, result_rated.alpha, "b-s", ms=2, label="Rated")
ax.set_xlabel("r [m]")
ax.set_ylabel("α [°]")
ax.set_title("Ángulo de ataque")
ax.legend()
ax.grid(True, alpha=0.3)

# Cl
ax = axes[1, 1]
ax.plot(result_parked.r, result_parked.cl, "r-o", ms=2, label="Parked")
ax.plot(result_rated.r, result_rated.cl, "b-s", ms=2, label="Rated")
ax.set_xlabel("r [m]")
ax.set_ylabel("Cl [-]")
ax.set_title("Coeficiente de sustentación")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("IEA 15 MW — Fuerzas aerodinámicas BEM (una pala)", fontsize=14)
fig.tight_layout()
plt.savefig("bem_single_blade_forces.png", dpi=150)
plt.show()

print("\nGráfica guardada en: bem_single_blade_forces.png")
