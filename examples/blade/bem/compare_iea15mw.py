"""
Comparación BEM: fem-shell (CCBlade) vs Referencia IEA 15 MW
=============================================================

Lee los datos de referencia de la tabla "Rotor Performance" del Excel
oficial del IEA-15-240-RWT y ejecuta nuestro BEMSolver en las mismas
condiciones (V, pitch, RPM).  Genera gráficas comparativas de:
  - Curva de potencia [MW]
  - Empuje (Thrust) [MN]
  - Par (Torque) [MN·m]
  - Coeficiente de potencia Cp
  - Coeficiente de empuje Ct

Requisitos:
  pip install openpyxl matplotlib
  El repo IEA-15-240-RWT clonado en /tmp o la ruta indicada abajo.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Ruta al Excel de referencia ──────────────────────────────────────
REF_XLSX = Path("/tmp/IEA-15-240-RWT/Documentation/IEA-15-240-RWT_tabular.xlsx")
YAML_FILE = str(Path(__file__).resolve().parent.parent / "solid" / "IEA-15-240-RWT.yaml")


def load_reference_data(xlsx_path: Path) -> dict[str, np.ndarray]:
    """Lee la hoja 'Rotor Performance' y devuelve un dict de arrays."""
    import openpyxl

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    ws = wb["Rotor Performance"]

    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data_rows = rows[1:]

    result = {}
    for col_idx, col_name in enumerate(header):
        if col_name is None:
            continue
        vals = []
        for row in data_rows:
            v = row[col_idx]
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                continue
        result[col_name] = np.array(vals, dtype=float)
    wb.close()
    return result


def run_bem_sweep(ref: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ejecuta el BEM a las mismas condiciones que la referencia."""
    from fem_shell.models.blade.aerodynamics import load_blade_aero
    from fem_shell.solvers.bem.engine import BEMSolver

    blade_aero = load_blade_aero(YAML_FILE)

    solver = BEMSolver(
        blade_aero,
        rho=1.225,
        mu=1.81206e-5,
        hub_height=150.0,
        shear_exp=0.2,
        tilt=6.0,
        precone=4.0,
    )

    v_arr = ref["Wind [m/s]"]
    pitch_arr = ref["Pitch [deg]"]
    rpm_arr = ref["Rotor Speed [rpm]"]

    n = len(v_arr)
    power = np.zeros(n)
    thrust = np.zeros(n)
    torque = np.zeros(n)

    for i in range(n):
        res = solver.compute(
            v_inf=float(v_arr[i]),
            omega=float(rpm_arr[i]),
            pitch=float(pitch_arr[i]),
        )
        power[i] = res.power
        thrust[i] = res.thrust
        torque[i] = res.torque

    # Coeficientes adimensionales
    rho = 1.225
    R = blade_aero.rotor_radius
    A = np.pi * R**2
    cp = power / (0.5 * rho * A * v_arr**3)
    ct = thrust / (0.5 * rho * A * v_arr**2)

    return {
        "Wind [m/s]": v_arr,
        "Power [MW]": power / 1e6,
        "Thrust [MN]": thrust / 1e6,
        "Torque [MNm]": torque / 1e6,
        "Cp": cp,
        "Ct": ct,
    }


def plot_comparison(ref: dict, bem: dict):
    """Genera las gráficas comparativas."""
    v_ref = ref["Wind [m/s]"]
    v_bem = bem["Wind [m/s]"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IEA 15 MW — fem-shell BEM vs Referencia WISDEM", fontsize=14)

    # 1) Potencia
    ax = axes[0, 0]
    ax.plot(v_ref, ref["Power [MW]"], "o-", ms=4, label="Ref WISDEM")
    ax.plot(v_bem, bem["Power [MW]"], "x--", ms=4, label="fem-shell BEM")
    ax.set_xlabel("Viento [m/s]")
    ax.set_ylabel("Potencia [MW]")
    ax.set_title("Curva de potencia")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Thrust
    ax = axes[0, 1]
    ax.plot(v_ref, ref["Thrust [MN]"], "o-", ms=4, label="Ref WISDEM")
    ax.plot(v_bem, bem["Thrust [MN]"], "x--", ms=4, label="fem-shell BEM")
    ax.set_xlabel("Viento [m/s]")
    ax.set_ylabel("Empuje [MN]")
    ax.set_title("Empuje del rotor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Torque
    ax = axes[0, 2]
    ax.plot(v_ref, ref["Torque [MNm]"], "o-", ms=4, label="Ref WISDEM")
    ax.plot(v_bem, bem["Torque [MNm]"], "x--", ms=4, label="fem-shell BEM")
    ax.set_xlabel("Viento [m/s]")
    ax.set_ylabel("Par [MN·m]")
    ax.set_title("Par del rotor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Cp
    ax = axes[1, 0]
    ax.plot(v_ref, ref["Power Coefficient [-]"], "o-", ms=4, label="Ref WISDEM")
    ax.plot(v_bem, bem["Cp"], "x--", ms=4, label="fem-shell BEM")
    ax.set_xlabel("Viento [m/s]")
    ax.set_ylabel("Cp [-]")
    ax.set_title("Coeficiente de potencia")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5) Ct
    ax = axes[1, 1]
    ax.plot(v_ref, ref["Thrust Coefficient [-]"], "o-", ms=4, label="Ref WISDEM")
    ax.plot(v_bem, bem["Ct"], "x--", ms=4, label="fem-shell BEM")
    ax.set_xlabel("Viento [m/s]")
    ax.set_ylabel("Ct [-]")
    ax.set_title("Coeficiente de empuje")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6) Errores relativos
    ax = axes[1, 2]
    err_power = (
        np.abs(bem["Power [MW]"] - ref["Power [MW]"]) / np.maximum(ref["Power [MW]"], 1e-6) * 100
    )
    err_thrust = (
        np.abs(bem["Thrust [MN]"] - ref["Thrust [MN]"]) / np.maximum(ref["Thrust [MN]"], 1e-6) * 100
    )
    ax.plot(v_ref, err_power, "s-", ms=4, label="Potencia")
    ax.plot(v_ref, err_thrust, "^-", ms=4, label="Empuje")
    ax.set_xlabel("Viento [m/s]")
    ax.set_ylabel("Error relativo [%]")
    ax.set_title("Errores relativos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "comparison_iea15mw.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nGráfica guardada en: {out_path}")
    plt.show()


def print_table(ref: dict, bem: dict):
    """Imprime una tabla comparativa resumida."""
    v = ref["Wind [m/s]"]

    print("\n" + "=" * 100)
    print(
        f"{'V [m/s]':>8s}  {'P_ref':>8s}  {'P_bem':>8s}  {'ΔP%':>6s}"
        f"  {'T_ref':>8s}  {'T_bem':>8s}  {'ΔT%':>6s}"
        f"  {'Q_ref':>8s}  {'Q_bem':>8s}  {'ΔQ%':>6s}"
    )
    print(
        f"{'':>8s}  {'[MW]':>8s}  {'[MW]':>8s}  {'':>6s}"
        f"  {'[MN]':>8s}  {'[MN]':>8s}  {'':>6s}"
        f"  {'[MNm]':>8s}  {'[MNm]':>8s}  {'':>6s}"
    )
    print("-" * 100)

    for i in range(len(v)):
        p_ref = ref["Power [MW]"][i]
        p_bem = bem["Power [MW]"][i]
        t_ref = ref["Thrust [MN]"][i]
        t_bem = bem["Thrust [MN]"][i]
        q_ref = ref["Torque [MNm]"][i]
        q_bem = bem["Torque [MNm]"][i]

        dp = (p_bem - p_ref) / max(abs(p_ref), 1e-9) * 100
        dt = (t_bem - t_ref) / max(abs(t_ref), 1e-9) * 100
        dq = (q_bem - q_ref) / max(abs(q_ref), 1e-9) * 100

        print(
            f"{v[i]:8.2f}  {p_ref:8.3f}  {p_bem:8.3f}  {dp:+5.1f}%"
            f"  {t_ref:8.4f}  {t_bem:8.4f}  {dt:+5.1f}%"
            f"  {q_ref:8.4f}  {q_bem:8.4f}  {dq:+5.1f}%"
        )
    print("=" * 100)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cargando datos de referencia IEA 15 MW ...")
    ref = load_reference_data(REF_XLSX)

    print(f"  {len(ref['Wind [m/s]'])} puntos operativos")
    print(f"  Rango de viento: {ref['Wind [m/s]'].min():.1f} – {ref['Wind [m/s]'].max():.1f} m/s")

    print("\nEjecutando BEM sweep con fem-shell ...")
    bem = run_bem_sweep(ref)

    print_table(ref, bem)
    plot_comparison(ref, bem)
