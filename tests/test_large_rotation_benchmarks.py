"""
Large-rotation benchmark suite for the MITC4+ shell element.

All benchmarks use the Updated-Lagrangian incremental solve path
(nonlinear_static_solve_coo + PyMeshAssembler.update_reference).

Analytical reference
--------------------
For a cantilever under pure end moment M (load parameter λ = M·L/EI):

    R    = EI / M = L / λ                  (radius of curvature)
    x(λ) = R · sin(λ) = L · sin(λ) / λ    (tip x-coord)
    z(λ) = R · (1 - cos(λ))               (tip z-coord)

    u_tip(λ) = x(λ) - L = L · (sin(λ)/λ - 1)
    w_tip(λ) = z(λ)     = L · (1 - cos(λ)) / λ

Key values (L = 10):
    λ = π/2:  u_tip ≈ -3.634, w_tip ≈  6.366
    λ = π:    u_tip = -10.0,   w_tip ≈  6.366
    λ = 3π/2: u_tip ≈ -12.12,  w_tip ≈  2.122
    λ = 2π:   u_tip = -10.0,   w_tip =   0.0

Note: at λ = 2π the tip closes the full circle, ending at (0, 0, 0) —
NOT back at its starting position (L, 0, 0).  The total u_tip is −L.

References
----------
- Simo, J.C. & Vu-Quoc, L. (1986). CMAME 58, 79-116. (Table 1, N=10)
- Bathe, K.J. & Bolourchi, S. (1979). Computers & Structures 11, 23-48.
"""

import math
import numpy as np
import pytest

_aeroelast = pytest.importorskip("_aeroelast", reason="_aeroelast Rust extension not built")


# ─────────────────────────────────────────────────────────────────────────────
# Analytical references
# ─────────────────────────────────────────────────────────────────────────────

def _analytical_tip(lam: float, L: float = 10.0):
    """Return (u_tip, w_tip) for load parameter λ = M·L/EI."""
    if abs(lam) < 1e-12:
        return 0.0, 0.0
    R = L / lam
    x_tip = R * math.sin(lam)
    z_tip = R * (1.0 - math.cos(lam))
    return x_tip - L, z_tip


# Tabulated control points (λ, u_tip_ref, w_tip_ref) — exact elastic curve
REFERENCE_TABLE = [
    (math.pi / 2,       -3.6338,  6.3662),   # quarter circle
    (math.pi,           -10.000,  6.3662),    # half circle
    (3 * math.pi / 2,   -12.122,  2.1221),    # three-quarter
    (2 * math.pi,       -10.000,  0.0000),    # full circle
]


# ─────────────────────────────────────────────────────────────────────────────
# Mesh helpers
# ─────────────────────────────────────────────────────────────────────────────

def cantilever_mitc4_mesh(n_elem: int, L: float, b: float):
    """
    Build a single-strip MITC4 mesh for a cantilever beam.

    Node layout (top view):
        row y=0:  nodes 0, 2, 4, ...  2*n_elem          (x = 0, L/n, 2L/n, ...)
        row y=b:  nodes 1, 3, 5, ...  2*n_elem+1

    Element connectivity: (2i, 2i+2, 2i+3, 2i+1) — CCW looking from above.

    Returns
    -------
    node_coords   : np.ndarray (n_nodes, 3)
    connectivity  : list[list[int]]
    elem_types    : list[int]  (all 4 = MITC4)
    clamped_dofs  : list[int]  (all DOFs of the two leftmost nodes)
    n_dof         : int
    tip_nodes     : list[int]  (the two rightmost nodes)
    """
    n_nodes = 2 * (n_elem + 1)
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_elem + 1):
        x = L * i / n_elem
        node_coords[2 * i]     = [x, 0.0, 0.0]
        node_coords[2 * i + 1] = [x, b,   0.0]

    connectivity = [[2*i, 2*i+2, 2*i+3, 2*i+1] for i in range(n_elem)]
    elem_types   = [4] * n_elem   # 4 = MITC4

    dofs_per_node = 6
    clamped_dofs  = [n * dofs_per_node + d for n in (0, 1) for d in range(dofs_per_node)]
    n_dof         = n_nodes * dofs_per_node
    tip_nodes     = [2 * n_elem, 2 * n_elem + 1]

    return node_coords, connectivity, elem_types, clamped_dofs, n_dof, tip_nodes


def make_assembler(node_coords, connectivity, elem_types, E, nu, rho, thickness,
                   shear_corr=5.0/6.0):
    """Construct a PyMeshAssembler with a uniform isotropic material."""
    n_nodes = len(node_coords)
    prop = {
        "type": "isotropic",
        "e":    E,              # lowercase — required by parse_material
        "nu":   nu,
        "rho":  rho,
        "thickness":        thickness,
        "shear_correction": shear_corr,
    }
    return _aeroelast.PyMeshAssembler(
        node_coords.astype(np.float64),
        connectivity,
        elem_types,
        [prop] * len(connectivity),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Incremental solver
# ─────────────────────────────────────────────────────────────────────────────

def incremental_solve(assembler, f_total, clamped_dofs, n_steps,
                      atol=1e-8, rtol=1e-6, max_it=60):
    """
    Updated-Lagrangian incremental Newton solve.

    Divides f_total into n_steps equal increments.  After each converged step
    the reference configuration is updated (UL shift) so the next step starts
    from equilibrium.

    Returns the accumulated displacement from the *original* configuration:
        u_total = Σ u_inc_k  = final_coords − initial_coords
    """
    f_inc    = np.asarray(f_total, dtype=np.float64) / n_steps
    dirichlet = np.array(clamped_dofs, dtype=np.int64)
    u_total  = np.zeros(len(f_total), dtype=np.float64)

    for step in range(n_steps):
        u_inc, iters, res_norm, reason = _aeroelast.nonlinear_static_solve_coo(
            assembler, f_inc, dirichlet,
            atol=atol, rtol=rtol, stol=1e-8, max_it=max_it,
        )
        u_inc = np.asarray(u_inc, dtype=np.float64)
        if reason <= 0:
            raise RuntimeError(
                f"SNES diverged at step {step+1}/{n_steps}: "
                f"reason={reason}, |R|={res_norm:.3e}"
            )
        assembler.update_reference(u_inc)
        u_total += u_inc

    return u_total


def tip_displacement(u_total, tip_nodes):
    """Return mean (u_x, w_z) of the given tip nodes."""
    u = np.mean([u_total[6*n + 0] for n in tip_nodes])
    w = np.mean([u_total[6*n + 2] for n in tip_nodes])
    return u, w


# ─────────────────────────────────────────────────────────────────────────────
# Beam parameters (shared by all nonlinear tests)
# ─────────────────────────────────────────────────────────────────────────────

L    = 10.0
B    = 1.0    # width
H    = 0.1    # thickness
E    = 1.2e6
NU   = 0.0
RHO  = 1.0

# Second moment of area: I = b·h³/12
I_beam = B * H**3 / 12.0


def _moment_load(lam, tip_nodes, n_dof):
    """Build f_ext for end moment M = λ·EI/L applied to the two tip nodes."""
    M_total = lam * E * I_beam / L
    f_ext = np.zeros(n_dof)
    m_per_node = M_total / len(tip_nodes)
    for n in tip_nodes:
        f_ext[6*n + 4] = m_per_node   # θy — moment about global Y
    return f_ext


# ─────────────────────────────────────────────────────────────────────────────
# 0. Linear sanity check — Euler-Bernoulli tip deflection
# ─────────────────────────────────────────────────────────────────────────────

def test_linear_tip_deflection_euler_bernoulli():
    """
    Cantilever under small transverse tip force: δ = PL³/(3EI).

    Verifies that the mesh, assembler, and linear solve are all wired up
    correctly *before* testing the nonlinear path.  Uses n=10 MITC4
    elements and expects ≤2 % error vs. Euler-Bernoulli.
    """
    n_elem = 10
    node_coords, connectivity, elem_types, clamped_dofs, n_dof, tip_nodes = \
        cantilever_mitc4_mesh(n_elem, L, B)

    assembler = make_assembler(node_coords, connectivity, elem_types, E, NU, RHO, H)

    # P chosen so δ/L < 0.1%  →  δ = P·L³/(3EI) ≤ 0.001·L
    # EI = 1.2e6 * (1*0.1³/12) = 100  →  P_max = 0.001·L·3EI/L³ = 0.003 N
    P = 0.001   # [N];  δ ≈ 3.3e-3 m  (δ/L ≈ 3.3e-4)
    f_ext = np.zeros(n_dof)
    for n in tip_nodes:
        f_ext[6*n + 2] = P / len(tip_nodes)   # z-direction (w)

    dirichlet = np.array(clamped_dofs, dtype=np.int64)
    u, _iters, _res, reason = _aeroelast.nonlinear_static_solve_coo(
        assembler, f_ext, dirichlet,
        atol=1e-10, rtol=1e-8, stol=1e-10, max_it=30,
    )
    assert reason > 0, f"Solver diverged: reason={reason}"

    u = np.asarray(u)
    w_tip = np.mean([u[6*n + 2] for n in tip_nodes])
    w_ref  = P * L**3 / (3.0 * E * I_beam)

    rel_err = abs(w_tip - w_ref) / w_ref
    assert rel_err < 0.02, (
        f"EB tip deflection: w_tip={w_tip:.6e}, reference={w_ref:.6e}, "
        f"rel error={rel_err:.2%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Half-circle (λ = π): Simo & Vu-Quoc Table 1
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_elem", [10])
def test_cantilever_large_rotation_half_circle(n_elem):
    """
    Cantilever under end moment: λ = M·L/EI = π  (half-circle).

    Analytical tip position:
        x_tip = L · sin(π)/π = 0      →  u_tip = 0 − L = −10
        z_tip = L · (1−cos(π))/π = 2L/π ≈ 6.366  →  w_tip ≈ 6.366
    """
    node_coords, conn, et, clamped, n_dof, tips = cantilever_mitc4_mesh(n_elem, L, B)
    assembler = make_assembler(node_coords, conn, et, E, NU, RHO, H)

    f_ext = _moment_load(math.pi, tips, n_dof)
    u_total = incremental_solve(assembler, f_ext, clamped, n_steps=20)

    u_tip, w_tip = tip_displacement(u_total, tips)

    u_ref, w_ref = _analytical_tip(math.pi, L)   # (−10.0, 6.366)
    tol = 0.05   # 5 % relative

    assert abs(u_tip - u_ref) / abs(u_ref) < tol, (
        f"u_tip={u_tip:.4f}, ref={u_ref:.4f}")
    assert abs(w_tip - w_ref) / abs(w_ref) < tol, (
        f"w_tip={w_tip:.4f}, ref={w_ref:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full equilibrium path — four control points
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lam,u_ref,w_ref", REFERENCE_TABLE)
def test_equilibrium_path(lam, u_ref, w_ref):
    """
    Verify the UL incremental solve tracks the exact elastic curve at
    λ = π/2, π, 3π/2, 2π.

    Each case uses n_steps proportional to λ so the angular increment
    per step stays ≤ 9°.  Tolerance is 5 % relative (or 0.5 m absolute
    when the reference is near zero).
    """
    n_elem  = 10
    n_steps = max(20, int(math.ceil(lam / (math.pi / 20))))   # ≤9° per step

    node_coords, conn, et, clamped, n_dof, tips = cantilever_mitc4_mesh(n_elem, L, B)
    assembler = make_assembler(node_coords, conn, et, E, NU, RHO, H)

    f_ext   = _moment_load(lam, tips, n_dof)
    u_total = incremental_solve(assembler, f_ext, clamped, n_steps=n_steps)

    u_tip, w_tip = tip_displacement(u_total, tips)

    tol_rel = 0.05
    tol_abs = 0.5   # [m] — used when |ref| < 1

    def _check(val, ref, label):
        if abs(ref) >= 1.0:
            err = abs(val - ref) / abs(ref)
            assert err < tol_rel, (
                f"λ={lam/math.pi:.2f}π  {label}: computed={val:.4f}, "
                f"ref={ref:.4f}, rel err={err:.2%}"
            )
        else:
            err = abs(val - ref)
            assert err < tol_abs, (
                f"λ={lam/math.pi:.2f}π  {label}: computed={val:.4f}, "
                f"ref={ref:.4f}, abs err={err:.4f}"
            )

    _check(u_tip, u_ref, "u_tip")
    _check(w_tip, w_ref, "w_tip")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Simo–Vu-Quoc 360° rollup (dedicated, explicit assertions)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_elem", [10])
def test_simo_vu_quoc_rollup_360(n_elem):
    """
    Full 360° rollup: λ = 2π.

    The beam forms a complete circle.  The free tip travels from (L, 0, 0)
    to (0, 0, 0), so the TOTAL displacement from the original configuration is:
        u_tip ≈ −L = −10   (full axial retraction)
        w_tip ≈  0          (tip returns to the base plane)

    Reference: Simo & Vu-Quoc (1986), CMAME 58, Fig. 9 / Table 1 (N=10).
    """
    node_coords, conn, et, clamped, n_dof, tips = cantilever_mitc4_mesh(n_elem, L, B)
    assembler = make_assembler(node_coords, conn, et, E, NU, RHO, H)

    lam   = 2.0 * math.pi
    f_ext = _moment_load(lam, tips, n_dof)

    u_total = incremental_solve(assembler, f_ext, clamped, n_steps=40)
    u_tip, w_tip = tip_displacement(u_total, tips)

    # Exact targets from the elastic-curve formula
    u_ref, w_ref = _analytical_tip(lam, L)   # (−10.0, 0.0)

    tol_rel = 0.05   # 5 %
    tol_abs = 0.5    # 0.5 m absolute (for w_tip ≈ 0)

    assert abs(u_tip - u_ref) / abs(u_ref) < tol_rel, (
        f"Full circle u_tip={u_tip:.4f}, expected {u_ref:.4f} (= −L)")
    assert abs(w_tip - w_ref) < tol_abs, (
        f"Full circle w_tip={w_tip:.4f}, expected {w_ref:.4f} (≈ 0)")
