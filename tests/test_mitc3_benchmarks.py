"""
Large-rotation benchmark suite for the MITC3+ shell element.

All benchmarks use the Updated-Lagrangian incremental solve path
(nonlinear_static_solve_coo + PyMeshAssembler.update_reference).

Mesh layout
-----------
Each "column" of n_elem spans is discretised with 2 triangles per column:

    y=0: nodes 0, 2, 4, ...  2*n_elem
    y=b: nodes 1, 3, 5, ...  2*n_elem+1

    Column i → lower tri  (2i,  2i+2, 2i+1)
             → upper tri  (2i+2, 2i+3, 2i+1)

Total elements: 2·n_elem  (elem_type=3 for all).

Analytical reference
--------------------
For a cantilever under pure end moment M (load parameter λ = M·L/EI):

    R    = EI / M = L / λ
    u_tip(λ) = L·(sin(λ)/λ − 1)
    w_tip(λ) = L·(1 − cos(λ)) / λ

Key values (L = 10):
    λ = π/2:  u_tip ≈ −3.634, w_tip ≈  6.366
    λ = π:    u_tip = −10.0,   w_tip ≈  6.366
    λ = 3π/2: u_tip ≈ −12.12,  w_tip ≈  2.122
    λ = 2π:   u_tip = −10.0,   w_tip =   0.0

References
----------
- Simo, J.C. & Vu-Quoc, L. (1986). CMAME 58, 79-116.
- Lee, P.S., Lee, Y., & Bathe, K.J. (2014). "The MITC3+ shell element and its performance." CAS 138, 12-23.
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


REFERENCE_TABLE = [
    (math.pi / 2, -3.6338, 6.3662),
    (math.pi, -10.000, 6.3662),
    (3 * math.pi / 2, -12.122, 2.1221),
    (2 * math.pi, -10.000, 0.0000),
]


# ─────────────────────────────────────────────────────────────────────────────
# Mesh helpers
# ─────────────────────────────────────────────────────────────────────────────


def cantilever_mitc3_mesh(n_elem: int, L: float, b: float):
    """
    Build a two-triangle-per-column MITC3 mesh for a cantilever beam.

    Node layout (top view):
        row y=0:  nodes 0, 2, 4, ...  2*n_elem
        row y=b:  nodes 1, 3, 5, ...  2*n_elem+1

    Each column i contributes two triangles:
        lower: (2i, 2i+2, 2i+1)
        upper: (2i+2, 2i+3, 2i+1)

    Returns
    -------
    node_coords   : np.ndarray (n_nodes, 3)
    connectivity  : list[list[int]]
    elem_types    : list[int]  (all 3 = MITC3)
    clamped_dofs  : list[int]  (all DOFs of the two leftmost nodes)
    n_dof         : int
    tip_nodes     : list[int]  (the two rightmost nodes)
    """
    n_nodes = 2 * (n_elem + 1)
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_elem + 1):
        x = L * i / n_elem
        node_coords[2 * i] = [x, 0.0, 0.0]
        node_coords[2 * i + 1] = [x, b, 0.0]

    connectivity = []
    for i in range(n_elem):
        connectivity.append([2 * i, 2 * i + 2, 2 * i + 1])  # lower triangle
        connectivity.append([2 * i + 2, 2 * i + 3, 2 * i + 1])  # upper triangle
    elem_types = [3] * len(connectivity)

    dofs_per_node = 6
    clamped_dofs = [n * dofs_per_node + d for n in (0, 1) for d in range(dofs_per_node)]
    n_dof = n_nodes * dofs_per_node
    tip_nodes = [2 * n_elem, 2 * n_elem + 1]

    return node_coords, connectivity, elem_types, clamped_dofs, n_dof, tip_nodes


def make_assembler(
    node_coords, connectivity, elem_types, E, nu, rho, thickness, shear_corr=5.0 / 6.0
):
    """Construct a PyMeshAssembler with a uniform isotropic material."""
    prop = {
        "type": "isotropic",
        "e": E,
        "nu": nu,
        "rho": rho,
        "thickness": thickness,
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


def incremental_solve(assembler, f_total, clamped_dofs, n_steps, atol=1e-8, rtol=1e-6, max_it=60):
    """
    Updated-Lagrangian incremental Newton solve.

    Divides f_total into n_steps equal increments.  After each converged step
    the reference configuration is updated so the next step starts from
    equilibrium.

    Returns the accumulated displacement from the original configuration.
    """
    f_inc = np.asarray(f_total, dtype=np.float64) / n_steps
    dirichlet = np.array(clamped_dofs, dtype=np.int64)
    u_total = np.zeros(len(f_total), dtype=np.float64)

    for step in range(n_steps):
        u_inc, iters, res_norm, reason = _aeroelast.nonlinear_static_solve_coo(
            assembler,
            f_inc,
            dirichlet,
            atol=atol,
            rtol=rtol,
            stol=1e-8,
            max_it=max_it,
        )
        u_inc = np.asarray(u_inc, dtype=np.float64)
        if reason <= 0:
            raise RuntimeError(
                f"SNES diverged at step {step + 1}/{n_steps}: reason={reason}, |R|={res_norm:.3e}"
            )
        assembler.update_reference(u_inc)
        u_total += u_inc

    return u_total


def tip_displacement(u_total, tip_nodes):
    """Return mean (u_x, w_z) over the given tip nodes."""
    u = np.mean([u_total[6 * n + 0] for n in tip_nodes])
    w = np.mean([u_total[6 * n + 2] for n in tip_nodes])
    return u, w


# ─────────────────────────────────────────────────────────────────────────────
# Beam parameters (shared by all tests)
# ─────────────────────────────────────────────────────────────────────────────

L = 10.0
B = 1.0
H = 0.1
E = 1.2e6
NU = 0.0
RHO = 1.0

I_beam = B * H**3 / 12.0  # second moment of area


def _moment_load(lam, tip_nodes, n_dof):
    """Build f_ext for end moment M = λ·EI/L applied to the two tip nodes."""
    M_total = lam * E * I_beam / L
    f_ext = np.zeros(n_dof)
    m_per_node = M_total / len(tip_nodes)
    for n in tip_nodes:
        f_ext[6 * n + 4] = m_per_node  # θy — moment about global Y
    return f_ext


# ─────────────────────────────────────────────────────────────────────────────
# 0. Linear sanity check — Euler-Bernoulli tip deflection
# ─────────────────────────────────────────────────────────────────────────────


def test_linear_tip_deflection_euler_bernoulli():
    """
    Cantilever under small transverse tip force: δ = PL³/(3EI).

    Uses n=10 columns (20 MITC3 triangles).  Expects ≤2 % error vs.
    the Euler-Bernoulli reference.
    """
    n_elem = 10
    node_coords, connectivity, elem_types, clamped_dofs, n_dof, tip_nodes = cantilever_mitc3_mesh(
        n_elem, L, B
    )

    assembler = make_assembler(node_coords, connectivity, elem_types, E, NU, RHO, H)

    P = 0.001  # small load — δ/L ≈ 3.3e-4 (linear regime)
    f_ext = np.zeros(n_dof)
    for n in tip_nodes:
        f_ext[6 * n + 2] = P / len(tip_nodes)

    dirichlet = np.array(clamped_dofs, dtype=np.int64)
    u, _iters, _res, reason = _aeroelast.nonlinear_static_solve_coo(
        assembler,
        f_ext,
        dirichlet,
        atol=1e-10,
        rtol=1e-8,
        stol=1e-10,
        max_it=30,
    )
    assert reason > 0, f"Solver diverged: reason={reason}"

    u = np.asarray(u)
    w_tip = np.mean([u[6 * n + 2] for n in tip_nodes])
    w_ref = P * L**3 / (3.0 * E * I_beam)

    rel_err = abs(w_tip - w_ref) / w_ref
    assert rel_err < 0.02, (
        f"EB tip deflection: w_tip={w_tip:.6e}, reference={w_ref:.6e}, rel error={rel_err:.2%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 0b. Linear moment sign check
# ─────────────────────────────────────────────────────────────────────────────


def test_linear_tip_moment_sign():
    """
    Positive moment about Y on the free tip must produce positive w (upward).

    This is the sign-convention regression test for the MITC3 shear B-matrix fix.
    Uses a small λ so the response stays in the linear regime.
    """
    n_elem = 10
    node_coords, connectivity, elem_types, clamped_dofs, n_dof, tip_nodes = cantilever_mitc3_mesh(
        n_elem, L, B
    )

    assembler = make_assembler(node_coords, connectivity, elem_types, E, NU, RHO, H)

    lam = 0.01  # small load — pure linear response
    f_ext = _moment_load(lam, tip_nodes, n_dof)

    dirichlet = np.array(clamped_dofs, dtype=np.int64)
    u, _iters, _res, reason = _aeroelast.nonlinear_static_solve_coo(
        assembler,
        f_ext,
        dirichlet,
        atol=1e-10,
        rtol=1e-8,
        stol=1e-10,
        max_it=30,
    )
    assert reason > 0, f"Solver diverged: reason={reason}"

    u = np.asarray(u)
    w_tip = np.mean([u[6 * n + 2] for n in tip_nodes])

    # Linear theory: w_tip = M·L²/(2EI) = λ·EI/L · L²/(2EI) = λ·L/2
    w_ref = lam * L / 2.0
    assert w_tip > 0, f"Sign bug: positive M_y produced w_tip={w_tip:.4e} (expected > 0)"
    rel_err = abs(w_tip - w_ref) / w_ref
    assert rel_err < 0.02, (
        f"Moment tip deflection: w_tip={w_tip:.6e}, ref={w_ref:.6e}, rel error={rel_err:.2%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Half-circle (λ = π)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_elem", [10])
def test_cantilever_large_rotation_half_circle(n_elem):
    """
    Cantilever under end moment: λ = M·L/EI = π  (half-circle).

    Analytical tip:  u_tip = −10.0,  w_tip ≈ 6.366.
    """
    node_coords, conn, et, clamped, n_dof, tips = cantilever_mitc3_mesh(n_elem, L, B)
    assembler = make_assembler(node_coords, conn, et, E, NU, RHO, H)

    f_ext = _moment_load(math.pi, tips, n_dof)
    u_total = incremental_solve(assembler, f_ext, clamped, n_steps=20)

    u_tip, w_tip = tip_displacement(u_total, tips)
    u_ref, w_ref = _analytical_tip(math.pi, L)  # (−10.0, 6.366)
    tol = 0.05

    assert abs(u_tip - u_ref) / abs(u_ref) < tol, f"u_tip={u_tip:.4f}, ref={u_ref:.4f}"
    assert abs(w_tip - w_ref) / abs(w_ref) < tol, f"w_tip={w_tip:.4f}, ref={w_ref:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full equilibrium path — four control points
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lam,u_ref,w_ref", REFERENCE_TABLE)
def test_equilibrium_path(lam, u_ref, w_ref):
    """
    Verify the UL incremental solve tracks the exact elastic curve at
    λ = π/2, π, 3π/2, 2π.  Tolerance: 5 % relative (0.5 m absolute near zero).
    """
    n_elem = 10
    n_steps = max(20, int(math.ceil(lam / (math.pi / 20))))

    node_coords, conn, et, clamped, n_dof, tips = cantilever_mitc3_mesh(n_elem, L, B)
    assembler = make_assembler(node_coords, conn, et, E, NU, RHO, H)

    f_ext = _moment_load(lam, tips, n_dof)
    u_total = incremental_solve(assembler, f_ext, clamped, n_steps=n_steps)

    u_tip, w_tip = tip_displacement(u_total, tips)

    tol_rel = 0.05
    tol_abs = 0.5

    def _check(val, ref, label):
        if abs(ref) >= 1.0:
            err = abs(val - ref) / abs(ref)
            assert err < tol_rel, (
                f"λ={lam / math.pi:.2f}π  {label}: computed={val:.4f}, "
                f"ref={ref:.4f}, rel err={err:.2%}"
            )
        else:
            err = abs(val - ref)
            assert err < tol_abs, (
                f"λ={lam / math.pi:.2f}π  {label}: computed={val:.4f}, "
                f"ref={ref:.4f}, abs err={err:.4f}"
            )

    _check(u_tip, u_ref, "u_tip")
    _check(w_tip, w_ref, "w_tip")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Simo–Vu-Quoc 360° rollup
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_elem", [10])
def test_simo_vu_quoc_rollup_360(n_elem):
    """
    Full 360° rollup: λ = 2π.

    The beam forms a complete circle.  The free tip travels from (L,0,0)
    to (0,0,0):
        u_tip ≈ −L = −10
        w_tip ≈  0
    """
    node_coords, conn, et, clamped, n_dof, tips = cantilever_mitc3_mesh(n_elem, L, B)
    assembler = make_assembler(node_coords, conn, et, E, NU, RHO, H)

    lam = 2.0 * math.pi
    f_ext = _moment_load(lam, tips, n_dof)

    u_total = incremental_solve(assembler, f_ext, clamped, n_steps=40)
    u_tip, w_tip = tip_displacement(u_total, tips)

    u_ref, w_ref = _analytical_tip(lam, L)  # (−10.0, 0.0)

    tol_rel = 0.05
    tol_abs = 0.5

    assert abs(u_tip - u_ref) / abs(u_ref) < tol_rel, (
        f"Full circle u_tip={u_tip:.4f}, expected {u_ref:.4f} (= −L)"
    )
    assert abs(w_tip - w_ref) < tol_abs, (
        f"Full circle w_tip={w_tip:.4f}, expected {w_ref:.4f} (≈ 0)"
    )
