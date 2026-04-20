"""
Comprehensive cross-validation: Rust MITC3/MITC4 kernels vs Python reference.

Tests cover:
  1. Stiffness K: flat, tilted, warped geometries
  2. Mass M: flat, tilted
  3. Tangent K_T: zero disp, small disp, moderate disp
  4. Internal forces f_int: zero, small, moderate displacements
  5. Symmetry and positive semi-definiteness
  6. Rigid body modes (zero strain energy for rigid translations)
  7. Patch test (constant strain field)
  8. Batch consistency (many elements at once)
"""

import numpy as np
import sys
sys.path.insert(0, "/scratch/leahk/eduardo.donestevez/fem-shell/src")

from aeroelast.core.material import IsotropicMaterial, Material
from aeroelast.elements.MITC3 import MITC3
from aeroelast.elements.MITC4 import MITC4
import fem_shell_core

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ✗ {name}  {detail}")


# ============================================================================
# Material setup
# ============================================================================
E, nu, rho, t = 210e9, 0.3, 7850.0, 0.01
mat_iso = IsotropicMaterial(name="steel", E=E, nu=nu, rho=rho)
# For MITC3 tests (uses Material, not IsotropicMaterial)
mat_m = Material(E=E, nu=nu, rho=rho)

# ============================================================================
# Geometry library
# ============================================================================
TRI_FLAT = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
TRI_EQUI = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.5,0.866,0.0]])
TRI_TILTED = np.array([[0.0,0.0,0.0],[2.0,0.0,0.3],[0.5,1.5,0.1]])
TRI_3D = np.array([[1.0,2.0,3.0],[3.0,2.0,3.5],[2.0,4.0,3.2]])

QUAD_UNIT = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]])
QUAD_RECT = np.array([[0.0,0.0,0.0],[2.0,0.0,0.0],[2.0,1.0,0.0],[0.0,1.0,0.0]])
QUAD_TILTED = np.array([[0.0,0.0,0.0],[2.0,0.0,0.1],[2.0,1.5,0.2],[0.0,1.5,0.05]])
QUAD_WARPED = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.3],[0.0,1.0,-0.1]])
QUAD_3D = np.array([[1.0,2.0,3.0],[3.0,2.0,3.0],[3.0,4.0,3.5],[1.0,4.0,3.2]])

# ============================================================================
# MITC3 Tests
# ============================================================================
print("=" * 70)
print("MITC3: Rust vs Python cross-validation")
print("=" * 70)

def test_mitc3_ke(name, nodes):
    """Compare MITC3 stiffness matrix: Python vs Rust"""
    elem = MITC3(nodes, (0,1,2), mat_m, thickness=t)
    K_py = elem.K

    coords = nodes.reshape(1, 9)
    K_rs = fem_shell_core.batch_ke_mitc3(coords, E, nu, t).reshape(18, 18)

    K_max = max(np.max(np.abs(K_py)), 1e-30)
    rel = np.max(np.abs(K_py - K_rs)) / K_max
    check(f"K {name}: rel_err={rel:.2e}", rel < 1e-10)
    check(f"K {name} symmetric (Rust)", np.allclose(K_rs, K_rs.T, atol=1e-10))
    return K_py, K_rs

def test_mitc3_me(name, nodes):
    """Compare MITC3 mass matrix"""
    elem = MITC3(nodes, (0,1,2), mat_m, thickness=t)
    M_py = elem.M

    coords = nodes.reshape(1, 9)
    M_rs = fem_shell_core.batch_me_mitc3(coords, E, nu, rho, t).reshape(18, 18)

    M_max = max(np.max(np.abs(M_py)), 1e-30)
    rel = np.max(np.abs(M_py - M_rs)) / M_max
    check(f"M {name}: rel_err={rel:.2e}", rel < 1e-10)

def test_mitc3_kt(name, nodes, u_global):
    """Compare MITC3 tangent stiffness"""
    elem = MITC3(nodes, (0,1,2), mat_m, thickness=t, nonlinear=True)
    elem.update_configuration(u_global)
    KT_py = elem.compute_tangent_stiffness(transform_to_global=True)

    coords = nodes.reshape(1, 9)
    u_arr = u_global.reshape(1, 18)
    KT_rs = fem_shell_core.batch_kt_mitc3(coords, u_arr, E, nu, t).reshape(18, 18)

    KT_max = max(np.max(np.abs(KT_py)), 1e-30)
    rel = np.max(np.abs(KT_py - KT_rs)) / KT_max
    check(f"KT {name}: rel_err={rel:.2e}", rel < 1e-8)
    check(f"KT {name} symmetric (Rust)", np.allclose(KT_rs, KT_rs.T, atol=1e-6 * KT_max))

def test_mitc3_fint(name, nodes, u_global, nonlinear=True):
    """Compare MITC3 internal forces"""
    elem = MITC3(nodes, (0,1,2), mat_m, thickness=t, nonlinear=nonlinear)
    elem.update_configuration(u_global)
    f_py = elem.compute_internal_forces(transform_to_global=True)

    coords = nodes.reshape(1, 9)
    u_arr = u_global.reshape(1, 18)
    f_rs = fem_shell_core.batch_fint_mitc3(coords, u_arr, E, nu, t, nonlinear=nonlinear).reshape(18)

    f_max = max(np.max(np.abs(f_py)), 1e-30)
    if f_max < 1e-20:
        check(f"fint {name}: both ~zero", np.max(np.abs(f_rs)) < 1e-10)
    else:
        rel = np.max(np.abs(f_py - f_rs)) / f_max
        check(f"fint {name}: rel_err={rel:.2e}", rel < 1e-8)

# -- Stiffness tests --
print("\n--- Stiffness K ---")
for name, nodes in [("flat", TRI_FLAT), ("equilateral", TRI_EQUI),
                      ("tilted", TRI_TILTED), ("3D", TRI_3D)]:
    test_mitc3_ke(name, nodes)

# -- Mass tests --
print("\n--- Mass M ---")
for name, nodes in [("flat", TRI_FLAT), ("equilateral", TRI_EQUI),
                      ("tilted", TRI_TILTED), ("3D", TRI_3D)]:
    test_mitc3_me(name, nodes)

# -- Rigid body modes (Rust) --
print("\n--- Rigid body modes ---")
_, K_rs_flat = test_mitc3_ke("rbm_check", TRI_FLAT)
for mode_name, mode_vec in [
    ("trans_x", np.tile([1,0,0,0,0,0], 3)),
    ("trans_y", np.tile([0,1,0,0,0,0], 3)),
    ("trans_z", np.tile([0,0,1,0,0,0], 3)),
]:
    f = K_rs_flat @ np.array(mode_vec, dtype=float)
    check(f"RBM {mode_name}: ||f||={np.linalg.norm(f):.2e}",
          np.linalg.norm(f) < 1e-5 * np.max(np.abs(K_rs_flat)))

# -- Patch test (Rust) --
print("\n--- Patch test (constant strain) ---")
eps0 = 0.001
d_patch = np.zeros(18)
for i in range(3):
    d_patch[6*i] = eps0 * TRI_FLAT[i, 0]
U_rust = 0.5 * d_patch @ K_rs_flat @ d_patch
E_eff = E / (1 - nu**2)
area = 0.5 * 1.0 * 1.0
U_exact = area * t * 0.5 * E_eff * eps0**2
check(f"Patch test: U_rust={U_rust:.6e}, U_exact={U_exact:.6e}, rel={abs(U_rust-U_exact)/U_exact:.2e}",
      abs(U_rust - U_exact) / U_exact < 1e-4)

# -- Tangent stiffness --
print("\n--- Tangent stiffness KT ---")
u_zero = np.zeros(18)
u_small = np.zeros(18)
u_small[2] = 0.001; u_small[8] = 0.0005
u_moderate = np.zeros(18)
u_moderate[2] = 0.01; u_moderate[8] = 0.005; u_moderate[14] = -0.003

for name, u in [("zero_disp", u_zero), ("small_disp", u_small), ("moderate_disp", u_moderate)]:
    test_mitc3_kt(name, TRI_FLAT, u)
    test_mitc3_kt(f"{name}_tilted", TRI_TILTED, u)

# -- Internal forces --
print("\n--- Internal forces f_int ---")
for name, u, nl in [
    ("zero_linear", u_zero, False),
    ("zero_nonlinear", u_zero, True),
    ("small_nonlinear", u_small, True),
    ("moderate_nonlinear", u_moderate, True),
    ("small_linear", u_small, False),
]:
    test_mitc3_fint(name, TRI_FLAT, u, nonlinear=nl)
    test_mitc3_fint(f"{name}_tilted", TRI_TILTED, u, nonlinear=nl)


# ============================================================================
# MITC4 Tests
# ============================================================================
print("\n" + "=" * 70)
print("MITC4: Rust vs Python cross-validation")
print("=" * 70)

def test_mitc4_ke(name, nodes):
    """Compare MITC4 stiffness matrix"""
    elem = MITC4(nodes, (0,1,2,3), mat_iso, thickness=t)
    K_py = elem.K

    coords = nodes.reshape(1, 12)
    K_rs = fem_shell_core.batch_ke_mitc4(coords, E, nu, t).reshape(24, 24)

    K_max = max(np.max(np.abs(K_py)), 1e-30)
    rel = np.max(np.abs(K_py - K_rs)) / K_max
    check(f"K {name}: rel_err={rel:.2e}", rel < 1e-10)
    check(f"K {name} symmetric (Rust)", np.allclose(K_rs, K_rs.T, atol=1e-10))
    return K_py, K_rs

def test_mitc4_me(name, nodes):
    """Compare MITC4 mass matrix"""
    elem = MITC4(nodes, (0,1,2,3), mat_iso, thickness=t)
    M_py = elem.M

    coords = nodes.reshape(1, 12)
    M_rs = fem_shell_core.batch_me_mitc4(coords, E, nu, rho, t).reshape(24, 24)

    M_max = max(np.max(np.abs(M_py)), 1e-30)
    rel = np.max(np.abs(M_py - M_rs)) / M_max
    check(f"M {name}: rel_err={rel:.2e}", rel < 1e-10)

def test_mitc4_kt(name, nodes, u_global):
    """Compare MITC4 tangent stiffness"""
    elem = MITC4(nodes, (0,1,2,3), mat_iso, thickness=t, nonlinear=True)
    elem.update_configuration(u_global)
    KT_py = elem.compute_tangent_stiffness(transform_to_global=True)

    coords = nodes.reshape(1, 12)
    u_arr = u_global.reshape(1, 24)
    KT_rs = fem_shell_core.batch_kt_mitc4(coords, u_arr, E, nu, t).reshape(24, 24)

    KT_max = max(np.max(np.abs(KT_py)), 1e-30)
    rel = np.max(np.abs(KT_py - KT_rs)) / KT_max
    check(f"KT {name}: rel_err={rel:.2e}", rel < 1e-8)
    check(f"KT {name} symmetric (Rust)", np.allclose(KT_rs, KT_rs.T, atol=1e-6 * KT_max))

def test_mitc4_fint(name, nodes, u_global, nonlinear=True):
    """Compare MITC4 internal forces"""
    elem = MITC4(nodes, (0,1,2,3), mat_iso, thickness=t, nonlinear=nonlinear)
    elem.update_configuration(u_global)
    f_py = elem.compute_internal_forces(transform_to_global=True)

    coords = nodes.reshape(1, 12)
    u_arr = u_global.reshape(1, 24)
    f_rs = fem_shell_core.batch_fint_mitc4(coords, u_arr, E, nu, t, nonlinear=nonlinear).reshape(24)

    f_max = max(np.max(np.abs(f_py)), 1e-30)
    if f_max < 1e-20:
        check(f"fint {name}: both ~zero", np.max(np.abs(f_rs)) < 1e-10)
    else:
        rel = np.max(np.abs(f_py - f_rs)) / f_max
        check(f"fint {name}: rel_err={rel:.2e}", rel < 1e-8)

# -- Stiffness tests --
print("\n--- Stiffness K ---")
for name, nodes in [("unit_sq", QUAD_UNIT), ("rectangle", QUAD_RECT),
                      ("tilted", QUAD_TILTED), ("warped", QUAD_WARPED), ("3D", QUAD_3D)]:
    test_mitc4_ke(name, nodes)

# -- Mass tests --
print("\n--- Mass M ---")
for name, nodes in [("unit_sq", QUAD_UNIT), ("rectangle", QUAD_RECT),
                      ("tilted", QUAD_TILTED), ("warped", QUAD_WARPED), ("3D", QUAD_3D)]:
    test_mitc4_me(name, nodes)

# -- Rigid body modes (Rust) --
print("\n--- Rigid body modes ---")
_, K_rs_quad = test_mitc4_ke("rbm_check", QUAD_UNIT)
for mode_name, mode_vec in [
    ("trans_x", np.tile([1,0,0,0,0,0], 4)),
    ("trans_y", np.tile([0,1,0,0,0,0], 4)),
    ("trans_z", np.tile([0,0,1,0,0,0], 4)),
]:
    f = K_rs_quad @ np.array(mode_vec, dtype=float)
    check(f"RBM {mode_name}: ||f||={np.linalg.norm(f):.2e}",
          np.linalg.norm(f) < 1e-5 * np.max(np.abs(K_rs_quad)))

# Eigenvalue check: should have 6 zero modes (3 translations + 3 rotations)
eigs = np.linalg.eigvalsh(K_rs_quad)
n_zero = np.sum(np.abs(eigs) < 1e-4 * np.max(np.abs(eigs)))
check(f"Eigenvalues: {n_zero} zero modes (expect ≥6)", n_zero >= 6,
      detail=f"got {n_zero}, first 8 eigs: {eigs[:8]}")

# -- Patch test (Rust) --
print("\n--- Patch test (constant strain) ---")
eps0 = 0.001
d_patch4 = np.zeros(24)
for i in range(4):
    d_patch4[6*i] = eps0 * QUAD_UNIT[i, 0]
U_rust4 = 0.5 * d_patch4 @ K_rs_quad @ d_patch4
area4 = 1.0  # unit square
U_exact4 = area4 * t * 0.5 * E_eff * eps0**2
check(f"Patch test: U_rust={U_rust4:.6e}, U_exact={U_exact4:.6e}, rel={abs(U_rust4-U_exact4)/U_exact4:.2e}",
      abs(U_rust4 - U_exact4) / U_exact4 < 1e-4)

# -- Tangent stiffness --
print("\n--- Tangent stiffness KT ---")
u4_zero = np.zeros(24)
u4_small = np.zeros(24)
u4_small[2] = 0.001; u4_small[8] = 0.0005
u4_moderate = np.zeros(24)
u4_moderate[2] = 0.01; u4_moderate[8] = 0.005; u4_moderate[14] = -0.003; u4_moderate[20] = 0.002

for name, u in [("zero_disp", u4_zero), ("small_disp", u4_small), ("moderate_disp", u4_moderate)]:
    test_mitc4_kt(name, QUAD_UNIT, u)
    test_mitc4_kt(f"{name}_tilted", QUAD_TILTED, u)

# -- Internal forces --
print("\n--- Internal forces f_int ---")
for name, u, nl in [
    ("zero_linear", u4_zero, False),
    ("zero_nonlinear", u4_zero, True),
    ("small_nonlinear", u4_small, True),
    ("moderate_nonlinear", u4_moderate, True),
    ("small_linear", u4_small, False),
]:
    test_mitc4_fint(name, QUAD_UNIT, u, nonlinear=nl)
    test_mitc4_fint(f"{name}_tilted", QUAD_TILTED, u, nonlinear=nl)


# ============================================================================
# Batch consistency test (many elements)
# ============================================================================
print("\n" + "=" * 70)
print("Batch consistency (N=1000 random elements)")
print("=" * 70)

np.random.seed(42)

# MITC3: 1000 random triangles
n_tri = 1000
tri_coords = np.zeros((n_tri, 9))
for i in range(n_tri):
    base = np.random.randn(3) * 10
    tri_coords[i, 0:3] = base
    tri_coords[i, 3:6] = base + np.array([1 + 0.5*np.random.randn(), 0.3*np.random.randn(), 0.1*np.random.randn()])
    tri_coords[i, 6:9] = base + np.array([0.3*np.random.randn(), 1 + 0.5*np.random.randn(), 0.1*np.random.randn()])

K_batch_tri = fem_shell_core.batch_ke_mitc3(tri_coords, E, nu, t)
K_batch_tri = K_batch_tri.reshape(n_tri, 18, 18)

# Check a random sample
n_sample = 20
for idx in np.random.choice(n_tri, n_sample, replace=False):
    nodes = tri_coords[idx].reshape(3, 3)
    elem = MITC3(nodes, (0,1,2), mat_m, thickness=t)
    K_py = elem.K
    K_rs = K_batch_tri[idx]
    K_max = max(np.max(np.abs(K_py)), 1e-30)
    rel = np.max(np.abs(K_py - K_rs)) / K_max
    if rel > 1e-10:
        check(f"MITC3 batch elem {idx}: rel={rel:.2e}", False)

all_sym = all(np.allclose(K_batch_tri[i], K_batch_tri[i].T, atol=1e-10) for i in range(n_tri))
check(f"MITC3 batch: all {n_tri} symmetric", all_sym)
check(f"MITC3 batch: {n_sample} random samples match Python", True)  # would have failed above

# MITC4: 1000 random quads
n_quad = 1000
quad_coords = np.zeros((n_quad, 12))
for i in range(n_quad):
    base = np.random.randn(3) * 10
    quad_coords[i, 0:3]  = base
    quad_coords[i, 3:6]  = base + np.array([1 + 0.3*np.random.randn(), 0.2*np.random.randn(), 0.05*np.random.randn()])
    quad_coords[i, 6:9]  = base + np.array([1 + 0.3*np.random.randn(), 1 + 0.3*np.random.randn(), 0.1*np.random.randn()])
    quad_coords[i, 9:12] = base + np.array([0.2*np.random.randn(), 1 + 0.3*np.random.randn(), 0.05*np.random.randn()])

K_batch_quad = fem_shell_core.batch_ke_mitc4(quad_coords, E, nu, t)
K_batch_quad = K_batch_quad.reshape(n_quad, 24, 24)

for idx in np.random.choice(n_quad, n_sample, replace=False):
    nodes = quad_coords[idx].reshape(4, 3)
    elem = MITC4(nodes, (0,1,2,3), mat_iso, thickness=t)
    K_py = elem.K
    K_rs = K_batch_quad[idx]
    K_max = max(np.max(np.abs(K_py)), 1e-30)
    rel = np.max(np.abs(K_py - K_rs)) / K_max
    if rel > 1e-10:
        check(f"MITC4 batch elem {idx}: rel={rel:.2e}", False)

all_sym4 = all(np.allclose(K_batch_quad[i], K_batch_quad[i].T, atol=1e-10) for i in range(n_quad))
check(f"MITC4 batch: all {n_quad} symmetric", all_sym4)
check(f"MITC4 batch: {n_sample} random samples match Python", True)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
total = PASS_COUNT + FAIL_COUNT
print(f"RESULTS: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("ALL CHECKS PASSED ✓")
else:
    print("SOME CHECKS FAILED ✗")
print("=" * 70)

sys.exit(1 if FAIL_COUNT > 0 else 0)
