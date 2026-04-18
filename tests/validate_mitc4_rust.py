"""Cross-validation: Python MITC4 vs Rust MITC4+ kernel."""
import numpy as np
import time
import sys
sys.path.insert(0, "/scratch/leahk/eduardo.donestevez/fem-shell/src")

from fem_shell.elements.MITC4 import MITC4
from fem_shell.core.material import IsotropicMaterial
import fem_shell_core

# ============================================================================
# Test 1: Single element stiffness (flat quad in XY-plane)
# ============================================================================
print("=" * 60)
print("Test 1: Flat quad in XY-plane — stiffness matrix")
print("=" * 60)

E, nu, rho, t = 210e9, 0.3, 7850.0, 0.01
mat = IsotropicMaterial(name="steel", E=E, nu=nu, rho=rho)

# Simple unit square in XY-plane
nodes = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
])

elem_py = MITC4(
    node_coords=nodes,
    node_ids=(0, 1, 2, 3),
    material=mat,
    thickness=t,
)
K_py = elem_py.K  # 24×24 global stiffness

# Rust: batch with 1 element
coords_flat = nodes.reshape(1, 12)
K_rust_flat = fem_shell_core.batch_ke_mitc4(coords_flat, E, nu, t)
K_rust = K_rust_flat.reshape(24, 24)

# Compare
diff = np.abs(K_py - K_rust)
max_abs_diff = np.max(diff)
K_max = np.max(np.abs(K_py))
max_rel_diff = max_abs_diff / K_max if K_max > 0 else 0

print(f"  K_py max:        {K_max:.6e}")
print(f"  Max abs diff:    {max_abs_diff:.6e}")
print(f"  Max rel diff:    {max_rel_diff:.6e}")
print(f"  Symmetric (py):  {np.allclose(K_py, K_py.T, atol=1e-10)}")
print(f"  Symmetric (rs):  {np.allclose(K_rust, K_rust.T, atol=1e-10)}")

eigvals_py = np.linalg.eigvalsh(K_py)
eigvals_rs = np.linalg.eigvalsh(K_rust)
print(f"  Min eigenvalue (py): {eigvals_py[0]:.6e}")
print(f"  Min eigenvalue (rs): {eigvals_rs[0]:.6e}")
print(f"  PASS: {max_rel_diff < 1e-8}")
print()

# ============================================================================
# Test 2: Tilted non-planar quad
# ============================================================================
print("=" * 60)
print("Test 2: Tilted/warped quad — stiffness matrix")
print("=" * 60)

nodes2 = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.1],
    [2.0, 1.5, 0.2],
    [0.0, 1.5, 0.05],
])

elem_py2 = MITC4(
    node_coords=nodes2,
    node_ids=(0, 1, 2, 3),
    material=mat,
    thickness=t,
)
K_py2 = elem_py2.K

coords_flat2 = nodes2.reshape(1, 12)
K_rust_flat2 = fem_shell_core.batch_ke_mitc4(coords_flat2, E, nu, t)
K_rust2 = K_rust_flat2.reshape(24, 24)

diff2 = np.abs(K_py2 - K_rust2)
max_abs_diff2 = np.max(diff2)
K_max2 = np.max(np.abs(K_py2))
max_rel_diff2 = max_abs_diff2 / K_max2 if K_max2 > 0 else 0

print(f"  K_py max:        {K_max2:.6e}")
print(f"  Max abs diff:    {max_abs_diff2:.6e}")
print(f"  Max rel diff:    {max_rel_diff2:.6e}")
print(f"  PASS: {max_rel_diff2 < 1e-8}")
print()

# ============================================================================
# Test 3: Mass matrix
# ============================================================================
print("=" * 60)
print("Test 3: Mass matrix (flat quad)")
print("=" * 60)

M_py = elem_py.M

M_rust_flat = fem_shell_core.batch_me_mitc4(coords_flat, E, nu, rho, t)
M_rust = M_rust_flat.reshape(24, 24)

diff_m = np.abs(M_py - M_rust)
max_abs_diff_m = np.max(diff_m)
M_max = np.max(np.abs(M_py))
max_rel_diff_m = max_abs_diff_m / M_max if M_max > 0 else 0

print(f"  M_py max:        {M_max:.6e}")
print(f"  Max abs diff:    {max_abs_diff_m:.6e}")
print(f"  Max rel diff:    {max_rel_diff_m:.6e}")
print(f"  PASS: {max_rel_diff_m < 1e-8}")
print()

# ============================================================================
# Test 4: Performance benchmark
# ============================================================================
print("=" * 60)
print("Test 4: Performance — 100K elements")
print("=" * 60)

n_elem = 100_000
np.random.seed(42)
base_coords = np.tile(nodes.reshape(1, 12), (n_elem, 1))
# Add small random perturbation to make elements slightly different
base_coords += np.random.randn(n_elem, 12) * 0.01

# Python timing (small subset)
n_py = 100
t0 = time.perf_counter()
for i in range(n_py):
    e = MITC4(
        node_coords=base_coords[i].reshape(4, 3),
        node_ids=(0, 1, 2, 3),
        material=mat,
        thickness=t,
    )
    _ = e.K
t_py = (time.perf_counter() - t0) / n_py

# Rust timing (full batch)
t0 = time.perf_counter()
_ = fem_shell_core.batch_ke_mitc4(base_coords, E, nu, t)
t_rust = time.perf_counter() - t0

py_estimated = t_py * n_elem
speedup = py_estimated / t_rust

print(f"  Python: {t_py*1000:.3f} ms/elem → estimated {py_estimated:.1f}s for {n_elem} elems")
print(f"  Rust:   {t_rust*1000:.1f} ms for {n_elem} elements")
print(f"  Speedup: {speedup:.0f}x")
print()

# ============================================================================
# Summary
# ============================================================================
all_pass = max_rel_diff < 1e-8 and max_rel_diff2 < 1e-8 and max_rel_diff_m < 1e-8
print("=" * 60)
print(f"ALL TESTS PASS: {all_pass}")
print("=" * 60)
