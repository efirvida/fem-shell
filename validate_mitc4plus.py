"""
Validation script for MITC4Plus implementation.

Runs comprehensive tests without pytest dependency.
"""

import numpy as np
import sys

from fem_shell.core.material import Material
from fem_shell.elements.MITC4 import MITC4, MITC4Plus


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_test(name, passed, details=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_api_compatibility():
    """Test that MITC4Plus shares the same API as MITC4."""
    print_header("1. API COMPATIBILITY TESTS")
    
    # Create test material and geometry
    material = Material(E=210e9, nu=0.3, rho=7850)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    node_ids = (1, 2, 3, 4)
    thickness = 0.01
    
    # Create MITC4Plus element
    elem = MITC4Plus(node_coords, node_ids, material, thickness)
    
    # Test 1.1: Element initialization
    test1 = (
        elem.element_type == "MITC4Plus" and
        elem.thickness == 0.01 and
        elem.dofs_count == 24 and
        elem.dofs_per_node == 6
    )
    print_test("Element initialization", test1, 
               f"type={elem.element_type}, DOFs={elem.dofs_count}")
    
    # Test 1.2: Tying points setup
    test2 = (
        len(elem._tying_points_eps_xx) == 4 and
        len(elem._tying_points_eps_yy) == 4 and
        len(elem._tying_points_gamma_xy) == 5
    )
    print_test("Tying points setup", test2,
               f"eps_xx={len(elem._tying_points_eps_xx)}, eps_yy={len(elem._tying_points_eps_yy)}, gamma_xy={len(elem._tying_points_gamma_xy)}")
    
    # Test 1.3: K matrix shape and symmetry
    K = elem.K
    test3 = (
        K.shape == (24, 24) and
        np.allclose(K, K.T, atol=1e-10)
    )
    print_test("K matrix shape and symmetry", test3,
               f"shape={K.shape}, symmetric={np.allclose(K, K.T, atol=1e-10)}")
    
    # Test 1.4: K matrix positive semi-definite
    eigs_K = np.linalg.eigvalsh(K)
    test4 = np.all(eigs_K >= -1e-10)
    print_test("K matrix positive semi-definite", test4,
               f"min eigenvalue={eigs_K[0]:.3e}, max={eigs_K[-1]:.3e}")
    
    # Test 1.5: M matrix shape and symmetry
    M = elem.M
    test5 = (
        M.shape == (24, 24) and
        np.allclose(M, M.T, atol=1e-10)
    )
    print_test("M matrix shape and symmetry", test5,
               f"shape={M.shape}, symmetric={np.allclose(M, M.T, atol=1e-10)}")
    
    # Test 1.6: M matrix positive semi-definite
    eigs_M = np.linalg.eigvalsh(M)
    test6 = np.all(eigs_M >= -1e-10)
    print_test("M matrix positive semi-definite", test6,
               f"min eigenvalue={eigs_M[0]:.3e}, max={eigs_M[-1]:.3e}")
    
    # Test 1.7: body_load method
    body_force = np.array([0, 0, -9.81])
    f = elem.body_load(body_force)
    test7 = (
        f.shape == (24,) and
        np.any(f != 0)
    )
    print_test("body_load method", test7,
               f"shape={f.shape}, has nonzero={np.any(f != 0)}")
    
    return all([test1, test2, test3, test4, test5, test6, test7])


def test_membrane_interpolation():
    """Test MITC4+ membrane interpolation functionality."""
    print_header("2. MEMBRANE INTERPOLATION TESTS")
    
    material = Material(E=210e9, nu=0.3, rho=7850)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    elem = MITC4Plus(node_coords, (1, 2, 3, 4), material, 0.01)
    
    # Test 2.1: Evaluate B_m at point
    B_m_point = elem._evaluate_B_m_at_point(0.0, 0.0)
    test1 = (
        B_m_point.shape == (3, 8) and
        np.all(np.isfinite(B_m_point))
    )
    print_test("Evaluate B_m at single point", test1,
               f"shape={B_m_point.shape}, all finite={np.all(np.isfinite(B_m_point))}")
    
    # Test 2.2: Get membrane strains at tying points
    eps_xx_list = elem._get_eps_xx_at_tying_points()
    eps_yy_list = elem._get_eps_yy_at_tying_points()
    gamma_xy_list = elem._get_gamma_xy_at_tying_points()
    
    test2 = (
        len(eps_xx_list) == 4 and
        all(e.shape == (8,) for e in eps_xx_list) and
        len(eps_yy_list) == 4 and
        all(e.shape == (8,) for e in eps_yy_list) and
        len(gamma_xy_list) == 5 and
        all(e.shape == (8,) for e in gamma_xy_list)
    )
    print_test("Get membrane strains at tying points", test2,
               f"eps_xx={len(eps_xx_list)}×8, eps_yy={len(eps_yy_list)}×8, gamma_xy={len(gamma_xy_list)}×8")
    
    # Test 2.3: Interpolation functions
    gp = 1 / np.sqrt(3)
    eps_xx_interp = elem._interpolate_eps_xx(gp, gp, eps_xx_list)
    eps_yy_interp = elem._interpolate_eps_yy(gp, gp, eps_yy_list)
    gamma_xy_interp = elem._interpolate_gamma_xy(gp, gp, gamma_xy_list)
    
    test3 = (
        eps_xx_interp.shape == (8,) and
        np.all(np.isfinite(eps_xx_interp)) and
        eps_yy_interp.shape == (8,) and
        np.all(np.isfinite(eps_yy_interp)) and
        gamma_xy_interp.shape == (8,) and
        np.all(np.isfinite(gamma_xy_interp))
    )
    print_test("Interpolation functions", test3,
               f"all shapes 8, all finite={np.all(np.isfinite(eps_xx_interp)) and np.all(np.isfinite(eps_yy_interp)) and np.all(np.isfinite(gamma_xy_interp))}")
    
    # Test 2.4: MITC4Plus B_m override
    B_m = elem.B_m(gp, gp)
    test4 = (
        B_m.shape == (3, 8) and
        np.all(np.isfinite(B_m))
    )
    print_test("MITC4Plus B_m override", test4,
               f"shape={B_m.shape}, all finite={np.all(np.isfinite(B_m))}")
    
    return all([test1, test2, test3, test4])


def test_mitc4plus_vs_mitc4():
    """Compare MITC4Plus vs MITC4."""
    print_header("3. MITC4Plus vs MITC4 COMPARISON")
    
    material = Material(E=210e9, nu=0.3, rho=7850)
    
    # Test 3.1: Flat element (should be very similar)
    print("\n  3.1 Flat rectangular element:")
    flat_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    
    mitc4_flat = MITC4(flat_coords, (1, 2, 3, 4), material, 0.01)
    mitc4plus_flat = MITC4Plus(flat_coords, (1, 2, 3, 4), material, 0.01)
    
    K_mitc4_flat = mitc4_flat.K
    K_mitc4plus_flat = mitc4plus_flat.K
    M_mitc4_flat = mitc4_flat.M
    M_mitc4plus_flat = mitc4plus_flat.M
    
    K_diff_flat = np.linalg.norm(K_mitc4_flat - K_mitc4plus_flat)
    M_diff_flat = np.linalg.norm(M_mitc4_flat - M_mitc4plus_flat)
    
    print(f"       K difference norm: {K_diff_flat:.3e}")
    print(f"       M difference norm: {M_diff_flat:.3e}")
    print(f"       M should be identical (both inherit same formulation)")
    
    test1 = np.allclose(M_mitc4_flat, M_mitc4plus_flat, atol=1e-10)
    print_test("M matrices identical for flat element", test1)
    
    # Test 3.2: Curved element (should differ more)
    print("\n  3.2 Curved cylindrical shell segment:")
    R = 1.0
    theta = 0.2
    L = 1.0
    curved_coords = np.array([
        [R * np.cos(-theta/2), -L/2, R * np.sin(-theta/2)],
        [R * np.cos(+theta/2), -L/2, R * np.sin(+theta/2)],
        [R * np.cos(+theta/2), +L/2, R * np.sin(+theta/2)],
        [R * np.cos(-theta/2), +L/2, R * np.sin(-theta/2)],
    ], dtype=float)
    
    mitc4_curved = MITC4(curved_coords, (1, 2, 3, 4), material, 0.05)
    mitc4plus_curved = MITC4Plus(curved_coords, (1, 2, 3, 4), material, 0.05)
    
    K_mitc4_curved = mitc4_curved.K
    K_mitc4plus_curved = mitc4plus_curved.K
    M_mitc4_curved = mitc4_curved.M
    M_mitc4plus_curved = mitc4plus_curved.M
    
    K_diff_curved = np.linalg.norm(K_mitc4_curved - K_mitc4plus_curved)
    M_diff_curved = np.linalg.norm(M_mitc4_curved - M_mitc4plus_curved)
    
    print(f"       K difference norm: {K_diff_curved:.3e}")
    print(f"       M difference norm: {M_diff_curved:.3e}")
    
    test2 = K_diff_curved > 1e-6
    print_test("K matrices differ for curved element", test2,
               f"difference > 1e-6: {K_diff_curved > 1e-6}")
    
    test3 = np.allclose(M_mitc4_curved, M_mitc4plus_curved, atol=1e-10)
    print_test("M matrices identical for curved element", test3)
    
    return all([test1, test2, test3])


def test_numerical_stability():
    """Test numerical stability for edge cases."""
    print_header("4. NUMERICAL STABILITY TESTS")
    
    material = Material(E=210e9, nu=0.3, rho=7850)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    
    # Test 4.1: Very thin element
    print("\n  4.1 Very thin element (h/L = 0.001):")
    elem_thin = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.001)
    
    K_thin = elem_thin.K
    M_thin = elem_thin.M
    eigs_K_thin = np.linalg.eigvalsh(K_thin)
    eigs_M_thin = np.linalg.eigvalsh(M_thin)
    
    test1 = np.all(eigs_K_thin >= -1e-10) and np.all(eigs_M_thin >= -1e-10)
    print_test("Thin element matrices positive semi-definite", test1,
               f"K min eig={eigs_K_thin[0]:.3e}, M min eig={eigs_M_thin[0]:.3e}")
    
    # Test 4.2: Thick element
    print("\n  4.2 Thick element (h/L = 0.1):")
    elem_thick = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.1)
    
    K_thick = elem_thick.K
    M_thick = elem_thick.M
    eigs_K_thick = np.linalg.eigvalsh(K_thick)
    eigs_M_thick = np.linalg.eigvalsh(M_thick)
    
    test2 = np.all(eigs_K_thick >= -1e-10) and np.all(eigs_M_thick >= -1e-10)
    print_test("Thick element matrices positive semi-definite", test2,
               f"K min eig={eigs_K_thick[0]:.3e}, M min eig={eigs_M_thick[0]:.3e}")
    
    # Test 4.3: Stiffness modifiers
    print("\n  4.3 Stiffness modifiers (kx_mod=0.8, ky_mod=1.2):")
    elem_mod1 = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.01,
                          kx_mod=1.0, ky_mod=1.0)
    elem_mod2 = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.01,
                          kx_mod=0.8, ky_mod=1.2)
    
    K_mod1 = elem_mod1.K
    K_mod2 = elem_mod2.K
    eigs_K_mod1 = np.linalg.eigvalsh(K_mod1)
    eigs_K_mod2 = np.linalg.eigvalsh(K_mod2)
    
    test3a = not np.allclose(K_mod1, K_mod2, atol=1e-10)
    test3b = np.all(eigs_K_mod1 >= -1e-10) and np.all(eigs_K_mod2 >= -1e-10)
    
    K_diff_mod = np.linalg.norm(K_mod1 - K_mod2)
    print_test("Different modifiers produce different K", test3a,
               f"K difference norm: {K_diff_mod:.3e}")
    print_test("Both with modifiers positive semi-definite", test3b,
               f"K1 min eig={eigs_K_mod1[0]:.3e}, K2 min eig={eigs_K_mod2[0]:.3e}")
    
    return all([test1, test2, test3a, test3b])


def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MITC4Plus Shell Element - Implementation Validation".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        result1 = test_api_compatibility()
        result2 = test_membrane_interpolation()
        result3 = test_mitc4plus_vs_mitc4()
        result4 = test_numerical_stability()
        
        print_header("SUMMARY")
        print(f"  1. API Compatibility:          {'✓ PASS' if result1 else '✗ FAIL'}")
        print(f"  2. Membrane Interpolation:     {'✓ PASS' if result2 else '✗ FAIL'}")
        print(f"  3. MITC4Plus vs MITC4:        {'✓ PASS' if result3 else '✗ FAIL'}")
        print(f"  4. Numerical Stability:        {'✓ PASS' if result4 else '✗ FAIL'}")
        
        all_passed = result1 and result2 and result3 and result4
        print("\n" + "=" * 70)
        if all_passed:
            print("  ✓ ALL TESTS PASSED - MITC4Plus implementation is complete!")
        else:
            print("  ✗ SOME TESTS FAILED - Please review the implementation")
        print("=" * 70 + "\n")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
