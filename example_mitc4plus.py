"""
Example: Using MITC4Plus Shell Element

This example demonstrates how to use the MITC4Plus element,
which is an enhanced version of MITC4 that eliminates membrane locking.
"""

import numpy as np
from fem_shell.core.material import Material
from fem_shell.elements.MITC4 import MITC4, MITC4Plus


def example_flat_plate():
    """Example 1: Flat plate element."""
    print("\n" + "="*70)
    print("Example 1: Flat Rectangular Plate")
    print("="*70)
    
    # Material properties (steel)
    material = Material(E=210e9, nu=0.3, rho=7850)
    
    # Element geometry: rectangular plate 1m × 1m
    node_coords = np.array([
        [0.0, 0.0, 0.0],  # Node 1
        [1.0, 0.0, 0.0],  # Node 2
        [1.0, 1.0, 0.0],  # Node 3
        [0.0, 1.0, 0.0],  # Node 4
    ], dtype=float)
    
    node_ids = (1, 2, 3, 4)
    thickness = 0.01  # 10 mm
    
    # Create MITC4Plus element
    elem = MITC4Plus(node_coords, node_ids, material, thickness)
    
    print(f"\nElement Information:")
    print(f"  Type: {elem.element_type}")
    print(f"  DOFs: {elem.dofs_count} (6 per node × 4 nodes)")
    print(f"  Thickness: {thickness} m")
    print(f"  Area: {elem.area():.4f} m²")
    
    # Get stiffness and mass matrices
    print(f"\nStiffness Matrix K:")
    K = elem.K
    print(f"  Shape: {K.shape}")
    print(f"  Symmetric: {np.allclose(K, K.T, atol=1e-10)}")
    eigs_K = np.linalg.eigvalsh(K)
    print(f"  Eigenvalues: min={eigs_K[0]:.3e}, max={eigs_K[-1]:.3e}")
    print(f"  Positive semi-definite: {np.all(eigs_K >= -1e-10)}")
    
    print(f"\nMass Matrix M:")
    M = elem.M
    print(f"  Shape: {M.shape}")
    print(f"  Symmetric: {np.allclose(M, M.T, atol=1e-10)}")
    eigs_M = np.linalg.eigvalsh(M)
    print(f"  Eigenvalues: min={eigs_M[0]:.3e}, max={eigs_M[-1]:.3e}")
    print(f"  Positive semi-definite: {np.all(eigs_M >= -1e-10)}")
    
    # Body load (gravity)
    print(f"\nBody Load (Gravity):")
    body_force = np.array([0, 0, -9.81])  # g = 9.81 m/s² in -z
    f = elem.body_load(body_force)
    print(f"  Gravity vector: {body_force}")
    print(f"  Load vector shape: {f.shape}")
    print(f"  Total load magnitude: {np.linalg.norm(f):.4f}")


def example_curved_shell():
    """Example 2: Curved cylindrical shell element."""
    print("\n" + "="*70)
    print("Example 2: Curved Cylindrical Shell")
    print("="*70)
    
    material = Material(E=210e9, nu=0.3, rho=7850)
    
    # Cylindrical shell segment: radius R = 1 m, angular span θ = 20°
    R = 1.0
    theta = np.radians(20)  # 20 degrees
    L = 1.0  # Length in z
    
    # Create curved element coordinates
    node_coords = np.array([
        [R * np.cos(-theta/2), -L/2, R * np.sin(-theta/2)],
        [R * np.cos(+theta/2), -L/2, R * np.sin(+theta/2)],
        [R * np.cos(+theta/2), +L/2, R * np.sin(+theta/2)],
        [R * np.cos(-theta/2), +L/2, R * np.sin(-theta/2)],
    ], dtype=float)
    
    thickness = 0.05  # 50 mm
    
    # Compare MITC4 vs MITC4Plus
    print(f"\nGeometry:")
    print(f"  Radius: {R} m")
    print(f"  Angular span: {np.degrees(theta):.1f}°")
    print(f"  Length: {L} m")
    print(f"  Thickness: {thickness} m")
    print(f"  Radius/Thickness ratio: {R/thickness:.1f}")
    
    print(f"\nCreating elements...")
    elem_mitc4 = MITC4(node_coords, (1, 2, 3, 4), material, thickness)
    elem_mitc4plus = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness)
    
    # Compute matrices
    print(f"  Computing MITC4 matrices...")
    K_mitc4 = elem_mitc4.K
    M_mitc4 = elem_mitc4.M
    
    print(f"  Computing MITC4Plus matrices...")
    K_mitc4plus = elem_mitc4plus.K
    M_mitc4plus = elem_mitc4plus.M
    
    # Compare
    print(f"\nComparison:")
    K_diff = np.linalg.norm(K_mitc4 - K_mitc4plus)
    M_diff = np.linalg.norm(M_mitc4 - M_mitc4plus)
    
    print(f"  K difference: {K_diff:.3e}")
    print(f"  M difference: {M_diff:.3e}")
    print(f"  Explanation:")
    print(f"    - K differs because MITC4+ adds membrane interpolation")
    print(f"    - M is identical (both use same mass formulation)")
    
    # Eigenvalue analysis
    print(f"\nMITC4 Stiffness Eigenvalues:")
    eigs_K_mitc4 = np.linalg.eigvalsh(K_mitc4)
    print(f"  Min: {eigs_K_mitc4[0]:.3e}, Max: {eigs_K_mitc4[-1]:.3e}")
    
    print(f"\nMITC4Plus Stiffness Eigenvalues:")
    eigs_K_mitc4plus = np.linalg.eigvalsh(K_mitc4plus)
    print(f"  Min: {eigs_K_mitc4plus[0]:.3e}, Max: {eigs_K_mitc4plus[-1]:.3e}")


def example_with_modifiers():
    """Example 3: Element with stiffness modifiers."""
    print("\n" + "="*70)
    print("Example 3: Element with Stiffness Modifiers")
    print("="*70)
    
    material = Material(E=210e9, nu=0.3, rho=7850)
    
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    
    # Create elements with different modifiers
    elem1 = MITC4Plus(node_coords, (1, 2, 3, 4), material, 0.01, 
                      kx_mod=1.0, ky_mod=1.0)
    elem2 = MITC4Plus(node_coords, (1, 2, 3, 4), material, 0.01,
                      kx_mod=0.8, ky_mod=1.2)
    
    print(f"\nElement 1 (kx_mod=1.0, ky_mod=1.0):")
    K1 = elem1.K
    eigs_K1 = np.linalg.eigvalsh(K1)
    print(f"  K eigenvalues: min={eigs_K1[0]:.3e}, max={eigs_K1[-1]:.3e}")
    
    print(f"\nElement 2 (kx_mod=0.8, ky_mod=1.2):")
    K2 = elem2.K
    eigs_K2 = np.linalg.eigvalsh(K2)
    print(f"  K eigenvalues: min={eigs_K2[0]:.3e}, max={eigs_K2[-1]:.3e}")
    
    K_diff = np.linalg.norm(K1 - K2)
    print(f"\nK difference: {K_diff:.3e}")
    print(f"  (Different modifiers produce different stiffness, as expected)")


def example_edge_cases():
    """Example 4: Edge cases - very thin and very thick elements."""
    print("\n" + "="*70)
    print("Example 4: Edge Cases - Thin and Thick Elements")
    print("="*70)
    
    material = Material(E=210e9, nu=0.3, rho=7850)
    
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    
    # Very thin shell (h/L = 0.001)
    print(f"\nVery Thin Shell (h/L = 0.001):")
    elem_thin = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.001)
    K_thin = elem_thin.K
    M_thin = elem_thin.M
    eigs_K_thin = np.linalg.eigvalsh(K_thin)
    eigs_M_thin = np.linalg.eigvalsh(M_thin)
    
    print(f"  K positive semi-definite: {np.all(eigs_K_thin >= -1e-10)}")
    print(f"  M positive semi-definite: {np.all(eigs_M_thin >= -1e-10)}")
    print(f"  K condition number: {eigs_K_thin[-1] / (eigs_K_thin[0] + 1e-16):.3e}")
    
    # Very thick shell (h/L = 0.1)
    print(f"\nVery Thick Shell (h/L = 0.1):")
    elem_thick = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.1)
    K_thick = elem_thick.K
    M_thick = elem_thick.M
    eigs_K_thick = np.linalg.eigvalsh(K_thick)
    eigs_M_thick = np.linalg.eigvalsh(M_thick)
    
    print(f"  K positive semi-definite: {np.all(eigs_K_thick >= -1e-10)}")
    print(f"  M positive semi-definite: {np.all(eigs_M_thick >= -1e-10)}")
    print(f"  K condition number: {eigs_K_thick[-1] / (eigs_K_thick[0] + 1e-16):.3e}")


if __name__ == "__main__":
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  MITC4Plus Shell Element - Usage Examples".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        example_flat_plate()
        example_curved_shell()
        example_with_modifiers()
        example_edge_cases()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
