"""
Example demonstrating automatic boundary closing and volumetric remeshing
with a simple cylinder that has an open top.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add src to sys.path
src_path = Path(__file__).resolve().parents[2] / "src"
if src_path.exists():
    sys.path.append(str(src_path))

from fem_shell.core.mesh import detect_open_boundaries, close_open_boundaries, volumetric_remesh
from fem_shell.core.mesh.model import MeshModel
from fem_shell.core.mesh.entities import Node, MeshElement, ElementType


def create_open_cylinder(radius=1.0, height=2.0, n_circum=16, n_height=8):
    """Create a cylinder surface mesh with open top and bottom."""
    mesh = MeshModel()
    Node._id_counter = 0
    
    # Create nodes
    nodes = []
    for j in range(n_height + 1):
        z = j * height / n_height
        for i in range(n_circum):
            theta = 2 * np.pi * i / n_circum
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            node = Node(coords=[x, y, z])
            mesh.add_node(node)
            nodes.append(node)
    
    # Create quad elements for the cylinder wall
    for j in range(n_height):
        for i in range(n_circum):
            i_next = (i + 1) % n_circum
            
            # Node indices
            n1 = j * n_circum + i
            n2 = j * n_circum + i_next
            n3 = (j + 1) * n_circum + i_next
            n4 = (j + 1) * n_circum + i
            
            # Create two triangles per quad
            elem1 = MeshElement(
                nodes=[nodes[n1], nodes[n2], nodes[n3]],
                element_type=ElementType.triangle
            )
            mesh.add_element(elem1)
            
            elem2 = MeshElement(
                nodes=[nodes[n1], nodes[n3], nodes[n4]],
                element_type=ElementType.triangle
            )
            mesh.add_element(elem2)
    
    return mesh


def main():
    print("=" * 70)
    print("Example: Automatic Boundary Closing and Volumetric Remeshing")
    print("=" * 70)
    
    # Create an open cylinder
    print("\n1. Creating open cylinder mesh...")
    open_mesh = create_open_cylinder(radius=1.0, height=2.0, n_circum=16, n_height=8)
    
    print(f"   Nodes:    {open_mesh.node_count:,}")
    print(f"   Elements: {open_mesh.elements_count:,}")
    
    # Detect open boundaries
    print("\n2. Detecting open boundaries...")
    has_open = detect_open_boundaries(open_mesh)
    print(f"   Open boundaries: {'Yes ⚠' if has_open else 'No ✓'}")
    
    # Close boundaries manually
    print("\n3. Closing open boundaries...")
    closed_mesh = close_open_boundaries(open_mesh)
    print(f"   Closed mesh nodes:    {closed_mesh.node_count:,}")
    print(f"   Closed mesh elements: {closed_mesh.elements_count:,}")
    
    # Verify closure
    still_open = detect_open_boundaries(closed_mesh)
    print(f"   Still has open boundaries: {'Yes ⚠' if still_open else 'No ✓'}")
    
    # Convert to volumetric mesh
    print("\n4. Converting to volumetric mesh...")
    print("   (This may take a moment...)")
    
    volumetric_mesh = volumetric_remesh(
        closed_mesh,
        target_edge_length=0.3,
        auto_close_boundaries=False  # Already closed
    )
    
    print(f"   ✓ Volumetric mesh generated!")
    print(f"   Nodes:    {volumetric_mesh.node_count:,}")
    print(f"   Elements: {volumetric_mesh.elements_count:,}")
    
    # Get element type distribution
    from collections import Counter
    element_types = Counter([elem.element_type.name for elem in volumetric_mesh.elements])
    print("\n   Element Types:")
    for elem_type, count in element_types.items():
        print(f"     {elem_type}: {count:,}")
    
    # Save meshes
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("\n5. Saving meshes...")
    
    open_file = output_dir / "cylinder_open.vtu"
    open_mesh.write_mesh(str(open_file))
    print(f"   Open mesh:       {open_file}")
    
    closed_file = output_dir / "cylinder_closed.vtu"
    closed_mesh.write_mesh(str(closed_file))
    print(f"   Closed mesh:     {closed_file}")
    
    volumetric_file = output_dir / "cylinder_volumetric.vtu"
    volumetric_mesh.write_mesh(str(volumetric_file))
    print(f"   Volumetric mesh: {volumetric_file}")
    
    print("\n" + "=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)
    print("\nYou can visualize the meshes in Paraview:")
    print(f"  paraview {open_file}")
    print(f"  paraview {closed_file}")
    print(f"  paraview {volumetric_file}")
    
    # Try to visualize
    try:
        print("\nOpening volumetric mesh viewer...")
        volumetric_mesh.view()
    except Exception as e:
        print(f"\nCould not open viewer: {e}")


if __name__ == "__main__":
    main()
