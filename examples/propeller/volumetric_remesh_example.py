"""
Example demonstrating volumetric remeshing from a closed surface mesh.

This script shows how to:
1. Load or generate a closed surface mesh
2. Detect if the mesh has open boundaries
3. Convert the surface mesh to a volumetric mesh
"""

import os
import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).resolve().parents[2] / "src"
if src_path.exists():
    sys.path.append(str(src_path))

from fem_shell.core.mesh import detect_open_boundaries, volumetric_remesh
from fem_shell.core.mesh.model import MeshModel
from fem_shell.core.mesh.entities import Node, MeshElement, ElementType
import numpy as np


def create_icosphere_surface(radius=1.0, subdivisions=2):
    """Create a closed icosphere surface mesh."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    a = 1.0
    b = 1.0 / phi
    
    # 12 vertices of an icosahedron
    vertices = np.array([
        [-a,  b,  0], [ a,  b,  0], [-a, -b,  0], [ a, -b,  0],
        [ 0, -a,  b], [ 0,  a,  b], [ 0, -a, -b], [ 0,  a, -b],
        [ b,  0, -a], [ b,  0,  a], [-b,  0, -a], [-b,  0,  a]
    ])
    
    # Normalize to unit sphere and scale by radius
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
    
    # 20 faces of an icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Simple subdivision (not implemented here, using base icosahedron)
    mesh = MeshModel()
    Node._id_counter = 0
    
    # Create nodes
    nodes = []
    for v in vertices:
        node = Node(coords=v)
        mesh.add_node(node)
        nodes.append(node)
    
    # Create elements
    for face in faces:
        element = MeshElement(
            nodes=[nodes[face[0]], nodes[face[1]], nodes[face[2]]],
            element_type=ElementType.triangle
        )
        mesh.add_element(element)
    
    return mesh


def main():
    # Generate a closed icosphere surface mesh
    print("Generating icosphere surface mesh...")
    surface_mesh = create_icosphere_surface(radius=1.0)
    
    print("\nSurface Mesh Information:")
    print(f"  Nodes:    {surface_mesh.node_count:,}")
    print(f"  Elements: {surface_mesh.elements_count:,}")
    
    # Detect open boundaries
    print("\nDetecting open boundaries...")
    has_open_boundaries = detect_open_boundaries(surface_mesh)
    
    if has_open_boundaries:
        print("  ⚠ Open boundaries detected!")
        print("  The surface mesh is not closed.")
        return
    else:
        print("  ✓ Surface mesh is closed.")
    
    # Convert to volumetric mesh
    print("\nConverting to volumetric mesh...")
    print("  (This may take a moment...)")
    
    try:
        volumetric_mesh = volumetric_remesh(surface_mesh, target_edge_length=0.3)
        
        print("\n✓ Volumetric mesh generated successfully!")
        print("\nVolumetric Mesh Information:")
        print(f"  Nodes:    {volumetric_mesh.node_count:,}")
        print(f"  Elements: {volumetric_mesh.elements_count:,}")
        
        # Get element type distribution
        from collections import Counter
        element_types = Counter([elem.element_type.name for elem in volumetric_mesh.elements])
        print("\nElement Types:")
        for elem_type, count in element_types.items():
            print(f"  {elem_type}: {count:,}")
        
        # Save the volumetric mesh
        output_dir = Path(__file__).resolve().parents[2] / "output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "sphere_volumetric.vtu"
        
        print(f"\nSaving volumetric mesh to: {output_file}")
        volumetric_mesh.write_mesh(str(output_file))
        print("  ✓ Mesh saved successfully!")
        
        # Visualize the mesh
        print("\nOpening mesh viewer...")
        try:
            volumetric_mesh.view()
        except Exception as gui_err:
            print(f"\nCould not open viewer: {gui_err}")
            print("\nYou can view the mesh in Paraview:")
            print(f"  paraview {output_file}")
            
    except Exception as e:
        print(f"\n✗ Error during volumetric remeshing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
