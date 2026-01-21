#!/usr/bin/env python3
"""
Boolean Union Example: Combine multiple propeller components into a single mesh.

This example demonstrates how to use boolean union operations to combine
multiple OBJ files (hub, propeller blade, transmission) into a single
surface mesh, and then convert it to a volumetric mesh.

Files used:
- hub.obj
- propeller.obj
- transmission.obj

Output:
- propeller_union_surface.vtu (combined surface mesh before volumetric conversion)
- propeller_union_volumetric.vtu (final volumetric mesh)
"""

import sys
from pathlib import Path

import numpy as np

# Add src to sys.path to ensure fem_shell can be imported if not installed
src_path = Path(__file__).resolve().parents[2] / "src"
if src_path.exists():
    sys.path.append(str(src_path))

from fem_shell.core.mesh import (
    load_mesh,
    boolean_union_meshes,
    detect_open_boundaries,
    volumetric_remesh,
)


def main():
    # Define paths to the propeller component files
    geometries_dir = Path(__file__).resolve().parents[1] / "propeller" / "Geometries" / "propeller2"
    
    hub_path = geometries_dir / "hub.obj"
    propeller_path = geometries_dir / "propeller.obj"
    transmission_path = geometries_dir / "transmission.obj"
    
    # Output directory
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Check that all input files exist
    print("=" * 70)
    print("BOOLEAN UNION EXAMPLE - Combine Propeller Components")
    print("=" * 70)
    
    files_to_load = [
        ("Hub", hub_path),
        ("Propeller", propeller_path),
        ("Transmission", transmission_path),
    ]
    
    for name, path in files_to_load:
        if not path.exists():
            print(f"✗ {name} file not found: {path}")
            return
        print(f"✓ {name} file found: {path}")
    
    print()
    
    try:
        # Step 1: Load individual components
        print("Step 1: Loading component meshes...")
        print("-" * 70)
        
        components = {}
        for name, path in files_to_load:
            print(f"  Loading {name}...", end=" ")
            mesh = load_mesh(str(path))
            components[name] = mesh
            
            has_open = detect_open_boundaries(mesh)
            status = "⚠ OPEN" if has_open else "✓ CLOSED"
            print(f"{status}")
            print(f"    Nodes: {mesh.node_count:,}, Elements: {mesh.elements_count:,}")
        
        print()
        
        # Step 2: Perform boolean union
        print("Step 2: Performing boolean union operation...")
        print("-" * 70)
        
        # Check if all meshes are closed (they should be for union operation)
        all_closed = all(not detect_open_boundaries(mesh) for mesh in components.values())
        
        if not all_closed:
            print("  ⚠ Warning: Some meshes have open boundaries.")
            print("  The boolean union operation requires closed surfaces.")
            print("  Attempting to proceed anyway (may fail)...")
        
        # Collect meshes in order for union
        meshes_to_union = [components[name] for name, _ in files_to_load]
        
        print("  Performing union: Hub ∪ Propeller ∪ Transmission...")
        print("  (This may take a minute...)")
        
        union_mesh = boolean_union_meshes(meshes_to_union)
        
        print(f"  ✓ Union completed!")
        print(f"    Nodes: {union_mesh.node_count:,}")
        print(f"    Elements: {union_mesh.elements_count:,}")
        print()
        
        # Step 3: Save surface union mesh
        print("Step 3: Saving surface union mesh...")
        print("-" * 70)
        
        surface_output = output_dir / "propeller_union_surface.vtu"
        print(f"  Saving to: {surface_output}")
        union_mesh.write_mesh(str(surface_output))
        print(f"  ✓ Surface mesh saved!")
        print()
        
        # Step 4: Convert to volumetric mesh using fTetWild (wildmeshing)
        print("Step 4: Converting to volumetric mesh...")
        print("-" * 70)
        print("  Note: This may take several minutes for large meshes...")
        print("  Using fTetWild (robust for low-quality input meshes)...")
        
        try:
            import wildmeshing as wm
            import tempfile
            import meshio
            from fem_shell.core.mesh.model import MeshModel
            from fem_shell.core.mesh.entities import Node, MeshElement, ElementType
            
            # Convert union mesh to numpy arrays for fTetWild
            node_id_to_idx = {node.id: idx for idx, node in enumerate(union_mesh.nodes)}
            vertices = np.array([[node.x, node.y, node.z] for node in union_mesh.nodes])
            
            faces = []
            for element in union_mesh.elements:
                face_indices = [node_id_to_idx[node.id] for node in element.nodes]
                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    # Split quad into 2 triangles
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])
            
            faces = np.array(faces, dtype=np.int32)
            
            print(f"  Input: {len(vertices):,} vertices, {len(faces):,} faces")
            print("  Running fTetWild (this may take a few minutes)...")
            
            # fTetWild requires file paths, not arrays
            with tempfile.TemporaryDirectory() as tmpdir:
                input_mesh_path = f"{tmpdir}/input.obj"
                output_mesh_path = f"{tmpdir}/output.msh"
                
                # Write input mesh
                meshio.write(input_mesh_path, meshio.Mesh(
                    points=vertices,
                    cells=[("triangle", faces)]
                ))
                
                # Run fTetWild
                # Key parameters for fine mesh:
                # - edge_length_r: relative edge length (smaller = finer)
                # - skip_simplify: don't simplify input mesh (preserves detail)
                # - coarsen: False to avoid mesh coarsening
                # - epsilon: envelope tolerance (smaller = closer to surface)
                success = wm.tetrahedralize(
                    input=input_mesh_path,
                    output=output_mesh_path,
                    edge_length_r=0.005,      # Fine mesh (5% of bbox diagonal)
                    epsilon=0.0005,           # Tight envelope
                    stop_quality=10,          # Better quality elements
                    max_its=100,              # More optimization iterations
                    skip_simplify=True,       # Don't simplify input!
                    coarsen=False,            # Don't coarsen the mesh
                    mute_log=False
                )
                
                if not success:
                    raise RuntimeError("fTetWild failed to create mesh")
                
                # Read output mesh
                result = meshio.read(output_mesh_path)
                tet_verts = result.points
                
                # Find tetra cells
                tet_tets = None
                for cell_block in result.cells:
                    if cell_block.type == "tetra":
                        tet_tets = cell_block.data
                        break
                
                if tet_tets is None:
                    raise RuntimeError("No tetrahedra found in output")
            
            print(f"  ✓ fTetWild completed!")
            print(f"    Vertices: {len(tet_verts):,}")
            print(f"    Tetrahedra: {len(tet_tets):,}")
            
            # Convert to MeshModel
            volumetric_mesh = MeshModel()
            Node._id_counter = 0
            
            node_map = {}
            for i, v in enumerate(tet_verts):
                node = Node(coords=v)
                node_map[i] = node
                volumetric_mesh.add_node(node)
            
            for tet in tet_tets:
                nodes = [node_map[idx] for idx in tet]
                element = MeshElement(nodes=nodes, element_type=ElementType.tetra)
                volumetric_mesh.add_element(element)
            
            print(f"  ✓ Volumetric mesh created!")
            print(f"    Nodes: {volumetric_mesh.node_count:,}")
            print(f"    Elements: {volumetric_mesh.elements_count:,}")
            
            volumetric_output = output_dir / "propeller_union_volumetric.vtu"
            volumetric_mesh.write_mesh(str(volumetric_output))
            print(f"  ✓ Volumetric mesh saved to: {volumetric_output}")
            
        except Exception as vol_err:
            import traceback
            print(f"  ✗ Error during volumetric conversion: {vol_err}")
            traceback.print_exc()
            volumetric_mesh = None
        
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Surface union mesh: {surface_output}")
        print()
        print("Boolean union results:")
        print(f"  Input elements:  {sum(m.elements_count for m in meshes_to_union):,}")
        print(f"  Output elements: {union_mesh.elements_count:,}")
        print(f"  Removed:         {sum(m.elements_count for m in meshes_to_union) - union_mesh.elements_count:,} (internal surfaces)")
        print()
        print("Next steps:")
        print("  1. Open the VTU file in Paraview to visualize")
        print("  2. Use a Slice filter to verify internal surfaces were removed")
        print("  3. Run volumetric_remesh() if volumetric mesh is needed")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
