import os
import sys
from pathlib import Path

# Add src to sys.path to ensure fem_shell can be imported if not installed
src_path = Path(__file__).resolve().parents[2] / "src"
if src_path.exists():
    sys.path.append(str(src_path))

from fem_shell.core.mesh import load_mesh

def main():
    # Define the path to the OBJ file
    # Path relative to the script location: ../Geometries/propeller/propellerTip.obj
    # Absolute path provided by user: /home/efirvida/fem-shell/examples/Geometries/propeller/propellerTip.obj
    obj_path = Path(__file__).resolve().parents[1] / "Geometries" / "propeller" / "propellerTip.obj"
    
    if not obj_path.exists():
        print(f"Error: File not found at {obj_path}")
        return

    print(f"Loading mesh from: {obj_path}")
    
    try:
        # Load the mesh using the newly implemented support
        mesh = load_mesh(str(obj_path), nodesets=[
            {"name": "base_nodes", "type": "coordinate", "axis": "y", "value": 0.049, "mode": "near"},
        ])
        mesh.create_node_set_by_geometry(
            "blades", 
            "direction", 
            point=[0, 0, 0],
            direction=[0, 1, 0],  # Eje de rotación (Y)
            method="radial",
            distance=0.0265,
            mode="outside",
        )
        
        print("\nMesh Information:")
        print(f"  Nodes:    {mesh.node_count:,}")
        print(f"  Elements: {mesh.elements_count:,}")
        
        # Visualize the mesh
        print("\nOpening mesh viewer...")
        try:
            mesh.view()
        except Exception as gui_err:
            print(f"\nCould not open viewer: {gui_err}")
            print("\nTip: This error often happens in WSL/Linux environments missing some Qt dependencies.")
            print("Try installing them with:")
            print("  sudo apt-get update && sudo apt-get install -y libxcb-cursor0")
            print("\nAlternatively, you can export the mesh to VTK and view it in Paraview:")
            print("  mesh.write_mesh('propeller_preview.vtu')")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
