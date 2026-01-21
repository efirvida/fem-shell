#!/usr/bin/env python3
"""
Simple Boolean Union Test: Hub + Transmission only
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[2] / "src"
if src_path.exists():
    sys.path.append(str(src_path))

from fem_shell.core.mesh import load_mesh, boolean_union_meshes, detect_open_boundaries

def main():
    geometries_dir = Path(__file__).resolve().parents[1] / "propeller" / "Geometries" / "propeller2"
    
    hub_path = geometries_dir / "hub.obj"
    transmission_path = geometries_dir / "transmission.obj"
    
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("SIMPLE BOOLEAN TEST - Hub + Transmission")
    print("=" * 70)
    
    # Load meshes
    print("\nLoading meshes...")
    hub = load_mesh(str(hub_path))
    transmission = load_mesh(str(transmission_path))
    
    print(f"  Hub: {hub.node_count:,} nodes, {hub.elements_count:,} elements")
    print(f"  Transmission: {transmission.node_count:,} nodes, {transmission.elements_count:,} elements")
    
    # Check closed
    print(f"  Hub closed: {not detect_open_boundaries(hub)}")
    print(f"  Transmission closed: {not detect_open_boundaries(transmission)}")
    
    # Boolean union
    print("\nPerforming boolean union (this may take time)...")
    try:
        union_mesh = boolean_union_meshes([hub, transmission])
        print(f"  ✓ Union completed!")
        print(f"    Nodes: {union_mesh.node_count:,}")
        print(f"    Elements: {union_mesh.elements_count:,}")
        
        # Save
        output_path = output_dir / "hub_transmission_union.vtu"
        union_mesh.write_mesh(str(output_path))
        print(f"\n  ✓ Saved to: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
