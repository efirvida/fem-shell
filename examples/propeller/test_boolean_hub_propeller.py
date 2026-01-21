#!/usr/bin/env python3
"""
Boolean Union Test: Hub + ONE propeller blade
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
    propeller_path = geometries_dir / "propeller.obj"
    
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("BOOLEAN TEST - Hub + Propeller")
    print("=" * 70)
    print("\nThis is the CRITICAL test: the propeller blades penetrate the hub,")
    print("so we should see internal surfaces being removed.")
    
    # Load meshes
    print("\nLoading meshes...")
    hub = load_mesh(str(hub_path))
    propeller = load_mesh(str(propeller_path))
    
    print(f"  Hub: {hub.node_count:,} nodes, {hub.elements_count:,} elements")
    print(f"  Propeller: {propeller.node_count:,} nodes, {propeller.elements_count:,} elements")
    
    # Check closed
    print(f"  Hub closed: {not detect_open_boundaries(hub)}")
    print(f"  Propeller closed: {not detect_open_boundaries(propeller)}")
    
    # Boolean union
    print("\nPerforming boolean union...")
    print("  (This will take a while - propeller has 144K elements)")
    try:
        union_mesh = boolean_union_meshes([hub, propeller])
        print(f"\n  ✓ Union completed!")
        print(f"    Input total:  {hub.elements_count + propeller.elements_count:,} elements")
        print(f"    Output:       {union_mesh.elements_count:,} elements")
        print(f"    Reduction:    {hub.elements_count + propeller.elements_count - union_mesh.elements_count:,} elements removed")
        
        # Save
        output_path = output_dir / "hub_propeller_union.vtu"
        union_mesh.write_mesh(str(output_path))
        print(f"\n  ✓ Saved to: {output_path}")
        print("\n  Open in Paraview and use a 'Slice' filter to verify")
        print("  that internal faces were removed where the blades")
        print("  intersect with the hub.")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
