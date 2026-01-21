# Solid Beam Analysis Example

This example demonstrates the behavior of 3D solid finite elements under different loading conditions applied to a cantilever beam.

## Loading Cases

1. **Tension**: Axial load pulling the beam along its length
2. **Compression**: Axial load pushing the beam
3. **Bending (Z)**: Transverse load causing bending about the Y-axis
4. **Bending (Y)**: Transverse load causing bending about the Z-axis
5. **Torsion**: Moment about the beam axis causing twist

## Element Types Compared

| Mesh Type | Element | Description |
|-----------|---------|-------------|
| `hex` | HEXA8 (8-node) | Structured hexahedral mesh, linear elements |
| `tet` | TETRA4 (4-node) | Structured tetrahedral mesh, linear elements |
| `unstructured` | TETRA4 | Unstructured tetrahedral mesh (Gmsh free meshing) |

### Additional Types (disabled by default, may require testing)

| Mesh Type | Element | Description |
|-----------|---------|-------------|
| `hex_q` | HEXA20 (20-node) | Quadratic hexahedron |
| `tet_q` | TETRA10 (10-node) | Quadratic tetrahedron |
| `wedge` | WEDGE6 (6-node) | Linear wedge/prism |
| `mixed` | Multiple | Mixed element mesh with transitions |

Hexahedral elements generally provide better accuracy per node, while tetrahedral elements offer more flexibility for complex geometries. Unstructured meshes are generated with random node placement, simulating real-world mesh generation.

## Running the Example

```bash
cd examples/solid_beam_analysis
python beam_loading_cases.py
```

## Output

The script generates:

1. **Console output**: Summary table with displacement results and comparison with analytical solutions
2. **VTK files**: Deformed meshes for visualization in ParaView (`output/beam_*.vtk`)
3. **Comparison plot**: Bar charts comparing element types (`output/beam_comparison.png`)

## Geometry

- Length (L): 10.0 m
- Width (W): 1.0 m  
- Height (H): 1.0 m

## Material Properties (Steel)

- Young's modulus (E): 210 GPa
- Poisson's ratio (ν): 0.3
- Density (ρ): 7850 kg/m³

## Analytical Solutions for Validation

For a cantilever beam with tip load P:

- **Tension/Compression**: δ = PL/(AE)
- **Bending**: δ = PL³/(3EI)
- **Torsion**: θ = TL/(GJ)

Where:
- A = cross-section area
- I = moment of inertia
- J = polar moment of inertia
- G = shear modulus = E/(2(1+ν))

## Visualization with ParaView

1. Open ParaView
2. File → Open → Select `output/beam_hex_bending_z.vtk`
3. Click "Apply"
4. Color by "displacement_magnitude"
5. Use "Warp by Vector" filter with "displacement" to see deformation

## Expected Results

| Load Case | Dominant Displacement |
|-----------|----------------------|
| Tension | Positive X (axial extension) |
| Compression | Negative X (axial contraction) |
| Bending Z | Positive Z (transverse) |
| Bending Y | Positive Y (transverse) |
| Torsion | Rotation about X-axis |

## Notes

- Hexahedral elements generally give more accurate results with fewer DOFs
- Tetrahedral elements are more flexible for complex geometries
- Linear elements may show shear locking in bending; use more elements or quadratic elements
- The mesh density can be adjusted via `NX`, `NY`, `NZ` parameters
