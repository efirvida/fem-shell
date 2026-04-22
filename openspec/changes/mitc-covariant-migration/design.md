# Design: MITC4 3D Covariant Migration

## Technical Approach

The goal is to migrate the MITC4 shell element from a projected "Flat-Shell" formulation to a full 3D Covariant formulation. This eliminates parasitic shear and membrane locking in warped/curved geometries by performing all membrane calculations in the covariant basis of the surface and mapping them to a point-wise local orthonormal frame for constitutive application.

The implementation will:
1. Replace the 2D projected Jacobian and coordinates with the 3D covariant metric tensor $g_{\alpha\beta} = \mathbf{x}_{,\alpha} \cdot \mathbf{x}_{,\beta}$ and the surface Jacobian $\sqrt{g}$.
2. Compute membrane strains in the covariant basis using MITC4+ blending.
3. Map these covariant strains to the local orthonormal frame $\{\mathbf{e}_1, \mathbf{e}_2\}$ using a point-wise transformation matrix derived from the projection of 3D tangents.
4. Integrate the resulting stiffness using the surface area element $\sqrt{g}$.

## Architecture Decisions

### Decision: Point-wise Local Frame Mapping
**Choice**: Compute a transformation matrix $\mathbf{T}_{cov \to loc}$ at each Gauss point based on the projection of the 3D tangents $\mathbf{g}_r, \mathbf{g}_s$ onto the element's orthonormal basis $\mathbf{e}_1, \mathbf{e}_2$.
**Alternatives considered**: Using a constant projection for the whole element.
**Rationale**: Required for accuracy in highly warped geometries where the natural basis deviates significantly from the orthonormal basis.

### Decision: Data Structure Update
**Choice**: Update `Mitc4Precomputed` and `GpJacobian` to store covariant metric components and the local projection matrix $\mathbf{J}_{loc}$.
**Alternatives considered**: Computing these on-the-fly in the integration loop.
**Rationale**: Precomputing the projection matrix and $\sqrt{g}$ at Gauss points minimizes redundant operations in the hot path, keeping assembly overhead within the 20% limit (NFR-01).

## Data Flow

```
3D Node Coords ──→ 3D Tangents (g_r, g_s) ──→ Covariant Metric (g_ab) ──→ sqrt(g)
                                       │
                                       └──→ Local Projection (J_loc) ──→ T_{cov -> loc}
                                                                            │
                                                                            ▼
Nodal DOFs ──→ Covariant B-matrices (MITC4+ Blending) ──→ Covariant Strains ──→ Local Strains ──→ Local Stress ──→ Ke
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `crates/aeroelast-core/src/elements/mitc4.rs` | Modify | Update `Mitc4Precomputed` and `GpJacobian` structures. |
| `crates/aeroelast-core/src/elements/mitc4.rs` | Modify | Implement `compute_covariant_to_local_transform` and update `compute_ke_local`. |
| `crates/aeroelast-core/src/elements/mitc4.rs` | Modify | Update `Mitc4Precomputed::new` to precompute 3D covariant data. |
| `crates/aeroelast-core/src/elements/mitc4.rs` | Modify | Update `b_m_mitc4_plus` to return purely covariant B-matrices. |

## Interfaces / Contracts

### Updated Data Structures

```rust
pub struct GpJacobian {
    /// Projection of 3D tangents onto [e1, e2]: [g_r.e1, g_s.e1; g_r.e2, g_s.e2]
    pub j_loc: Matrix2<f64>,
    /// Inverse of j_loc
    pub j_inv: Matrix2<f64>,
    /// Surface Jacobian sqrt(g) = sqrt(g_rr*g_ss - g_rs^2)
    pub sqrt_g: f64,
    /// Shape function derivatives in local orthonormal coords [2×4]
    pub dh: SMatrix<f64, 2, 4>,
}

pub struct Mitc4Precomputed {
    // ... other fields ...
    /// Removed: local_coords (projected 2D)
    pub initial_coords_3d: [[f64; 3]; 4],
    pub e1: Vector3<f64>,
    pub e2: Vector3<f64>,
    pub e3: Vector3<f64>,
    pub gp_jacobians: [GpJacobian; N_GAUSS],
    // ...
}
```

### Transformation Logic

The transformation from covariant strains $\boldsymbol{\epsilon}_{cov}$ to local orthonormal strains $\boldsymbol{\epsilon}_{loc}$ is:
$\boldsymbol{\epsilon}_{loc} = \mathbf{J}_{loc}^{-1} \boldsymbol{\epsilon}_{cov} \mathbf{J}_{loc}^{-T}$
where $\mathbf{J}_{loc}$ is the projection matrix stored in `GpJacobian`.

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Covariant $\to$ Local transform | Verify that a known covariant strain state maps correctly to orthonormal components. |
| Integration | "Twisted Beam" Benchmark | Compare displacement at tip against Ko et al. (2017) values. Error $\le 5\%$. |
| Integration | "Pinched Cylinder" Benchmark | Compare displacement field against Ko et al. (2017) values. Error $\le 5\%$. |
| Parity | Cantilever Plate | Run `test_ccx_cantilever_plate_parity.py` to ensure no regression vs CalculiX ($\le 1\%$). |
| Stability | Locking Test | Reduce thickness $t \to 10^{-4}L$ and verify convergence of displacement. |

## Migration / Rollout

No migration required. This is a replacement of the kernel logic within the existing MITC4 implementation.

## Open Questions

- [ ] Verify if bubble enrichment needs any adjustments for the covariant formulation (the current spec assumes standard bubble behavior).
