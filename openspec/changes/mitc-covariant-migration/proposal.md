# Proposal: MITC4 Shell Migration to 3D Covariant Formulation

## Intent

The current MITC4 implementation uses a "Flat-Shell" approach, projecting the 3D element geometry onto a 2D local plane. This projection introduces parasitic shear and membrane locking in warped or highly curved geometries (e.g., Twisted Beam, Pinched Cylinder), causing discrepancies compared to CalculiX and the Ko et al. (2017) benchmarks. This change migrates the kernel to a full 3D covariant formulation to eliminate projection errors.

## Scope

### In Scope
- **Kernel Refactoring**: Remove dependency on `compute_local_coordinate_system` and 2D `local_coords` for stiffness matrix calculation.
- **Covariant B-Matrix**: Redefine B-matrix construction using 3D characteristic vectors ($x_r, x_s, x_d$) and the covariant metric tensor $g_{\alpha\beta}$.
- **Integration Update**: Use the 3D surface area element $dA = \sqrt{g} \, d\xi d\eta$ for numerical integration.
- **Membrane Blending**: Ensure MITC4+ blending is applied to covariant strains before transforming to the local orthonormal frame.
- **Validation**:
    - Implement "Twisted Beam" and "Pinched Cylinder" benchmarks in `tests/benchmarks/test_ko2017_performance.py`.
    - Resolve parity gaps in `tests/benchmarks/test_ccx_cantilever_plate_parity.py`.

### Out of Scope
- Migration of MITC3+ elements.
- Full non-linear covariant formulation (focus is on linear/tangent kernel alignment).

## Capabilities

### New Capabilities
- None

### Modified Capabilities
- `mitc4-element`: Migration from projected 2D "Flat-Shell" to full 3D covariant formulation.

## Approach

1. **Metric Tensor**: Replace 2D Jacobian with the covariant metric tensor $g_{\alpha\beta} = \mathbf{x}_{,\alpha} \cdot \mathbf{x}_{,\beta}$ where $\alpha, \beta \in \{r, s\}$.
2. **Integration**: Replace `det_j` with $\sqrt{g} = \sqrt{g_{rr}g_{ss} - g_{rs}^2}$.
3. **Covariant Strains**: Construct the membrane B-matrix rows directly using 3D tangents $\mathbf{g}_r, \mathbf{g}_s$.
4. **Local Frame**: Instead of a fixed element-wide projection plane, compute a point-wise local orthonormal basis $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ at each Gauss point to transform covariant strains for constitutive integration.
5. **Blending**: Apply the MITC4+ blending coefficients $a_a \dots a_e$ to the covariant strain components $\epsilon_{rr}, \epsilon_{ss}, \epsilon_{rs}$.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `crates/aeroelast-core/src/elements/mitc4.rs` | Modified | Complete refactor of `Mitc4Precomputed`, `compute_ke_local`, and B-matrix logic. |
| `tests/benchmarks/test_ko2017_performance.py` | New | Addition of Twisted Beam and Pinched Cylinder test cases. |
| `tests/benchmarks/test_ccx_cantilever_plate_parity.py` | Modified | Updates to parity checks for better alignment with CalculiX. |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Performance Regression | Medium | Use `SMatrix` and avoid redundant 3D $\to$ 2D projections in the hot loop. |
| Numerical Singularity | Low | Implement checks for $\sqrt{g} \approx 0$ and add small epsilons during metric inversion. |
| Regression in Flat Plates | Low | Verify against existing flat plate benchmarks to ensure backward compatibility. |

## Rollback Plan

Revert changes to `crates/aeroelast-core/src/elements/mitc4.rs` using git. The previous "Flat-Shell" implementation is stable for simple geometries.

## Dependencies

- Exploration results at `sdd/mitc-theory-alignment/explore`.
- Ko, Lee & Bathe (2017) theoretical references in `elements_theory/`.

## Success Criteria

- [ ] Removal of `compute_local_coordinate_system` from the stiffness kernel.
- [ ] Successful convergence and accuracy in the "Twisted Beam" benchmark (locking removed).
- [ ] Successful convergence and accuracy in the "Pinched Cylinder" benchmark.
- [ ] Parity with CalculiX results for the cantilever plate benchmark improved to $<1\%$ error.
