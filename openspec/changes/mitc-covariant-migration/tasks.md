# Tasks: MITC4 3D Covariant Migration

## Phase 1: Foundation / Infrastructure
- [x] 1.1 Modify `GpJacobian` in `mitc4.rs`: replace `j_mat` (2D) with `metric_tensor` (2x2 covariant $g_{\alpha\beta}$) and add `sqrt_g`.
- [x] 1.2 Update `Mitc4Precomputed` struct: remove `local_coords`.
- [x] 1.3 Implement `compute_covariant_metric(coords_3d, xi, eta)`: compute $\mathbf{x}_{,\alpha}$, $g_{\alpha\beta}$, and $\sqrt{g}$.
- [x] 1.4 Update `Mitc4Precomputed::new` to call `compute_covariant_metric` for each GP and populate `gp_jacobians`.
- [x] 1.5 Remove `compute_jacobian` and `compute_dh` (2D projection) as they are no longer used.
- [x] 1.6 Verification: Ensure `Mitc4Precomputed::new` compiles and `sqrt_g` is positive for unit square.

## Phase 2: Core Implementation (Membrane)
- [x] 2.1 Refactor `compute_covariant_membrane_b_row`: use 3D tangents $\mathbf{x}_{,\alpha}$ directly.
- [x] 2.2 Implement `covariant_to_local_mapping(metric_tensor, e1, e2)`: create transformation $\mathbf{J}_{loc}$ for orthonormal frame $\{\mathbf{e}_1, \mathbf{e}_2\}$.
- [x] 2.3 Update `b_m_mitc4_plus`: replace `covariant_to_cartesian_transform` with point-wise `covariant_to_local_mapping`.
- [x] 2.4 Update `compute_ke_local` integration loop: replace `det_j` with `sqrt_g`.
- [ ] 2.5 Verification: Compare `compute_ke_local` with existing implementation for a flat plate (should be identical).

## Phase 3: Integration & Global Assembly
- [x] 3.1 Review `compute_kt_global` and `compute_fint_global`: remove any legacy dependencies on 2D projected Jacobians.
- [x] 3.2 Verify `transform_to_global` and `build_t24` consistency with updated `Mitc4Precomputed`.
- [x] 3.3 Verification: Run existing unit tests in `mitc4.rs` to ensure no regressions.

## Phase 4: Testing & Validation
- [ ] 4.1 Implement "Twisted Beam" benchmark in `tests/benchmarks/`: verify relative error $\le 5\%$ vs Ko et al. (2017).
- [ ] 4.2 Implement "Pinched Cylinder" benchmark in `tests/benchmarks/`: verify relative error $\le 5\%$ vs Ko et al. (2017).
- [ ] 4.3 Verify "Cantilever Plate Parity": relative error $\le 1\%$ vs CalculiX.
- [ ] 4.4 Perform "Locking Test": verify displacement stability as $t/L \to 10^{-4}$.
- [ ] 4.5 Verification: All VAL-XX criteria from spec are met.
