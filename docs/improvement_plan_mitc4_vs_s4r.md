# Plan: Correcciones a la formulación no lineal del MITC4+

## Context

`crates/aeroelast-core/src/elements/mitc4.rs` has a solid linear path (`compute_ke_local`) and a broken nonlinear path (`compute_fint_global` when `nonlinear = true` and `compute_kt_global`). The defects fall into two groups: **wrong results** (D1–D3, critical) and **K/f inconsistency** that blocks quadratic Newton convergence (D4–D5). D6–D7 are isolated utility bugs not yet wired to any production path.

---

## Dependency order

```
D1 ─┐
D2  ├── implement together in a single nonlinear f_int rewrite
D3 ─┘
 │
 └─► D4 (K_L: cm_raw·h → cm) — depends on D2/D3 being understood
      └─► D5 (K_sigma multi-GP) — independent of D2/D3, but place after D4

D6 (quaternion skew-sym) — independent utility fix
D7 (log strain factor)   — independent utility fix
```

---

## D1 + D2 + D3 — Rewrite the nonlinear Gauss loop in `compute_fint_global`

**File:** `mitc4.rs:1339–1416`

These three defects share the same per-Gauss-point block and must be fixed atomically. The current loop:
- computes membrane strain from standard Green-Lagrange (`h_mat → e_gl`) rather than from the MITC4+ assumed-strain operator **(D2)**
- zeroes the bubble curvature DOFs **(D3)**
- omits the transverse shear contribution `bs^T · Q` entirely **(D1)**

### Pre-loop: compute condensed bubble displacement operator

The bubble static condensation used in `compute_ke_local` (line 1026–1031) yields `u_b = -K_bb^{-1} · K_bn · u`. We need the same operator inside `f_int`.

Build it once before the Gauss loop (reusing the same assembly as `compute_bending_shear_condensed` at line 1426, which already computes `knn`, `knb`, `kbb`):

```rust
// Pre-loop (before `for g in 0..N_GAUSS`)
let mut knb: SMatrix<f64, 24, 2> = SMatrix::zeros();
let mut kbb = Matrix2::zeros();
for g in 0..N_GAUSS {
    let gj = &pre.gp_jacobians[g];
    let gb = &pre.gp_bubble[g];
    let factor_pre = GAUSS_W[g] * gj.sqrt_g;
    let bk  = b_kappa(&gj.dh);
    let bkb = b_kappa_bubble(&gj.j_inv, gb.dnb_dxi, gb.dnb_deta);
    let (bs_nodal, bs_bubble) = b_gamma_mitc4_plus(
        &pre.local_coords, GAUSS_XI[g], GAUSS_ETA[g], gj.sqrt_g, gb.nb,
    );
    knb += (bk.transpose() * cb * &bkb + bs_nodal.transpose() * cs * &bs_bubble) * factor_pre;
    kbb += (bkb.transpose() * cb * &bkb + bs_bubble.transpose() * cs * &bs_bubble) * factor_pre;
}
let kbb_inv = regularized_inverse_2x2(&kbb);
let bubble_op: SMatrix<f64, 2, 24> = -(kbb_inv * knb.transpose()); // u_b = bubble_op · u_local
```

### Per-Gauss-point changes

Replace lines 1362–1404 with:

```rust
// D2: membrane strain from MITC4+ assumed strain + nonlinear GL correction
let h_mat = displacement_gradient(&gj.dh, &u_local);  // still needed for bm_nl
let bm_l = b_m_mitc4_plus(pre, xi, eta);
let eps_m_linear = bm_l * &u_local;
let eps_m_nl = Vector3::new(
    0.5 * (h_mat[(0,0)].powi(2) + h_mat[(1,0)].powi(2) + h_mat[(2,0)].powi(2)),
    0.5 * (h_mat[(0,1)].powi(2) + h_mat[(1,1)].powi(2) + h_mat[(2,1)].powi(2)),
    h_mat[(0,0)]*h_mat[(0,1)] + h_mat[(1,0)]*h_mat[(1,1)] + h_mat[(2,0)]*h_mat[(2,1)],
);
let eps_m = eps_m_linear + eps_m_nl;

// D2 continued: B_NL for virtual work
let bnl = compute_b_nl(&gj.dh, &h_mat);
let bm_nl = extract_membrane_rows(&bnl);
let bm_total = bm_l + bm_nl;

// D3: condensed bubble DOFs
let u_b = bubble_op * &u_local;

// D3: curvature with bubble
let bk = b_kappa(&gj.dh);
let bkb = b_kappa_bubble(&gj.j_inv, gb.dnb_dxi, gb.dnb_deta);
let mut u_rot = Vec24::zeros();
for i in 0..4 {
    u_rot[6*i+3] = u_local[6*i+3];
    u_rot[6*i+4] = u_local[6*i+4];
}
let kappa = bk * &u_rot + bkb * u_b;

// Resultants with ABD coupling
let n_resultant = cm * &eps_m + cb_coupling * &kappa;
let m_resultant = cb_coupling * &eps_m + cb * &kappa;

// D1: transverse shear
let (bs_nodal, bs_bubble) = b_gamma_mitc4_plus(
    &pre.local_coords, xi, eta, sqrt_g, gb.nb,
);
let gamma = bs_nodal * &u_local + bs_bubble * u_b;
let q_resultant = cs * gamma;

// Accumulate virtual work
let factor = w * sqrt_g;
f += bm_total.transpose() * &n_resultant * factor;
f += bk.transpose() * &m_resultant * factor;
f += bs_nodal.transpose() * &q_resultant * factor;  // D1
```

---

## D4 — Fix `K_L` constitutive matrix in `compute_kt_global`

**File:** `mitc4.rs:1265–1269`

Current:
```rust
k_l += (
    bm_l.transpose() * cm_raw * &bm_nl
    + bm_nl.transpose() * cm_raw * &bm_l
    + bm_nl.transpose() * cm_raw * &bm_nl
) * (w * sqrt_g * pre.thickness);
```

`cm_raw` is the stress-strain matrix (no thickness); `cm` already integrates through thickness. For laminates these differ. The linearization of the corrected `f_int` (which uses `cm`) gives:

```rust
k_l += (
    bm_l.transpose() * cm * &bm_nl
    + bm_nl.transpose() * cm * &bm_l
    + bm_nl.transpose() * cm * &bm_nl
) * (w * sqrt_g);  // pre.thickness dropped — cm already includes it
```

Also: update `compute_membrane_stress` (called for `K_sigma`) to use `cm` instead of `cm_raw`:
```rust
fn compute_membrane_stress(pre: &Mitc4Precomputed, u_local: &Vec24) -> Vector3<f64> {
    let bm = b_m_mitc4_plus(pre, 0.0, 0.0);
    let eps_m = bm * u_local;
    &pre.constitutive.cm * eps_m  // was cm_raw
}
```

---

## D5 — Integrate `K_sigma` over 4 Gauss points

**File:** `mitc4.rs:1272–1276`

Currently: `compute_membrane_stress` evaluates stress at element center `(0,0)` only.

Replace the two-line call with a loop over Gauss points that averages (or accumulates with weights) the geometric stiffness:

```rust
// Replace lines 1272-1276
let k_sigma = {
    let mut ks = Mat24::zeros();
    for g in 0..N_GAUSS {
        let xi = GAUSS_XI[g];
        let eta = GAUSS_ETA[g];
        let w = GAUSS_W[g];
        let sqrt_g = pre.gp_jacobians[g].sqrt_g;
        // sigma at this Gauss point
        let bm = b_m_mitc4_plus(pre, xi, eta);
        let eps_m = bm * &u_local;
        let sigma_g = &pre.constitutive.cm * eps_m;
        // geometric stiffness contribution
        ks += compute_geometric_stiffness_contribution(pre, g, &sigma_g);
    }
    ks
};
```

Extract the per-point computation from `compute_geometric_stiffness_local` into a helper `compute_geometric_stiffness_contribution(pre, g, sigma)` that builds the `S_tilde` and accumulates `bg^T S_tilde bg * (w * sqrt_g)` for a single Gauss point.

---

## D6 — Fix skew-symmetric matrix in `quaternion_to_matrix`

**File:** `mitc4.rs:2061–2065`

Current (`q_hat` is wrong — applies skew of `[qz, -qy, qx]`):
```rust
let q_hat = Matrix3::new(
     0.0, -qx, -qy,
    qx,   0.0, -qz,
    qy,  qz,   0.0,
);
```

Correct skew-symmetric of `[qx, qy, qz]`:
```rust
let q_hat = Matrix3::new(
     0.0, -qz,  qy,
     qz,  0.0, -qx,
    -qy,  qx,   0.0,
);
```

---

## D7 — Fix log strain approximation in `log_strain_from_polar`

**File:** `mitc4.rs:2185`

For small strains, `ln(U) ≈ U - I` (not `½·(U - I)`). Change:
```rust
0.5 * u_minus_i   // wrong — no standard measure
```
to:
```rust
u_minus_i         // correct small-strain approximation of ln(U)
```

---

## Verification tests (per defect)

Each test lives in `#[cfg(test)]` at the bottom of `mitc4.rs`. All tests use a standard 1×1 m square plate, thickness 0.01, isotropic E=210e9, ν=0.3, simply-supported or cantilever BCs.

| Fix | Test name | What it checks |
|-----|-----------|----------------|
| D1 | `test_fint_nonlinear_pure_shear` | cantilever: transverse tip load, `f_int(nonlinear) ≈ K·u` for small `|u|` |
| D2+D3 | `test_fint_linear_nonlinear_small_disp_parity` | `‖f_int(nonlinear, u=ε·û) - K·u‖ = O(ε²)` |
| D4 | `test_kt_fint_directional_derivative` | `‖K_T·δu - (f_int(u+δu)-f_int(u))‖/‖δu‖² → 0` as `|δu|→0` |
| D5 | `test_buckling_load_convergence` | biaxial compression: critical load converges monotonically with mesh refinement (2×2, 4×4, 8×8) |
| D6 | `test_quaternion_to_matrix_90deg_z` | `quaternion_to_matrix(from_vector([0,0,π/2]))` = rotation-90-Z exactly |
| D7 | `test_log_strain_small_deformation` | `log_strain_from_polar(I + ε·H) ≈ ε·sym(H)` within O(ε²) for ε=1e-6 |

---

## Critical files

| File | Sections to change |
|------|--------------------|
| `crates/aeroelast-core/src/elements/mitc4.rs` | L1339–1416 (f_int nonlinear loop), L1251–1282 (K_T), L1285–1322 (K_sigma), L2061–2065 (quat), L2185 (log strain) |
| `crates/aeroelast-core/src/materials/mod.rs` | Read-only — `ShellConstitutive` for reference |

## Existing functions to reuse (do not re-implement)

- `b_m_mitc4_plus(pre, xi, eta)` — L694: MITC4+ assumed membrane strain B
- `b_gamma_mitc4_plus(local_coords, xi, eta, sqrt_g, nb)` — L896: shear B with bubble
- `b_kappa(dh)` — L758: bending curvature B (nodal)
- `b_kappa_bubble(j_inv, dnb_dxi, dnb_deta)` — L775: bubble curvature B
- `compute_b_nl(dh, h)` — L1182: nonlinear B_NL
- `extract_membrane_rows(b6)` — L1226: extract rows 0,1,3 from 6×24
- `displacement_gradient(dh, u)` — L1139: H = du/dX
- `regularized_inverse_2x2(kbb)` — used at L1030, L1466: safe 2×2 inverse
- `compute_geometric_stiffness_local(pre, sigma)` — L1295: refactor into per-point helper for D5