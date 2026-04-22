# AeroElast ↔ CalculiX Solver Parity Findings

Last updated: 2026-04-21

## Scope

This document consolidates current parity evidence and the next implementation priorities for AeroElast solver parity against CalculiX (CCX), focusing on:

- 3D solids (core solver sanity)
- Shell cantilever plate (current mismatch area)

## Evidence Summary

### 1) 3D solid cantilever bar parity (C3D8 / Hexa8)

Targeted command:

```bash
CCX_BIN=/scratch/leahk/eduardo.donestevez/venv/bin/ccx \
  /scratch/leahk/eduardo.donestevez/venv/bin/python -m pytest -q \
  tests/benchmarks/test_ccx_square_bar_solid_parity.py
```

Result: `2 passed`

Detailed numeric evidence:

| Case | Location | AE | CCX | Relative error |
|---|---|---:|---:|---:|
| free_face_fx | center | 1.875834e-04 | 1.875830e-04 | 0.000239% |
| free_face_fx | tip | 1.908949e-04 | 1.908950e-04 | 0.000042% |
| free_face_fz | center | 1.173112e-02 | 1.173110e-02 | 0.000155% |
| free_face_fz | tip | 1.173968e-02 | 1.173110e-02 | 0.073099% |

Modal (first 5):

- AE: [67.53723227, 67.53723227, 423.62120326, 423.62120326, 805.30812153]
- CCX: [67.53723, 67.53723, 423.6212, 423.6212, 805.3081]
- Relative error per mode: ~7.7e-09 to 3.36e-08

**Interpretation:** core assembly/solve path is in excellent agreement for 3D solids.

---

### 2) Shell cantilever plate parity (FY/FZ focused)

Targeted linear command:

```bash
CCX_BIN=/scratch/leahk/eduardo.donestevez/venv/bin/ccx \
  /scratch/leahk/eduardo.donestevez/venv/bin/python -m pytest -q \
  tests/benchmarks/test_ccx_cantilever_plate_parity.py::TestCantileverPlateCCXParity::test_linear_static_center_free_edge_metric_fy_fz
```

Targeted nonlinear command:

```bash
CCX_BIN=/scratch/leahk/eduardo.donestevez/venv/bin/ccx \
  /scratch/leahk/eduardo.donestevez/venv/bin/python -m pytest -q \
  tests/benchmarks/test_ccx_cantilever_plate_parity.py::TestCantileverPlateCCXParity::test_nonlinear_static_center_free_edge_metric_fy_fz
```

Status: both tests fail (diagnostic failures expected/useful).

Representative mismatches from latest run:

| Regime | Material | Case | q-order | Key mismatch |
|---|---|---|---|---|
| Linear | isotropic | free_edge_fy | q1/q2 | ~37–38% on \|u\| and Uy component |
| Linear | isotropic | free_edge_fz | q1 | ~5.9% on Uz |
| Linear | composite | free_edge_fy | q1/q2 | ~37–38% on \|u\| and Uy component |
| Linear | composite | free_edge_fz | q1 | ~36.9% on Uz |
| Nonlinear | isotropic | free_edge_fy | q1/q2 | ~37–38% on \|u\| and Uy component |
| Nonlinear | isotropic | free_edge_fz | q1 | CCX runtime failed (code 201) |
| Nonlinear | composite | free_edge_fy | q1/q2 | ~37–38% on \|u\| and Uy component |
| Nonlinear | composite | free_edge_fz | q1 | CCX runtime failed (code 201) |

Load/resultant consistency checks added in-test:

- AE nodal distributed resultant = intended load6
- CCX writer resultant via *CLOAD distribution = intended load6
- Relative mismatch: effectively zero within numerical tolerance (`<= 1e-12`), so major discrepancy is not from gross resultant scaling.

**Interpretation:** shell mismatch is persistent and directional (strong FY/Uy bias), while loads are represented equivalently at resultant level.

## Diagnosis

### Solver core vs shell formulation/export

- **Core solver likely OK**: 3D solid static and modal parity is very tight.
- **Shell path likely culprit**: discrepancies are concentrated in shell FY/FZ behavior and some modal mapping outliers (from prior runs).
- **Likely root areas**:
  1. shell formulation assumptions (kinematics/shear treatment/local basis)
  2. shell section/orientation parity details between AE and CCX export
  3. load application convention subtleties beyond global resultant (e.g., local axes/sign conventions, through-thickness interpretation)

## Implemented Improvements (this iteration)

1. **Configurable nonlinear CCX step controls in writer**
   - Added parameters to `write_ccx_mesh(...)`:
     - `nl_initial_increment`
     - `nl_min_increment`
     - `nl_max_increment`
     - `nl_max_increments`
   - Backward-compatible defaults preserved:
     - initial=1.0, min=1e-5, max=1.0, max increments=100
   - Exported into CCX cards as:
     - `*STEP, NLGEOM, INC=...`
     - `*STATIC` line with `(initial, 1.0, min, max)`

2. **Threaded controls through runner config path**
   - Added optional solver config fields in `SolverConfig` and YAML parser mapping.
   - Threaded from `FSIRunner.export_calculix(...)` to writer.

3. **Plate diagnostics expanded**
   - Added resultant consistency check (AE vs CCX representation) before solve.
   - Added per-component center free-edge displacement comparisons (Ux, Uy, Uz), not only norm.
   - Added targeted FY/FZ-only parity tests (linear and nonlinear) for faster diagnosis cycles.

## Prioritized Implementation Roadmap

### Priority 0 — keep diagnostics actionable (done, continue using)

- **Files**:
  - `tests/benchmarks/test_ccx_cantilever_plate_parity.py`
  - `src/aeroelast/core/mesh/io/writers.py`
- **Functions**:
  - `_ccx_static_center_disp`
  - `test_linear_static_center_free_edge_metric_fy_fz`
  - `test_nonlinear_static_center_free_edge_metric_fy_fz`
  - `_write_ccx_nonlinear_static_step`

### Priority 1 — isolate shell local-axis / orientation effects per load direction

- Run FY/FZ with controlled orientation variants (span direction and reference axis alternatives) and compare component-wise center displacement and selected mode MAC.
- **Likely touchpoints**:
  - `src/aeroelast/core/mesh/io/writers.py`
    - `_build_angle_bucket_sets`
    - `_write_ccx_orientations`
    - `_write_ccx_sections`

### Priority 2 — robust nonlinear shell parity harness for FZ

- Add per-case nonlinear load scaling sweep (e.g., 0.1x/0.25x/0.5x/1.0x) to determine stable comparability envelope and separate divergence from model mismatch.
- Keep CCX runtime-failure capture as structured diagnostics (do not abort matrix evaluation).
- **Likely touchpoints**:
  - `tests/benchmarks/test_ccx_cantilever_plate_parity.py`
    - nonlinear FY/FZ targeted test methods

### Priority 3 — richer shell equivalence checks (beyond resultant)

- Add checks for distributed edge-load reconstruction (per-node load vectors and local/global direction audit) to detect convention mismatches not visible in global resultant.
- **Likely touchpoints**:
  - `tests/benchmarks/test_ccx_cantilever_plate_parity.py`
  - `src/aeroelast/core/mesh/io/writers.py` (`_write_ccx_cload` usage contexts)

### Priority 4 — modal shell diagnostics hardening

- Extend existing MAC diagnostics with mode family tags and constrained-DOF handling notes to avoid false modal pair assumptions when frequencies are close.
- **Likely touchpoints**:
  - `tests/benchmarks/test_ccx_cantilever_plate_parity.py`
    - `_mac_matrix`
    - `_best_mac_mode_mapping`
