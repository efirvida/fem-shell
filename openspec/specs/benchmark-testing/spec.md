# Delta for Benchmark Testing

## MODIFIED Requirements

### Requirement: test_beam_4cases_parity — Test Structure and Coverage

The test file `tests/benchmarks/test_beam_4cases_parity.py` MUST provide a 3-way parity check (AeroElast vs Analytical vs CCX) for static displacement and a 2-way parity check (AeroElast vs CCX) for modal frequencies. The modal comparison logic MUST NOT be duplicated more than once.

(Previously: test contained 4 verbatim copy-paste blocks of modal comparison code at lines 582–625)

#### Scenario: Static Parity — 4 Load Cases

- GIVEN a cantilever beam mesh with steel material (E=2.1e11 Pa, nu=0.3, rho=7850 kg/m³) of dimensions B=0.05m × H=0.05m × L=1.0m, clamped at z=0
- WHEN `test_linear_static_with_analytical` runs the 4 load cases (tension, compression, bending Fx, shear Fy) through both AeroElast (Rust backend) and CCX
- THEN the test MUST print a 3-way comparison (Analytical, AeroElast, CCX) for each load case
- AND report a pytest failure if the relative difference between AeroElast and CCX exceeds 5%

#### Scenario: Modal Parity — First 5 Frequencies (Single Comparison)

- GIVEN the same mesh setup with clamped boundary conditions
- WHEN `test_modal_first_five_modes` computes the first 5 eigenfrequencies via AeroElast (Rust backend) and CCX
- THEN the test MUST run the comparison exactly once
- AND the helper function `_compare_modal_frequencies(ae_freqs, ccx_freqs, tol)` MUST be defined at module level and called exactly once
- AND report a pytest failure if any relative difference exceeds 8%

#### Scenario: Modal Parity — CCX Mesh Writing

- GIVEN the modal test computes AeroElast frequencies and is about to run CCX
- THEN the `write_ccx_mesh` call (lines 552–560) MUST be present and unchanged
- AND it MUST produce a `beam_modal.inp` file in the modal working directory
- AND CCX MUST be invoked via `_run_ccx` producing `beam_modal.dat`

## ADDED Requirements

### Requirement: _compare_modal_frequencies Helper

The module-level function `_compare_modal_frequencies(ae_freqs, ccx_freqs, tol)` MUST be defined near `_analytical_solution` and MUST:

1. Print a summary showing AeroElast frequencies, CCX frequencies, and relative errors
2. Compute relative errors as `|ccx - ae| / max(|ccx|, 1e-14)`
3. Call `pytest.fail` with formatted error message if any error exceeds `tol`
4. Use `tol=0.08` as the default tolerance

### Requirement: Redundant Modal Blocks Removed

The file MUST NOT contain more than one block of modal frequency comparison code. Lines 582–625 (three duplicate blocks) MUST be removed. The single correct block at lines 567–580 MUST remain and be replaced by a call to `_compare_modal_frequencies`.

## REMOVED Requirements

### Requirement: Duplicate Modal Comparison Blocks

(Reason: verbatim copy-paste duplication; replaced by a single helper function call)

The three duplicate modal comparison blocks previously at lines 582–625 MUST NOT exist in the file.

## Requirements Summary Table

| Requirement | Strength | Scenarios |
|-------------|----------|-----------|
| test_beam_4cases_parity — Test Structure | MUST | 3 |
| _compare_modal_frequencies Helper | MUST | 1 |
| Redundant Modal Blocks Removed | MUST NOT | 0 |