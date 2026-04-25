# Tasks: beam-test-cleanup

## Phase 1: Analysis & Planning

- [ ] 1.1 Verify the single correct modal comparison block is at lines 567–580 in `tests/benchmarks/test_beam_4cases_parity.py`
- [ ] 1.2 Confirm all three duplicate blocks to remove: 582–595, 597–610, 612–625
- [ ] 1.3 Identify placement location for `_compare_modal_frequencies` helper (near `_analytical_solution` at line 321)

## Phase 2: Core Implementation

- [ ] 2.1 Define `_compare_modal_frequencies(ae_freqs, ccx_freqs, tol=0.08)` as module-level helper after `_analytical_solution` (after line 335), extracting: print summary, relative error computation, tolerance check, pytest.fail on failure
- [ ] 2.2 In `test_modal_first_five_modes`, replace the first block (lines 567–580) with a call to `_compare_modal_frequencies(freqs_ae, freqs_ccx)`
- [ ] 2.3 Delete duplicate block pair at 582–595
- [ ] 2.4 Delete duplicate block pair at 597–610
- [ ] 2.5 Delete duplicate block pair at 612–625

## Phase 3: Verification

- [ ] 3.1 Verify `test_linear_static_with_analytical` (lines 436–518) is untouched — the 3-way parity print (Analytical/AeroElast/CCX) remains intact
- [ ] 3.2 Verify all `write_ccx_mesh` calls unchanged: line 38, lines 488–497, lines 552–560
- [ ] 3.3 Run `pytest tests/benchmarks/test_beam_4cases_parity.py::TestBeamCantilever::test_modal_first_five_modes -v` to confirm test passes with same tolerance (0.08)
- [ ] 3.4 Run `pytest tests/benchmarks/test_beam_4cases_parity.py::TestBeamCantilever::test_linear_static_with_analytical -v` to confirm static test still passes
- [ ] 3.5 Confirm file length reduced from 625 to ~582 lines

## Phase 4: Cleanup

- [ ] 4.1 Final review: no duplicate comparison logic remains, no `write_ccx_mesh` calls modified