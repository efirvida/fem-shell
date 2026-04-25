# Proposal: beam-test-cleanup

## Intent

Clean up `tests/benchmarks/test_beam_4cases_parity.py` by removing three verbatim copy-paste blocks of modal comparison code (lines 582–625) and restructuring the 3-way comparison (Analytical vs AeroElast vs CCX) to eliminate duplication and improve readability. The existing, correct usage of `write_ccx_mesh` (lines 38, 488–497, 552–560) is preserved verbatim.

## Scope

### In Scope
- Remove 3x identical copy-paste blocks from `test_modal_first_five_modes` (lines 582–625), keeping only the single correct block (lines 567–580)
- Add a helper function `_compare_modal_frequencies(ae_freqs, ccx_freqs, tol)` to encapsulate the comparison logic
- Ensure `test_modal_first_five_modes` calls this helper once after computing both AeroElast and CCX frequencies
- Preserve all `write_ccx_mesh` calls exactly as they are

### Out of Scope
- Any changes to the static displacement comparison in `test_static_beam_cases`
- Any changes to mesh building, CCX invocation, or the analytical solution logic
- Changes to other test files

## Capabilities

### New Capabilities
- `_compare_modal_frequencies` helper: extracted comparison logic (print summary, relative error computation, tolerance check, pytest.fail on failure)

### Modified Capabilities
- `test_modal_first_five_modes`: streamlined to a single comparison call instead of four redundant blocks

## Approach

1. Identify the **single correct block** (lines 567–580) in `test_modal_first_five_modes`.
2. Delete the three duplicate copy-paste blocks (the two pairs at 582–595 and 597–610, and the partial pair at 612–625).
3. Extract the remaining block into a module-level helper `_compare_modal_frequencies(ae_freqs, ccx_freqs, tol=0.08)` in the static comparison area (near `_analytical_solution`).
4. Replace the comparison block with a single call: `_compare_modal_frequencies(freqs_ae, freqs_ccx)`.
5. Confirm no logic changes — only structural cleanup.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `tests/benchmarks/test_beam_4cases_parity.py` | Modified | Lines 582–625 removed; helper added |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Deleting the wrong block changes test semantics | Low | Keep the block at 567–580; delete only the 3 copies after it |
| Helper placement causes import/scope issues | Low | Define at module level alongside `_analytical_solution` |

## Rollback Plan

Revert to the original file via `git checkout tests/benchmarks/test_beam_4cases_parity.py`. All `write_ccx_mesh` calls are untouched by this change, so rollback restores full original behavior.

## Dependencies

- None — pure refactor, no new dependencies

## Success Criteria

- [ ] Lines 582–625 removed; file ends at line 580 (or ~581 with helper)
- [ ] `write_ccx_mesh` calls at lines 38, 488–497, 552–560 unchanged
- [ ] `_compare_modal_frequencies` helper defined and called exactly once
- [ ] Test still passes (same parity assertions, same tolerance 0.08)