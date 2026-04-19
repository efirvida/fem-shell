# Proposal: Migrate Python `MeshAssembler` to Rust `PyMeshAssembler`

## Intent

`assembler.py` is the last significant Python glue layer: it calls Rust batch functions but orchestrates topology, load vectors, and geometric stiffness in Python. This creates a maintenance split and prevents future Rust-only usage. The goal is to move all assembly logic into `aeroelast-solvers` and expose it via a `PyMeshAssembler` PyO3 class, then delete `assembler.py`.

## Scope

### In Scope
- Rust `MeshAssembler` struct in `crates/aeroelast-solvers` with all public methods
- PyO3 `PyMeshAssembler` binding in `crates/aeroelast-py`
- Port `body_load` (load vectors) to Rust
- Port `compute_geometric_stiffness` / `compute_centrifugal_prestress` to Rust
- Port `ElementFactory` / `MeshModel` topology iteration to Rust
- Python shim in `assembler.py` delegating 1:1 to `PyMeshAssembler` (compatibility layer)
- Delete `assembler.py` once all tests pass against Rust impl

### Out of Scope
- Solver refactor (`solver.py`, `stress_stiffened_dynamic.py`)
- Postprocessing (`stress_recovery.py`) beyond import compatibility
- Non-assembler Python code

## Capabilities

### New Capabilities
- `rust-mesh-assembler`: Full Rust-side mesh assembly (stiffness, mass, load, geometric stiffness, tangent stiffness, internal forces, DOF count)

### Modified Capabilities
- None — Python API surface is unchanged; only implementation moves

## Approach

**Incremental replacement with a compatibility shim:**

1. **Phase 1 — Port missing Rust functions**: `body_load`, `compute_geometric_stiffness`, `compute_centrifugal_prestress` into `aeroelast-core` or `aeroelast-solvers`.
2. **Phase 2 — Topology in Rust**: Port `ElementFactory` / `MeshModel` element iteration into a `Mesh` struct in `aeroelast-core`.
3. **Phase 3 — `MeshAssembler` struct**: Implement all public methods in `aeroelast-solvers`, unit-tested in Rust.
4. **Phase 4 — PyO3 binding**: `PyMeshAssembler` in `aeroelast-py`, wrapping `MeshAssembler`.
5. **Phase 5 — Shim + validation**: `assembler.py` becomes a 1:1 delegate to `PyMeshAssembler`. All existing tests pass.
6. **Phase 6 — Delete**: Remove `assembler.py`; update imports in consumers.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `crates/aeroelast-core/` | Modified | New `Mesh`, `body_load`, geometric stiffness fns |
| `crates/aeroelast-solvers/` | New | `MeshAssembler` struct, all assembly methods |
| `crates/aeroelast-py/` | New | `PyMeshAssembler` PyO3 binding |
| `src/fem_shell/assembler.py` | Shim → Deleted | Delegate → remove |
| `src/fem_shell/solvers/solver.py` | Modified (Phase 6) | Import swap |
| `src/fem_shell/solvers/fsi/stress_stiffened_dynamic.py` | Modified (Phase 6) | Import swap |
| `src/fem_shell/postprocess/stress_recovery.py` | Modified (Phase 6) | Import swap |
| `tests/test_rust_assembler.py` | Modified | Update to target `PyMeshAssembler` |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Topology port breaks element ordering / DOF numbering | Med | Bit-exact comparison tests against Python impl before switching |
| `body_load` numerical parity with Python | Med | Property-based tests; compare vectors element-by-element |
| PETSc matrix lifecycle differs between Python and Rust sides | Med | Reuse existing `aeroelast-solvers` PETSc patterns |
| `compute_geometric_stiffness` depends on stress field shape assumptions | High | Capture Python behavior with integration tests first |
| Build time increase from topology structs | Low | Accept; CI caching mitigates |

## Rollback Plan

Every phase is independently mergeable. If any phase breaks tests:
- Revert to previous phase's branch
- `assembler.py` shim remains functional until Phase 6 is explicitly merged
- Phase 6 (deletion) is a separate, isolated PR gated on all tests green

## Dependencies

- `aeroelast-solvers` PETSc/SLEPc linking (already in place)
- PyO3 feature parity for `PetscMat` / `PetscVec` wrappers (verify in `aeroelast-py`)

## Success Criteria

- [ ] All existing tests in `test_rust_assembler.py`, `test_rust_modal.py`, `test_stress_stiffened_solver.py` pass with `PyMeshAssembler`
- [ ] `assembler.py` deleted from the codebase
- [ ] No Python code remains in the assembly path
- [ ] Numerical outputs (matrices, vectors) are bit-exact or within `1e-12` of Python baseline
