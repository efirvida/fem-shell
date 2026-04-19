# Tasks: Assembler Migration (Python → Rust)

## Phase 1: Element Physics

- [x] 1.1 **Add** `compute_body_load_global` to `crates/aeroelast-core/src/elements/mitc3.rs` — integrates `Nᵀ·(ρ·h·g)` over element area; returns `Vec18`. Verify: unit test compares output vs Python reference at 1e-10 rtol.
- [x] 1.2 **Add** `compute_k_sigma_global` (geometric stiffness) to `crates/aeroelast-core/src/elements/mitc3.rs` — takes membrane stress `σ: Vector3`; returns `Mat18`. Verify: unit test vs Python 18×18 matrix at 1e-10 rtol.
- [x] 1.3 **Add** `compute_centrifugal_prestress` to `crates/aeroelast-core/src/elements/mitc3.rs` — takes `ω, axis, center, centroid, ρ`; returns `Vector3` prestress tensor. Verify: unit test vs Python at 1e-10 rtol.
- [x] 1.4 **Add** `compute_body_load_global` to `crates/aeroelast-core/src/elements/mitc4.rs` — same formula for quad; returns `Vec24`. Verify: unit test vs Python at 1e-10 rtol.
- [x] 1.5 **Add** `compute_k_sigma_global` to `crates/aeroelast-core/src/elements/mitc4.rs` — returns `Mat24`. Verify: unit test vs Python 24×24 matrix at 1e-10 rtol.
- [x] 1.6 **Add** `compute_centrifugal_prestress` to `crates/aeroelast-core/src/elements/mitc4.rs` — same signature as MITC3. Verify: unit test vs Python at 1e-10 rtol.

## Phase 2: MeshTopology Struct

- [x] 2.1 **Create** `crates/aeroelast-core/src/assembly/topology.rs` — define `ElemType` enum (`Mitc3 | Mitc4`) and `MeshTopology` struct with `node_coords`, `connectivity`, `elem_types`, `node_id_to_index`, `dofs_per_node`. Verify: `cargo test -p aeroelast-core` compiles cleanly.
- [x] 2.2 **Update** `crates/aeroelast-core/src/assembly/mod.rs` — add `pub mod topology; pub use topology::{MeshTopology, ElemType};`. Verify: public API visible from crate root.
- [x] 2.3 **Implement** DOF index method on `MeshTopology` — `fn global_dof_indices(&self, elem_idx: usize) -> Vec<usize>` mapping node DOFs (6 per node) to global indices. Verify: unit test on 2-element patch matches Python mapping exactly.

## Phase 3: MeshAssembler Struct

- [x] 3.1 **Create** `crates/aeroelast-core/src/assembly/assembler.rs` — define `ElementPrecomputed` enum and `MeshAssembler` struct with `topology`, `dofs_count`, `dof_connectivity`, `precomputed`, `nnz_per_row`. Verify: `cargo build -p aeroelast-core` passes.
- [x] 3.2 **Implement** `MeshAssembler::new` — constructs from `MeshTopology + &[MaterialSpec]`, precomputes `Mitc3/4Precomputed` per element, builds `dof_connectivity`, computes `nnz_per_row`. Verify: instantiation on 4-element patch without panic.
- [x] 3.3 **Implement** `MeshAssembler::assemble_k` and `assemble_m` — return `(Vec<i64>, Vec<i64>, Vec<f64>)` COO triplets. Verify: Rust-only test on 4-element patch; K/M COO matches Python assembler output.
- [x] 3.4 **Implement** `MeshAssembler::assemble_f_body` — takes `gravity: [f64; 3]`; returns `Vec<f64>`. Verify: unit test vs Python body load vector on same 4-element patch.
- [x] 3.5 **Implement** `MeshAssembler::assemble_geometric_k` — takes `sigma: &[[f64; 3]]`; returns COO triplets. Verify: unit test vs Python geometric stiffness on patch.
- [x] 3.6 **Implement** `MeshAssembler::assemble_kt` and `assemble_fint` — nonlinear tangent stiffness and internal forces. Verify: unit test on known displacement state; values plausible (finite, symmetric).
- [x] 3.7 **Update** `crates/aeroelast-core/src/assembly/mod.rs` — add `pub mod assembler; pub use assembler::MeshAssembler;`. Verify: `coo_assembly` and `MeshAssembler` both accessible from `aeroelast_core::assembly`.

## Phase 4: PyO3 Binding + Python Shim

- [ ] 4.1 **Add** `PyMeshAssembler` PyO3 class to `crates/aeroelast-py/src/lib.rs` — `__init__` accepts flat numpy arrays (`node_coords[N×3]`, `connectivity[E×max_nodes]`, `elem_types[E]`) via `PyReadonlyArray`; constructs `MeshAssembler` internally. Verify: Python `PyMeshAssembler(mesh_model)` instantiates without error.
- [ ] 4.2 **Expose** `assemble_k()`, `assemble_m()`, `assemble_f_body(gravity)`, `assemble_geometric_k(sigma)` methods on `PyMeshAssembler` in `crates/aeroelast-py/src/lib.rs` — call `assemble_seq_aij` from `aeroelast-solvers` and return `PyCapsule`. Verify: each method returns capsule; shape/nnz match legacy Python assembler.
- [ ] 4.3 **Expose** `dofs_count` property on `PyMeshAssembler` in `crates/aeroelast-py/src/lib.rs`. Verify: Python `asm.dofs_count == 6 * n_nodes`.
- [ ] 4.4 **Update** `src/fem_shell/core/assembler.py` — replace implementation body with thin shim that instantiates `PyMeshAssembler` and delegates all method calls. Keep existing class name and method signatures unchanged. Verify: `test_rust_assembler.py`, `test_rust_modal.py`, and solver tests all pass unmodified.

## Phase 5: Cleanup

- [ ] 5.1 **Update** all consumers (`solver.py`, `stress_recovery.py`, etc.) to import `PyMeshAssembler` from `aeroelast_py` directly, removing dependency on `assembler.py`. Verify: each consumer imports resolve; run full test suite.
- [ ] 5.2 **Delete** `src/fem_shell/core/assembler.py`. Verify: CI passes with no import errors; `test_rust_assembler.py`, `test_rust_modal.py`, and all solver tests green.
