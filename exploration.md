## Exploration: fem-shell Rust Migration of assembler.py

### Summary of what assembler.py does
`src/fem_shell/core/assembler.py` manages the finite element assembly process. It takes a mesh and property definitions, instantiates elements via a factory, maps nodes to global degrees of freedom (DOF), and builds the global sparse stiffness, mass, load, and geometric stiffness matrices. It leverages PETSc for scalable sparse matrices and Newton-Raphson iteration support. It already delegates heavy computation (like element-level stiffness/mass integration) to Rust for MITC3, MITC4, and their composite variants.

### Migration Status Table

| Method in `assembler.py` | Rust Coverage Status | Rust Modules/Functions |
|---|---|---|
| `__init__` / `_precompute_elements` | ❌ Python | Element initialization and DOF mapping are in Python |
| `_compute_sparsity_pattern` | ⚠️ Mixed | Logic exists in `aeroelast_core::assembly::compute_nnz_per_row` but Python still uses sets |
| `_prepare_rust_batch_data` | ❌ Python | Grouping logic (by coords, material, element type) is in Python |
| `assemble_stiffness_matrix` | ⚠️ Mixed | Uses `batch_ke_*` and `coo_assembly` (Rust), but converts to `petsc4py.Mat` in Python |
| `assemble_mass_matrix` | ⚠️ Mixed | Uses `batch_me_*` and `coo_assembly` (Rust), but converts to `petsc4py.Mat` in Python |
| `assemble_load_vector` | ❌ Python | Calls Python `Element.body_load` |
| `assemble_geometric_stiffness` | ❌ Python | Calls Python `Element.compute_geometric_stiffness` |
| `assemble_tangent_stiffness` | ⚠️ Mixed | Uses `batch_kt_*` and `coo_assembly` (Rust), but uses `petsc4py` |
| `assemble_internal_forces` | ⚠️ Mixed | Uses `batch_fint_*` (Rust), but scatters into vector in Python |

### List of Blockers (Python dependencies not yet in Rust)
1. **Python Element Classes**: `assembler.py` directly instantiates Python classes (`MITC3`, `MITC4`, `MITC3Composite`, `MITC4Composite`) via `ElementFactory.get_element()`. The Rust backend calculates matrices, but the object representation lives in Python.
2. **Missing Rust Methods**:
   - `body_load` (load vectors)
   - `compute_geometric_stiffness` and `compute_centrifugal_prestress` (geometric stiffness)
3. **Mesh Data Structures**: The input `MeshModel` and `MeshElement` are Python objects. Rust needs a way to parse/receive the global mesh topology directly instead of Python feeding it element by element.
4. **PETSc Boundary**: Python currently creates the `petsc4py.Mat` objects. While `petsc_assemble_matrix` exists in `aeroelast-py` (which returns a `PyCapsule`), `assembler.py` currently relies on `petsc4py` wrappers for matrix insertion and assembly. Migrating `assembler.py` completely requires wrapping the `PetscMat` capsule into a usable class or moving the full assembly and solver logic into Rust.

### Recommended Migration Strategy
1. **Migrate Mesh and Element Definitions**: Create a Rust representation of the global `Mesh` and the `ElementFactory` logic in `aeroelast-core`. Instead of passing Python `Element` objects, pass node coordinates and connectivity to Rust once.
2. **Implement Missing Element Physics in Rust**:
   - Add geometric stiffness and centrifugal prestress to `aeroelast-core::elements::mitc3` and `mitc4`.
   - Add load vector integration (`body_load`) to Rust.
3. **Build the Rust Assembler**: Create a pure Rust `Assembler` struct in `aeroelast-core::assembly` that:
   - Takes the Rust `Mesh`.
   - Groups elements internally.
   - Computes the DOF mappings.
   - Outputs COO triplets.
4. **Expose `PyMeshAssembler`**: Create a PyO3 class in `aeroelast-py` that mirrors the API of `MeshAssembler`. This class will hold the Rust `Assembler` and return either `petsc4py` objects (via conversion) or `PyCapsule`s for the PETSc solver.
5. **Update Consumers**: Swap out the Python `MeshAssembler` for the PyO3 `PyMeshAssembler` in the solvers.

### Files That Will Need to Change
**Consumers of `MeshAssembler` (to update imports/types):**
- `src/fem_shell/solvers/solver.py`
- `src/fem_shell/solvers/fsi/stress_stiffened_dynamic.py`
- `src/fem_shell/postprocess/stress_recovery.py`
- `tests/test_rust_assembler.py`
- `tests/test_rust_modal.py`
- `tests/test_stress_stiffened_solver.py`

**Files to Delete/Replace:**
- `src/fem_shell/core/assembler.py` (Replaced by `aeroelast_py.MeshAssembler`)

**Rust Files to Expand:**
- `crates/aeroelast-core/src/assembly/mod.rs` (or new `assembler.rs`)
- `crates/aeroelast-core/src/elements/mitc3.rs` and `mitc4.rs` (add load and geometric stiffness)
- `crates/aeroelast-py/src/lib.rs` (expose the new `PyMeshAssembler` class)
