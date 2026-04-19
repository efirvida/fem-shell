# Design: Assembler Migration (Python → Rust)

## Technical Approach

Migrate `MeshAssembler` incrementally: first add missing element physics (body_load, geometric stiffness, centrifugal prestress) to `aeroelast-core`, then build a `MeshAssembler` struct in `aeroelast-core` that owns topology + element data, expose via PyO3 as `PyMeshAssembler`, and finally replace the Python shim.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| MeshTopology location | `aeroelast-core` | `aeroelast-py` only | Pure Rust struct enables unit testing without Python; solvers crate can use it directly |
| Python→Rust mesh transfer | Pass flat numpy arrays (coords, connectivity, elem types) | JSON serialization; PyO3 class wrapping Python obj | Zero-copy via `PyReadonlyArray`; matches existing batch pattern in `aeroelast-py` |
| MeshAssembler location | `aeroelast-core` (pure, returns COO) | `aeroelast-solvers` (has PETSc) | Keep core PETSc-free; solvers consumes COO from core. Assembler doesn't need PETSc directly |
| PETSc matrix return to Python | `PyCapsule` wrapping `PetscMat` (existing pattern) | Return COO + Python converts; petsc4py interop | Already proven in `petsc_assemble_matrix`; zero overhead |
| Shim structure | `assembler.py` delegates to `PyMeshAssembler` | Direct import replacement | Transparent to consumers; enables gradual migration |

## Data Flow

```
Python MeshModel
    │
    ▼  (flat numpy arrays: node_coords[N×3], connectivity[E×max_nodes], elem_types[E])
PyMeshAssembler.__init__(...)     ← aeroelast-py PyO3 class
    │
    ▼  constructs
MeshTopology + MeshAssembler      ← aeroelast-core (pure Rust)
    │  precomputes Mitc3/4Precomputed per element
    │  stores DOF connectivity
    │
    ▼  assemble_K() / assemble_M() / assemble_F_body() / assemble_geometric_K()
COO triplets (rows, cols, vals)   ← aeroelast-core
    │
    ▼  passed to
assemble_seq_aij(...)             ← aeroelast-solvers (PETSc)
    │
    ▼  wrapped in
PyCapsule<PetscMat>               ← returned to Python
```

## Key Rust Types

```rust
// aeroelast-core::assembly::topology
pub struct MeshTopology {
    pub node_coords: Vec<[f64; 3]>,       // N nodes
    pub connectivity: Vec<Vec<u32>>,       // E elements, variable node count
    pub elem_types: Vec<ElemType>,         // MITC3 | MITC4
    pub node_id_to_index: Vec<usize>,      // identity for 0-based
    pub dofs_per_node: usize,              // 6 for shells
}

pub enum ElemType { Mitc3, Mitc4 }

// aeroelast-core::assembly::assembler
pub struct MeshAssembler {
    pub topology: MeshTopology,
    pub dofs_count: usize,
    dof_connectivity: Vec<Vec<i64>>,       // per-element DOF indices
    precomputed: Vec<ElementPrecomputed>,   // enum over Mitc3/4
    nnz_per_row: Vec<i64>,
}

enum ElementPrecomputed {
    Tri(Mitc3Precomputed),
    Quad(Mitc4Precomputed),
}

impl MeshAssembler {
    pub fn new(topology: MeshTopology, materials: &[MaterialSpec]) -> Self;
    pub fn dofs_count(&self) -> usize;
    pub fn assemble_k(&self) -> (Vec<i64>, Vec<i64>, Vec<f64>);
    pub fn assemble_m(&self) -> (Vec<i64>, Vec<i64>, Vec<f64>);
    pub fn assemble_f_body(&self, gravity: [f64; 3]) -> Vec<f64>;
    pub fn assemble_geometric_k(&self, sigma: &[[f64; 3]]) -> (Vec<i64>, Vec<i64>, Vec<f64>);
    pub fn assemble_kt(&self, u: &[f64]) -> (Vec<i64>, Vec<i64>, Vec<f64>);
    pub fn assemble_fint(&self, u: &[f64], nonlinear: bool) -> Vec<f64>;
    pub fn nnz_per_row(&self) -> &[i64];
}
```

### New Element Functions (aeroelast-core)

```rust
// mitc3.rs additions
pub fn compute_body_load_global(pre: &Mitc3Precomputed, gravity: &Vector3<f64>, rho: f64) -> Vec18;
pub fn compute_k_sigma_global(pre: &Mitc3Precomputed, sigma: &Vector3<f64>) -> Mat18;
pub fn compute_centrifugal_prestress(pre: &Mitc3Precomputed, omega: f64,
    axis: &Vector3<f64>, center: &Vector3<f64>, centroid: &Vector3<f64>, rho: f64) -> Vector3<f64>;

// mitc4.rs additions (same signatures with Mat24/Vec24)
pub fn compute_body_load_global(pre: &Mitc4Precomputed, gravity: &Vector3<f64>, rho: f64) -> Vec24;
pub fn compute_k_sigma_global(pre: &Mitc4Precomputed, sigma: &Vector3<f64>) -> Mat24;
pub fn compute_centrifugal_prestress(pre: &Mitc4Precomputed, omega: f64,
    axis: &Vector3<f64>, center: &Vector3<f64>, centroid: &Vector3<f64>, rho: f64) -> Vector3<f64>;
```

Body load formula: `f_body = ∫ Nᵀ · (ρ·h·g) dA` where N maps shape functions to translational DOFs.

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `crates/aeroelast-core/src/elements/mitc3.rs` | Modify | Add `compute_body_load_global`, `compute_k_sigma_global`, `compute_centrifugal_prestress` |
| `crates/aeroelast-core/src/elements/mitc4.rs` | Modify | Same three functions for quad element |
| `crates/aeroelast-core/src/assembly/topology.rs` | Create | `MeshTopology`, `ElemType` |
| `crates/aeroelast-core/src/assembly/assembler.rs` | Create | `MeshAssembler` struct + all assembly methods |
| `crates/aeroelast-core/src/assembly/mod.rs` | Modify | Re-export topology + assembler modules |
| `crates/aeroelast-py/src/lib.rs` | Modify | Add `PyMeshAssembler` PyO3 class + batch body_load/geometric_k functions |
| `src/fem_shell/core/assembler.py` | Modify | Phase 4: thin shim delegating to `PyMeshAssembler` |

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit (Rust) | body_load, geometric_K, centrifugal prestress per element | Compare vs Python reference values (1e-10 rtol) |
| Unit (Rust) | MeshAssembler COO output for small mesh (4-elem patch) | Verify COO matches Python assembler output |
| Integration (Python) | `PyMeshAssembler` vs `MeshAssembler` on real meshes | Run existing `test_rust_assembler.py`, `test_rust_modal.py` |
| E2E | Full modal solve pipeline through shim | Existing test suite must pass unmodified |

## Migration Phases

**Phase 1: Element physics** — Add body_load, geometric_K, centrifugal_prestress to mitc3.rs/mitc4.rs. Go/no-go: unit tests pass at 1e-10 rtol vs Python.

**Phase 2: MeshAssembler in core** — Build `MeshTopology` + `MeshAssembler` in `aeroelast-core`. Go/no-go: Rust-only integration test assembles K/M for a 4-element patch matching Python COO output.

**Phase 3: PyO3 binding** — Expose `PyMeshAssembler` class in `aeroelast-py`. Go/no-go: Python can instantiate, call assemble_K(), get PyCapsule back. Existing tests still pass.

**Phase 4: Python shim** — `assembler.py` delegates to `PyMeshAssembler`. Go/no-go: ALL existing tests pass unmodified (test_rust_assembler, test_rust_modal, solver tests).

**Phase 5: Cleanup** — Remove shim, point consumers directly to `PyMeshAssembler`. Go/no-go: CI green with no Python assembler code.

## Open Questions

- [ ] Mixed meshes (MITC3 + MITC4 in same model): store heterogeneous connectivity in MeshTopology via `Vec<Vec<u32>>` or separate per-type arrays? Per-type arrays are better for batch parallelism.
- [ ] Composite materials in MeshAssembler: pass per-element ABD matrices or material enum? Lean toward per-element constitutive to match existing batch pattern.
