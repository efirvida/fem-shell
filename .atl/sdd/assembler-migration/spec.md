# Assembler Migration Specification

## Purpose
This specification covers the migration of the Python `MeshAssembler` to Rust in the `fem-shell` project. The migration occurs in stages, preserving compatibility with existing Python consumers by introducing a Rust-backed assembler `PyMeshAssembler` via PyO3, which eventually replaces the current Python implementation.

## ADDED Requirements

### Requirement: REQ-1 Element Physics Completeness

The system MUST implement `body_load`, `compute_geometric_stiffness`, and `compute_centrifugal_prestress` in the Rust `MITC3` and `MITC4` elements (for both isotropic and composite materials). 

#### Scenario: Geometric Stiffness matches Python
- GIVEN a Rust MITC3 or MITC4 element 
- WHEN `compute_geometric_stiffness` is called with a membrane stress tensor
- THEN the resulting 24x24 matrix MUST match the Python implementation within 1e-10 relative tolerance.

#### Scenario: Centrifugal Prestress matches Python
- GIVEN a Rust MITC3 or MITC4 element
- WHEN `compute_centrifugal_prestress` is called with rotation vector and center of rotation
- THEN the resulting prestress tensor MUST match the Python implementation within 1e-10 relative tolerance.

#### Scenario: Body Load computation
- GIVEN a Rust MITC3 or MITC4 element
- WHEN `body_load` is called with a 3D gravity vector
- THEN the resulting 24-element force vector MUST accurately integrate the body force over the element area (accounting for thickness and density).

### Requirement: REQ-2 Rust MeshTopology

The system MUST define a Rust data structure (`MeshTopology`) to receive and store mesh data transferred from Python.

#### Scenario: Mesh data transfer
- GIVEN a Python `MeshModel` with nodes and elements
- WHEN initializing the Rust assembler
- THEN the Python mesh data (node IDs, coordinates, element IDs, node connectivities, element types) MUST be correctly copied to `MeshTopology` in Rust.

#### Scenario: DOF mapping
- GIVEN a `MeshTopology` in Rust
- WHEN calculating global DOF indices
- THEN it MUST correctly map each node's DOFs (6 per node) to global indices identical to the Python logic.

### Requirement: REQ-3 Rust MeshAssembler Struct

The system MUST implement a `MeshAssembler` struct in the `aeroelast-core` crate containing all assembly methods required by the solvers.

#### Scenario: Assembly Methods
- GIVEN a fully populated `MeshAssembler` in Rust
- WHEN methods like `assemble_K`, `assemble_M`, `assemble_geometric_K`, `assemble_F_body`, `assemble_internal_forces`, and `assemble_residual` are called
- THEN it MUST aggregate the element-level matrices/vectors into global COO or PETSc-compatible formats.

#### Scenario: DOFs Count
- GIVEN a `MeshAssembler` in Rust
- WHEN querying `dofs_count`
- THEN it MUST return the total number of degrees of freedom in the system.

### Requirement: REQ-4 PyO3 PyMeshAssembler Binding

The system MUST expose the Rust `MeshAssembler` to Python as a PyO3 class `PyMeshAssembler` in the `aeroelast-py` crate.

#### Scenario: Initialization from Python
- GIVEN a Python `MeshModel`
- WHEN `PyMeshAssembler(mesh_model)` is called
- THEN a valid `PyMeshAssembler` object MUST be instantiated holding the underlying Rust assembler.

#### Scenario: Returning PETSc matrices
- GIVEN a `PyMeshAssembler` in Python
- WHEN calling `assemble_K()` or similar matrix-returning methods
- THEN it MUST return a PETSc matrix (either as a PyCapsule compatible with `petsc4py` or a standard petsc4py object) identical in shape and non-zero pattern to the legacy assembler.

### Requirement: REQ-5 Python Shim Compatibility

The system MUST replace the logic in `src/fem_shell/core/assembler.py` with a thin shim that delegates to `PyMeshAssembler` without breaking existing API contracts.

#### Scenario: Transparent delegation
- GIVEN an existing consumer (e.g., `solver.py` or `stress_recovery.py`)
- WHEN it uses `MeshAssembler` from `assembler.py`
- THEN the shim MUST transparently route calls to the PyO3 binding and return expected types, allowing all existing tests to pass unmodified.

### Requirement: REQ-6 Python Assembler Deletion

The system MUST eventually remove the shim `assembler.py` and point consumers directly to `PyMeshAssembler` once full compatibility is verified.

#### Scenario: Deprecation and Removal
- GIVEN all test suites (`test_rust_assembler.py`, `test_rust_modal.py`, etc.) pass with the shim
- WHEN the shim is deleted
- THEN consumers MUST directly import and use the Rust-backed `PyMeshAssembler` without failures.
