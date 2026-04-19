/// PETSc/SLEPc infrastructure: raw FFI bindings and RAII wrappers.
///
/// These modules are pure infrastructure — no solver logic lives here.
///
/// ```text
/// infra/
/// ├── ffi  — Raw C bindings for PETSc and SLEPc
/// ├── mat  — RAII PetscMat wrapper (owns Mat handle, calls MatDestroy on drop)
/// └── vec  — RAII PetscVec wrapper (owns Vec handle, calls VecDestroy on drop)
/// ```
pub mod ffi;
pub mod mat;
pub mod vec;
