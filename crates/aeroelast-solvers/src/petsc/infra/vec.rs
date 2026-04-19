/// Safe wrapper around a PETSc `Vec` handle.
use super::ffi::{self, Vec as RawVec};
use super::mat::PetscError;
use std::fmt;

/// A RAII wrapper for a PETSc vector.
pub struct PetscVec {
    pub(crate) raw: RawVec,
    pub(crate) size: usize,
}

impl PetscVec {
    /// Wrap a raw PETSc Vec handle (takes ownership).
    ///
    /// # Safety
    /// `raw` must be a valid, initialized PETSc Vec that has not yet been destroyed.
    pub unsafe fn from_raw(raw: RawVec, size: usize) -> Self {
        Self { raw, size }
    }

    /// Return the raw handle.
    pub fn as_raw(&self) -> RawVec {
        self.raw
    }

    /// Number of global DOFs.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy the vector's values into a Rust Vec<f64>.
    pub fn to_vec(&self) -> Result<Vec<f64>, PetscError> {
        let mut ptr: *mut f64 = std::ptr::null_mut();
        // SAFETY: self.raw is a valid PETSc Vec.
        unsafe {
            crate::petsc::infra::mat::check(
                ffi::VecGetArray(self.raw, &mut ptr),
                "VecGetArray",
            )?;
            let slice = std::slice::from_raw_parts(ptr, self.size);
            let result = slice.to_vec();
            crate::petsc::infra::mat::check(
                ffi::VecRestoreArray(self.raw, &mut ptr),
                "VecRestoreArray",
            )?;
            Ok(result)
        }
    }
}

impl Drop for PetscVec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a valid non-null Vec owned by this wrapper.
            unsafe {
                ffi::VecDestroy(&mut self.raw);
            }
        }
    }
}

impl fmt::Debug for PetscVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PetscVec({:p}, size={})", self.raw, self.size)
    }
}

unsafe impl Send for PetscVec {}
