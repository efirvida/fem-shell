/// Safe wrapper around a PETSc `Mat` handle.
///
/// Owns the matrix and calls `MatDestroy` on drop.
use super::ffi::{self, Mat as RawMat, PetscErrorCode};
use std::fmt;

/// A RAII wrapper for a PETSc matrix.
pub struct PetscMat {
    pub(crate) raw: RawMat,
}

impl PetscMat {
    /// Wrap a raw PETSc Mat handle (takes ownership).
    ///
    /// # Safety
    /// `raw` must be a valid, initialized PETSc Mat that has not yet been destroyed.
    pub unsafe fn from_raw(raw: RawMat) -> Self {
        Self { raw }
    }

    /// Return the raw handle (for passing to FFI calls).
    pub fn as_raw(&self) -> RawMat {
        self.raw
    }
}

impl Drop for PetscMat {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a valid non-null Mat owned by this wrapper.
            unsafe {
                ffi::MatDestroy(&mut self.raw);
            }
        }
    }
}

impl fmt::Debug for PetscMat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PetscMat({:p})", self.raw)
    }
}

// PetscMat is not Clone — matrices are unique resources.
// It is Send because PETSc Mat handles can be passed between threads
// as long as there is no concurrent access (caller's responsibility).
unsafe impl Send for PetscMat {}

/// Error type for PETSc operations.
#[derive(Debug)]
pub struct PetscError {
    pub code: PetscErrorCode,
    pub context: &'static str,
}

impl fmt::Display for PetscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PETSc error {} in context '{}'",
            self.code, self.context
        )
    }
}

impl std::error::Error for PetscError {}

/// Convert a PETSc error code into a Result.
pub(crate) fn check(code: PetscErrorCode, context: &'static str) -> Result<(), PetscError> {
    if code == 0 {
        Ok(())
    } else {
        Err(PetscError { code, context })
    }
}
