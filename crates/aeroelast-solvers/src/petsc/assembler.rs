/// PETSc assembler: converts COO triplets from `aeroelast-core` into an
/// assembled PETSc `Mat`.
///
/// Uses `MatSetPreallocationCOO` + `MatSetValuesCOO` (PETSc 3.20+ API) which
/// is more efficient than `MatSetValues` for batch insertion.
use super::infra::ffi::{self, ADD_VALUES, MAT_FINAL_ASSEMBLY, MAT_SYMMETRIC, PETSC_DECIDE, PETSC_TRUE, PetscInt};
use super::infra::mat::{check, PetscError, PetscMat};

// "seqaij" matrix type — sequential AIJ (CSR) sparse format (single process)
const MATAIJ: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"seqaij\0") };

/// Ensure PETSc and SLEPc are initialized (idempotent).
pub fn ensure_initialized() -> Result<(), PetscError> {
    let mut flag: i32 = 0;
    // SAFETY: PetscInitialized is safe to call before/after init.
    unsafe {
        check(ffi::PetscInitialized(&mut flag), "PetscInitialized")?;
        if flag == 0 {
            check(
                ffi::PetscInitializeNoArguments(),
                "PetscInitializeNoArguments",
            )?;
        }
        // Always check SLEPc separately — it may not be initialized even if PETSc is.
        check(ffi::SlepcInitialized(&mut flag), "SlepcInitialized")?;
        if flag == 0 {
            check(
                ffi::SlepcInitializeNoArguments(),
                "SlepcInitializeNoArguments",
            )?;
        }
    }
    Ok(())
}

/// Assemble a PETSc `Mat` (AIJ, sequential) from COO triplets.
///
/// # Arguments
/// * `rows`   – row indices (PETSc 0-based `i32`)
/// * `cols`   – column indices
/// * `vals`   – non-zero values
/// * `n_dof`  – global matrix dimension (n_dof × n_dof)
///
/// # Notes
/// * Duplicate entries at the same (row, col) are SUMMED.
/// * Uses `PETSC_COMM_SELF` — single-process sequential matrix.
///   For MPI use, replace with `petsc_comm_world()` and adjust prealloc.
pub fn assemble_seq_aij(
    rows: &[i32],
    cols: &[i32],
    vals: &[f64],
    n_dof: usize,
) -> Result<PetscMat, PetscError> {
    assert_eq!(rows.len(), cols.len());
    assert_eq!(rows.len(), vals.len());

    ensure_initialized()?;

    let n = n_dof as PetscInt;
    let ncoo = rows.len() as PetscInt;

    unsafe {
        let comm = ffi::petsc_comm_self();
        let mut raw_mat = std::ptr::null_mut();

        // Create and configure
        check(ffi::MatCreate(comm, &mut raw_mat), "MatCreate")?;
        check(
            ffi::MatSetType(raw_mat, MATAIJ.as_ptr()),
            "MatSetType",
        )?;
        check(
            ffi::MatSetSizes(raw_mat, n, n, n, n),
            "MatSetSizes",
        )?;

        // Preallocate via COO pattern — PETSc deduces sparsity from the triplets
        check(
            ffi::MatSetPreallocationCOO(raw_mat, ncoo, rows.as_ptr(), cols.as_ptr()),
            "MatSetPreallocationCOO",
        )?;

        // Insert values — ADD_VALUES: duplicate (row,col) entries are SUMMED.
        // This is essential for FEM: shared DOFs between elements accumulate contributions.
        check(
            ffi::MatSetValuesCOO(raw_mat, vals.as_ptr(), ADD_VALUES),
            "MatSetValuesCOO",
        )?;

        // Assemble
        check(
            ffi::MatAssemblyBegin(raw_mat, MAT_FINAL_ASSEMBLY),
            "MatAssemblyBegin",
        )?;
        check(
            ffi::MatAssemblyEnd(raw_mat, MAT_FINAL_ASSEMBLY),
            "MatAssemblyEnd",
        )?;

        // Mark symmetric — enables Cholesky factorization path in PETSc PC
        // FEM stiffness and mass matrices are always symmetric.
        check(
            ffi::MatSetOption(raw_mat, MAT_SYMMETRIC, PETSC_TRUE),
            "MatSetOption(MAT_SYMMETRIC)",
        )?;

        Ok(PetscMat::from_raw(raw_mat))
    }
}

/// Convenience: assemble from `aeroelast_core::assembly::coo_from_batch` output.
///
/// Takes the (rows, cols, vals) tuple returned by `coo_from_batch`, casts
/// from `i64` to `i32` (PetscInt), and calls `assemble_seq_aij`.
pub fn assemble_from_coo(
    coo: (Vec<i64>, Vec<i64>, Vec<f64>),
    n_dof: usize,
) -> Result<PetscMat, PetscError> {
    let (rows_i64, cols_i64, vals) = coo;
    let rows: Vec<i32> = rows_i64.iter().map(|&v| v as i32).collect();
    let cols: Vec<i32> = cols_i64.iter().map(|&v| v as i32).collect();
    assemble_seq_aij(&rows, &cols, &vals, n_dof)
}

/// Create an uninitialized PETSc Vec of the given size (PETSC_COMM_SELF).
pub fn create_vec(n_dof: usize) -> Result<crate::petsc::infra::vec::PetscVec, PetscError> {
    ensure_initialized()?;
    let n = n_dof as PetscInt;

    unsafe {
        let comm = ffi::petsc_comm_self();
        let mut raw_vec = std::ptr::null_mut();

        check(ffi::VecCreate(comm, &mut raw_vec), "VecCreate")?;
        check(ffi::VecSetSizes(raw_vec, n, PETSC_DECIDE), "VecSetSizes")?;
        check(ffi::VecSetFromOptions(raw_vec), "VecSetFromOptions")?;

        Ok(crate::petsc::infra::vec::PetscVec::from_raw(raw_vec, n_dof))
    }
}
