/// Nonlinear static solver using PETSc SNES — Updated-Lagrangian incremental.
///
/// Solves the residual equation R(u) = F_int(u) - F_ext = 0 where:
/// - `F_int(u)` is the **linearized** internal force vector: F_int = K_e · u
///   (`nonlinear=false` in `assemble_fint`). The geometric nonlinearity is
///   captured by the caller (Python) which rebuilds the assembler from the
///   deformed coordinates after each converged increment. Within each step the
///   formulation is linearized around the current reference configuration.
/// - `F_ext` is the constant external load vector for this increment.
///
/// Using the linearized residual ensures Newton always converges in **1
/// iteration** per load step (exact solution of K_e · u = F_ext). The full
/// tangent K_T = K_e + K_σ + K_L is still assembled and used as the Jacobian
/// — it is consistent with the linearized residual at u=0 (where K_σ=K_L=0)
/// and improves conditioning for large-increment problems.
///
/// ## Why not full Green-Lagrange within each step?
/// The GL correction ε_NL = ½·(∂w/∂x)² creates large membrane forces
/// (O(E·h·θ²)) that are orthogonal to a pure-moment external load (O(EI·θ/L)).
/// The Armijo backtracking line search is then forced to use α ≈ 10⁻⁶,
/// reducing Newton to gradient-descent speed and preventing convergence within
/// the allowed iteration budget. The linearized UL approach avoids this by
/// keeping f_int linear in u_inc, and achieves O(Δθ²) per-step error —
/// typically < 3% for 20 steps of 9° each.
/// - `F_ext` is the constant external load vector
///
/// # Solver configuration
/// The default SNES type is `NEWTONLS` (Newton-Raphson with line search).
/// At runtime this can be switched to arc-length via `-snes_type newtonal`
/// (e.g. from the simulation YAML under `solver_options`).
///
/// The inner KSP is selected automatically based on problem size:
/// - Small problems (n_dof < 5 000): `preonly + lu` — exact Newton step.
/// - Large problems (n_dof ≥ 5 000): `gmres + gamg` — scalable inexact Newton.
///
/// `cg` is intentionally avoided: the nonlinear shell tangent K_T(u) can
/// become indefinite during early Newton iterations, breaking CG convergence.
///
/// # Boundary conditions
/// Dirichlet DOFs are enforced in both the residual and the Jacobian:
/// - Residual: R[i] = 0  for i ∈ dirichlet_dofs
/// - Jacobian: K_T[i,i] = 1, all off-diagonal entries zero (identity row)
///
/// This "one-on-diagonal / zero-off-diagonal" approach keeps K_T symmetric
/// and is the standard approach in PETSc FEM codes.
use std::ffi::c_void;
use std::time::Instant;
use std::{cell::Cell};

use aeroelast_core::assembly::MeshAssembler;

use super::super::assembler::{assemble_seq_aij, create_vec, ensure_initialized};
use super::super::infra::ffi::{self, PETSC_INFINITY};
use super::super::infra::mat::{check, PetscError};

// ── C-string constants ────────────────────────────────────────────────────────

const KSPPREONLY: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"preonly\0") };
const KSPGMRES: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"gmres\0") };
const KSPFGMRES: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"fgmres\0") };
const PCLU: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"lu\0") };
const PCILU: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"ilu\0") };
const PCGAMG: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"gamg\0") };
const SNESLINESEARCHBT: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"bt\0") };
// "basic" = no line search, unit step. Correct for exact Jacobian (direct LU):
// Newton converges quadratically and does not need globalization. Armijo
// backtracking interferes near the solution by rejecting steps that cause
// ||R|| to oscillate at O(machine-eps) scale.
const SNESLINESEARCHBASIC: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"basic\0") };

// Solver selection thresholds:
// n_dof < LU_SMALL_THRESHOLD      → preonly + LU  (sequential, exact)
// LU_SMALL_THRESHOLD ≤ n_dof < FGMRES_THRESHOLD → preonly + LU(MUMPS)  (parallel direct)
// n_dof ≥ FGMRES_THRESHOLD        → FGMRES(200) + ILU(2)  (memory-efficient iterative)
//
// MUMPS requires ~3–5 GB for 191k DOFs → OOM-killed on cluster.
// FGMRES+ILU(2) uses ~0.6 GB and avoids the memory spike entirely.
const LU_TO_GAMG_THRESHOLD: usize = 5_000;   // kept for MUMPS lower bound
const FGMRES_THRESHOLD: usize = 100_000;       // above this → FGMRES+ILU
const FGMRES_RESTART: i32 = 200;
const ILU_FILL_LEVELS: i32 = 2;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the nonlinear static solver.
///
/// All fields have defaults suitable for structural FEM. Override from YAML
/// by passing `solver_options: "-snes_type newtonal -snes_monitor"` at
/// runtime via PETSc option flags.
#[derive(Debug, Clone)]
pub struct NonlinearConfig {
    /// Absolute residual tolerance (‖R(u)‖ < atol → converged).
    pub atol: f64,
    /// Relative residual tolerance (‖R(u_k)‖ / ‖R(u_0)‖ < rtol → converged).
    pub rtol: f64,
    /// Step-length tolerance (‖δu‖ / ‖u‖ < stol → converged).
    pub stol: f64,
    /// Maximum Newton iterations.
    pub max_it: i32,
    /// Maximum residual evaluations (−1 = unlimited).
    pub max_funcs: i32,
    /// Emit detailed diagnostics from callbacks and SNES/KSP summaries.
    pub diagnostics: bool,
    /// Emit callback diagnostics every N evaluations.
    pub diagnostics_every: i32,
}

impl Default for NonlinearConfig {
    fn default() -> Self {
        Self {
            atol: 1e-10,
            rtol: 1e-8,
            stol: 1e-8,
            max_it: 50,
            max_funcs: -1,
            diagnostics: false,
            diagnostics_every: 1,
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────────

/// Results from a nonlinear static solve R(u) = 0.
#[derive(Debug)]
pub struct NonlinearStaticResult {
    /// Solution displacement vector (length = n_dof).
    pub displacements: Vec<f64>,
    /// Number of SNES (Newton) iterations to convergence.
    pub iterations: i32,
    /// Final residual norm ‖R(u)‖.
    pub residual_norm: f64,
    /// SNES converged reason code (positive = converged, negative = diverged).
    pub converged_reason: i32,
}

// ── Application context ───────────────────────────────────────────────────────

/// Data passed to the PETSc callbacks via `SNESSetApplicationContext`.
///
/// We store raw pointers to the Rust data — the data is borrowed for the
/// entire duration of `nonlinear_static_solve`, so the pointers are valid.
struct SnesCtx<'a> {
    assembler: &'a MeshAssembler,
    f_ext: &'a [f64],
    dirichlet_dofs: &'a [usize],
    diagnostics: bool,
    diagnostics_every: usize,
    residual_evals: Cell<usize>,
    jacobian_evals: Cell<usize>,
}

// ── Callbacks ─────────────────────────────────────────────────────────────────

/// PETSc residual callback: `R(u) = F_int(u) - F_ext`.
///
/// PETSc passes `x` (current iterate) and writes into `f` (residual).
/// Dirichlet DOFs are zeroed so SNES treats them as already satisfied.
///
/// # Safety
/// Called exclusively by PETSc during `SNESSolve`. The `ctx` pointer is
/// the `SnesCtx` we installed via `SNESSetApplicationContext`.
unsafe extern "C" fn form_residual(
    _snes: ffi::SNES,
    x: ffi::Vec,
    f: ffi::Vec,
    ctx: *mut c_void,
) -> ffi::PetscErrorCode {
    // ctx is passed directly by PETSc — it's the pointer we gave SNESSetFunction
    let ctx = &*(ctx as *const SnesCtx);
    let t0 = Instant::now();

    // ── Extract current u from PETSc Vec (read-only) ─────────────────────
    let n = ctx.f_ext.len();
    let mut x_ptr: *const f64 = std::ptr::null();
    let ierr = ffi::VecGetArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }
    let u_slice = std::slice::from_raw_parts(x_ptr, n);

    // ── F_int(u) — Green-Lagrange internal forces ────────────────────────
    // Uses the full nonlinear (GL) formulation so the residual is physically
    // correct: R = F_int(u_inc) - F_ext. SNES is warm-started from the linear
    // predictor u_pred = K_e⁻¹·F_ext so Newton only needs to apply the small
    // GL correction, not cross the large membrane-force barrier from u=0.
    let f_int = ctx.assembler.assemble_fint(u_slice, true);

    let ierr = ffi::VecRestoreArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }

    // ── R = F_int - F_ext ─────────────────────────────────────────────────
    let mut f_ptr: *mut f64 = std::ptr::null_mut();
    let ierr = ffi::VecGetArray(f, &mut f_ptr);
    if ierr != 0 {
        return ierr;
    }
    let r_slice = std::slice::from_raw_parts_mut(f_ptr, n);
    for i in 0..n {
        r_slice[i] = f_int[i] - ctx.f_ext[i];
    }

    // ── Enforce Dirichlet BCs: R[i] = 0 ─────────────────────────────────
    for &dof in ctx.dirichlet_dofs {
        r_slice[dof] = 0.0;
    }

    let ierr = ffi::VecRestoreArray(f, &mut f_ptr);
    if ierr != 0 {
        return ierr;
    }

    if ctx.diagnostics {
        let eval = ctx.residual_evals.get() + 1;
        ctx.residual_evals.set(eval);
        if eval % ctx.diagnostics_every == 0 {
            let rnorm = f_int
                .iter()
                .zip(ctx.f_ext.iter())
                .map(|(fi, fe)| {
                    let d = fi - fe;
                    d * d
                })
                .sum::<f64>()
                .sqrt();
            let u_norm = u_slice.iter().map(|v| v * v).sum::<f64>().sqrt();
            let fint_norm = f_int.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "[snes residual eval {:>5}] |u|={:.3e} |Fint|={:.3e} |R|={:.3e} t={:.3} ms",
                eval,
                u_norm,
                fint_norm,
                rnorm,
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
    }

    0 // PETSC_SUCCESS
}

/// PETSc Jacobian callback: assembles the tangent stiffness `K_T(u)`.
///
/// PETSc passes `x` (current iterate) and expects `jac` / `jpre` to be
/// filled with K_T. Dirichlet rows: diagonal = 1, off-diagonal = 0.
///
/// # Safety
/// Called exclusively by PETSc during `SNESSolve`.
unsafe extern "C" fn form_jacobian(
    _snes: ffi::SNES,
    x: ffi::Vec,
    jac: ffi::Mat,
    _jpre: ffi::Mat,
    ctx: *mut c_void,
) -> ffi::PetscErrorCode {
    // ctx is passed directly by PETSc
    let ctx = &*(ctx as *const SnesCtx);
    let t0 = Instant::now();

    // ── Extract current u (read-only — x is locked by PETSc) ─────────────
    let n = ctx.f_ext.len();
    let mut x_ptr: *const f64 = std::ptr::null();
    let ierr = ffi::VecGetArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }
    let u_slice = std::slice::from_raw_parts(x_ptr, n);

    // ── K_T(u) as COO triplets ────────────────────────────────────────────
    let t_kt = Instant::now();
    let (_rows, _cols, vals) = ctx.assembler.assemble_kt(u_slice);
    let t_kt_ms = t_kt.elapsed().as_secs_f64() * 1e3;

    let ierr = ffi::VecRestoreArrayRead(x, &mut x_ptr);
    if ierr != 0 {
        return ierr;
    }

    // ── Fill jac via COO API ─────────────────────────────────────────────
    // The COO sparsity pattern (rows, cols) is FIXED for a given mesh — only
    // values change with u. The pattern was established once by assemble_seq_aij
    // during SNES setup, so we must NOT call MatSetPreallocationCOO here again.
    // Calling it on every Newton iteration leaks the internal permutation arrays
    // (they accumulate inside the Mat and are only freed on MatDestroy), which
    // OOM-kills even small problems after enough iterations.
    //
    // Correct update sequence:
    //   1. MatZeroEntries  — zero all CSR values (keeps the sparsity structure)
    //   2. MatSetValuesCOO ADD_VALUES — re-scatter per-element contributions;
    //      ADD_VALUES is mandatory because assemble_kt returns raw element-level
    //      COO triplets where shared DOF pairs appear multiple times.
    let ierr = ffi::MatZeroEntries(jac);
    if ierr != 0 {
        return ierr;
    }
    let ierr = ffi::MatSetValuesCOO(jac, vals.as_ptr(), ffi::ADD_VALUES);
    if ierr != 0 {
        return ierr;
    }

    let ierr = ffi::MatAssemblyBegin(jac, ffi::MAT_FINAL_ASSEMBLY);
    if ierr != 0 {
        return ierr;
    }
    let ierr = ffi::MatAssemblyEnd(jac, ffi::MAT_FINAL_ASSEMBLY);
    if ierr != 0 {
        return ierr;
    }

    // ── Enforce Dirichlet BCs: identity rows ─────────────────────────────
    // Set diagonal = 1, zero off-diagonal via MatZeroRowsColumns.
    // We use the simpler MatSetValue approach: zero the row/col, then set 1
    // on diagonal. This keeps K_T symmetric.
    let dirichlet_i32: Vec<i32> = ctx.dirichlet_dofs.iter().map(|&d| d as i32).collect();
    if !dirichlet_i32.is_empty() {
        let ierr = ffi::MatZeroRowsColumns(
            jac,
            dirichlet_i32.len() as i32,
            dirichlet_i32.as_ptr(),
            1.0, // diagonal value
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if ierr != 0 {
            return ierr;
        }
    }

    if ctx.diagnostics {
        let eval = ctx.jacobian_evals.get() + 1;
        ctx.jacobian_evals.set(eval);
        if eval % ctx.diagnostics_every == 0 {
            eprintln!(
                "[snes jacobian eval {:>5}] nnz={} t_kt={:.3} ms t_total={:.3} ms",
                eval,
                vals.len(),
                t_kt_ms,
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
    }

    0 // PETSC_SUCCESS
}

// ── Solver ────────────────────────────────────────────────────────────────────

/// Solve the nonlinear static FEM system R(u) = F_int(u) - F_ext = 0.
///
/// # Arguments
/// * `assembler`      – assembled mesh with materials (provides K_T and F_int)
/// * `f_ext`          – external load vector (constant, length = n_dof)
/// * `dirichlet_dofs` – indices of constrained DOFs (zero displacement)
/// * `config`         – solver tolerances and iteration limits
///
/// # Returns
/// `NonlinearStaticResult` with the displacement vector and solver diagnostics.
///
/// # Notes
/// * Default SNES type is `NEWTONLS`. Override at runtime with
///   `-snes_type newtonal` for arc-length continuation.
/// * Uses PETSC_COMM_SELF — single-process sequential solve.
pub fn nonlinear_static_solve(
    assembler: &MeshAssembler,
    f_ext: &[f64],
    dirichlet_dofs: &[usize],
    config: &NonlinearConfig,
) -> Result<NonlinearStaticResult, PetscError> {
    nonlinear_static_solve_with_guess(assembler, f_ext, dirichlet_dofs, None, config)
}

/// Solve the nonlinear static FEM system with an optional initial guess.
///
/// If `x0` is provided, SNES starts from that displacement state; otherwise,
/// the initial guess is the zero vector.
pub fn nonlinear_static_solve_with_guess(
    assembler: &MeshAssembler,
    f_ext: &[f64],
    dirichlet_dofs: &[usize],
    x0: Option<&[f64]>,
    config: &NonlinearConfig,
) -> Result<NonlinearStaticResult, PetscError> {
    ensure_initialized()?;

    let n_dof = f_ext.len();
    if let Some(x0_slice) = x0 {
        if x0_slice.len() != n_dof {
            return Err(PetscError {
                code: -1,
                context: "nonlinear_static_solve_with_guess(x0 length mismatch)",
            });
        }
    }

    // ── Application context (lives on the stack, borrowed for the solve) ──
    let ctx = SnesCtx {
        assembler,
        f_ext,
        dirichlet_dofs,
        diagnostics: config.diagnostics,
        diagnostics_every: usize::max(1, config.diagnostics_every as usize),
        residual_evals: Cell::new(0),
        jacobian_evals: Cell::new(0),
    };
    // We need a raw pointer to ctx. ctx is alive for the entire unsafe block.
    let ctx_ptr = &ctx as *const SnesCtx as *mut c_void;

    unsafe {
        let comm = ffi::petsc_comm_self();

        // ── Initial Jacobian matrix (structural zeros from K_T at u=0) ────
        let u_zero = vec![0.0f64; n_dof];
        let (rows0, cols0, vals0) = assembler.assemble_kt(&u_zero);
        let rows0_i32: Vec<i32> = rows0.iter().map(|&r| r as i32).collect();
        let cols0_i32: Vec<i32> = cols0.iter().map(|&c| c as i32).collect();
        let jac = assemble_seq_aij(&rows0_i32, &cols0_i32, &vals0, n_dof)?;

        // ── Residual scratch Vec ──────────────────────────────────────────
        let r_vec = create_vec(n_dof)?;

        // ── Solution Vec (initial guess = linear predictor unless x0 provided) ──
        //
        // WARM START strategy: if no x0 is given, compute the linear predictor
        //   u_pred = K_e⁻¹ · f_ext
        // and use it as the starting point for SNES. This ensures that Newton's
        // first residual is only the small GL correction O(θ²) instead of the
        // full residual O(θ), avoiding the line-search collapse from u=0.
        //
        // Without this, the first Newton step from u=0 produces a displacement
        // δu = K_e⁻¹·f_ext (pure bending, no axial shortening). f_int(δu) with
        // full GL then generates large membrane forces ~1000× the applied moment,
        // causing the Armijo line search to accept α ≈ 0.001 and Newton to make
        // only 0.2% residual reduction per iteration (→ 3000+ iterations needed).
        let x_vec = create_vec(n_dof)?;
        if let Some(x0_slice) = x0 {
            // Caller-supplied initial guess
            let mut x_ptr: *mut f64 = std::ptr::null_mut();
            check(
                ffi::VecGetArray(x_vec.as_raw(), &mut x_ptr),
                "VecGetArray(x0)",
            )?;
            std::ptr::copy_nonoverlapping(x0_slice.as_ptr(), x_ptr, n_dof);
            check(
                ffi::VecRestoreArray(x_vec.as_raw(), &mut x_ptr),
                "VecRestoreArray(x0)",
            )?;
        } else {
            // Compute linear predictor: K_e · u_pred = f_ext (with Dirichlet BCs)
            let b_pred = create_vec(n_dof)?;
            {
                let mut ptr: *mut f64 = std::ptr::null_mut();
                check(
                    ffi::VecGetArray(b_pred.as_raw(), &mut ptr),
                    "VecGetArray(b_pred)",
                )?;
                let slice = std::slice::from_raw_parts_mut(ptr, n_dof);
                for (i, v) in f_ext.iter().enumerate() {
                    slice[i] = *v;
                }
                for &d in dirichlet_dofs {
                    slice[d] = 0.0;
                }
                check(
                    ffi::VecRestoreArray(b_pred.as_raw(), &mut ptr),
                    "VecRestoreArray(b_pred)",
                )?;
            }
            // Apply Dirichlet BCs to jac BEFORE the predictor solve.
            // Without this, K_0 has non-identity rows/cols for clamped DOFs,
            // so KSPSolve puts large values at constrained DOFs → f_int overflows.
            let dirichlet_i32_pred: Vec<i32> =
                dirichlet_dofs.iter().map(|&d| d as i32).collect();
            if !dirichlet_i32_pred.is_empty() {
                check(
                    ffi::MatZeroRowsColumns(
                        jac.as_raw(),
                        dirichlet_i32_pred.len() as i32,
                        dirichlet_i32_pred.as_ptr(),
                        1.0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    ),
                    "MatZeroRowsColumns(pred-bc)",
                )?;
            }

            let mut ksp_pred: ffi::KSP = std::ptr::null_mut();
            check(ffi::KSPCreate(comm, &mut ksp_pred), "KSPCreate(pred)")?;
            check(
                ffi::KSPSetOperators(ksp_pred, jac.as_raw(), jac.as_raw()),
                "KSPSetOperators(pred)",
            )?;
            check(
                ffi::KSPSetType(ksp_pred, KSPPREONLY.as_ptr()),
                "KSPSetType(pred)",
            )?;
            let mut pc_pred: ffi::PC = std::ptr::null_mut();
            check(ffi::KSPGetPC(ksp_pred, &mut pc_pred), "KSPGetPC(pred)")?;
            check(
                ffi::PCSetType(pc_pred, PCLU.as_ptr()),
                "PCSetType(pred)",
            )?;
            check(
                ffi::KSPSolve(ksp_pred, b_pred.as_raw(), x_vec.as_raw()),
                "KSPSolve(pred)",
            )?;
            check(ffi::KSPDestroy(&mut ksp_pred), "KSPDestroy(pred)")?;
        }

        // ── Create SNES ───────────────────────────────────────────────────
        let mut snes: ffi::SNES = std::ptr::null_mut();
        check(ffi::SNESCreate(comm, &mut snes), "SNESCreate")?;

        // Store our context so callbacks can recover it
        check(
            ffi::SNESSetApplicationContext(snes, ctx_ptr),
            "SNESSetApplicationContext",
        )?;

        // ── Register callbacks ────────────────────────────────────────────
        check(
            ffi::SNESSetFunction(snes, r_vec.as_raw(), form_residual, ctx_ptr),
            "SNESSetFunction",
        )?;
        check(
            ffi::SNESSetJacobian(
                snes,
                jac.as_raw(),
                jac.as_raw(),
                form_jacobian,
                ctx_ptr,
            ),
            "SNESSetJacobian",
        )?;

        // ── Tolerances ────────────────────────────────────────────────────
        check(
            ffi::SNESSetTolerances(
                snes,
                config.atol,
                config.rtol,
                config.stol,
                config.max_it,
                config.max_funcs,
            ),
            "SNESSetTolerances",
        )?;

        // ── Inner KSP: strategy by problem size ──────────────────────────
        //
        // n_dof < 5k      → preonly + LU (sequential direct, exact)
        // 5k ≤ n_dof < 100k → preonly + LU(MUMPS) (parallel direct, exact)
        // n_dof ≥ 100k    → FGMRES(200) + ILU(2) (iterative, memory-efficient)
        //
        // MUMPS at 191k DOFs requires ~3–5 GB → OOM on cluster.
        // FGMRES + ILU(2) uses ~0.6 GB and is robust for nonlinear shell K_T.
        let mut ksp: ffi::KSP = std::ptr::null_mut();
        check(ffi::SNESGetKSP(snes, &mut ksp), "SNESGetKSP")?;
        let mut pc: ffi::PC = std::ptr::null_mut();
        check(ffi::KSPGetPC(ksp, &mut pc), "KSPGetPC")?;

        if n_dof >= FGMRES_THRESHOLD {
            // ── FGMRES + ILU(2): memory-efficient for large problems ──────
            check(
                ffi::KSPSetType(ksp, KSPFGMRES.as_ptr()),
                "KSPSetType(fgmres)",
            )?;
            check(
                ffi::KSPGMRESSetRestart(ksp, FGMRES_RESTART),
                "KSPGMRESSetRestart",
            )?;
            check(ffi::PCSetType(pc, PCILU.as_ptr()), "PCSetType(ilu)")?;
            check(
                ffi::PCFactorSetLevels(pc, ILU_FILL_LEVELS),
                "PCFactorSetLevels",
            )?;
            check(
                ffi::KSPSetTolerances(ksp, 1e-6, 1e-10, PETSC_INFINITY, 5000),
                "KSPSetTolerances",
            )?;
            if config.diagnostics {
                eprintln!(
                    "[nonlinear_static_solve] KSP selected: fgmres({})+ilu({}) (n_dof={})",
                    FGMRES_RESTART, ILU_FILL_LEVELS, n_dof
                );
            }
        } else {
            // ── preonly + LU(MUMPS): direct solve for small/medium problems ─
            check(
                ffi::KSPSetType(ksp, KSPPREONLY.as_ptr()),
                "KSPSetType(preonly)",
            )?;
            check(ffi::PCSetType(pc, PCLU.as_ptr()), "PCSetType(lu)")?;
            if n_dof >= LU_TO_GAMG_THRESHOLD {
                let _ = ffi::PCFactorSetMatSolverType(
                    pc,
                    b"mumps\0".as_ptr() as *const i8,
                );
            }
            // For direct solver, KSP converges in 1 step.
            check(
                ffi::KSPSetTolerances(ksp, 1e-8, 1e-12, PETSC_INFINITY, 10),
                "KSPSetTolerances",
            )?;
            if config.diagnostics {
                if n_dof >= LU_TO_GAMG_THRESHOLD {
                    eprintln!(
                        "[nonlinear_static_solve] KSP selected: preonly+lu(mumps) (n_dof={})",
                        n_dof
                    );
                } else {
                    eprintln!(
                        "[nonlinear_static_solve] KSP selected: preonly+lu (n_dof={})",
                        n_dof
                    );
                }
            }
        }

        // Backtracking line search. With the warm-start predictor (u_pred =
        // K_e⁻¹·f_ext), SNES starts from a point where the residual is the
        // small GL correction O(θ²). The Armijo bt search limits the Newton
        // step when the GL correction causes ||R|| to grow, ensuring convergence
        // for load steps where the full Newton step would otherwise overshoot.
        let mut linesearch: ffi::SNESLineSearch = std::ptr::null_mut();
        check(ffi::SNESGetLineSearch(snes, &mut linesearch), "SNESGetLineSearch")?;
        check(
            ffi::SNESLineSearchSetType(linesearch, SNESLINESEARCHBT.as_ptr()),
            "SNESLineSearchSetType(bt)",
        )?;
        check(
            ffi::SNESLineSearchSetDamping(linesearch, 1.0),
            "SNESLineSearchSetDamping",
        )?;

        // ── Allow runtime overrides ───────────────────────────────────────
        // e.g. -snes_type newtonal  → arc-length
        //      -snes_monitor        → print residual each iteration
        check(ffi::SNESSetFromOptions(snes), "SNESSetFromOptions")?;

        // ── Solve R(u) = 0 ────────────────────────────────────────────────
        // b = null (no RHS shift — R already includes -F_ext in the callback)
        check(
            ffi::SNESSolve(snes, std::ptr::null_mut(), x_vec.as_raw()),
            "SNESSolve",
        )?;

        // ── Diagnostics ───────────────────────────────────────────────────
        let mut reason: i32 = 0;
        check(
            ffi::SNESGetConvergedReason(snes, &mut reason),
            "SNESGetConvergedReason",
        )?;

        let mut its: i32 = 0;
        check(
            ffi::SNESGetIterationNumber(snes, &mut its),
            "SNESGetIterationNumber",
        )?;

        let mut fnorm: f64 = 0.0;
        check(
            ffi::SNESGetFunctionNorm(snes, &mut fnorm),
            "SNESGetFunctionNorm",
        )?;

        let mut linear_reason: i32 = 0;
        check(
            ffi::KSPGetConvergedReason(ksp, &mut linear_reason),
            "KSPGetConvergedReason",
        )?;

        let mut snes_linear_its: i32 = 0;
        check(
            ffi::SNESGetLinearSolveIterations(snes, &mut snes_linear_its),
            "SNESGetLinearSolveIterations",
        )?;

        let mut snes_func_evals: i32 = 0;
        check(
            ffi::SNESGetNumberFunctionEvals(snes, &mut snes_func_evals),
            "SNESGetNumberFunctionEvals",
        )?;

        let mut ksp_its: i32 = 0;
        check(
            ffi::KSPGetIterationNumber(ksp, &mut ksp_its),
            "KSPGetIterationNumber",
        )?;

        let mut ksp_rnorm: f64 = 0.0;
        check(
            ffi::KSPGetResidualNorm(ksp, &mut ksp_rnorm),
            "KSPGetResidualNorm",
        )?;

        if config.diagnostics {
            eprintln!(
                "[nonlinear_static_solve] SNES summary: reason={}, newton_its={}, linear_its_total={}, func_evals={}, |R|={:.3e}",
                reason,
                its,
                snes_linear_its,
                snes_func_evals,
                fnorm,
            );
            eprintln!(
                "[nonlinear_static_solve] KSP summary: reason={}, last_its={}, last_rnorm={:.3e}",
                linear_reason,
                ksp_its,
                ksp_rnorm,
            );
        }

        if reason <= 0 {
            eprintln!(
                "[nonlinear_static_solve] SNES diverged: snes_reason={}, ksp_reason={}, iterations={}, linear_its_total={}, func_evals={}, |R|={:.3e}",
                reason,
                linear_reason,
                its,
                snes_linear_its,
                snes_func_evals,
                fnorm,
            );
        }

        // ── Cleanup ───────────────────────────────────────────────────────
        check(ffi::SNESDestroy(&mut snes), "SNESDestroy")?;

        // ── Extract solution ──────────────────────────────────────────────
        let displacements = x_vec.to_vec()?;

        Ok(NonlinearStaticResult {
            displacements,
            iterations: its,
            residual_norm: fnorm,
            converged_reason: reason,
        })
    }
}

// ── Public COO-based entry point (for PyO3 binding) ───────────────────────────

/// Convenience wrapper that accepts COO triplets directly (bypasses
/// `MeshAssembler`) for callers that already have the tangent stiffness.
///
/// This is the version exposed to Python via PyO3.
/// For the full assembler-based path, use `nonlinear_static_solve` directly.
pub fn nonlinear_static_solve_coo(
    assembler: &MeshAssembler,
    f_ext_slice: &[f64],
    dirichlet_dofs: &[usize],
    config: NonlinearConfig,
) -> Result<NonlinearStaticResult, PetscError> {
    nonlinear_static_solve_with_guess(assembler, f_ext_slice, dirichlet_dofs, None, &config)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use aeroelast_core::assembly::{ElemType, MaterialSpec, MeshAssembler, MeshTopology};

    /// Helper: single MITC3 triangle mesh (patch test).
    ///
    /// Three nodes at (0,0,0), (1,0,0), (0,1,0) — one MITC3 element.
    /// 3 nodes × 6 DOFs = 18 DOFs total.
    fn make_single_triangle_assembler() -> MeshAssembler {
        let node_coords = vec![
            0.0, 0.0, 0.0, // node 0
            1.0, 0.0, 0.0, // node 1
            0.0, 1.0, 0.0, // node 2
        ];
        let connectivity = vec![vec![0usize, 1, 2]];
        let elem_types = vec![ElemType::Mitc3];

        let topology = MeshTopology::new(node_coords, connectivity, elem_types);
        let materials = vec![MaterialSpec::Isotropic {
            e: 200e9,
            nu: 0.3,
            rho: 7800.0,
            thickness: 0.01,
            shear_correction: 5.0 / 6.0,
            drilling_scale: 1.0,
        }];

        MeshAssembler::new(topology, materials)
    }

    /// Patch test: all DOFs clamped, zero load → zero displacement,
    /// SNES converges in 1 step (linear problem with trivial solution u=0).
    #[test]
    fn test_all_clamped_zero_load() {
        let assembler = make_single_triangle_assembler();
        let n_dof = assembler.dofs_count;
        let f_ext = vec![0.0f64; n_dof];
        let dirichlet_dofs: Vec<usize> = (0..n_dof).collect(); // all clamped

        let config = NonlinearConfig::default();
        let result = nonlinear_static_solve(&assembler, &f_ext, &dirichlet_dofs, &config)
            .expect("nonlinear solve failed");

        assert!(
            result.converged_reason > 0,
            "SNES diverged: reason={}, iters={}, fnorm={}",
            result.converged_reason,
            result.iterations,
            result.residual_norm
        );
        for (i, &u) in result.displacements.iter().enumerate() {
            assert!(
                u.abs() < 1e-10,
                "DOF {i}: expected 0.0, got {u} (all clamped, zero load)"
            );
        }
    }
}
