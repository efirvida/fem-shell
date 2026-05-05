/// Stress-stiffened FSI solver — incremental linearization with per-step K_G update.
///
/// Extends [`LinearElasticFsiSolver`] with a geometric stiffness term K_G that is
/// rebuilt from the converged displacement at the end of each time window:
///
///   K_eff^{n+1} = K + K_G(σ^n) + a₀·M + a₁·C
///
/// where σ^n is the membrane stress field recovered from u^n.
///
/// This is a *frozen-tangent* (linearized incremental) approach:
/// * ONE linear solve per time step — no Newton-Raphson inner iterations.
/// * ONE K_eff factorization per `kg_update_interval` converged steps.
/// * Converges to the correct nonlinear solution as Δt → 0.
///
/// # Feature gate
/// Compiled only with `--features fsi`.

use aeroelast_core::assembly::assembler::MeshAssembler;

use crate::petsc::elasticity::dynamic_newmark::{NewmarkCheckpoint, NewmarkStepper};
use crate::petsc::fsi::force_utils::{apply_cap, apply_ramp};
use crate::petsc::fsi::linear_elastic::{FsiConfig, FsiError, FsiInitialState, FsiResult, StepCallback};
use crate::petsc::fsi::setup;

// ── Solver struct ─────────────────────────────────────────────────────────────

/// FSI solver with geometric-stiffness (stress-stiffening) update.
///
/// Rebuilds `K_eff = K + K_G(σ) + a₀·M + a₁·C` every
/// `kg_update_interval` converged time windows using the current membrane
/// stress recovered from the displacement solution.
pub struct StressStiffenedFsiSolver {
    // ── Newmark structural integrator ────────────────────────────────────────
    stepper: NewmarkStepper,

    // ── preCICE coupling ─────────────────────────────────────────────────────
    config: FsiConfig,
    interface_coords: Vec<f64>,
    interface_dofs: Vec<usize>,
    mesh_dims: usize,

    // ── Optional restart state ───────────────────────────────────────────────
    initial_state: Option<FsiInitialState>,

    // ── Optional per-step callback ───────────────────────────────────────────
    step_callback: Option<StepCallback>,

    // ── Stress-stiffening fields ─────────────────────────────────────────────
    assembler: MeshAssembler,
    /// Reduced DOF indices for every free DOF: `free_dofs_i32[i]` = global DOF index.
    free_dofs_i32: Vec<i32>,
    /// Total DOF count in the **full** (unreduced) system.
    n_full_dofs: usize,
    /// Rebuild K_G every N converged steps (1 = every step).
    kg_update_interval: usize,
    /// Precomputed mapping: full K_G COO index → output index in reduced K sparsity
    /// (or -1 for entries outside the free-DOF set).  Eliminates per-timestep HashMap.
    kg_coo_map: Vec<i32>,
}

impl StressStiffenedFsiSolver {
    /// Create a new `StressStiffenedFsiSolver`.
    ///
    /// # Arguments
    /// Most arguments are identical to `LinearElasticFsiSolver::new`.
    ///
    /// * `assembler`          — mesh assembler used for stress recovery and K_G assembly
    /// * `free_dofs_i32`      — free (unrestrained) global DOF indices, ascending
    /// * `n_full_dofs`        — total DOF count in the unreduced system
    /// * `kg_update_interval` — rebuild K_G every N converged steps (≥ 1)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stepper: NewmarkStepper,
        config: FsiConfig,
        interface_coords: Vec<f64>,
        interface_dofs: Vec<usize>,
        mesh_dims: usize,
        assembler: MeshAssembler,
        free_dofs_i32: Vec<i32>,
        n_full_dofs: usize,
        kg_update_interval: usize,
    ) -> Self {
        // Precompute K_G COO → output index map (once, avoids per-timestep HashMap).
        let dummy_sigma: Vec<[f64; 3]> = vec![[0.0; 3]; assembler.topology.n_elems];
        let (kg_rows_full, kg_cols_full, _) = assembler.assemble_geometric_k(&dummy_sigma);
        let kg_coo_map = crate::petsc::fsi::setup::build_kg_coo_map(
            &kg_rows_full,
            &kg_cols_full,
            &free_dofs_i32,
            stepper.k_rows(),
            stepper.k_cols(),
        );

        Self {
            stepper,
            config,
            interface_coords,
            interface_dofs,
            mesh_dims,
            initial_state: None,
            step_callback: None,
            assembler,
            free_dofs_i32,
            n_full_dofs,
            kg_update_interval: kg_update_interval.max(1),
            kg_coo_map,
        }
    }

    /// Set the initial structural state for a restart.
    pub fn with_initial_state(mut self, state: FsiInitialState) -> Self {
        self.initial_state = Some(state);
        self
    }

    /// Register a per-step callback invoked after each converged time window.
    ///
    /// The callback signature is identical to [`LinearElasticFsiSolver`]:
    /// `(t, time_step, dt, u_red, v_red, a_red, force_mag, forces_iface)`.
    pub fn with_step_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn(f64, usize, f64, &[f64], &[f64], &[f64], f64, &[f64]) -> Result<(), FsiError>
            + Send
            + 'static,
    {
        self.step_callback = Some(Box::new(cb));
        self
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Expand a reduced-DOF vector to the full DOF vector.
    fn expand_u(&self, u_red: &[f64]) -> Vec<f64> {
        let mut u_full = vec![0.0f64; self.n_full_dofs];
        for (i, &dof) in self.free_dofs_i32.iter().enumerate() {
            u_full[dof as usize] = u_red[i];
        }
        u_full
    }

    /// Rebuild K_G from the current deformation state and update the stepper.
    ///
    /// Does nothing when the stress field is essentially zero (avoids a
    /// no-op refactorization at t ≈ 0 when loads are still ramping up).
    fn update_kg(&mut self, u_red: &[f64], time_step: usize) -> Result<(), FsiError> {
        if time_step % self.kg_update_interval != 0 {
            return Ok(());
        }

        let u_full = self.expand_u(u_red);

        // Recover element-centroid membrane stresses (z=0, membrane-only).
        let (sigma, _) = self.assembler.compute_stress_field(&u_full, 0.0, 0);

        // Skip K_G update when stresses are negligible (e.g. at t≈0).
        let max_s = sigma
            .iter()
            .flat_map(|s| s.iter().copied())
            .fold(0.0_f64, f64::max);
        if max_s <= 1e-20 {
            return Ok(());
        }

        // Convert to membrane 3-vector [σxx, σyy, σxy] (Voigt 6→3).
        // compute_stress_field returns [σxx, σyy, σzz, τxy, τyz, τzx],
        // for shell membrane: σzz=τyz=τzx=0, so index 3 = τxy = σxy.
        let sigma_m: Vec<[f64; 3]> = sigma.iter().map(|s| [s[0], s[1], s[3]]).collect();

        // Assemble full K_G COO values and accumulate via precomputed map.
        let (_, _, kg_vals_full) =
            self.assembler.assemble_geometric_k(&sigma_m);
        let kg_red = crate::petsc::fsi::setup::apply_kg_coo_map(
            &self.kg_coo_map,
            &kg_vals_full,
            self.stepper.k_rows().len(),
        );

        self.stepper
            .update_geometric_stiffness(&kg_red)
            .map_err(FsiError::StepperError)?;

        let kg_norm: f64 = kg_red.iter().map(|x| x * x).sum::<f64>().sqrt();
        log::info!(
            "StressStiffened step {time_step}: ||K_G||_F (reduced) = {kg_norm:.3e}"
        );

        Ok(())
    }

    // ── Main loop ─────────────────────────────────────────────────────────────

    /// Run the preCICE FSI coupling loop with stress-stiffening.
    ///
    /// Identical to `LinearElasticFsiSolver::run()` except that after each
    /// **converged** time window the geometric stiffness K_G is rebuilt and
    /// `K_eff` is refactorized.
    pub fn run(&mut self) -> Result<FsiResult, FsiError> {
        let mut participant = precice::Participant::new(
            &self.config.participant_name,
            &self.config.config_file,
            0, // solver rank
            1, // communicator size
        )?;

        let n_vertices = self.interface_coords.len() / self.mesh_dims.max(1);
        let mut vertex_ids = vec![0i32; n_vertices];
        participant.set_mesh_vertices(
            &self.config.coupling_mesh,
            &self.interface_coords,
            &mut vertex_ids,
        )?;

        participant.initialize()?;
        let mut dt = participant.get_max_time_step_size()?;

        if let Some(ref state) = self.initial_state {
            self.stepper.set_state(&state.u, &state.v, &state.a, state.t);
        }

        let n_dofs = self.stepper.n_dofs();
        let n_data = n_vertices * self.mesh_dims;
        let mut checkpoint: Option<NewmarkCheckpoint> = None;
        let mut result = FsiResult::default();

        while participant.is_coupling_ongoing()? {
            // ── Save checkpoint before implicit coupling iteration ─────────────
            if participant.requires_writing_checkpoint()? {
                checkpoint = Some(self.stepper.checkpoint());
            }

            // ── Read forces from preCICE ───────────────────────────────────────
            let mut forces = vec![0.0f64; n_data];
            participant.read_data(
                &self.config.coupling_mesh,
                &self.config.read_data,
                &vertex_ids,
                dt,
                &mut forces,
            )?;

            // ── Force pre-processing ───────────────────────────────────────────
            let t = self.stepper.current_time();
            apply_ramp(&mut forces, t, self.config.ramp_time);
            if let Some(max_f) = self.config.force_max {
                apply_cap(&mut forces, max_f, self.mesh_dims);
            }

            // ── Scatter interface forces → global DOF vector ───────────────────
            let mut f_global = vec![0.0f64; n_dofs];
            for (local_idx, &global_dof) in self.interface_dofs.iter().enumerate() {
                if local_idx < forces.len() && global_dof < n_dofs {
                    f_global[global_dof] += forces[local_idx];
                }
            }

            // ── Advance structural state ───────────────────────────────────────
            let step_t = self.stepper.step(&f_global, dt)?.t;

            // ── Gather interface displacements ─────────────────────────────────
            let disp_interface: Vec<f64> = self
                .interface_dofs
                .iter()
                .map(|&dof| {
                    if dof < self.stepper.n_dofs() {
                        self.stepper.current_u()[dof]
                    } else {
                        0.0
                    }
                })
                .collect();

            // ── Write displacements to preCICE ────────────────────────────────
            participant.write_data(
                &self.config.coupling_mesh,
                &self.config.write_data,
                &vertex_ids,
                &disp_interface,
            )?;

            participant.advance(dt)?;

            // ── Implicit coupling: restore or commit ──────────────────────────
            if participant.requires_reading_checkpoint()? {
                match checkpoint {
                    Some(ref cp) => self.stepper.restore(cp),
                    None => {
                        return Err(FsiError::PreciceError(
                            "preCICE requires checkpoint restore but no checkpoint was saved"
                                .to_string(),
                        ))
                    }
                }
            } else {
                // Converged time window — overwrite final state (no history accumulation).
                result.u_final = self.stepper.current_u().to_vec();
                result.v_final = self.stepper.current_v().to_vec();
                result.a_final = self.stepper.current_a().to_vec();
                result.times.push(step_t);

                let time_step = result.times.len(); // 1-based

                // ── K_G update ────────────────────────────────────────────────
                // Clone u to release the immutable borrow before the mutable update_kg call.
                let u_snapshot = self.stepper.current_u().to_vec();
                self.update_kg(&u_snapshot, time_step)?;

                // ── Per-step callback ─────────────────────────────────────────
                if let Some(ref cb) = self.step_callback {
                    let force_mag = forces.iter().map(|x| x * x).sum::<f64>().sqrt();
                    cb(
                        step_t,
                        time_step,
                        dt,
                        self.stepper.current_u(),
                        self.stepper.current_v(),
                        self.stepper.current_a(),
                        force_mag,
                        &forces,
                    )?;
                }

                dt = participant.get_max_time_step_size()?;
            }
        }

        participant.finalize()?;
        Ok(result)
    }

    /// Number of free DOFs in the structural system.
    pub fn n_dofs(&self) -> usize {
        self.stepper.n_dofs()
    }
}
