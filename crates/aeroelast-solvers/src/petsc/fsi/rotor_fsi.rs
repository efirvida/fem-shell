/// Co-rotational FSI solver for rotating structures (rotor blades).
///
/// Implements the full preCICE coupling loop for a rotor in a rotating reference
/// frame, including:
/// - Rodrigues coordinate transforms (forces: global → rotating, disps: rotating → global)
/// - Centrifugal, Coriolis, and Euler inertial body forces on the RHS
/// - Spin-softening K_SP diagonal update on significant Δω
/// - Geometric stiffness K_G update at configurable intervals
/// - Four angular-velocity provider modes: Constant, Ramped, Computed, RampedComputed
/// - Checkpoint/restore for preCICE implicit IQN-ILS coupling
///
/// # Feature gate
/// Compiled only with `--features fsi`.

use aeroelast_core::assembly::assembler::MeshAssembler;

use crate::petsc::elasticity::dynamic_newmark::{NewmarkCheckpoint, NewmarkStepper};
use crate::petsc::fsi::force_utils::{apply_cap, apply_ramp};
use crate::petsc::fsi::linear_elastic::{FsiConfig, FsiError, FsiInitialState, FsiResult};
use crate::petsc::fsi::rotor_physics::{
    OmegaCheckpoint, OmegaProvider, PerformanceCoefficients, RotorTransforms, build_ksp_vals,
    compute_centrifugal_force, compute_coriolis_force, compute_euler_force, compute_gravity_force,
    compute_performance_coefficients, compute_thrust, compute_torque,
};

// ── Step callback type ─────────────────────────────────────────────────────────

/// Per-step callback for the rotor FSI solver.
///
/// Invoked once per **converged** time window with the full rotor state.
///
/// Arguments:
/// `(t, time_step, dt, u_red, v_red, a_red, force_mag,
///   forces_iface_global, omega_state, alpha_state, theta, tau_aero, perf)`
pub type RotorStepCallback = Box<
    dyn Fn(
            f64,                     // t: converged time
            usize,                   // time_step: 1-based step index
            f64,                     // dt: time window width
            &[f64],                  // u_red: displacement (reduced DOFs)
            &[f64],                  // v_red: velocity (reduced DOFs)
            &[f64],                  // a_red: acceleration (reduced DOFs)
            f64,                     // force_mag: ||F_aero|| Frobenius
            &[f64],                  // forces_iface_global: aero forces, global frame
            f64,                     // omega_state: dynamic state after convergence
            f64,                     // alpha_state: dynamic state after convergence
            f64,                     // theta: cumulative rotation angle
            f64,                     // tau_aero: aerodynamic torque (scalar, about axis)
            PerformanceCoefficients, // ct, cp, cq, tsr
        ) -> Result<(), FsiError>
        + Send,
>;

// ── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the rotor FSI solver.
#[derive(Debug, Clone)]
pub struct RotorFsiConfig {
    /// Base preCICE FSI configuration (participant, mesh, read/write data names).
    pub fsi: FsiConfig,

    // ── Rotor geometry ────────────────────────────────────────────────────────
    /// Rotation axis unit vector (will be normalized if not already).
    pub rotation_axis: [f64; 3],
    /// Rotation center in the global frame.
    pub rotation_center: [f64; 3],

    // ── Inertial forces ───────────────────────────────────────────────────────
    /// Gravity vector in the GLOBAL (inertial) frame [m/s²]. Use `[0,0,0]` to disable.
    pub gravity: [f64; 3],
    /// Enable centrifugal forces on the RHS.
    pub include_centrifugal: bool,
    /// Enable Coriolis forces on the RHS.
    pub include_coriolis: bool,
    /// Enable Euler forces on the RHS (angular-acceleration term).
    pub include_euler: bool,

    // ── Stiffness updates ─────────────────────────────────────────────────────
    /// Enable geometric stiffness K_G updates (requires `assembler` to be `Some`).
    pub include_kg: bool,
    /// Rebuild K_G every N converged steps. `0` and `1` both mean every step.
    pub kg_update_interval: usize,
    /// Enable spin-softening K_SP diagonal update.
    pub include_ksp: bool,
    /// Rebuild K_SP when |Δω| exceeds this threshold [rad/s].
    pub ksp_omega_threshold: f64,

    // ── DOF layout ────────────────────────────────────────────────────────────
    /// DOFs per FEM node (typically 6 for shells).
    pub dofs_per_node: usize,

    // ── Performance coefficients ──────────────────────────────────────────────
    /// Fluid (air) density [kg/m³].
    pub fluid_density: f64,
    /// Freestream flow velocity [m/s].
    pub flow_velocity: f64,
    /// Rotor outer radius [m].
    pub rotor_radius: f64,

    // ── Optional GlobalSolidMesh for ω communication to the CFD solver ─────────
    /// preCICE mesh name for the GlobalSolidMesh that communicates ω.
    /// `None` disables this feature.
    pub omega_mesh_name: Option<String>,
    /// Data name to write ω to on the GlobalSolidMesh.
    pub omega_write_data: Option<String>,
    /// Single vertex coordinate for the GlobalSolidMesh (typically the rotation center).
    pub omega_vertex_coord: Option<[f64; 3]>,
}

impl RotorFsiConfig {
    /// Return the gravity magnitude squared.
    fn gravity_active(&self) -> bool {
        self.gravity.iter().map(|x| x * x).sum::<f64>() > 1e-24
    }

    /// Normalize the K_G cadence so legacy `0` means "every converged step".
    fn normalized_kg_update_interval(&self) -> usize {
        self.kg_update_interval.max(1)
    }

    /// Whether K_G should be rebuilt on this 1-based converged step index.
    fn should_update_kg_on_step(&self, time_step: usize) -> bool {
        time_step % self.normalized_kg_update_interval() == 0
    }
}

// ── Solver struct ──────────────────────────────────────────────────────────────

/// Co-rotational FSI solver for rotating blades.
///
pub struct RotorFsiSolver {
    // ── Newmark structural integrator ─────────────────────────────────────────
    stepper: NewmarkStepper,

    // ── Configuration ─────────────────────────────────────────────────────────
    config: RotorFsiConfig,

    // ── preCICE interface ─────────────────────────────────────────────────────
    /// Flat interface node reference coordinates in the rotating frame, length `n_iface * 3`.
    ///
    /// Numerically these match the initial global coordinates at `t = 0`, but the
    /// values are treated as the co-rotational reference geometry `X0` when
    /// evaluating torques and deformed positions.
    interface_coords_global: Vec<f64>,
    /// Indices into the REDUCED DOF vector for each interface component
    /// (length `n_iface * dofs_dim`, typically 3 per node).
    interface_dofs: Vec<usize>,
    /// Number of spatial dimensions on the coupling mesh (3 for 3D).
    mesh_dims: usize,
    // ── Coordinate transforms ─────────────────────────────────────────────────
    transforms: RotorTransforms,

    // ── Angular velocity ──────────────────────────────────────────────────────
    omega_provider: OmegaProvider,
    /// Cumulative rotation angle θ = ∫ω dt [rad].
    theta: f64,

    // ── Inertial forces: full-mesh nodal data ─────────────────────────────────
    /// Reference (undeformed) coordinates of ALL structural nodes in the
    /// ROTATING frame, flat `[x0,y0,z0,…]`, length `n_all_nodes * 3`.
    all_node_coords: Vec<f64>,
    /// Per-node scalar lumped masses, length `n_all_nodes`.
    all_node_masses: Vec<f64>,

    // ── Matrix updates ────────────────────────────────────────────────────────
    /// Optional mesh assembler for K_G computation.
    assembler: Option<MeshAssembler>,
    /// Lumped mass diagonal in the FULL (unreduced) DOF space, length `n_full_dofs`.
    m_lumped_full: Vec<f64>,
    /// Free global DOF indices (ascending), length `n_reduced_dofs`.
    free_dofs_i32: Vec<i32>,
    /// Lookup: `full_to_red[global_dof]` = reduced index, or -1 if constrained.
    full_to_red: Vec<i32>,
    /// Total DOF count in the full (unreduced) system.
    n_full_dofs: usize,
    /// COO row indices of the REDUCED stiffness matrix (for K_SP and K_G expansion).
    k_rows: Vec<i32>,
    /// COO column indices of the REDUCED stiffness matrix.
    k_cols: Vec<i32>,
    /// Precomputed mapping: full K_G COO index → output index in reduced K sparsity
    /// (or -1 for entries outside the free-DOF set). Computed once in `new()`,
    /// eliminates all per-timestep HashMap allocations in `update_geometric_stiffness`.
    kg_coo_map: Vec<i32>,
    /// ω at the last K_SP rebuild (to detect Δω > threshold).
    omega_at_last_ksp: f64,

    // ── Optional restart state ────────────────────────────────────────────────
    initial_state: Option<FsiInitialState>,
    /// Initial cumulative angle for restarts.
    initial_theta: f64,

    // ── Optional per-step callback ────────────────────────────────────────────
    step_callback: Option<RotorStepCallback>,
}

impl RotorFsiSolver {
    /// Create a new `RotorFsiSolver`.
    ///
    /// # Arguments
    ///
    /// * `stepper`                 — pre-assembled Newmark integrator (assembled with K, M, C)
    /// * `config`                  — rotor FSI configuration
    /// * `interface_coords_global` — flat interface node reference coords (`X0`) in the
    ///   rotating frame, `n_iface * 3`
    /// * `interface_dofs`          — reduced DOF indices for each interface component
    /// * `mesh_dims`               — coupling mesh spatial dimension (3)
    /// * `omega_provider`          — angular-velocity provider (Constant/Ramped/Computed/…)
    /// * `all_node_coords`         — ALL structural node reference coords (rotating frame), `n * 3`
    /// * `all_node_masses`         — per-node scalar masses, length `n_all_nodes`
    /// * `m_lumped_full`           — lumped mass diagonal in the full DOF space
    /// * `k_rows`, `k_cols`        — COO sparsity of the reduced K (for K_SP expansion)
    /// * `free_dofs_i32`           — sorted free global DOF indices
    /// * `n_full_dofs`             — total DOF count (unreduced)
    /// * `assembler`               — mesh assembler for K_G; `None` to disable K_G updates
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stepper: NewmarkStepper,
        config: RotorFsiConfig,
        interface_coords_global: Vec<f64>,
        interface_dofs: Vec<usize>,
        mesh_dims: usize,
        omega_provider: OmegaProvider,
        all_node_coords: Vec<f64>,
        all_node_masses: Vec<f64>,
        m_lumped_full: Vec<f64>,
        k_rows: Vec<i32>,
        k_cols: Vec<i32>,
        free_dofs_i32: Vec<i32>,
        n_full_dofs: usize,
        assembler: Option<MeshAssembler>,
    ) -> Self {
        // Build full→reduced lookup.
        let mut full_to_red = vec![-1i32; n_full_dofs];
        for (i, &dof) in free_dofs_i32.iter().enumerate() {
            full_to_red[dof as usize] = i as i32;
        }

        // Precompute K_G COO → output index map (once, avoids per-timestep HashMap).
        let kg_coo_map = if let Some(ref asm) = assembler {
            let dummy = vec![[0.0f64; 3]; asm.topology.n_elems];
            let (rows, cols, _) = asm.assemble_geometric_k(&dummy);
            crate::petsc::fsi::setup::build_kg_coo_map(
                &rows,
                &cols,
                &free_dofs_i32,
                &k_rows,
                &k_cols,
            )
        } else {
            Vec::new()
        };

        let axis = config.rotation_axis;
        let center = config.rotation_center;
        let transforms = RotorTransforms::new(axis, center);
        let initial_omega = omega_provider.initial_omega();

        Self {
            stepper,
            config,
            interface_coords_global,
            interface_dofs,
            mesh_dims,
            transforms,
            omega_provider,
            theta: 0.0,
            all_node_coords,
            all_node_masses,
            assembler,
            m_lumped_full,
            free_dofs_i32,
            full_to_red,
            n_full_dofs,
            k_rows,
            k_cols,
            kg_coo_map,
            omega_at_last_ksp: initial_omega,
            initial_state: None,
            initial_theta: 0.0,
            step_callback: None,
        }
    }

    /// Set the initial structural state for a restart.
    pub fn with_initial_state(mut self, state: FsiInitialState, theta: f64) -> Self {
        self.initial_state = Some(state);
        self.initial_theta = theta;
        self
    }

    /// Register a per-step callback invoked after each converged time window.
    pub fn with_step_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn(
            f64,
            usize,
            f64,
            &[f64],
            &[f64],
            &[f64],
            f64,
            &[f64],
            f64,
            f64,
            f64,
            f64,
            PerformanceCoefficients,
        ) -> Result<(), FsiError>
            + Send
            + 'static,
    {
        self.step_callback = Some(Box::new(cb));
        self
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Expand a reduced-DOF vector to the full DOF space (zeros at constrained DOFs).
    fn expand_to_full(&self, v_red: &[f64]) -> Vec<f64> {
        let mut v_full = vec![0.0f64; self.n_full_dofs];
        for (i, &dof) in self.free_dofs_i32.iter().enumerate() {
            v_full[dof as usize] = v_red[i];
        }
        v_full
    }

    /// Extract per-node translational components from a full-DOF vector.
    ///
    /// Returns a flat array `[vx0,vy0,vz0,…]` of length `n_all_nodes * 3`.
    fn extract_node_translations(&self, v_full: &[f64]) -> Vec<f64> {
        let n_nodes = self.all_node_masses.len();
        let dofs = self.config.dofs_per_node;
        let mut out = vec![0.0f64; n_nodes * 3];
        for i in 0..n_nodes {
            for j in 0..3 {
                let gdof = i * dofs + j;
                if gdof < v_full.len() {
                    out[i * 3 + j] = v_full[gdof];
                }
            }
        }
        out
    }

    /// Scatter flat node forces (n_nodes × 3) into the reduced-DOF force vector.
    ///
    /// Only free DOFs receive contributions; constrained DOFs are silently skipped.
    fn scatter_node_forces(&self, f_node: &[f64], f_global: &mut [f64]) {
        let n_nodes = self.all_node_masses.len();
        let dofs = self.config.dofs_per_node;
        for i in 0..n_nodes {
            let nb = i * 3;
            for j in 0..3 {
                let gdof = i * dofs + j;
                if gdof < self.full_to_red.len() {
                    let red = self.full_to_red[gdof];
                    if red >= 0 {
                        f_global[red as usize] += f_node[nb + j];
                    }
                }
            }
        }
    }

    /// Build spin-softening K_SP values (K sparsity pattern length) and apply to stepper.
    fn apply_ksp(&mut self, omega: f64) -> Result<(), FsiError> {
        if !self.config.include_ksp {
            return Ok(());
        }
        let ksp = build_ksp_vals(
            &self.m_lumped_full,
            &self.transforms.axis,
            omega,
            self.config.dofs_per_node,
            &self.k_rows,
            &self.k_cols,
            &self.free_dofs_i32,
            self.n_full_dofs,
        );
        self.stepper
            .update_spin_softening(&ksp)
            .map_err(FsiError::StepperError)?;
        self.omega_at_last_ksp = omega;
        Ok(())
    }

    /// Rebuild K_G from centrifugal prestress when the configured cadence says it is due.
    ///
    /// Calls `stepper.set_initial_geometric_stiffness` (writes to `kg_base_vals`) so the
    /// centrifugal prestress is always included in K_eff regardless of runtime deformation.
    fn update_kg_if_needed(&mut self, omega: f64, time_step: usize) -> Result<(), FsiError> {
        if !self.config.include_kg {
            return Ok(());
        }
        if !self.config.should_update_kg_on_step(time_step) {
            return Ok(());
        }
        let asm = match &self.assembler {
            Some(a) => a,
            None => return Ok(()),
        };

        // Build rho_per_elem from assembler materials.
        use aeroelast_core::assembly::assembler::MaterialSpec;
        let rho_per_elem: Vec<f64> = asm
            .materials
            .iter()
            .map(|m| match m {
                MaterialSpec::Isotropic { rho, .. } => *rho,
                MaterialSpec::Composite { mass_per_area, .. } => *mass_per_area,
                MaterialSpec::PlaneStress { rho, .. } => *rho,
                MaterialSpec::Solid3D { rho, .. } => *rho,
            })
            .collect();

        let axis = self.config.rotation_axis;
        let center = self.config.rotation_center;
        let (_, _, kg_vals_full) = asm.assemble_centrifugal_k(omega, axis, center, &rho_per_elem);
        let kg_red = crate::petsc::fsi::setup::apply_kg_coo_map(
            &self.kg_coo_map,
            &kg_vals_full,
            self.k_rows.len(),
        );

        self.stepper
            .set_initial_geometric_stiffness(&kg_red)
            .map_err(FsiError::StepperError)?;

        let kg_norm: f64 = kg_red.iter().map(|x| x * x).sum::<f64>().sqrt();
        log::info!(
            "RotorFsi: K_G rebuilt at step {time_step}, ω={omega:.4} rad/s, ||K_G||_F = {kg_norm:.3e}"
        );
        Ok(())
    }

    // ── Main coupling loop ────────────────────────────────────────────────────

    /// Run the preCICE co-rotational FSI coupling loop.
    ///
    /// # Returns
    /// [`FsiResult`] containing the time history of displacement, velocity, and
    /// acceleration in the ROTATING frame (reduced DOF space).
    pub fn run(&mut self) -> Result<FsiResult, FsiError> {
        // ── Initialize preCICE ────────────────────────────────────────────────
        let mut participant = precice::Participant::new(
            &self.config.fsi.participant_name,
            &self.config.fsi.config_file,
            0,
            1,
        )?;

        let n_vertices = self.interface_coords_global.len() / self.mesh_dims.max(1);
        let mut vertex_ids = vec![0i32; n_vertices];
        participant.set_mesh_vertices(
            &self.config.fsi.coupling_mesh,
            &self.interface_coords_global,
            &mut vertex_ids,
        )?;

        // Optional GlobalSolidMesh for ω communication.
        let omega_vertex_ids: Option<Vec<i32>> =
            if let (Some(mesh), Some(coord)) = (
                &self.config.omega_mesh_name,
                &self.config.omega_vertex_coord,
            ) {
                let mut ids = vec![0i32; 1];
                participant.set_mesh_vertices(mesh, coord, &mut ids)?;
                Some(ids)
            } else {
                None
            };

        let restart_time = self.initial_state.as_ref().map(|state| state.t).unwrap_or(0.0);

        // Write initial omega before initialize() when preCICE requires it.
        // This satisfies the initialize="true" exchange for AngularVelocity.
        if participant.requires_initial_data()? {
            if let (Some(mesh), Some(wdata), Some(ids)) = (
                &self.config.omega_mesh_name,
                &self.config.omega_write_data,
                &omega_vertex_ids,
            ) {
                let omega_init = self.omega_provider.get(restart_time).0;
                participant.write_data(mesh, wdata, ids, &[omega_init])?;
            }
        }

        participant.initialize()?;
        let mut dt = participant.get_max_time_step_size()?;

        // ── Optional restart ──────────────────────────────────────────────────
        if let Some(ref state) = self.initial_state {
            self.stepper.set_state(&state.u, &state.v, &state.a, state.t);
            self.theta = self.initial_theta;
        }

        // ── Initial K_SP build ────────────────────────────────────────────────
        let omega_init = self.omega_provider.get(restart_time).0;
        self.apply_ksp(omega_init)?;

        let n_dofs = self.stepper.n_dofs();
        let n_data = n_vertices * self.mesh_dims;

        // Checkpoint storage.
        let mut newmark_cp: Option<NewmarkCheckpoint> = None;
        let mut omega_cp: Option<OmegaCheckpoint> = None;
        let mut theta_cp: f64 = self.theta;

        let mut result = FsiResult::default();

        // ── Coupling loop ─────────────────────────────────────────────────────
        while participant.is_coupling_ongoing()? {
            // ── Save checkpoint ───────────────────────────────────────────────
            if participant.requires_writing_checkpoint()? {
                newmark_cp = Some(self.stepper.checkpoint());
                omega_cp = Some(self.omega_provider.checkpoint());
                theta_cp = self.theta;
            }

            let t = self.stepper.current_time();
            let window_kin = self.omega_provider.kinematics_over_step(t, dt);
            let omega_step = window_kin.omega_step;
            let alpha_step = window_kin.alpha_step;
            let theta_target = self.theta + window_kin.theta_increment;

            // ── Read forces from preCICE (global frame) ───────────────────────
            let mut forces_global = vec![0.0f64; n_data];
            participant.read_data(
                &self.config.fsi.coupling_mesh,
                &self.config.fsi.read_data,
                &vertex_ids,
                dt,
                &mut forces_global,
            )?;

            // ── Transform aero forces to rotating frame ───────────────────────
            let mut forces_local = self
                .transforms
                .forces_to_rotating(&forces_global, theta_target);

            // ── Force pre-processing (ramp + cap in rotating frame) ───────────
            apply_ramp(&mut forces_local, t, self.config.fsi.ramp_time);
            if let Some(max_f) = self.config.fsi.force_max {
                apply_cap(&mut forces_local, max_f, self.mesh_dims);
            }

            // ── Assemble global reduced force vector ──────────────────────────
            let mut f_red = vec![0.0f64; n_dofs];

            // Scatter aero forces at interface DOFs.
            for (li, &rdof) in self.interface_dofs.iter().enumerate() {
                if li < forces_local.len() && rdof < n_dofs {
                    f_red[rdof] += forces_local[li];
                }
            }

            // Inertial forces (all nodes, rotating frame).
            if self.config.include_centrifugal {
                let f_cf = compute_centrifugal_force(
                    &self.all_node_coords,
                    &self.all_node_masses,
                    &self.transforms.axis,
                    &self.transforms.center,
                    omega_step,
                );
                self.scatter_node_forces(&f_cf, &mut f_red);
            }

            if self.config.include_coriolis {
                let v_full = self.expand_to_full(self.stepper.current_v());
                let v_nodes = self.extract_node_translations(&v_full);
                let f_cor = compute_coriolis_force(
                    &v_nodes,
                    &self.all_node_masses,
                    &self.transforms.axis,
                    omega_step,
                );
                self.scatter_node_forces(&f_cor, &mut f_red);
            }

            if self.config.include_euler && alpha_step.abs() > 1e-14 {
                // Euler evaluated at deformed coords X₀ + u.
                let u_full = self.expand_to_full(self.stepper.current_u());
                let deformed: Vec<f64> = self
                    .all_node_coords
                    .iter()
                    .enumerate()
                    .map(|(k, &x0)| {
                        let node = k / 3;
                        let comp = k % 3;
                        let gdof = node * self.config.dofs_per_node + comp;
                        if gdof < u_full.len() {
                            x0 + u_full[gdof]
                        } else {
                            x0
                        }
                    })
                    .collect();
                let f_euler = compute_euler_force(
                    &deformed,
                    &self.all_node_masses,
                    &self.transforms.axis,
                    &self.transforms.center,
                    alpha_step,
                );
                self.scatter_node_forces(&f_euler, &mut f_red);
            }

            if self.config.gravity_active() {
                let g_rot =
                    self.transforms.gravity_to_rotating(self.config.gravity, theta_target);
                let f_g = compute_gravity_force(&self.all_node_masses, &g_rot);
                self.scatter_node_forces(&f_g, &mut f_red);
            }

            // ── Advance structural state ──────────────────────────────────────
            // step() returns only the updated time; the structural state (u,v,a)
            // is stored in self.stepper and accessed via current_u/v/a() below.
            let step_t = self.stepper.step(&f_red, dt)?.t;

            // ── Gather interface displacements (rotating frame) ───────────────
            let disp_iface_local: Vec<f64> = self
                .interface_dofs
                .iter()
                .map(|&rdof| {
                    if rdof < self.stepper.n_dofs() {
                        self.stepper.current_u()[rdof]
                    } else {
                        0.0
                    }
                })
                .collect();

            // ── Transform displacements to global frame ───────────────────────
            let disp_iface_global = self
                .transforms
                .disps_to_inertial(&disp_iface_local, theta_target);

            // ── Write displacements to preCICE ────────────────────────────────
            participant.write_data(
                &self.config.fsi.coupling_mesh,
                &self.config.fsi.write_data,
                &vertex_ids,
                &disp_iface_global,
            )?;

            // ── Write ω to GlobalSolidMesh (if used) ──────────────────────────
            if let (Some(ref ids), Some(ref mesh), Some(ref wdata)) = (
                &omega_vertex_ids,
                &self.config.omega_mesh_name,
                &self.config.omega_write_data,
            ) {
                participant.write_data(mesh, wdata, ids, &[omega_step])?;
            }

            participant.advance(dt)?;

            // ── Implicit coupling: restore or commit ──────────────────────────
            if participant.requires_reading_checkpoint()? {
                match (newmark_cp.as_ref(), omega_cp.as_ref()) {
                    (Some(ncp), Some(ocp)) => {
                        self.stepper.restore(ncp);
                        self.omega_provider.restore(ocp);
                        self.theta = theta_cp;
                    }
                    _ => {
                        return Err(FsiError::PreciceError(
                            "preCICE requires checkpoint restore but no checkpoint was saved"
                                .to_string(),
                        ))
                    }
                }
            } else {
                // ── Converged time window ─────────────────────────────────────
                // 1. Commit the converged target angle for this time window.
                self.theta = theta_target;

                // 2. Compute aerodynamic torque (for ω dynamics and logging).
                //    Use interface reference coords + displacements in the ROTATING frame.
                let n_iface = self.interface_dofs.len() / self.mesh_dims.max(1);
                let iface_coords_rot: Vec<f64> = self
                    .interface_coords_global
                    .iter()
                    .take(n_iface * 3)
                    .copied()
                    .collect();
                let iface_disps_rot: Vec<f64> = {
                    let mut buf = vec![0.0f64; n_iface * 3];
                    for k in 0..n_iface * 3 {
                        let rdof = self.interface_dofs.get(k).copied().unwrap_or(0);
                        buf[k] = if rdof < self.stepper.n_dofs() {
                            self.stepper.current_u()[rdof]
                        } else {
                            0.0
                        };
                    }
                    buf
                };
                let (_, tau_aero) = compute_torque(
                    &iface_coords_rot,
                    &iface_disps_rot,
                    &forces_local[..n_iface * 3],
                    &self.transforms.axis,
                    &self.transforms.center,
                );

                // 3. Gravity torque in the rotating frame.
                //    τ_gravity = torque from gravity body forces about the rotation axis.
                //    Needed so that ω dynamics use τ_driving = τ_aero + τ_gravity
                //    (gravitational sag changes the driving torque; centrifugal/Coriolis/Euler
                //    are fictitious forces and must NOT be included).
                let tau_gravity = if self.config.gravity_active() {
                    let g_rot = self.transforms.gravity_to_rotating(self.config.gravity, self.theta);
                    let f_grav = compute_gravity_force(&self.all_node_masses, &g_rot);
                    let u_full = self.expand_to_full(self.stepper.current_u());
                    let u_nodes = self.extract_node_translations(&u_full);
                    // Torque about axis at the deformed nodal positions X0 + u.
                    let (_, tau_g) = compute_torque(
                        &self.all_node_coords,
                        &u_nodes,
                        &f_grav,
                        &self.transforms.axis,
                        &self.transforms.center,
                    );
                    tau_g
                } else {
                    0.0
                };

                // 4. Update ω from driving torque = τ_aero + τ_gravity.
                //    (excludes centrifugal, Coriolis, Euler — those are fictitious forces)
                self.omega_provider.update_from_torque(tau_aero + tau_gravity, dt, t + dt);

                let (omega_new, _) = self.omega_provider.get(t + dt);
                let time_step = result.times.len() + 1; // 1-based, before push

                // 5. Rebuild K_SP if |Δω| > threshold.
                if (omega_new - self.omega_at_last_ksp).abs() > self.config.ksp_omega_threshold {
                    self.apply_ksp(omega_new)?;
                }

                // 6. Rebuild K_G from centrifugal prestress on the configured cadence.
                self.update_kg_if_needed(omega_new, time_step)?;

                // 7. Overwrite final state (no history accumulation — O(n_dofs) RAM).
                result.u_final = self.stepper.current_u().to_vec();
                result.v_final = self.stepper.current_v().to_vec();
                result.a_final = self.stepper.current_a().to_vec();
                result.times.push(step_t);

                // 8. Performance coefficients.
                let thrust = compute_thrust(&forces_global, &self.transforms.axis);
                let power_aero = tau_aero * omega_step;
                let perf = compute_performance_coefficients(
                    thrust,
                    power_aero,
                    tau_aero,
                    omega_step,
                    self.config.rotor_radius,
                    self.config.fluid_density,
                    self.config.flow_velocity,
                );

                let (omega_state, alpha_state) = self.omega_provider.get(t + dt);

                log::info!(
                    "RotorFsi step {time_step}: t={:.4} θ={:.4}rad ω_window={omega_step:.4}rad/s ω_state={omega_state:.4}rad/s \
                     τ_aero={tau_aero:.3e}N·m τ_grav={tau_gravity:.3e}N·m  Ct={:.4} Cp={:.4} TSR={:.3}",
                    step_t, self.theta, perf.ct, perf.cp, perf.tsr
                );

                // 9. Per-step callback.
                if let Some(ref cb) = self.step_callback {
                    let force_mag = forces_global.iter().map(|x| x * x).sum::<f64>().sqrt();
                    cb(
                        step_t,
                        time_step,
                        dt,
                        self.stepper.current_u(),
                        self.stepper.current_v(),
                        self.stepper.current_a(),
                        force_mag,
                        &forces_global,
                        omega_state,
                        alpha_state,
                        self.theta,
                        tau_aero,
                        perf,
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

    /// Number of interface nodes registered with the coupling mesh.
    pub fn n_interface_nodes(&self) -> usize {
        self.interface_coords_global.len() / self.mesh_dims.max(1)
    }

    /// Current cumulative rotation angle θ [rad].
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Current angular velocity ω [rad/s] from the provider.
    pub fn omega(&self) -> f64 {
        self.omega_provider.get(self.stepper.current_time()).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config(kg_update_interval: usize) -> RotorFsiConfig {
        RotorFsiConfig {
            fsi: FsiConfig {
                participant_name: "Solid".to_string(),
                config_file: "precice-config.xml".to_string(),
                coupling_mesh: "Solid-Mesh".to_string(),
                write_data: "Displacement".to_string(),
                read_data: "Force".to_string(),
                ramp_time: 0.0,
                force_max: None,
            },
            rotation_axis: [0.0, 0.0, 1.0],
            rotation_center: [0.0, 0.0, 0.0],
            gravity: [0.0, 0.0, 0.0],
            include_centrifugal: true,
            include_coriolis: false,
            include_euler: false,
            include_kg: true,
            kg_update_interval,
            include_ksp: true,
            ksp_omega_threshold: 1e-4,
            dofs_per_node: 6,
            fluid_density: 1.225,
            flow_velocity: 10.0,
            rotor_radius: 1.0,
            omega_mesh_name: None,
            omega_write_data: None,
            omega_vertex_coord: None,
        }
    }

    #[test]
    fn kg_update_interval_zero_means_every_step() {
        let cfg = dummy_config(0);
        assert_eq!(cfg.normalized_kg_update_interval(), 1);
        assert!(cfg.should_update_kg_on_step(1));
        assert!(cfg.should_update_kg_on_step(2));
        assert!(cfg.should_update_kg_on_step(3));
    }

    #[test]
    fn kg_update_interval_updates_on_multiples_only() {
        let cfg = dummy_config(3);
        assert_eq!(cfg.normalized_kg_update_interval(), 3);
        assert!(!cfg.should_update_kg_on_step(1));
        assert!(!cfg.should_update_kg_on_step(2));
        assert!(cfg.should_update_kg_on_step(3));
        assert!(!cfg.should_update_kg_on_step(4));
        assert!(!cfg.should_update_kg_on_step(5));
        assert!(cfg.should_update_kg_on_step(6));
    }
}
