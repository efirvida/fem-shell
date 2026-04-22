/// Linear elastic FSI solver using preCICE for coupling.
///
/// This module implements the full preCICE coupling loop for a linear elastic
/// structural solver using the Newmark-β time integrator.
///
/// # Feature gate
/// All items in this module are only compiled with `--features fsi`.
/// The module itself is declared under `#[cfg(feature = "fsi")]` in `fsi/mod.rs`.
///
/// # API
/// - [`FsiConfig`] — coupling configuration (participant, mesh, data names)
/// - [`FsiResult`] — time history of displacement, velocity, acceleration
/// - [`LinearElasticFsiSolver`] — the solver struct with a `run()` method
/// - [`FsiError`] — error type for FSI operations

use crate::petsc::elasticity::dynamic_newmark::{NewmarkCheckpoint, NewmarkStepper};
use crate::petsc::fsi::force_utils::{apply_cap, apply_ramp};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during the FSI coupling loop.
#[derive(Debug)]
pub enum FsiError {
    /// A PETSc error occurred inside the Newmark stepper.
    StepperError(crate::petsc::infra::mat::PetscError),
    /// preCICE reported a fatal error.
    PreciceError(String),
}

impl std::fmt::Display for FsiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FsiError::StepperError(e) => write!(f, "Newmark stepper error: {e}"),
            FsiError::PreciceError(msg) => write!(f, "preCICE error: {msg}"),
        }
    }
}

impl std::error::Error for FsiError {}

impl From<crate::petsc::infra::mat::PetscError> for FsiError {
    fn from(e: crate::petsc::infra::mat::PetscError) -> Self {
        FsiError::StepperError(e)
    }
}

impl From<precice::Error> for FsiError {
    fn from(e: precice::Error) -> Self {
        FsiError::PreciceError(e.to_string())
    }
}
// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the preCICE FSI coupling.
///
/// # Example
/// ```rust,no_run
/// use aeroelast_solvers::petsc::fsi::linear_elastic::FsiConfig;
///
/// let config = FsiConfig {
///     participant_name: "Solid".to_string(),
///     config_file: "precice-config.xml".to_string(),
///     coupling_mesh: "Solid-Mesh".to_string(),
///     write_data: "Displacement".to_string(),
///     read_data: "Force".to_string(),
///     ramp_time: 0.1,
///     force_max: Some(1e6),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FsiConfig {
    /// preCICE participant name (must match the config XML).
    pub participant_name: String,
    /// Path to the preCICE XML configuration file.
    pub config_file: String,
    /// Name of the coupling mesh registered with preCICE.
    pub coupling_mesh: String,
    /// Name of the data field written to preCICE (e.g. "Displacement").
    pub write_data: String,
    /// Name of the data field read from preCICE (e.g. "Force").
    pub read_data: String,
    /// Duration of the force ramp-up at the start of the simulation.
    /// Set to `0.0` to disable ramping.
    pub ramp_time: f64,
    /// Maximum absolute value for any force component (capping).
    /// Set to `None` to disable capping.
    pub force_max: Option<f64>,
}

// ── Result type ───────────────────────────────────────────────────────────────

/// Time history of the structural response from the FSI coupling loop.
///
/// Only converged windows are stored — iterations that are rolled back via
/// checkpoint/restore are discarded.
#[derive(Debug, Default)]
pub struct FsiResult {
    /// Displacement vector at each converged time step: `displacement_history[step][dof]`
    pub displacement_history: Vec<Vec<f64>>,
    /// Velocity vector at each converged time step: `velocity_history[step][dof]`
    pub velocity_history: Vec<Vec<f64>>,
    /// Acceleration vector at each converged time step: `acceleration_history[step][dof]`
    pub acceleration_history: Vec<Vec<f64>>,
    /// Simulation time at each converged step.
    pub times: Vec<f64>,
}

// ── Solver ────────────────────────────────────────────────────────────────────

/// Linear elastic FSI solver coupled via preCICE.
///
/// Wraps a [`NewmarkStepper`] and orchestrates the preCICE coupling loop:
/// reading forces, stepping the structure, writing displacements, and
/// handling implicit coupling iteration (checkpoint/restore).
///
/// # Usage
/// 1. Build a `NewmarkStepper` from the structural matrices.
/// 2. Construct `LinearElasticFsiSolver::new(stepper, config, coords, dofs, dims)`.
/// 3. Call `run()` — this blocks until the coupled simulation finishes.
pub struct LinearElasticFsiSolver {
    /// Structural time integrator.
    stepper: NewmarkStepper,
    /// preCICE coupling configuration.
    config: FsiConfig,
    /// Flat interface node coordinates `[x0, y0, z0, x1, y1, z1, ...]`.
    interface_coords: Vec<f64>,
    /// DOF indices (in the reduced system) for each interface node component.
    ///
    /// Length = `n_interface_nodes × mesh_dims`.
    interface_dofs: Vec<usize>,
    /// Spatial dimension of the coupling mesh (2 or 3).
    mesh_dims: usize,
}

impl LinearElasticFsiSolver {
    /// Create a new `LinearElasticFsiSolver`.
    ///
    /// # Arguments
    /// * `stepper`          — pre-built Newmark-β stepper
    /// * `config`           — preCICE coupling configuration
    /// * `interface_coords` — flat node coordinates for the coupling mesh
    /// * `interface_dofs`   — DOF indices (in reduced system) for each interface component
    /// * `mesh_dims`        — spatial dimension (2 or 3)
    pub fn new(
        stepper: NewmarkStepper,
        config: FsiConfig,
        interface_coords: Vec<f64>,
        interface_dofs: Vec<usize>,
        mesh_dims: usize,
    ) -> Self {
        Self {
            stepper,
            config,
            interface_coords,
            interface_dofs,
            mesh_dims,
        }
    }

    /// Run the preCICE FSI coupling loop.
    ///
    /// This method:
    /// 1. Initializes a preCICE `Participant`.
    /// 2. Registers the coupling mesh vertices.
    /// 3. Iterates the FSI loop (read forces → step structure → write displacements).
    /// 4. Handles implicit coupling via checkpoint/restore.
    /// 5. Finalizes preCICE and returns the full time history.
    ///
    /// # Returns
    /// [`FsiResult`] containing displacement, velocity and acceleration histories
    /// at each *converged* time window (rolled-back iterations are discarded).
    ///
    /// # Errors
    /// Returns [`FsiError`] if preCICE initialization fails or the Newmark
    /// stepper encounters a PETSc error.
    pub fn run(&mut self) -> Result<FsiResult, FsiError> {
        let mut participant = precice::Participant::new(
            &self.config.participant_name,
            &self.config.config_file,
            0, // solver rank
            1, // communicator size
        )?;

        // Register interface mesh vertices.
        // set_mesh_vertices writes IDs into the provided out-param slice.
        let n_vertices = self.interface_coords.len() / self.mesh_dims.max(1);
        let mut vertex_ids = vec![0i32; n_vertices];
        participant.set_mesh_vertices(
            &self.config.coupling_mesh,
            &self.interface_coords,
            &mut vertex_ids,
        )?;

        participant.initialize()?;
        let mut dt = participant.get_max_time_step_size()?;

        let n_dofs = self.stepper.n_dofs();
        // n_data = n_vertices * data_dims (for displacements and forces)
        let n_data = n_vertices * self.mesh_dims;
        let mut checkpoint: Option<NewmarkCheckpoint> = None;
        let mut result = FsiResult::default();

        while participant.is_coupling_ongoing()? {
            // ── Implicit coupling: save state before iteration ────────────────
            if participant.requires_writing_checkpoint()? {
                checkpoint = Some(self.stepper.checkpoint());
            }

            // ── Read forces from preCICE (flat: n_vertices × mesh_dims) ───────
            // read_data fills an out-param buffer.
            let mut forces = vec![0.0f64; n_data];
            participant.read_data(
                &self.config.coupling_mesh,
                &self.config.read_data,
                &vertex_ids,
                dt,
                &mut forces,
            )?;

            // ── Force pre-processing: ramp and cap ────────────────────────────
            let t = self.stepper.current_time();
            apply_ramp(&mut forces, t, self.config.ramp_time);
            if let Some(max_f) = self.config.force_max {
                apply_cap(&mut forces, max_f);
            }

            // ── Scatter interface forces into the global DOF vector ───────────
            let mut f_global = vec![0.0f64; n_dofs];
            for (local_idx, &global_dof) in self.interface_dofs.iter().enumerate() {
                if local_idx < forces.len() && global_dof < n_dofs {
                    f_global[global_dof] = forces[local_idx];
                }
            }

            // ── Advance the structural state by one time step ─────────────────
            let step_result = self.stepper.step(&f_global, dt)?;

            // ── Gather interface displacements (DOF → vertex component) ───────
            let disp_interface: Vec<f64> = self
                .interface_dofs
                .iter()
                .map(|&dof| {
                    if dof < step_result.u.len() {
                        step_result.u[dof]
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

            // ── Advance the coupling ──────────────────────────────────────────
            participant.advance(dt)?;
            dt = participant.get_max_time_step_size()?;

            // ── Implicit coupling: restore or commit ──────────────────────────
            if participant.requires_reading_checkpoint()? {
                // Iteration not converged — restore state and retry
                if let Some(ref cp) = checkpoint {
                    self.stepper.restore(cp);
                }
            } else {
                // Converged time window — commit to history
                result.displacement_history.push(step_result.u.clone());
                result.velocity_history.push(step_result.v.clone());
                result.acceleration_history.push(step_result.a.clone());
                result.times.push(step_result.t);
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
        self.interface_coords.len() / self.mesh_dims.max(1)
    }
}
