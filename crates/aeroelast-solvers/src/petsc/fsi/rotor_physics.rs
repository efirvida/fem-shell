/// Rotor physics utilities for the co-rotational FSI solver.
///
/// Provides:
/// - [`RotorTransforms`]  — Rodrigues rotation, force/displacement coordinate transforms.
/// - Inertial force functions — centrifugal, Coriolis, Euler (operating on flat arrays).
/// - [`build_ksp_diag`]   — spin-softening K_SP diagonal expanded to the K sparsity pattern.
/// - [`compute_torque`]   — scalar torque about the rotation axis from nodal forces.
/// - [`compute_performance_coefficients`] — Ct, Cp, Cq, TSR.
/// - [`OmegaProvider`]    — angular-velocity provider (constant, ramped, computed, mixed).
///
/// All coordinate arrays use the flat layout `[x0, y0, z0, x1, y1, z1, …]` with stride 3.
/// Force arrays returned by the inertial-force functions follow the same layout (n_nodes × 3).
///
/// # Feature gate
/// Compiled only with `--features fsi`.

use crate::petsc::fsi::setup::expand_diag_to_sparsity;

// ── Skew-symmetric helper ──────────────────────────────────────────────────────

/// Build the 3×3 skew-symmetric (cross-product) matrix K such that `K·x = axis × x`.
fn skew(axis: &[f64; 3]) -> [[f64; 3]; 3] {
    let [nx, ny, nz] = *axis;
    [
        [0.0, -nz, ny],
        [nz, 0.0, -nx],
        [-ny, nx, 0.0],
    ]
}

/// 3×3 matrix product A·B.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

// ── RotorTransforms ────────────────────────────────────────────────────────────

/// Coordinate transformation utilities for a rotating reference frame.
///
/// Uses Rodrigues' formula to compute rotation matrices for an arbitrary
/// rotation axis:
///
/// ```text
/// R(θ) = I + sin(θ)·K + (1 − cos(θ))·K²
/// ```
///
/// where K is the skew-symmetric matrix of the rotation axis.
///
/// # Conventions
/// * `to_rotating`   — `v_local  = R^T · v_global`
/// * `to_inertial`   — `v_global = R   · v_local`
#[derive(Debug, Clone)]
pub struct RotorTransforms {
    /// Normalized rotation axis unit vector.
    pub axis: [f64; 3],
    /// Rotation center (default: origin).
    pub center: [f64; 3],
    /// Precomputed skew-symmetric matrix K of the axis.
    k: [[f64; 3]; 3],
    /// Precomputed K² = K·K.
    k2: [[f64; 3]; 3],
}

impl RotorTransforms {
    /// Create a new `RotorTransforms` with the given rotation axis.
    ///
    /// # Panics
    /// Panics if `axis` has zero norm (cannot normalize).
    pub fn new(axis: [f64; 3], center: [f64; 3]) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        assert!(norm > 1e-12, "rotation axis must be non-zero");
        let n = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        let k = skew(&n);
        let k2 = mat3_mul(&k, &k);
        Self { axis: n, center, k, k2 }
    }

    /// Compute the 3×3 rotation matrix for angle `theta` (radians).
    ///
    /// `R(θ) = I + sin(θ)·K + (1 − cos(θ))·K²`
    pub fn rotation_matrix(&self, theta: f64) -> [[f64; 3]; 3] {
        let s = theta.sin();
        let c = 1.0 - theta.cos();
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let ident = if i == j { 1.0 } else { 0.0 };
                r[i][j] = ident + s * self.k[i][j] + c * self.k2[i][j];
            }
        }
        r
    }

    /// Transform a flat force array from the global (inertial) frame to the
    /// rotating (local) frame: `F_local = R^T · F_global`.
    ///
    /// `forces_global` — flat `[fx0, fy0, fz0, fx1, fy1, fz1, …]`.
    /// Returns a new flat array with the same layout.
    pub fn forces_to_rotating(&self, forces_global: &[f64], theta: f64) -> Vec<f64> {
        let r = self.rotation_matrix(theta);
        let n = forces_global.len() / 3;
        let mut out = vec![0.0f64; n * 3];
        for i in 0..n {
            let b = i * 3;
            let f = [forces_global[b], forces_global[b + 1], forces_global[b + 2]];
            // v_local = R^T · v_global  ↔  dot with rows of R
            for j in 0..3 {
                out[b + j] = r[0][j] * f[0] + r[1][j] * f[1] + r[2][j] * f[2];
            }
        }
        out
    }

    /// Transform a flat displacement array from the rotating (local) frame to
    /// the global (inertial) frame: `u_global = R · u_local`.
    ///
    /// `disps_local` — flat `[ux0, uy0, uz0, ux1, uy1, uz1, …]`.
    /// Returns a new flat array with the same layout.
    pub fn disps_to_inertial(&self, disps_local: &[f64], theta: f64) -> Vec<f64> {
        let r = self.rotation_matrix(theta);
        let n = disps_local.len() / 3;
        let mut out = vec![0.0f64; n * 3];
        for i in 0..n {
            let b = i * 3;
            let u = [disps_local[b], disps_local[b + 1], disps_local[b + 2]];
            for j in 0..3 {
                out[b + j] = r[j][0] * u[0] + r[j][1] * u[1] + r[j][2] * u[2];
            }
        }
        out
    }

    /// Transform gravity vector from the inertial frame to the rotating frame.
    ///
    /// `g_global` — gravity vector `[gx, gy, gz]` in the inertial frame.
    /// Returns `R^T · g_global`.
    pub fn gravity_to_rotating(&self, g_global: [f64; 3], theta: f64) -> [f64; 3] {
        let r = self.rotation_matrix(theta);
        let mut out = [0.0f64; 3];
        for j in 0..3 {
            out[j] = r[0][j] * g_global[0] + r[1][j] * g_global[1] + r[2][j] * g_global[2];
        }
        out
    }
}

// ── Inertial forces ────────────────────────────────────────────────────────────

/// Compute centrifugal forces for all nodes.
///
/// `F_cf,i = m_i · ω² · r_⊥,i`  where  `r_⊥ = r − (r·n̂)·n̂`
///
/// # Arguments
/// * `coords`  — flat node coordinates `[x0,y0,z0,…]` in the rotating frame
/// * `masses`  — per-node scalar masses, length `n_nodes`
/// * `axis`    — rotation axis unit vector
/// * `center`  — rotation center coordinates
/// * `omega`   — angular velocity (rad/s)
///
/// # Returns
/// Flat force array `[fx0,fy0,fz0,…]` length `n_nodes * 3`.
pub fn compute_centrifugal_force(
    coords: &[f64],
    masses: &[f64],
    axis: &[f64; 3],
    center: &[f64; 3],
    omega: f64,
) -> Vec<f64> {
    let n = masses.len();
    let mut out = vec![0.0f64; n * 3];
    if omega == 0.0 {
        return out;
    }
    let omega_sq = omega * omega;
    for i in 0..n {
        let b = i * 3;
        // r = coords - center
        let rx = coords[b] - center[0];
        let ry = coords[b + 1] - center[1];
        let rz = coords[b + 2] - center[2];
        // r_dot_axis = r · n̂
        let rda = rx * axis[0] + ry * axis[1] + rz * axis[2];
        // r_perp = r - (r·n̂) * n̂
        let rpx = rx - rda * axis[0];
        let rpy = ry - rda * axis[1];
        let rpz = rz - rda * axis[2];
        let m = masses[i];
        out[b] = m * omega_sq * rpx;
        out[b + 1] = m * omega_sq * rpy;
        out[b + 2] = m * omega_sq * rpz;
    }
    out
}

/// Compute Coriolis forces for all nodes.
///
/// `F_cor,i = −2 · m_i · (ω_vec × v_i)`
///
/// # Arguments
/// * `velocities` — flat node velocities `[vx0,vy0,vz0,…]` in the rotating frame
/// * `masses`     — per-node scalar masses
/// * `axis`       — rotation axis unit vector
/// * `omega`      — angular velocity (rad/s)
///
/// # Returns
/// Flat force array, length `n_nodes * 3`.
pub fn compute_coriolis_force(
    velocities: &[f64],
    masses: &[f64],
    axis: &[f64; 3],
    omega: f64,
) -> Vec<f64> {
    let n = masses.len();
    let mut out = vec![0.0f64; n * 3];
    if omega == 0.0 {
        return out;
    }
    // ω_vec = ω · n̂
    let wx = omega * axis[0];
    let wy = omega * axis[1];
    let wz = omega * axis[2];
    for i in 0..n {
        let b = i * 3;
        let vx = velocities[b];
        let vy = velocities[b + 1];
        let vz = velocities[b + 2];
        // ω × v
        let cx = wy * vz - wz * vy;
        let cy = wz * vx - wx * vz;
        let cz = wx * vy - wy * vx;
        let m = masses[i];
        out[b] = -2.0 * m * cx;
        out[b + 1] = -2.0 * m * cy;
        out[b + 2] = -2.0 * m * cz;
    }
    out
}

/// Compute Euler forces for all nodes (due to angular acceleration α = dω/dt).
///
/// `F_euler,i = −m_i · (α_vec × r_i)`
///
/// Evaluate at **deformed** coordinates `coords = X₀ + u`.
///
/// # Arguments
/// * `coords` — flat node coordinates in the rotating frame (may be deformed)
/// * `masses` — per-node scalar masses
/// * `axis`   — rotation axis unit vector
/// * `center` — rotation center
/// * `alpha`  — angular acceleration (rad/s²)
///
/// # Returns
/// Flat force array, length `n_nodes * 3`.
pub fn compute_euler_force(
    coords: &[f64],
    masses: &[f64],
    axis: &[f64; 3],
    center: &[f64; 3],
    alpha: f64,
) -> Vec<f64> {
    let n = masses.len();
    let mut out = vec![0.0f64; n * 3];
    if alpha == 0.0 {
        return out;
    }
    // α_vec = α · n̂
    let ax = alpha * axis[0];
    let ay = alpha * axis[1];
    let az = alpha * axis[2];
    for i in 0..n {
        let b = i * 3;
        let rx = coords[b] - center[0];
        let ry = coords[b + 1] - center[1];
        let rz = coords[b + 2] - center[2];
        // α_vec × r
        let cx = ay * rz - az * ry;
        let cy = az * rx - ax * rz;
        let cz = ax * ry - ay * rx;
        let m = masses[i];
        out[b] = -m * cx;
        out[b + 1] = -m * cy;
        out[b + 2] = -m * cz;
    }
    out
}

/// Compute gravity forces for all nodes in the rotating frame.
///
/// `F_g,i = m_i · g_rot`
///
/// # Arguments
/// * `masses` — per-node scalar masses
/// * `g_rot`  — gravity vector already transformed to the rotating frame
///
/// # Returns
/// Flat force array, length `n_nodes * 3`.
pub fn compute_gravity_force(masses: &[f64], g_rot: &[f64; 3]) -> Vec<f64> {
    let n = masses.len();
    let mut out = vec![0.0f64; n * 3];
    for i in 0..n {
        let b = i * 3;
        let m = masses[i];
        out[b] = m * g_rot[0];
        out[b + 1] = m * g_rot[1];
        out[b + 2] = m * g_rot[2];
    }
    out
}

// ── Spin-softening K_SP ────────────────────────────────────────────────────────

/// Build the spin-softening K_SP diagonal in the **full** (unreduced) DOF space,
/// then expand it to the K sparsity pattern so it can be passed to
/// [`NewmarkStepper::update_spin_softening`].
///
/// ANSYS Eq. 3-74 / 14-55 for lumped mass:
/// ```text
/// K_SP[dof_base + j] = −ω² · m_node · (1 − n̂ⱼ²)   for j = 0, 1, 2  (translational)
/// K_SP[dof_base + j] = 0                              for j ≥ 3  (rotational)
/// ```
///
/// The result is negative, reducing the effective stiffness perpendicular to
/// the rotation axis (spin-softening effect).
///
/// # Arguments
/// * `m_lumped_full` — lumped mass diagonal of the **full** (unreduced) system,
///   length `n_full_dofs`.  Only translational DOFs carry mass (rotational = 0).
/// * `axis`          — rotation axis unit vector
/// * `omega`         — angular velocity (rad/s)
/// * `dofs_per_node` — DOFs per FEM node (typically 6 for shells)
/// * `k_rows`, `k_cols` — COO sparsity of the **reduced** stiffness matrix
/// * `free_dofs`     — sorted free-DOF global indices (length = n_reduced_dofs)
///
/// # Returns
/// `Vec<f64>` of length `k_rows.len()` ready to pass to `update_spin_softening`.
#[allow(clippy::too_many_arguments)]
pub fn build_ksp_vals(
    m_lumped_full: &[f64],
    axis: &[f64; 3],
    omega: f64,
    dofs_per_node: usize,
    k_rows: &[i32],
    k_cols: &[i32],
    free_dofs: &[i32],
    n_full_dofs: usize,
) -> Vec<f64> {
    // 1. Build the K_SP diagonal in the full DOF space.
    let omega_sq = omega * omega;
    let n_nodes = n_full_dofs / dofs_per_node;
    let mut ksp_full = vec![0.0f64; n_full_dofs];
    for i in 0..n_nodes {
        let base = i * dofs_per_node;
        for j in 0..3_usize {
            let dof = base + j;
            if dof < n_full_dofs {
                let m = m_lumped_full[dof];
                ksp_full[dof] = -omega_sq * m * (1.0 - axis[j] * axis[j]);
            }
        }
        // Rotational DOFs (j >= 3) remain zero.
    }

    // 2. Reduce to the free DOFs (BC reduction): extract the diagonal at free DOF positions.
    let n_red = free_dofs.len();
    let mut ksp_red = vec![0.0f64; n_red];
    for (i, &global_dof) in free_dofs.iter().enumerate() {
        if (global_dof as usize) < n_full_dofs {
            ksp_red[i] = ksp_full[global_dof as usize];
        }
    }

    // 3. Expand the reduced diagonal to the K sparsity pattern.
    expand_diag_to_sparsity(&ksp_red, k_rows, k_cols)
}

// ── Torque computation ─────────────────────────────────────────────────────────

/// Compute the scalar driving torque about the rotation axis from interface nodal forces.
///
/// `τ_scalar = (Σᵢ (X₀,ᵢ + uᵢ − center) × Fᵢ) · n̂`
///
/// # Arguments
/// * `iface_coords` — flat interface node coordinates X₀ in the rotating frame,
///   length `n_nodes * 3`
/// * `iface_disps`  — flat interface displacements u, length `n_nodes * 3`
/// * `iface_forces` — flat interface forces F, length `n_nodes * 3`
/// * `axis`         — rotation axis unit vector
/// * `center`       — rotation center coordinates
///
/// # Returns
/// `(torque_vec [f64; 3], torque_scalar f64)`
pub fn compute_torque(
    iface_coords: &[f64],
    iface_disps: &[f64],
    iface_forces: &[f64],
    axis: &[f64; 3],
    center: &[f64; 3],
) -> ([f64; 3], f64) {
    let n = iface_coords.len() / 3;
    let mut tau = [0.0f64; 3];
    for i in 0..n {
        let b = i * 3;
        // r = (X₀ + u) − center
        let rx = iface_coords[b] + iface_disps[b] - center[0];
        let ry = iface_coords[b + 1] + iface_disps[b + 1] - center[1];
        let rz = iface_coords[b + 2] + iface_disps[b + 2] - center[2];
        let fx = iface_forces[b];
        let fy = iface_forces[b + 1];
        let fz = iface_forces[b + 2];
        // r × F
        tau[0] += ry * fz - rz * fy;
        tau[1] += rz * fx - rx * fz;
        tau[2] += rx * fy - ry * fx;
    }
    let scalar = tau[0] * axis[0] + tau[1] * axis[1] + tau[2] * axis[2];
    (tau, scalar)
}

// ── Performance coefficients ───────────────────────────────────────────────────

/// Non-dimensional rotor performance coefficients.
#[derive(Debug, Clone, Copy)]
pub struct PerformanceCoefficients {
    pub ct: f64,
    pub cp: f64,
    pub cq: f64,
    pub tsr: f64,
}

/// Kinematic quantities used to keep the rotor angle update consistent over one time window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowKinematics {
    /// Representative angular velocity held constant during the converged window.
    pub omega_step: f64,
    /// Angular acceleration used in the Euler body-force term during the window.
    pub alpha_step: f64,
    /// Exact or approximated angular increment over the full window.
    pub theta_increment: f64,
}

fn ramp_window_kinematics(omega_target: f64, t_ramp: f64, t: f64, dt: f64) -> WindowKinematics {
    if dt <= 0.0 {
        let omega = if t >= t_ramp {
            omega_target
        } else {
            omega_target * (t / t_ramp)
        };
        let alpha = if t < t_ramp { omega_target / t_ramp } else { 0.0 };
        return WindowKinematics {
            omega_step: omega,
            alpha_step: alpha,
            theta_increment: 0.0,
        };
    }

    if t_ramp <= 0.0 || t >= t_ramp {
        return WindowKinematics {
            omega_step: omega_target,
            alpha_step: 0.0,
            theta_increment: omega_target * dt,
        };
    }

    let alpha = omega_target / t_ramp;
    let omega_start = omega_target * (t / t_ramp);
    let t_end = t + dt;

    let theta_increment = if t_end <= t_ramp {
        omega_start * dt + 0.5 * alpha * dt * dt
    } else {
        let dt_ramp = t_ramp - t;
        let dtheta_ramp = omega_start * dt_ramp + 0.5 * alpha * dt_ramp * dt_ramp;
        let dtheta_const = omega_target * (dt - dt_ramp);
        dtheta_ramp + dtheta_const
    };

    WindowKinematics {
        omega_step: theta_increment / dt,
        alpha_step: alpha,
        theta_increment,
    }
}

/// Compute rotor performance coefficients.
///
/// Standard aerodynamic definitions:
/// ```text
/// Ct  = Thrust      / (½·ρ·V∞²·π·R²)
/// Cp  = P_aero      / (½·ρ·V∞³·π·R²)
/// Cq  = τ_aero      / (½·ρ·V∞²·π·R²·R)
/// TSR = |ω|·R / V∞
/// ```
///
/// All coefficients use AERODYNAMIC forces only (not total).
///
/// # Arguments
/// * `thrust`      — axial thrust [N] (aero force projected on rotation axis)
/// * `power_aero`  — aerodynamic power [W] = τ_aero · ω
/// * `torque_aero` — aerodynamic torque [N·m]
/// * `omega`       — angular velocity [rad/s]
/// * `radius`      — rotor radius [m]
/// * `fluid_density`   — fluid density [kg/m³]
/// * `flow_velocity`   — freestream velocity [m/s]
pub fn compute_performance_coefficients(
    thrust: f64,
    power_aero: f64,
    torque_aero: f64,
    omega: f64,
    radius: f64,
    fluid_density: f64,
    flow_velocity: f64,
) -> PerformanceCoefficients {
    let min_denom = 1e-6_f64;
    let area = std::f64::consts::PI * radius * radius;
    let q_dyn = 0.5 * fluid_density * flow_velocity * flow_velocity;
    let denom_force = q_dyn * area;
    let denom_power = q_dyn * area * flow_velocity;
    let denom_torque = q_dyn * area * radius;

    let ct = if denom_force.abs() > min_denom { thrust / denom_force } else { 0.0 };
    let cp = if denom_power.abs() > min_denom { power_aero / denom_power } else { 0.0 };
    let cq = if denom_torque.abs() > min_denom { torque_aero / denom_torque } else { 0.0 };
    let tsr = if flow_velocity.abs() > min_denom {
        omega.abs() * radius / flow_velocity
    } else {
        0.0
    };

    PerformanceCoefficients { ct, cp, cq, tsr }
}

// ── OmegaProvider ──────────────────────────────────────────────────────────────

/// Saved state for checkpoint/restore of an `OmegaProvider`.
#[derive(Debug, Clone)]
pub struct OmegaCheckpoint {
    pub omega: f64,
    pub alpha: f64,
    /// For `RampedComputed`: whether the ramp phase has been completed.
    pub ramp_completed: bool,
    /// Internal time tracked by the provider (for `RampedComputed` phase detection).
    pub current_time: f64,
}

/// Angular-velocity provider for the rotor FSI solver.
///
/// The provider is updated **once per converged time window** (not per sub-iteration).
/// During implicit FSI sub-iterations, ω is held constant.
///
/// # Variants
/// * `Constant`      — ω = const, α = 0 always.
/// * `Ramped`        — linear ramp ω(t) = ω_target · min(t/t_ramp, 1).
/// * `Computed`      — dynamic ω from `I·dω/dt = τ_driving + τ_shaft` (Euler).
/// * `RampedComputed`— ramp until `t ≥ t_ramp`, then `Computed`.
#[derive(Debug, Clone)]
pub enum OmegaProvider {
    Constant {
        omega: f64,
    },
    Ramped {
        omega_target: f64,
        t_ramp: f64,
    },
    Computed {
        moment_of_inertia: f64,
        shaft_torque: f64,
        /// Current dynamic state.
        omega: f64,
        alpha: f64,
    },
    RampedComputed {
        omega_target: f64,
        t_ramp: f64,
        moment_of_inertia: f64,
        shaft_torque: f64,
        /// Current dynamic state (used after ramp completion).
        omega: f64,
        alpha: f64,
        ramp_completed: bool,
        current_time: f64,
    },
}

impl OmegaProvider {
    /// Get the current (ω, α) at time `t`.
    ///
    /// For `Computed` and `RampedComputed` (post-ramp), `t` is ignored —
    /// the state is maintained internally and updated via [`Self::update_from_torque`].
    pub fn get(&self, t: f64) -> (f64, f64) {
        match self {
            OmegaProvider::Constant { omega } => (*omega, 0.0),
            OmegaProvider::Ramped { omega_target, t_ramp } => {
                if t >= *t_ramp {
                    (*omega_target, 0.0)
                } else {
                    let ratio = t / t_ramp;
                    (*omega_target * ratio, *omega_target / t_ramp)
                }
            }
            OmegaProvider::Computed { omega, alpha, .. } => (*omega, *alpha),
            OmegaProvider::RampedComputed {
                omega_target,
                t_ramp,
                omega,
                alpha,
                ramp_completed,
                ..
            } => {
                if !ramp_completed {
                    if t < *t_ramp {
                        let ratio = t / t_ramp;
                        (*omega_target * ratio, *omega_target / t_ramp)
                    } else {
                        (*omega_target, 0.0)
                    }
                } else {
                    (*omega, *alpha)
                }
            }
        }
    }

    /// Return the kinematics used during a single converged time window.
    ///
    /// The rotor solver holds a representative angular velocity constant during
    /// the window, while the cumulative angle is advanced with a second-order
    /// accurate increment whenever the provider supplies a meaningful `α`.
    pub fn kinematics_over_step(&self, t: f64, dt: f64) -> WindowKinematics {
        match self {
            OmegaProvider::Constant { omega } => WindowKinematics {
                omega_step: *omega,
                alpha_step: 0.0,
                theta_increment: *omega * dt,
            },
            OmegaProvider::Ramped { omega_target, t_ramp } => {
                ramp_window_kinematics(*omega_target, *t_ramp, t, dt)
            }
            OmegaProvider::Computed { omega, alpha, .. } => {
                let theta_increment = *omega * dt + 0.5 * *alpha * dt * dt;
                WindowKinematics {
                    omega_step: if dt > 0.0 { theta_increment / dt } else { *omega },
                    alpha_step: *alpha,
                    theta_increment,
                }
            }
            OmegaProvider::RampedComputed {
                omega_target,
                t_ramp,
                omega,
                alpha,
                ramp_completed,
                ..
            } => {
                if !ramp_completed {
                    ramp_window_kinematics(*omega_target, *t_ramp, t, dt)
                } else {
                    let theta_increment = *omega * dt + 0.5 * *alpha * dt * dt;
                    WindowKinematics {
                        omega_step: if dt > 0.0 { theta_increment / dt } else { *omega },
                        alpha_step: *alpha,
                        theta_increment,
                    }
                }
            }
        }
    }

    /// Integrate the equation of motion for one converged time window.
    ///
    /// `I · dω/dt = τ_driving + τ_shaft`  →  Euler step: `ω += α · dt`
    ///
    /// **`tau_driving` must equal τ_aero + τ_gravity.**
    /// Gravitational sag creates a non-zero torque about the rotation axis that
    /// contributes to the angular acceleration and must be included here.
    /// Fictitious forces (centrifugal, Coriolis, Euler) appear on the RHS of the
    /// structural equation but are NOT physical driving torques and must be excluded.
    ///
    /// For `Constant` and `Ramped` this is a no-op.
    /// For `RampedComputed`, ignored while still in the ramp phase.
    pub fn update_from_torque(&mut self, tau_driving: f64, dt: f64, t: f64) {
        match self {
            OmegaProvider::Computed { moment_of_inertia, shaft_torque, omega, alpha } => {
                *alpha = (tau_driving + *shaft_torque) / *moment_of_inertia;
                *omega += *alpha * dt;
            }
            OmegaProvider::RampedComputed {
                omega_target,
                t_ramp,
                moment_of_inertia,
                shaft_torque,
                omega,
                alpha,
                ramp_completed,
                current_time,
            } => {
                *current_time = t;
                if !*ramp_completed {
                    if t >= *t_ramp {
                        // Transition to dynamic phase.
                        *omega = *omega_target;
                        *alpha = 0.0;
                        *ramp_completed = true;
                    }
                    // Still ramping — torque update ignored.
                    return;
                }
                *alpha = (tau_driving + *shaft_torque) / *moment_of_inertia;
                *omega += *alpha * dt;
            }
            _ => {}
        }
    }

    /// Save the current state for implicit coupling checkpoint/restore.
    pub fn checkpoint(&self) -> OmegaCheckpoint {
        match self {
            OmegaProvider::Constant { omega } => OmegaCheckpoint {
                omega: *omega,
                alpha: 0.0,
                ramp_completed: true,
                current_time: 0.0,
            },
            OmegaProvider::Ramped { .. } => OmegaCheckpoint {
                omega: 0.0,
                alpha: 0.0,
                ramp_completed: false,
                current_time: 0.0,
            },
            OmegaProvider::Computed { omega, alpha, .. } => OmegaCheckpoint {
                omega: *omega,
                alpha: *alpha,
                ramp_completed: true,
                current_time: 0.0,
            },
            OmegaProvider::RampedComputed {
                omega,
                alpha,
                ramp_completed,
                current_time,
                ..
            } => OmegaCheckpoint {
                omega: *omega,
                alpha: *alpha,
                ramp_completed: *ramp_completed,
                current_time: *current_time,
            },
        }
    }

    /// Restore the state from a checkpoint.
    pub fn restore(&mut self, cp: &OmegaCheckpoint) {
        match self {
            OmegaProvider::Computed { omega, alpha, .. } => {
                *omega = cp.omega;
                *alpha = cp.alpha;
            }
            OmegaProvider::RampedComputed {
                omega,
                alpha,
                ramp_completed,
                current_time,
                ..
            } => {
                *omega = cp.omega;
                *alpha = cp.alpha;
                *ramp_completed = cp.ramp_completed;
                *current_time = cp.current_time;
            }
            _ => {}
        }
    }

    /// Return the angular velocity at t=0 (used for initial K_SP / K_G assembly).
    pub fn initial_omega(&self) -> f64 {
        match self {
            OmegaProvider::Constant { omega } => *omega,
            OmegaProvider::Ramped { .. } => 0.0,
            OmegaProvider::Computed { omega, .. } => *omega,
            OmegaProvider::RampedComputed { .. } => 0.0,
        }
    }
}

// ── Thrust calculation ──────────────────────────────────────────────────────────

/// Compute the axial thrust (aerodynamic force projected on the rotation axis).
///
/// `Thrust = (Σᵢ F_aero,i) · n̂`
///
/// # Arguments
/// * `aero_forces` — flat aerodynamic force array in the **inertial** frame,
///   length `n_nodes * 3`
/// * `axis`        — rotation axis unit vector
pub fn compute_thrust(aero_forces: &[f64], axis: &[f64; 3]) -> f64 {
    let n = aero_forces.len() / 3;
    let mut fsum = [0.0f64; 3];
    for i in 0..n {
        let b = i * 3;
        fsum[0] += aero_forces[b];
        fsum[1] += aero_forces[b + 1];
        fsum[2] += aero_forces[b + 2];
    }
    fsum[0] * axis[0] + fsum[1] * axis[1] + fsum[2] * axis[2]
}

// ── Moment of inertia ────────────────────────────────────────────────────────────

/// Estimate the moment of inertia about the rotation axis from nodal masses
/// and coordinates (lumped mass approximation).
///
/// `I = Σᵢ mᵢ · |r_⊥,i|²`
///
/// # Arguments
/// * `coords` — flat node coordinates in the rotating frame, length `n_nodes * 3`
/// * `masses` — per-node scalar masses, length `n_nodes`
/// * `axis`   — rotation axis unit vector
/// * `center` — rotation center
pub fn estimate_moment_of_inertia(
    coords: &[f64],
    masses: &[f64],
    axis: &[f64; 3],
    center: &[f64; 3],
) -> f64 {
    let n = masses.len();
    let mut inertia = 0.0f64;
    for i in 0..n {
        let b = i * 3;
        let rx = coords[b] - center[0];
        let ry = coords[b + 1] - center[1];
        let rz = coords[b + 2] - center[2];
        let rda = rx * axis[0] + ry * axis[1] + rz * axis[2];
        let rpx = rx - rda * axis[0];
        let rpy = ry - rda * axis[1];
        let rpz = rz - rda * axis[2];
        inertia += masses[i] * (rpx * rpx + rpy * rpy + rpz * rpz);
    }
    inertia
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matrix_identity_at_zero() {
        let t = RotorTransforms::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        let r = t.rotation_matrix(0.0);
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - expected[i][j]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn rotation_90deg_z_axis() {
        // R(90°) about Z: X→Y, Y→-X, Z→Z
        use std::f64::consts::FRAC_PI_2;
        let t = RotorTransforms::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        let r = t.rotation_matrix(FRAC_PI_2);
        // R·[1,0,0] should be [0,1,0]
        let rx0 = r[0][0]; // should be ≈0
        let ry0 = r[1][0]; // should be ≈1
        assert!(rx0.abs() < 1e-14, "R[0][0] = {rx0}");
        assert!((ry0 - 1.0).abs() < 1e-14, "R[1][0] = {ry0}");
    }

    #[test]
    fn centrifugal_zero_at_rest() {
        let axis = [0.0, 0.0, 1.0];
        let center = [0.0, 0.0, 0.0];
        let coords = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let masses = vec![1.0, 1.0];
        let f = compute_centrifugal_force(&coords, &masses, &axis, &center, 0.0);
        for v in &f {
            assert!(*v == 0.0);
        }
    }

    #[test]
    fn centrifugal_radially_outward() {
        // Node at [1, 0, 0], axis Z, omega=2.0 → r_perp=[1,0,0]
        // F = m * omega^2 * r_perp = 1 * 4 * [1,0,0]
        let axis = [0.0, 0.0, 1.0];
        let center = [0.0, 0.0, 0.0];
        let coords = vec![1.0, 0.0, 0.0];
        let masses = vec![1.0];
        let f = compute_centrifugal_force(&coords, &masses, &axis, &center, 2.0);
        assert!((f[0] - 4.0).abs() < 1e-13, "fx={}", f[0]);
        assert!(f[1].abs() < 1e-14, "fy={}", f[1]);
        assert!(f[2].abs() < 1e-14, "fz={}", f[2]);
    }

    #[test]
    fn coriolis_force_z_axis() {
        // Node velocity [1,0,0], axis Z, omega=1 → ω×v=[0,0,1]×[1,0,0]=[0*0-1*0, 1*1-0*0, 0*0-0*1]
        // ω=[0,0,1], v=[1,0,0] → ω×v = [0*0-1*0, 1*1-0*0, 0*0-0*1] = [0,1,0]
        // F_cor = -2*1*[0,1,0] = [0,-2,0]
        let axis = [0.0, 0.0, 1.0];
        let velocities = vec![1.0, 0.0, 0.0];
        let masses = vec![1.0];
        let f = compute_coriolis_force(&velocities, &masses, &axis, 1.0);
        assert!(f[0].abs() < 1e-14, "fx={}", f[0]);
        assert!((f[1] + 2.0).abs() < 1e-14, "fy={}", f[1]);
        assert!(f[2].abs() < 1e-14, "fz={}", f[2]);
    }

    #[test]
    fn torque_single_node() {
        // Node at [1,0,0], force=[0,0,1] → r×F = [1,0,0]×[0,0,1] = [0*1-0*0, 0*0-1*1, 1*0-0*0] = [0,-1,0]
        // axis=[0,0,1] → τ_scalar = [0,-1,0]·[0,0,1] = 0
        let coords = vec![1.0, 0.0, 0.0];
        let disps = vec![0.0, 0.0, 0.0];
        let forces = vec![0.0, 0.0, 1.0];
        let axis = [0.0, 0.0, 1.0];
        let center = [0.0, 0.0, 0.0];
        let (tau, scalar) = compute_torque(&coords, &disps, &forces, &axis, &center);
        assert!((tau[1] + 1.0).abs() < 1e-14, "tau_y={}", tau[1]);
        assert!(scalar.abs() < 1e-14, "scalar={scalar}");
    }

    #[test]
    fn torque_uses_deformed_coords_and_rotation_center() {
        // center = [1,0,0], X0 = [2,0,0], u = [0,1,0], F = [10,0,0]
        // r = (X0 + u - center) = [1,1,0]
        // r × F = [1,1,0] × [10,0,0] = [0,0,-10]
        // τ_scalar about +Z = -10
        let coords = vec![2.0, 0.0, 0.0];
        let disps = vec![0.0, 1.0, 0.0];
        let forces = vec![10.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let center = [1.0, 0.0, 0.0];

        let (tau, scalar) = compute_torque(&coords, &disps, &forces, &axis, &center);

        assert!(tau[0].abs() < 1e-14, "tau_x={}", tau[0]);
        assert!(tau[1].abs() < 1e-14, "tau_y={}", tau[1]);
        assert!((tau[2] + 10.0).abs() < 1e-14, "tau_z={}", tau[2]);
        assert!((scalar + 10.0).abs() < 1e-14, "scalar={scalar}");
    }

    #[test]
    fn omega_provider_constant() {
        let p = OmegaProvider::Constant { omega: 5.0 };
        assert_eq!(p.get(0.0), (5.0, 0.0));
        assert_eq!(p.get(100.0), (5.0, 0.0));
    }

    #[test]
    fn omega_provider_ramped() {
        let p = OmegaProvider::Ramped { omega_target: 10.0, t_ramp: 5.0 };
        let (w, a) = p.get(2.5);
        assert!((w - 5.0).abs() < 1e-12, "w={w}");
        assert!((a - 2.0).abs() < 1e-12, "a={a}");
        let (w2, a2) = p.get(10.0);
        assert_eq!((w2, a2), (10.0, 0.0));
    }

    #[test]
    fn omega_provider_ramped_window_kinematics_inside_ramp() {
        let p = OmegaProvider::Ramped { omega_target: 10.0, t_ramp: 5.0 };
        let kin = p.kinematics_over_step(2.0, 1.0);
        assert!((kin.omega_step - 5.0).abs() < 1e-12, "omega_step={}", kin.omega_step);
        assert!((kin.alpha_step - 2.0).abs() < 1e-12, "alpha_step={}", kin.alpha_step);
        assert!((kin.theta_increment - 5.0).abs() < 1e-12, "dtheta={}", kin.theta_increment);
    }

    #[test]
    fn omega_provider_ramped_window_kinematics_crossing_ramp_end() {
        let p = OmegaProvider::Ramped { omega_target: 10.0, t_ramp: 5.0 };
        let kin = p.kinematics_over_step(4.5, 1.0);
        assert!((kin.omega_step - 9.75).abs() < 1e-12, "omega_step={}", kin.omega_step);
        assert!((kin.alpha_step - 2.0).abs() < 1e-12, "alpha_step={}", kin.alpha_step);
        assert!((kin.theta_increment - 9.75).abs() < 1e-12, "dtheta={}", kin.theta_increment);
    }

    #[test]
    fn omega_provider_computed_update() {
        let mut p = OmegaProvider::Computed {
            moment_of_inertia: 100.0,
            shaft_torque: 0.0,
            omega: 0.0,
            alpha: 0.0,
        };
        // tau_driving = 100 N·m, dt = 1.0 → alpha = 1.0, omega = 1.0
        p.update_from_torque(100.0, 1.0, 0.0);
        let (w, a) = p.get(0.0);
        assert!((w - 1.0).abs() < 1e-12, "w={w}");
        assert!((a - 1.0).abs() < 1e-12, "a={a}");
    }

    #[test]
    fn omega_provider_computed_window_kinematics_is_second_order() {
        let p = OmegaProvider::Computed {
            moment_of_inertia: 100.0,
            shaft_torque: 0.0,
            omega: 5.0,
            alpha: 2.0,
        };
        let kin = p.kinematics_over_step(0.0, 0.5);
        assert!((kin.omega_step - 5.5).abs() < 1e-12, "omega_step={}", kin.omega_step);
        assert!((kin.alpha_step - 2.0).abs() < 1e-12, "alpha_step={}", kin.alpha_step);
        assert!((kin.theta_increment - 2.75).abs() < 1e-12, "dtheta={}", kin.theta_increment);
    }

    #[test]
    fn omega_provider_checkpoint_restore() {
        let mut p = OmegaProvider::Computed {
            moment_of_inertia: 100.0,
            shaft_torque: 0.0,
            omega: 5.0,
            alpha: 1.0,
        };
        let cp = p.checkpoint();
        p.update_from_torque(200.0, 1.0, 0.0); // changes state
        p.restore(&cp);
        let (w, a) = p.get(0.0);
        assert!((w - 5.0).abs() < 1e-12, "w={w}");
        assert!((a - 1.0).abs() < 1e-12, "a={a}");
    }
}
