/// Force pre-processing utilities for FSI coupling.
///
/// These functions are applied to the externally-read nodal force vector
/// *before* passing it to the structural stepper:
///
/// 1. **Ramping** — gradually increases forces from zero over `ramp_time`
///    to avoid impulsive loading at the start of a simulation.
/// 2. **Capping** — clamps each force component to `[-max_force, max_force]`
///    to guard against unphysical spikes from the fluid solver.

/// Scale forces by `min(t / ramp_time, 1.0)` when `t < ramp_time`.
///
/// # Arguments
/// * `forces`    – force vector to modify in place (nodal, 3-component interleaved or DOF-ordered)
/// * `t`         – current simulation time
/// * `ramp_time` – duration of the ramp; if `<= 0.0` this function is a no-op
pub fn apply_ramp(forces: &mut [f64], t: f64, ramp_time: f64) {
    if ramp_time <= 0.0 {
        return;
    }
    if t < ramp_time {
        let scale = t / ramp_time;
        for f in forces.iter_mut() {
            *f *= scale;
        }
    }
}

/// Cap force magnitude per node, preserving the direction of each nodal force vector.
///
/// Forces are assumed to be stored as interleaved components:
/// `[Fx0, Fy0, Fz0, Fx1, Fy1, Fz1, ...]` with `mesh_dims` components per node.
///
/// If a node's force vector magnitude exceeds `max_force`, all its components
/// are scaled down uniformly so that `||F_node|| == max_force`.
/// This is consistent with the Python `ForceClipper` which caps by nodal magnitude.
///
/// # Arguments
/// * `forces`    – force vector to modify in place (interleaved, `mesh_dims` components per node)
/// * `max_force` – maximum allowed nodal force magnitude
/// * `mesh_dims` – number of force components per node (2 or 3)
pub fn apply_cap(forces: &mut [f64], max_force: f64, mesh_dims: usize) {
    if mesh_dims == 0 {
        return;
    }
    for chunk in forces.chunks_mut(mesh_dims) {
        let magnitude: f64 = chunk.iter().map(|c| c * c).sum::<f64>().sqrt();
        if magnitude > max_force {
            let scale = max_force / magnitude;
            for c in chunk.iter_mut() {
                *c *= scale;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_ramp_half_time() {
        let mut forces = vec![10.0, -20.0, 30.0];
        apply_ramp(&mut forces, 0.5, 1.0);
        assert!((forces[0] - 5.0).abs() < 1e-12);
        assert!((forces[1] + 10.0).abs() < 1e-12);
        assert!((forces[2] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_ramp_zero_ramp_time_noop() {
        let mut forces = vec![10.0, -20.0];
        apply_ramp(&mut forces, 0.5, 0.0);
        assert_eq!(forces, vec![10.0, -20.0]);
    }

    #[test]
    fn test_apply_ramp_after_ramp_time_noop() {
        let mut forces = vec![10.0, -20.0];
        apply_ramp(&mut forces, 2.0, 1.0);
        assert_eq!(forces, vec![10.0, -20.0]);
    }

    #[test]
    fn test_apply_cap_1d_clamps() {
        // mesh_dims=1: each scalar is its own "node", behaves like per-component clamp
        let mut forces = vec![5.0, -15.0, 3.0, -3.0];
        apply_cap(&mut forces, 10.0, 1);
        assert_eq!(forces, vec![5.0, -10.0, 3.0, -3.0]);
    }

    #[test]
    fn test_apply_cap_3d_vector_magnitude() {
        // Node with force [3.0, 4.0, 0.0]: magnitude = 5.0, cap = 4.0
        // Expected scale = 4/5 = 0.8 → [2.4, 3.2, 0.0]
        let mut forces = vec![3.0, 4.0, 0.0];
        apply_cap(&mut forces, 4.0, 3);
        assert!((forces[0] - 2.4).abs() < 1e-12);
        assert!((forces[1] - 3.2).abs() < 1e-12);
        assert!((forces[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_cap_below_threshold_no_change() {
        // Node magnitude = sqrt(3^2 + 4^2) = 5.0, cap = 10.0 → no change
        let mut forces = vec![3.0, 4.0, 0.0];
        apply_cap(&mut forces, 10.0, 3);
        assert_eq!(forces, vec![3.0, 4.0, 0.0]);
    }
}
