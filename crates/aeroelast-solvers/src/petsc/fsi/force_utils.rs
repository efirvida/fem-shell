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

/// Clamp each force component to `[-max_force, max_force]`.
///
/// # Arguments
/// * `forces`    – force vector to modify in place
/// * `max_force` – upper bound for the absolute value of each component
pub fn apply_cap(forces: &mut [f64], max_force: f64) {
    for f in forces.iter_mut() {
        *f = f.clamp(-max_force, max_force);
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
    fn test_apply_cap_clamps() {
        let mut forces = vec![5.0, -15.0, 3.0, -3.0];
        apply_cap(&mut forces, 10.0);
        assert_eq!(forces, vec![5.0, -10.0, 3.0, -3.0]);
    }
}
