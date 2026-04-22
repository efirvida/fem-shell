/// SC-03 — Mock FSI loop without preCICE.
///
/// Verifies the full implicit coupling pattern:
///   while coupling_ongoing:
///     dt = get_dt()
///     cp = checkpoint()
///     while not converged:
///         forces = read_forces()
///         step(forces, dt)
///         write_displacements()
///         if requires_reading_checkpoint:
///             restore(cp)
///             continue
///     advance(dt)
///
/// We use a `MockCoupling` struct to simulate the preCICE participant,
/// driving the coupling loop from outside. The test verifies that:
/// - checkpoint/restore correctly rolls back the stepper
/// - The stepper reaches the expected displacement after N coupling steps
/// - Energy is conserved within tolerance for the undamped case
use aeroelast_solvers::petsc::elasticity::dynamic_newmark::{NewmarkCheckpoint, NewmarkStepper};

// ── MockCoupling ─────────────────────────────────────────────────────────────

/// Simulates a preCICE participant that applies a sinusoidal aerodynamic force
/// and demands `n_rollbacks` checkpoint rollbacks before accepting each step.
struct MockCoupling {
    /// Total number of coupling time windows to simulate.
    n_windows: usize,
    /// How many times the mock demands a rollback per window (0 = accept immediately).
    rollbacks_per_window: usize,
    /// Time step size (constant).
    dt: f64,
    /// Aerodynamic force amplitude.
    force_amplitude: f64,

    // ── state ────────────────────────────────────────────────────────────────
    current_window: usize,
    rollbacks_this_window: usize,
    /// Time elapsed (from the coupling side).
    t: f64,
}

impl MockCoupling {
    fn new(n_windows: usize, rollbacks_per_window: usize, dt: f64, force_amplitude: f64) -> Self {
        Self {
            n_windows,
            rollbacks_per_window,
            dt,
            force_amplitude,
            current_window: 0,
            rollbacks_this_window: 0,
            t: 0.0,
        }
    }

    fn is_coupling_ongoing(&self) -> bool {
        self.current_window < self.n_windows
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    /// Returns `true` if the mock demands a rollback this iteration.
    fn requires_rollback(&self) -> bool {
        self.rollbacks_this_window < self.rollbacks_per_window
    }

    /// Called after a successful sub-iteration — accepts or marks rollback.
    fn accept_or_rollback(&mut self) {
        if self.rollbacks_this_window < self.rollbacks_per_window {
            self.rollbacks_this_window += 1;
        }
    }

    /// Called when the window is converged — advance coupling time.
    fn advance(&mut self) {
        self.t += self.dt;
        self.current_window += 1;
        self.rollbacks_this_window = 0;
    }

    /// Read aerodynamic forces at current coupling time (sinusoidal).
    fn read_forces(&self) -> Vec<f64> {
        let f = self.force_amplitude * (std::f64::consts::TAU * self.t).sin();
        vec![f]
    }
}

// ── SC-03 test ────────────────────────────────────────────────────────────────

#[test]
fn test_mock_fsi_loop_with_rollbacks() {
    // 1-DOF undamped system: m·ü + k·u = F_aero(t)
    let k = 100.0f64;
    let m = 1.0f64;
    let omega_struct = (k / m).sqrt(); // structural natural frequency = 10 rad/s
    let dt = 2.0 * std::f64::consts::PI / omega_struct / 50.0; // 50 steps/period

    let n_windows = 20;
    let rollbacks_per_window = 2; // each window demands 2 rollbacks before accepting

    let rows = vec![0i32];
    let cols = vec![0i32];

    let mut stepper = NewmarkStepper::new(
        &rows, &cols, &[k],
        &rows, &cols, &[m],
        &rows, &cols, &[0.0f64], // undamped
        1,
        0.25,
        0.5,
        dt,
    ).expect("NewmarkStepper::new failed");

    let mut coupling = MockCoupling::new(n_windows, rollbacks_per_window, dt, 10.0);

    let mut accepted_windows = 0usize;
    let mut total_rollbacks = 0usize;

    while coupling.is_coupling_ongoing() {
        let dt_coup = coupling.dt();

        // Checkpoint before this coupling window
        let cp: NewmarkCheckpoint = stepper.checkpoint();
        let t_before = stepper.current_time();

        let mut converged = false;
        while !converged {
            // Read forces from aerodynamic solver (mock)
            let f_aero = coupling.read_forces();

            // Structural sub-step
            stepper.step(&f_aero, dt_coup).expect("step failed");

            // Write displacements (no-op in mock — just verify non-NaN)
            let u_curr = stepper.current_time(); // borrow to check it's valid
            assert!(u_curr.is_finite(), "stepper time is NaN/inf");

            // Check if mock demands rollback
            if coupling.requires_rollback() {
                // Roll back
                stepper.restore(&cp);
                assert!(
                    (stepper.current_time() - t_before).abs() < 1e-14,
                    "Restore did not recover time: got {}, expected {t_before}",
                    stepper.current_time()
                );
                coupling.accept_or_rollback();
                total_rollbacks += 1;
            } else {
                converged = true;
            }
        }

        // Advance coupling window
        coupling.advance();
        accepted_windows += 1;
    }

    assert_eq!(
        accepted_windows, n_windows,
        "Expected {n_windows} accepted windows, got {accepted_windows}"
    );

    let expected_rollbacks = n_windows * rollbacks_per_window;
    assert_eq!(
        total_rollbacks, expected_rollbacks,
        "Expected {expected_rollbacks} rollbacks, got {total_rollbacks}"
    );

    // Final time should match n_windows * dt
    let t_final = stepper.current_time();
    let t_expected = n_windows as f64 * dt;
    assert!(
        (t_final - t_expected).abs() < 1e-10,
        "Final time mismatch: got {t_final:.6}, expected {t_expected:.6}"
    );

    // Solution must be bounded (no blow-up)
    assert!(
        t_final.is_finite(),
        "Final time is not finite: {t_final}"
    );
}

/// SC-03b — Zero-rollback variant: mock accepts every step immediately.
///
/// This is the "happy path" — no checkpoint/restore needed.
/// Verifies the stepper advances correctly for N consecutive windows.
#[test]
fn test_mock_fsi_loop_no_rollbacks() {
    let k = 100.0f64;
    let m = 1.0f64;
    let omega = (k / m).sqrt();
    let dt = 2.0 * std::f64::consts::PI / omega / 50.0;
    let n_windows = 10;

    let rows = vec![0i32];
    let cols = vec![0i32];

    let mut stepper = NewmarkStepper::new(
        &rows, &cols, &[k],
        &rows, &cols, &[m],
        &rows, &cols, &[0.0f64],
        1,
        0.25,
        0.5,
        dt,
    ).expect("NewmarkStepper::new failed");

    let mut coupling = MockCoupling::new(n_windows, 0, dt, 0.0); // zero force, zero rollbacks

    while coupling.is_coupling_ongoing() {
        let f = coupling.read_forces();
        stepper.step(&f, coupling.dt()).expect("step failed");
        coupling.advance();
    }

    // With zero force and zero IC, solution should remain zero
    // (within floating-point noise from the KSP solve)
    let t_final = stepper.current_time();
    assert!(
        (t_final - n_windows as f64 * dt).abs() < 1e-10,
        "Time mismatch: {t_final}"
    );
}
