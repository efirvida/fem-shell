"""
StressStiffenedFSISolver — incremental linearization with per-step K_G update.

Physical motivation
-------------------
A standard ``LinearDynamicFSISolver`` assembles the elastic stiffness K once at
t = 0 and keeps it frozen throughout the simulation:

    K_eff = K + a₀·M + a₁·C                (constant for all time)

This neglects *stress stiffening* (geometric nonlinearity): under large membrane
tension (e.g. aerodynamic loading on a flexible blade) the structure is stiffer
than the linearized K predicts, leading to over-estimated tip deformations.

``StressStiffenedFSISolver`` adds a geometric stiffness term K_G that is updated
from the deformed state at the end of each converged time window:

    K_eff^{n+1} = K + K_G(σ^n) + a₀·M + a₁·C

where σ^n is the membrane stress field recovered from the displacement u^n.
This is a *linearized incremental* (also called "frozen-tangent") approach:

* ONE linear solve per time step — no Newton-Raphson inner iterations.
* ONE K_eff factorization per ``update_interval`` converged steps.
* Converges to the correct nonlinear solution as Δt → 0.
* Compatible with direct (MUMPS) and iterative (GAMG) solvers.

Cost relative to base solver
-----------------------------
Let T = MUMPS factorization time, t_s = time per back-substitution,
t_KG = K_G assembly time (one element loop — fast), n = total time steps.

Base:            T + n·t_s
StressStiffened: n/N · (T + t_KG) + n·t_s      (N = update_interval)

For small meshes (≤ 50 k DOFs) where T ≈ 10 · t_s and N = 1:
  Overhead ≈ 10× factorizations — wall-time roughly doubles.
For N = 5: overhead ≈ 2×.
For N = 10: overhead ≈ marginal.

Configuration
-------------
.. code-block:: yaml

    solver:
      type: StressStiffenedDynamicFSI
      geometric_stiffness:
        update_interval: 1   # K_G rebuilt every N converged steps (default 1)

References
----------
- Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in
  geometric nonlinear analysis." Computers & Structures, 185, 1-14.
- Bathe, K.J. (1996). Finite Element Procedures. §6.3 Updated Lagrangian.
"""

from __future__ import annotations

import logging

import numpy as np

from .linear_dynamic import LinearDynamicFSISolver

_logger = logging.getLogger(__name__)


class StressStiffenedFSISolver(LinearDynamicFSISolver):
    """Implicit Newmark FSI solver with per-step geometric stiffness update.

    Inherits all FSI coupling, checkpointing, force clipping/ramping, and
    damping logic from ``LinearDynamicFSISolver``.  The geometric stiffness
    K_G is rebuilt every ``kg_update_interval`` converged steps directly in
    Rust via ``run_stress_stiffened_fsi_solver``.

    Parameters
    ----------
    domain : MeshAssembler
        FEM domain (mesh + elements + DOF map).
    solver_params : dict
        Solver configuration dictionary.  In addition to the base-class
        keys, recognises:

        ``geometric_stiffness.update_interval`` : int, default 1
            Rebuild K_G every *N* converged time windows.
            1 = every step (maximum accuracy).
            N > 1 = amortize factorization cost over N steps.
    """

    def __init__(self, domain, solver_params: dict) -> None:
        super().__init__(domain, solver_params)
        gs_cfg = solver_params.get("geometric_stiffness", {})
        self._kg_update_interval: int = max(1, int(gs_cfg.get("update_interval", 1)))
        _logger.info(
            "StressStiffenedFSISolver: K_G update every %d converged step(s).",
            self._kg_update_interval,
        )


    # ------------------------------------------------------------------
    # Rust fast-path
    # ------------------------------------------------------------------

    def _solve_via_rust(
        self,
        bc_manager,
        interface_coords_flat,
        interface_dofs_global_flat,
    ):
        """Run the stress-stiffened FSI loop via the Rust binding.

        Delegates to ``_aeroelast.run_stress_stiffened_fsi_solver``, which
        internally rebuilds K_G every ``kg_update_interval`` converged steps
        using the Rust assembler — no Python callbacks required for the
        geometric stiffness update.

        Falls back to the Python path (with a warning) if the Rust assembler
        is not initialised.
        """
        import _aeroelast  # noqa: PLC0415

        rust_asm = getattr(self.domain, "_rust", None)
        if rust_asm is None:
            _logger.warning(
                "StressStiffenedFSISolver: Rust assembler not available — "
                "falling back to Python FSI path."
            )
            return super()._solve_via_rust(
                bc_manager, interface_coords_flat, interface_dofs_global_flat
            )

        import numpy as np  # noqa: PLC0415

        k_rows, k_cols, k_vals = self._petsc_to_coo(self.K)
        m_rows, m_cols, m_vals = self._petsc_to_coo(self.M)

        free_dofs = bc_manager.free_dofs.astype(np.int32)
        self.free_dofs = bc_manager.free_dofs

        n_full_dofs: int = int(self.K.getSize()[0])

        checkpoint_state = self._try_restore_checkpoint()
        u0 = v0 = a0 = None
        t0 = self.solver_params.get("start_time", 0.0)

        if checkpoint_state is not None:
            stored_dofs = len(checkpoint_state.get("u_red", []))
            expected_dofs = int(free_dofs.shape[0])
            if stored_dofs and stored_dofs != expected_dofs:
                print(
                    f"  ⚠️ Checkpoint DOF mismatch ({stored_dofs} vs {expected_dofs}). "
                    "Checkpoint ignored.",
                    flush=True,
                )
                checkpoint_state = None

        if checkpoint_state is not None and "u_red" in checkpoint_state:
            u0 = checkpoint_state["u_red"].astype(np.float64)
            v0 = checkpoint_state["v_red"].astype(np.float64)
            a0 = checkpoint_state["a_red"].astype(np.float64)
            t0 = float(checkpoint_state["t"])
            print(f"  ✓ Restored from checkpoint at t = {t0:.6f} s", flush=True)

        cfg = self._coupling_cfg
        mesh_name = cfg["coupling_mesh"]
        write_data = (
            cfg["write_data"] if isinstance(cfg["write_data"], str) else cfg["write_data"][0]
        )
        read_data = cfg["read_data"] if isinstance(cfg["read_data"], str) else cfg["read_data"][0]

        beta = float(self.solver_params["beta"])
        gamma = float(self.solver_params["gamma"])
        dt_hint = float(self.solver_params.get("dt", 0.01))

        print(
            f"  → Delegating stress-stiffened FSI loop to Rust: "
            f"η_k={self._eta_k:.3e}  η_m={self._eta_m:.3e}  "
            f"K_G every {self._kg_update_interval} step(s)  "
            f"β={beta}  γ={gamma}",
            flush=True,
        )

        # ── Per-step callback (identical to base class) ────────────────────
        n_total: int = n_full_dofs
        _free_dofs_arr: np.ndarray = self.free_dofs
        _fixed_dof_vals: dict = dict(bc_manager.fixed_dofs)

        def _step_cb(
            t,
            time_step,
            dt,
            u_red,
            v_red,
            a_red,
            force_mag,
            forces_iface,
        ):
            u_full = np.zeros(n_total, dtype=np.float64)
            u_full[_free_dofs_arr] = u_red
            for dof, val in _fixed_dof_vals.items():
                u_full[dof] = val

            v_full = np.zeros(n_total, dtype=np.float64)
            v_full[_free_dofs_arr] = v_red

            a_full = np.zeros(n_total, dtype=np.float64)
            a_full[_free_dofs_arr] = a_red

            stress_fields = self._compute_stress_fields(u_full)

            iface_dofs_flat = interface_dofs_global_flat
            force_fields = {}
            if forces_iface is not None and len(forces_iface) > 0:
                f_raw_full = np.zeros(n_total, dtype=np.float64)
                for local_idx in range(min(len(forces_iface), len(iface_dofs_flat))):
                    gdof = int(iface_dofs_flat[local_idx])
                    if gdof < n_total:
                        f_raw_full[gdof] += forces_iface[local_idx]
                force_fields["F_AERO_RAW"] = f_raw_full
                force_fields["F_AERO"] = f_raw_full
                force_fields["F_TOTAL"] = f_raw_full.copy()
            force_fields.update(stress_fields)

            self._log_structural_report(
                t=t,
                time_step=time_step,
                u_full=u_full,
                v_full=v_full,
                a_full=a_full,
                stress_fields=force_fields,
                applied_force_mag=force_mag,
            )
            self._log_probe_data(
                t=t,
                time_step=time_step,
                u_full=u_full,
                v_full=v_full,
                stress_fields=force_fields,
            )
            self._handle_checkpoint(
                t=t,
                time_step=time_step,
                dt=dt,
                u_red=u_red,
                v_red=v_red,
                a_red=a_red,
                u_full=u_full,
                v_full=v_full,
                a_full=a_full,
                extra_fields=force_fields,
            )

        disp_hist, vel_hist, acc_hist, times = _aeroelast.run_stress_stiffened_fsi_solver(
            rust_asm,
            n_full_dofs,
            self._kg_update_interval,
            k_rows,
            k_cols,
            k_vals,
            m_rows,
            m_cols,
            m_vals,
            free_dofs,
            self._eta_k,
            self._eta_m,
            beta,
            gamma,
            dt_hint,
            interface_coords_flat,
            interface_dofs_global_flat,
            self.domain.spatial_dim,
            cfg["participant"],
            cfg["config_file"],
            mesh_name,
            write_data,
            read_data,
            self._force_ramp_time,
            getattr(self, "_force_max_magnitude", None),
            u0,
            v0,
            a0,
            t0,
            step_callback=_step_cb,
        )

        n_steps = len(times)
        if n_steps > 0:
            print(
                f"  ✓ Stress-stiffened FSI loop complete: "
                f"{n_steps} converged steps, t_final={times[-1]:.4f} s",
                flush=True,
            )
        else:
            print("  ⚠️ FSI loop returned 0 converged steps.", flush=True)

        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        if n_steps > 0:
            u_final_full = np.zeros(n_total, dtype=np.float64)
            u_final_full[_free_dofs_arr] = np.asarray(disp_hist[-1], dtype=np.float64)
            for dof, val in _fixed_dof_vals.items():
                u_final_full[dof] = val

            v_final_full = np.zeros(n_total, dtype=np.float64)
            v_final_full[_free_dofs_arr] = np.asarray(vel_hist[-1], dtype=np.float64)

            a_final_full = np.zeros(n_total, dtype=np.float64)
            a_final_full[_free_dofs_arr] = np.asarray(acc_hist[-1], dtype=np.float64)
        else:
            u_final_full = np.zeros(n_total, dtype=np.float64)
            v_final_full = np.zeros(n_total, dtype=np.float64)
            a_final_full = np.zeros(n_total, dtype=np.float64)

        self.u = u_final_full
        self.v = v_final_full
        self.a = a_final_full
        return self.u, self.v, self.a
