"""
Linear Dynamic FSI Solver.

This module provides the LinearDynamicFSISolver for fluid-structure interaction
problems using preCICE coupling.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import meshio
import numpy as np
from petsc4py import PETSc
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aeroelast.core.bc import BoundaryConditionManager
from aeroelast.core.mesh import MeshModel
from aeroelast.elements import ElementFamily
from aeroelast.postprocess.stress_recovery import StressRecovery, StressType
from aeroelast.solvers.linear import LinearDynamicSolver

from .base import Adapter

_console = Console(highlight=False)
_logger = logging.getLogger(__name__)


class LinearDynamicFSISolver(LinearDynamicSolver):
    """
    Linear dynamic solver for Fluid-Structure Interaction (FSI) problems.

    Inherits from LinearDynamicSolver and adds functionality for FSI problems
    using preCICE coupling.

    Parameters
    ----------
    mesh : MeshModel
        The finite element mesh.
    fem_model_properties : Dict
        Configuration dictionary containing solver and element properties.

    Configuration
    -------------
    The fem_model_properties dictionary should contain:
        solver.coupling.participant : str
            Name of this participant in preCICE.
        solver.coupling.config_file : str
            Path to preCICE configuration XML.
        solver.coupling.coupling_mesh : str
            Name of the coupling mesh.
        solver.coupling.write_data : str or list
            Data field(s) to write to preCICE.
        solver.coupling.read_data : str or list
            Data field(s) to read from preCICE.
        solver.coupling_boundaries : list
            Node set names for coupling interface.
        solver.force_max_cap : float, optional
            Maximum force magnitude per node (for clipping).
        solver.force_ramp_time : float, optional
            Time over which to ramp forces from 0 to 1.
        solver.solver_type : str, optional
            Linear solver type: "auto", "direct", "iterative". Default: "auto".
        solver.damping.enabled : bool, optional
            Enable Rayleigh damping. Default: True.
        solver.damping.eta_m : float, optional
            Mass-proportional Rayleigh damping coefficient.
        solver.damping.eta_k : float, optional
            Stiffness-proportional Rayleigh damping coefficient.
        solver.damping.zeta : float, optional
            Target damping ratio for auto-computation. Default: 0.02.
        solver.damping.mode_i : int, optional
            First reference mode for auto-computation. Default: 1.
        solver.damping.mode_j : int, optional
            Second reference mode for auto-computation. Default: 2.
        solver.force_max_magnitude : float, optional
            Absolute upper bound on total force magnitude (divergence check).
        solver.force_jump_factor : float, optional
            Relative spike detector threshold. Default: 1000.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        super().__init__(mesh, fem_model_properties)

        # Store coupling configuration — Adapter is created in _initialize_precice
        # once interface coordinates are known.
        self._coupling_cfg = fem_model_properties["solver"]["coupling"]
        self.precice_participant: Optional[Adapter] = None
        self._init_solver_config()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_solver_config(self) -> None:
        """Initialize solver configuration parameters.

        Reads damping configuration from a nested ``damping`` dict with
        fallback to flat ``eta_m``/``eta_k`` keys for backward compatibility.
        Also reads force sanity-check parameters.
        """
        # --- Damping configuration ---
        damping_cfg = self.solver_params.get("damping") or {}

        self._damping_enabled: bool = damping_cfg.get("enabled", True)
        # Auto mode: compute coefficients from modal analysis at assembly time
        self._damping_auto: bool = (
            self._damping_enabled
            and damping_cfg.get("eta_m") is None
            and damping_cfg.get("eta_k") is None
            and bool(damping_cfg)  # only auto when damping section is present
        )
        self._damping_cfg: dict = damping_cfg

        if not self._damping_enabled:
            self._eta_m = 0.0
            self._eta_k = 0.0
        elif self._damping_auto:
            # Placeholder — will be overwritten in solve() via _compute_rayleigh_auto
            self._eta_m = 0.0
            self._eta_k = 0.0
        else:
            # Manual mode: read from nested damping dict, fall back to flat keys
            self._eta_m = float(
                damping_cfg["eta_m"]
                if damping_cfg.get("eta_m") is not None
                else self.solver_params.get("eta_m", 0.0)
            )
            self._eta_k = float(
                damping_cfg["eta_k"]
                if damping_cfg.get("eta_k") is not None
                else self.solver_params.get("eta_k", 0.0)
            )

        # --- Force sanity checks ---
        self._force_ramp_time = float(self.solver_params.get("force_ramp_time", 0.0))

        _fmax = self.solver_params.get("force_max_magnitude", None)
        self._force_max_magnitude: Optional[float] = float(_fmax) if _fmax is not None else None
        # --- Probe monitoring ---
        self._probe_node_ids: list[int] = []
        self._probe_file: Optional[str] = None

    # =========================================================================
    # Matrix Assembly Helpers
    # =========================================================================

    @staticmethod
    def _petsc_to_coo(mat: "PETSc.Mat"):
        """Extract (rows, cols, vals) COO triplets from a PETSc matrix.

        Returns numpy arrays with dtype int32 (rows, cols) and float64 (vals).
        Works for both sparse and diagonal PETSc matrices.
        """
        ai, aj, av = mat.getValuesCSR()
        counts = np.diff(ai.astype(np.int64))
        rows = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        cols = aj.astype(np.int32)
        vals = av.astype(np.float64)
        return rows, cols, vals

    def _solve_via_rust(
        self,
        bc_manager,
        interface_coords_flat: "np.ndarray",
        interface_dofs_global_flat: "np.ndarray",
    ):
        """Run the FSI coupling loop via the Rust ``run_fsi_solver`` binding.

        Delegates the entire coupling loop to Rust and fires a Python per-step
        callback after each converged time window to handle structural reports,
        probe logging, and checkpoint writing.

        Returns ``(u, v, a)`` as full-DOF numpy arrays — consistent with the
        Python coupling path.
        """
        import _aeroelast  # noqa: PLC0415

        if not hasattr(_aeroelast, "run_fsi_solver"):
            raise RuntimeError(
                "_aeroelast was built without FSI bindings (missing run_fsi_solver). "
                "Rebuild aeroelast-py enabling the feature: "
                "cd crates/aeroelast-py && maturin build --release --features fsi"
            )

        # Extract global COO from assembled PETSc matrices.
        # self.M has already been row-sum lumped in Phase 1.
        k_rows, k_cols, k_vals = self._petsc_to_coo(self.K)
        m_rows, m_cols, m_vals = self._petsc_to_coo(self.M)

        free_dofs = bc_manager.free_dofs.astype(np.int32)
        self.free_dofs = bc_manager.free_dofs

        # Optional restart state from checkpoint.
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
        # dt hint; preCICE will override via get_max_time_step_size at each window.
        dt_hint = float(self.solver_params.get("dt", 0.01))

        print(
            f"  → Delegating FSI loop to Rust: "
            f"η_k={self._eta_k:.3e}  η_m={self._eta_m:.3e}  "
            f"β={beta}  γ={gamma}",
            flush=True,
        )

        # ── Per-step callback ─────────────────────────────────────────────────
        # Capture expansion info once (avoids PETSc calls inside the hot path).
        n_total: int = self.K.getSize()[0]
        _free_dofs_arr: np.ndarray = self.free_dofs  # int64 indices
        _fixed_dof_vals: dict = dict(bc_manager.fixed_dofs)  # {dof: prescribed_value}

        def _step_cb(
            t: float,
            time_step: int,
            dt: float,
            u_red: np.ndarray,
            v_red: np.ndarray,
            a_red: np.ndarray,
            force_mag: float,
            forces_iface: np.ndarray,
        ) -> None:
            # Expand reduced vectors to full DOF space (pure numpy — no PETSc).
            u_full = np.zeros(n_total, dtype=np.float64)
            u_full[_free_dofs_arr] = u_red
            for dof, val in _fixed_dof_vals.items():
                u_full[dof] = val

            v_full = np.zeros(n_total, dtype=np.float64)
            v_full[_free_dofs_arr] = v_red

            a_full = np.zeros(n_total, dtype=np.float64)
            a_full[_free_dofs_arr] = a_red

            stress_fields = self._compute_stress_fields(u_full)

            # Reconstruct aerodynamic force fields for VTU export.
            # forces_iface is flat [fx0,fy0,fz0, ...] over interface nodes.
            iface_dofs_flat = interface_dofs_global_flat  # captured from outer scope
            mesh_dim = self.domain.spatial_dim
            force_fields = {}
            if forces_iface is not None and len(forces_iface) > 0:
                f_raw_full = np.zeros(n_total, dtype=np.float64)
                n_iface = len(iface_dofs_flat)
                for local_idx in range(min(len(forces_iface), n_iface)):
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

        u_final_red, v_final_red, a_final_red, times = _aeroelast.run_fsi_solver(
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
                f"  ✓ FSI loop complete: {n_steps} converged steps, t_final={times[-1]:.4f} s",
                flush=True,
            )
        else:
            print("  ⚠️ FSI loop returned 0 converged steps.", flush=True)

        # Flush async checkpoints and create PVD index — mirrors the Python path.
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        # Expand final state to full DOF space and store on self (consistent
        # with the Python coupling path which returns PETSc vecs).
        if n_steps > 0:
            u_final_full = np.zeros(n_total, dtype=np.float64)
            u_final_full[_free_dofs_arr] = np.asarray(u_final_red, dtype=np.float64)
            for dof, val in _fixed_dof_vals.items():
                u_final_full[dof] = val

            v_final_full = np.zeros(n_total, dtype=np.float64)
            v_final_full[_free_dofs_arr] = np.asarray(v_final_red, dtype=np.float64)

            a_final_full = np.zeros(n_total, dtype=np.float64)
            a_final_full[_free_dofs_arr] = np.asarray(a_final_red, dtype=np.float64)
        else:
            u_final_full = np.zeros(n_total, dtype=np.float64)
            v_final_full = np.zeros(n_total, dtype=np.float64)
            a_final_full = np.zeros(n_total, dtype=np.float64)

        self.u = u_final_full
        self.v = v_final_full
        self.a = a_final_full
        return self.u, self.v, self.a

    def _compute_rayleigh_auto(self) -> Tuple[float, float]:
        """Compute Rayleigh damping coefficients automatically from modal analysis.

        Uses the two-point method: given damping ratios ζ_i, ζ_j at two
        natural frequencies ω_i, ω_j, solves the system:

            α (η_k) = 2·(ζ_i·ω_i − ζ_j·ω_j) / (ω_i² − ω_j²)
            β (η_m) = 2·ω_i·ω_j·(ζ_j·ω_i − ζ_i·ω_j) / (ω_i² − ω_j²)

        Tries the Rust fast-path (``_aeroelast.compute_rayleigh_auto``) first.
        Falls back to SLEPc if the Rust extension is unavailable.

        Returns
        -------
        Tuple[float, float]
            (eta_k, eta_m)
        """
        from slepc4py import SLEPc

        cfg = self._damping_cfg
        zeta = float(cfg.get("zeta", 0.02))
        zeta_i = float(cfg["zeta_1"]) if cfg.get("zeta_1") is not None else zeta
        zeta_j = float(cfg["zeta_2"]) if cfg.get("zeta_2") is not None else zeta
        mode_i = int(cfg.get("mode_i", 1))
        mode_j = int(cfg.get("mode_j", 2))
        num_modes = int(cfg.get("num_modes", max(mode_j + 2, 6)))

        if mode_i == mode_j:
            raise ValueError("mode_i and mode_j must be different for Rayleigh auto-computation.")

        # ── Rust fast-path ─────────────────────────────────────────────────
        try:
            import _aeroelast  # noqa: PLC0415
            import numpy as _np  # noqa: PLC0415

            if hasattr(_aeroelast, "compute_rayleigh_auto"):
                k_rows, k_cols, k_vals = self._petsc_to_coo(self.K)
                m_rows, m_cols, m_vals = self._petsc_to_coo(self.M)

                # Temporary BC manager only to obtain free_dofs
                K_dup_tmp = self.K.copy()
                F_tmp2 = K_dup_tmp.createVecRight()
                F_tmp2.set(0.0)
                bc_tmp2 = BoundaryConditionManager(
                    K_dup_tmp, F_tmp2, None, self.domain.dofs_per_node
                )
                bc_tmp2.apply_dirichlet(self.dirichlet_conditions)
                free_dofs_i32 = bc_tmp2.free_dofs.astype(_np.int32)
                K_dup_tmp.destroy()
                F_tmp2.destroy()

                eta_k, eta_m = _aeroelast.compute_rayleigh_auto(
                    k_rows,
                    k_cols,
                    k_vals,
                    m_rows,
                    m_cols,
                    m_vals,
                    free_dofs_i32,
                    num_modes,
                    mode_i,
                    mode_j,
                    zeta_i,
                    zeta_j,
                )
                if eta_k < 0.0 or eta_m < 0.0:
                    _console.print(
                        f"  [yellow bold]⚠ Rayleigh auto: negative coefficient "
                        f"η_k={eta_k:.3e}  η_m={eta_m:.3e}[/yellow bold]"
                    )
                if self._is_primary_rank():
                    print(f"  [Rayleigh auto/Rust] η_k (stiffness) = {eta_k:.4e} s", flush=True)
                    print(f"  [Rayleigh auto/Rust] η_m (mass)      = {eta_m:.4e} 1/s", flush=True)
                return eta_k, eta_m
        except Exception as _rust_err:
            _logger.warning(
                "Rayleigh auto: Rust path failed (%s) — falling back to SLEPc.", _rust_err
            )

        K_dup = self.K.copy()
        M_dup = self.M.copy()
        F_tmp = K_dup.createVecRight()
        F_tmp.set(0.0)

        bc_tmp = BoundaryConditionManager(K_dup, F_tmp, M_dup, self.domain.dofs_per_node)
        bc_tmp.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = bc_tmp.reduced_system

        eps = SLEPc.EPS().create(self.comm)
        eps.setOperators(K_red, M_red)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)
        ksp_st = st.getKSP()
        ksp_st.setType("preonly")
        pc_st = ksp_st.getPC()
        pc_st.setType("lu")
        pc_st.setFactorSolverType("petsc")

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(0.0)
        eps.setDimensions(num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-8, max_it=500)
        try:
            eps.solve()
        except Exception as exc:
            K_red.destroy()
            if M_red is not None:
                M_red.destroy()
            F_red.destroy()
            K_dup.destroy()
            M_dup.destroy()
            F_tmp.destroy()
            raise RuntimeError(f"Rayleigh auto: SLEPc eigenvalue solve failed — {exc}.") from exc

        nconv = eps.getConverged()
        raw_eigs = sorted(eps.getEigenvalue(i).real for i in range(nconv))
        positive_eigs = [lam for lam in raw_eigs if lam > 1e-8]

        K_red.destroy()
        if M_red is not None:
            M_red.destroy()
        F_red.destroy()
        K_dup.destroy()
        M_dup.destroy()
        F_tmp.destroy()

        needed = max(mode_i, mode_j)
        if len(positive_eigs) < needed:
            raise RuntimeError(
                f"Rayleigh auto: modal solve found only {len(positive_eigs)} "
                f"positive eigenvalues but mode {needed} is required."
            )

        omega_i = float(np.sqrt(positive_eigs[mode_i - 1]))
        omega_j = float(np.sqrt(positive_eigs[mode_j - 1]))

        denom = omega_i**2 - omega_j**2
        if abs(denom) < 1e-12:
            raise ValueError(
                f"Rayleigh auto: modes {mode_i} and {mode_j} share the same "
                f"natural frequency (ω ≈ {omega_i:.4e} rad/s)."
            )

        alpha = 2.0 * (zeta_i * omega_i - zeta_j * omega_j) / denom  # η_k
        beta = 2.0 * omega_i * omega_j * (zeta_j * omega_i - zeta_i * omega_j) / denom  # η_m

        if alpha < 0.0 or beta < 0.0:
            _console.print(
                f"  [yellow bold]⚠ Rayleigh auto: negative coefficient "
                f"η_k={alpha:.3e}  η_m={beta:.3e}[/yellow bold]"
            )

        if self._is_primary_rank():
            f_i = omega_i / (2.0 * np.pi)
            f_j = omega_j / (2.0 * np.pi)
            print(
                f"  [Rayleigh auto] Mode {mode_i}: f={f_i:.3f} Hz "
                f"(ω={omega_i:.3f} rad/s), ζ={zeta_i:.4f}",
                flush=True,
            )
            print(
                f"  [Rayleigh auto] Mode {mode_j}: f={f_j:.3f} Hz "
                f"(ω={omega_j:.3f} rad/s), ζ={zeta_j:.4f}",
                flush=True,
            )
            print(f"  [Rayleigh auto] η_k (stiffness) = {alpha:.4e} s", flush=True)
            print(f"  [Rayleigh auto] η_m (mass)      = {beta:.4e} 1/s", flush=True)

        return alpha, beta

    # =========================================================================
    # Utility
    # =========================================================================

    def _is_primary_rank(self) -> bool:
        """Always True in serial."""
        return True

    def _export_interface_debug_data(
        self,
        step: int,
        time: float,
        forces_raw: Optional[np.ndarray],
        forces_applied: np.ndarray,
        directory: str = "debug_interface",
    ):
        """Export interface debugging data (points + vectors) to VTU."""
        # Only proceed if enabled
        if not self.solver_params.get("debug_interface", False):
            return

        # Only proceed if interface coordinates are available
        if not hasattr(self, "_interface_coords"):
            return

        out_dir = Path(directory)
        out_dir.mkdir(exist_ok=True, parents=True)

        points = self._interface_coords
        n_points = len(points)

        # Prepare point data
        point_data = {}

        if forces_raw is not None:
            point_data["Force CFD Raw"] = (
                forces_raw.reshape(-1, 3) if forces_raw.ndim == 1 else forces_raw
            )

        if forces_applied is not None:
            point_data["Force Applied"] = (
                forces_applied.reshape(-1, 3) if forces_applied.ndim == 1 else forces_applied
            )

        if not point_data:
            return

        # Create vertices for point cloud visualization
        # Note: meshio needs cells. For points, use 'vertex' cells.
        cells = [("vertex", np.arange(n_points).reshape(-1, 1))]

        mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)

        # Write VTU file (points only)
        # Using VTU instead of VTP because Paraview handles UnstructuredGrid of vertices well
        # and it's consistent with other outputs
        filename = f"interface_{step:06d}.vtu"
        full_path = out_dir / filename
        mesh.write(str(full_path))

        self._update_debug_pvd(out_dir, "interface_forces.pvd", filename, time)

    def _update_debug_pvd(self, folder: Path, pvd_name: str, filename: str, time: float):
        """Update PVD file for debug stream."""
        pvd_path = folder / pvd_name

        header = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
  <Collection>
"""
        footer = """  </Collection>
</VTKFile>"""

        entry = f'    <DataSet timestep="{time}" group="" part="0" file="{filename}"/>\n'

        if not pvd_path.exists():
            with open(pvd_path, "w") as f:
                f.write(header + entry + footer)
        else:
            # Read existing content
            with open(pvd_path, "r") as f:
                lines = f.readlines()

            # Remove footer lines to append new entry
            valid_lines = [
                line for line in lines if "</Collection>" not in line and "</VTKFile>" not in line
            ]

            # Append new entry and footer
            with open(pvd_path, "w") as f:
                f.writelines(valid_lines)
                f.write(entry)
                f.write(footer)

    def _expand_interface_forces_to_full(
        self,
        interface_forces: np.ndarray,
    ) -> np.ndarray:
        """
        Expand interface forces to full mesh array for visualization.

        Parameters
        ----------
        interface_forces : np.ndarray
            Forces at interface nodes, shape (n_interface_indices, dim).

        Returns
        -------
        np.ndarray
            Full force array, shape (n_nodes, 3) for VTU visualization.
        """
        n_nodes = self.domain.node_count
        full_forces = np.zeros((n_nodes, 3), dtype=np.float64)
        dim = interface_forces.shape[1]

        # Get the proper node indices from the solver
        node_id_to_index = self.domain.node_id_to_index
        interface_node_indices = np.array(
            [node_id_to_index[nid] for nid in self._interface_node_ids],
            dtype=np.int64,
        )

        # Map interface forces to their corresponding mesh node indices
        for i, node_idx in enumerate(interface_node_indices):
            if node_idx < n_nodes:
                full_forces[node_idx, :dim] = interface_forces[i, :]

        return full_forces

    def _compute_stress_fields(self, u_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute stress and strain fields for checkpoint VTU export."""
        sr = StressRecovery(self.domain, u_full)
        has_shell = self.domain.element_family == ElementFamily.SHELL
        has_solid = self.domain.element_family == ElementFamily.SOLID

        out: Dict[str, np.ndarray] = {}
        if has_shell and not has_solid:
            out.update(sr.compute_nodal_stresses_all_layers_dict(stress_type=StressType.TOTAL))
            out.update(sr.compute_nodal_strains_all_layers_dict())
        elif has_solid and not has_shell:
            result = sr.compute_nodal_stresses()
            out.update(result.to_dict())
            out.update({f"strain_{k}": v for k, v in sr.compute_nodal_strains().to_dict().items()})
        else:
            out.update(sr.compute_nodal_stresses_all_layers_dict(stress_type=StressType.TOTAL))
            out.update(sr.compute_nodal_strains_all_layers_dict())
        return out

    # =========================================================================
    # CSV Report & Probes
    # =========================================================================

    def _init_probes(self) -> None:
        """Resolve probe node IDs from solver_params['probes'].

        Accepts a list of ``[x, y, z]`` coordinates.  Each coordinate is
        matched to the nearest mesh node.  Results are written once per
        converged time window to ``<output_folder>/probes.csv``.
        """
        probe_cfg = self.solver_params.get("probes")
        if not probe_cfg:
            return

        coords = np.array([[n.x, n.y, n.z] for n in self.domain.nodes])
        resolved: list[int] = []
        probe_table = Table(show_header=True, box=None, padding=(0, 1))
        probe_table.add_column("#", style="bold")
        probe_table.add_column("Target (x, y, z)")
        probe_table.add_column("Node ID", justify="right")
        probe_table.add_column("Distance", justify="right")
        for i, pt in enumerate(probe_cfg):
            pt = np.asarray(pt, dtype=float)
            dists = np.linalg.norm(coords - pt, axis=1)
            idx = int(np.argmin(dists))
            node = self.domain.nodes[idx]
            resolved.append(idx)
            probe_table.add_row(
                str(i),
                f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})",
                str(node.id),
                f"{dists[idx]:.4e}",
            )
        self._probe_node_ids = resolved
        _console.print(Panel(probe_table, title="Probe Monitoring", border_style="cyan"))

        output_folder = self.solver_params.get("output_folder", "results")
        self._probe_file = str(Path(output_folder) / "probes.csv")

    def _log_structural_report(
        self,
        t: float,
        time_step: int,
        u_full: np.ndarray,
        v_full: np.ndarray,
        a_full: np.ndarray,
        stress_fields: Dict[str, np.ndarray],
        applied_force_mag: float,
    ) -> None:
        """Append one row to ``structural_report.csv`` with key mechanical metrics.

        Written once per converged time window (rank 0 only).

        Columns
        -------
        Time, TimeStep, Max Displacement + components, Max Displacement Node,
        Max Velocity, Max Acceleration,
        Max VonMises TOP/MID/BOT + node, Max Sigma1 TOP,
        Applied Force Magnitude.
        """
        if not self._is_primary_rank():
            return

        output_folder = self.solver_params.get("output_folder", "results")
        csv_path = Path(output_folder) / "structural_report.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        nodes = self.domain.nodes
        n_nodes = len(nodes)
        dofs_per_node = u_full.size // n_nodes

        def _translational_components(field: np.ndarray) -> np.ndarray:
            field_mat = field.reshape(n_nodes, dofs_per_node)
            xyz = np.zeros((n_nodes, 3), dtype=np.float64)
            n_comp = min(3, dofs_per_node)
            xyz[:, :n_comp] = field_mat[:, :n_comp]
            return xyz

        # Displacement magnitude per node
        u_mat = _translational_components(u_full)
        u_mag = np.linalg.norm(u_mat, axis=1)
        max_disp_idx = int(np.argmax(u_mag))
        max_disp = float(u_mag[max_disp_idx])
        max_disp_components = u_mat[max_disp_idx]
        max_disp_node = nodes[max_disp_idx].id

        # Velocity / acceleration magnitudes
        v_mat = _translational_components(v_full)
        a_mat = _translational_components(a_full)
        max_vel = float(np.max(np.linalg.norm(v_mat, axis=1)))
        max_acc = float(np.max(np.linalg.norm(a_mat, axis=1)))

        # Stress peaks — TOP/MID/BOT von Mises + TOP sigma_1
        def _peak(key: str) -> tuple[float, int]:
            arr = stress_fields.get(key)
            if arr is None:
                return 0.0, -1
            idx = int(np.argmax(np.abs(arr)))
            return float(arr[idx]), nodes[idx].id if idx < n_nodes else idx

        vm_top, vm_top_nd = _peak("TOP_von_mises")
        vm_mid, vm_mid_nd = _peak("MID_von_mises")
        vm_bot, vm_bot_nd = _peak("BOT_von_mises")
        s1_top, s1_top_nd = _peak("TOP_sigma_1")

        # Position of max displacement node
        nd = nodes[max_disp_idx]
        max_pos = f"{nd.x:.4f};{nd.y:.4f};{nd.z:.4f}"

        write_header = not csv_path.exists()
        try:
            with open(csv_path, "a") as f:
                if write_header:
                    f.write(
                        "Time [s],TimeStep,"
                        "Max Disp [m],Max Disp X [m],Max Disp Y [m],Max Disp Z [m],"
                        "Max Disp Node,Max Disp Pos (x;y;z),"
                        "Max Vel [m/s],Max Acc [m/s2],"
                        "VonMises TOP [Pa],VonMises TOP Node,"
                        "VonMises MID [Pa],VonMises MID Node,"
                        "VonMises BOT [Pa],VonMises BOT Node,"
                        "Sigma1 TOP [Pa],Sigma1 TOP Node,"
                        "Applied Force [N]\n"
                    )
                f.write(
                    f"{t:.6f},{time_step},"
                    f"{max_disp:.6e},"
                    f"{max_disp_components[0]:.6e},{max_disp_components[1]:.6e},"
                    f"{max_disp_components[2]:.6e},{max_disp_node},{max_pos},"
                    f"{max_vel:.6e},{max_acc:.6e},"
                    f"{vm_top:.6e},{vm_top_nd},"
                    f"{vm_mid:.6e},{vm_mid_nd},"
                    f"{vm_bot:.6e},{vm_bot_nd},"
                    f"{s1_top:.6e},{s1_top_nd},"
                    f"{applied_force_mag:.6e}\n"
                )
        except Exception as e:
            _console.print(f"  [yellow]⚠ Failed to write structural report: {e}[/yellow]")

    def _log_probe_data(
        self,
        t: float,
        time_step: int,
        u_full: np.ndarray,
        v_full: np.ndarray,
        stress_fields: Dict[str, np.ndarray],
    ) -> None:
        """Append probe data for all monitored nodes to ``probes.csv``.

        One row per time step.  For each probe node: displacement (3),
        velocity magnitude, von Mises TOP.
        """
        if not self._probe_node_ids or not self._is_primary_rank():
            return

        nodes = self.domain.nodes
        n_nodes = len(nodes)
        dofs_per_node = u_full.size // n_nodes
        u_mat = u_full.reshape(n_nodes, dofs_per_node)
        v_mat = v_full.reshape(n_nodes, dofs_per_node)

        vm_top = stress_fields.get("TOP_von_mises")

        csv_path = self._probe_file
        write_header = not Path(csv_path).exists()

        try:
            with open(csv_path, "a") as f:
                if write_header:
                    cols = ["Time [s]", "TimeStep"]
                    for i, nid in enumerate(self._probe_node_ids):
                        nd = nodes[nid]
                        tag = f"P{i}(n{nd.id})"
                        cols.extend([
                            f"{tag} Ux [m]",
                            f"{tag} Uy [m]",
                            f"{tag} Uz [m]",
                            f"{tag} |V| [m/s]",
                            f"{tag} VonMises TOP [Pa]",
                        ])
                    f.write(",".join(cols) + "\n")

                parts = [f"{t:.6f}", str(time_step)]
                for nid in self._probe_node_ids:
                    ux, uy, uz = float(u_mat[nid, 0]), float(u_mat[nid, 1]), float(u_mat[nid, 2])
                    vmag = float(np.linalg.norm(v_mat[nid, :3]))
                    vm = float(vm_top[nid]) if vm_top is not None else 0.0
                    parts.extend([
                        f"{ux:.6e}",
                        f"{uy:.6e}",
                        f"{uz:.6e}",
                        f"{vmag:.6e}",
                        f"{vm:.6e}",
                    ])
                f.write(",".join(parts) + "\n")
        except Exception as e:
            _console.print(f"  [yellow]⚠ Failed to write probe data: {e}[/yellow]")

    def solve(self):
        """Perform dynamic analysis using improved Newmark-β method."""

        _console.print()
        _console.rule("[bold]FSI Dynamic Analysis — Structural Solver[/bold]", style="blue")

        # =====================================================================
        # Phase 1: Matrix assembly
        # =====================================================================
        print("  [1/5] Assembling stiffness matrix...", flush=True)
        self.K = self.domain.assemble_stiffness_matrix()

        print("  [2/5] Assembling mass matrix (lumped in Rust)...", flush=True)
        self.M = self.domain.assemble_mass_matrix_lumped()

        # --- mass diagnostic: sum translational DOFs of M_lumped ---
        _m_diag = self.M.createVecRight()
        self.M.getDiagonal(_m_diag)
        _diag_arr = _m_diag.array
        _dofs_per_node = self.domain.dofs_per_node
        _m_total = (
            _diag_arr[0::_dofs_per_node].sum()
            + _diag_arr[1::_dofs_per_node].sum()
            + _diag_arr[2::_dofs_per_node].sum()
        ) / 3.0
        print(
            f"        Total blade mass (M_lumped translational DOFs): {_m_total:.4e} kg", flush=True
        )
        _m_diag.destroy()

        # =====================================================================
        # Phase 2: Damping coefficients
        # =====================================================================
        if not self._damping_enabled:
            print("  [3/5] Rayleigh damping: disabled", flush=True)
        elif self._damping_auto:
            print("  [3/5] Rayleigh damping: auto-computing from modal analysis...", flush=True)
            self._eta_k, self._eta_m = self._compute_rayleigh_auto()
            print(
                f"        Rayleigh damping set (η_m={self._eta_m:.4e}, η_k={self._eta_k:.4e}).",
                flush=True,
            )
        elif self._eta_m != 0.0 or self._eta_k != 0.0:
            print(
                f"  [3/5] Rayleigh damping: η_m={self._eta_m}, η_k={self._eta_k}",
                flush=True,
            )
        else:
            print("  [3/5] Rayleigh damping: disabled (zero coefficients)", flush=True)

        force_temp = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        force_temp.set(0.0)
        self.F = force_temp

        # =====================================================================
        # Phase 3: Boundary conditions
        # =====================================================================
        print("  [4/5] Applying boundary conditions...", flush=True)
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)
        print(
            f"        Fixed: {len(bc_manager.fixed_dofs)} DOFs, Free: {len(bc_manager.free_dofs)} DOFs",
            flush=True,
        )

        # =====================================================================
        # Phase 4: preCICE interface extraction
        # =====================================================================
        print("  [5/5] Extracting preCICE interface nodes...", flush=True)

        # Extract interface information from domain (solver responsibility)
        coupling_boundaries = self.model_properties["solver"]["coupling_boundaries"]
        mesh = self.domain.mesh
        node_sets = [mesh.node_sets[name] for name in coupling_boundaries]
        nodes = {node.id: node.coords for _set in node_sets for node in _set.nodes.values()}
        sorted_node_ids = sorted(nodes.keys())

        self._interface_node_ids = np.array(sorted_node_ids, dtype=np.int64)
        interface_coords = np.array([nodes[nid] for nid in sorted_node_ids])
        if self.domain.spatial_dim == 2 and interface_coords.shape[1] > 2:
            interface_coords = interface_coords[:, :2]
        self._interface_coords = interface_coords

        raw_dofs = np.array([self.domain._node_dofs_map[nid] for nid in sorted_node_ids])
        # For shell elements (6 DOFs/node), keep only translational DOFs
        if raw_dofs.ndim == 2 and raw_dofs.shape[1] > 3:
            self._interface_dofs = raw_dofs[:, :3].astype(PETSc.IntType)
        else:
            self._interface_dofs = raw_dofs.astype(PETSc.IntType)

        # Debug CSV export (solver side)
        np.savetxt(
            "interface_coords.csv",
            interface_coords,
            header="X,Y,Z" if self.domain.spatial_dim == 3 else "X,Y",
            delimiter=",",
        )

        # Create and initialize the Adapter now that coordinates are known
        cfg = self._coupling_cfg
        self._coupling_mesh_name = cfg["coupling_mesh"]
        self._write_data_name = (
            cfg["write_data"] if isinstance(cfg["write_data"], str) else cfg["write_data"][0]
        )
        self._read_data_name = (
            cfg["read_data"] if isinstance(cfg["read_data"], str) else cfg["read_data"][0]
        )

        # ── Delegate to Rust ─────────────────────────────────────────────────
        return self._solve_via_rust(
            bc_manager=bc_manager,
            interface_coords_flat=interface_coords.ravel().astype(np.float64),
            interface_dofs_global_flat=self._interface_dofs.ravel().astype(np.uint64),
        )
