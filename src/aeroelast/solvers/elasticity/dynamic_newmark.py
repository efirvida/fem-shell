"""Dynamic solver — Newmark-β time integration with PETSc KSP."""

import os
from typing import Dict, Optional, Tuple

import meshio
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from rich.progress import track
from slepc4py import SLEPc

from aeroelast.core.bc import BodyForce, BoundaryConditionManager
from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.checkpoint import CheckpointManager
from aeroelast.solvers.solver import Solver


# FIXME: This class is not well implemented
class DynamicNewmarkSolver(Solver):
    """
    Transient structural solver using the Newmark-β scheme with PETSc KSP.

    Integrates  M·ü + C·u̇ + K·u = F(t)  in time using the implicit
    constant-average-acceleration method (β=0.25, γ=0.5 by default).

    Parameters
    ----------
    mesh : MeshModel
        The computational mesh model.
    fem_model_properties : dict
        Dictionary with material and element properties. Required ``solver``
        sub-keys: ``total_time``, and either ``time_step`` or
        ``use_critical_dt: true``.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        super().__init__(mesh, fem_model_properties)
        self._validate_params(self.solver_params)
        self.time_history: Dict[float, np.ndarray] = {}
        self.critical_dt: float = 0.0
        self.free_dofs = []
        self.comm = MPI.COMM_WORLD
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._init_checkpoint_manager()

    def _validate_params(self, params: Dict) -> Dict:
        """Validate and set default solver parameters."""
        defaults = {
            "beta": 0.25,
            "gamma": 0.5,
            "eta_m": 1e-4,
            "eta_k": 1e-4,
            "save_history": True,
            "use_critical_dt": False,
            "safety_factor": 0.8,
            "output_folder": "results",
            "write_interval": 0.0,
            "start_from": "startTime",
            "save_deformed_mesh": True,
            "deformed_mesh_scale": 1.0,
            "reset_state_on_restart": False,
            "start_time": 0.0,
            "write_initial_state": True,
        }

        params = dict(self.solver_params)
        if "startFrom" in params and "start_from" not in params:
            params["start_from"] = params["startFrom"]
        if "startTime" in params and "start_time" not in params:
            params["start_time"] = params["startTime"]

        validated = {**defaults, **params}

        numeric_params = [
            "beta", "gamma", "eta_m", "eta_k", "safety_factor",
            "total_time", "time_step", "force_max_cap", "force_ramp_time",
            "write_interval",
        ]
        for param in numeric_params:
            if param in validated and validated[param] is not None:
                validated[param] = float(validated[param])

        if "total_time" not in validated:
            raise KeyError("'total_time' is a required parameter")

        if not validated["use_critical_dt"] and "time_step" not in validated:
            raise ValueError(
                "Either 'use_critical_dt' must be True or 'time_step' must be provided"
            )

        if validated["safety_factor"] <= 0 or validated["safety_factor"] > 1:
            raise ValueError("Safety factor must be in (0, 1]")

        if validated["write_interval"] < 0:
            raise ValueError("write_interval must be >= 0")

        if validated["start_from"] not in ("startTime", "latestTime", "firstTime"):
            raise ValueError("start_from must be 'startTime', 'latestTime', or 'firstTime'")

        if not isinstance(validated.get("start_time", 0.0), (int, float)):
            raise ValueError("start_time must be numeric")

        if not isinstance(validated.get("write_initial_state", True), bool):
            raise ValueError("write_initial_state must be a boolean")

        self.solver_params = validated

    def _init_checkpoint_manager(self) -> None:
        """Initialize checkpoint manager if write_interval > 0 or start_from == 'latestTime'."""
        write_interval = self.solver_params.get("write_interval", 0)
        start_from = self.solver_params.get("start_from", "startTime")
        write_initial_state = self.solver_params.get("write_initial_state", True)

        if write_interval > 0 or start_from in ("latestTime", "firstTime") or write_initial_state:
            self._checkpoint_manager = CheckpointManager(
                output_folder=self.solver_params["output_folder"],
                write_interval=write_interval,
                mesh_obj=self.mesh_obj,
                vector_form=self.vector_form,
                dofs_per_node=self.domain.dofs_per_node,
                save_deformed_mesh=self.solver_params.get("save_deformed_mesh", True),
                deformed_mesh_scale=self.solver_params.get("deformed_mesh_scale", 1.0),
                write_initial_state=write_initial_state,
            )

    def _try_restore_checkpoint(self) -> Optional[Dict]:
        """Attempt to restore from latest checkpoint if start_from == 'latestTime'."""
        start_from = self.solver_params.get("start_from", "startTime")

        if start_from not in ("latestTime", "firstTime"):
            return None
        if self._checkpoint_manager is None:
            return None

        if start_from == "firstTime":
            checkpoint_info = self._checkpoint_manager.find_first()
        else:
            checkpoint_info = self._checkpoint_manager.find_latest()

        if checkpoint_info is None:
            print("  ⚠️  No checkpoint found, starting from t=0", flush=True)
            return None

        print(f"  🔄 Restarting from checkpoint: t={checkpoint_info.time:.6f} s", flush=True)

        checkpoint_data = self._checkpoint_manager.load(checkpoint_info.path)
        self._checkpoint_manager.rebuild_pvd_from_disk()

        if start_from == "firstTime":
            return {
                "t": checkpoint_info.time,
                "time_step": checkpoint_info.time_step,
                "dt": checkpoint_data.get("dt", None),
            }

        return checkpoint_data

    def _handle_checkpoint(
        self,
        t: float,
        time_step: int,
        dt: float,
        u_red: np.ndarray,
        v_red: np.ndarray,
        a_red: np.ndarray,
        u_full: np.ndarray,
        v_full: Optional[np.ndarray] = None,
        a_full: Optional[np.ndarray] = None,
        extra_fields: Optional[Dict[str, np.ndarray]] = None,
        **kwargs,
    ) -> None:
        """Handle checkpoint writing if conditions are met."""
        if self._checkpoint_manager is None:
            return
        if not self._checkpoint_manager.should_write(t):
            return

        state = {"u_red": u_red, "v_red": v_red, "a_red": a_red, **kwargs}
        self._checkpoint_manager.write(
            t=t,
            time_step=time_step,
            dt=dt,
            state=state,
            u_full=u_full,
            v_full=v_full,
            a_full=a_full,
            extra_fields=extra_fields,
        )

    def _compute_critical_timestep(self, K: PETSc.Mat, M: PETSc.Mat) -> float:
        """Compute critical time step using SLEPc (largest eigenvalue)."""
        eps = SLEPc.EPS().create(self.comm)
        eps.setOperators(K, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
        eps.setDimensions(nev=1)
        eps.solve()

        nconv = eps.getConverged()
        if nconv < 1:
            raise RuntimeError("No se encontraron valores propios")

        eigval = eps.getEigenvalue(0)
        omega_max = np.sqrt(eigval.real)
        return self.solver_params["safety_factor"] * 2.0 / omega_max

    def solve(self) -> Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec]:
        """Run the Newmark-β time integration loop."""
        K = self.domain.assemble_stiffness_matrix()
        M = self.domain.assemble_mass_matrix()
        F = self.domain.assemble_load_vector(BodyForce([1.0, 1.0, 1.0]))
        C = self._create_damping_matrix(K, M)

        bc_applier = BoundaryConditionManager(K, F, M)
        bc_applier.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = bc_applier.reduced_system
        C_red = bc_applier.reduce_matrix(C)
        self.free_dofs = bc_applier.free_dofs

        ksp = PETSc.KSP().create(self.comm)
        ksp.setType("cg")
        ksp.getPC().setType("ilu")
        ksp.setFromOptions()

        if self.solver_params["use_critical_dt"]:
            dt = self._compute_critical_timestep(K_red, M_red)
        else:
            dt = self.solver_params["time_step"]

        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        a0 = 1.0 / (beta * dt**2)
        a1_v = 1.0 / (beta * dt)
        a1_c = gamma / (beta * dt)
        a3 = 1.0 / (2 * beta) - 1.0

        K_eff = self._assemble_effective_matrix(K_red, M_red, C_red, a0, a1_c)
        ksp.setOperators(K_eff)

        u, v, a = self._initialize_state_vectors(K_red)
        u_new, v_new, a_new = [K_red.createVecRight() for _ in range(3)]

        n_steps = int(self.solver_params["total_time"] / dt) + 1
        for step in track(range(n_steps), description="Processing..."):
            t = step * dt

            F_eff = self._compute_effective_force(M_red, C_red, u, v, a, a0, a1_v, a3, t)
            ksp.solve(F_eff, u_new)
            self._update_acceleration(u_new, u, v, a, a_new, a0, a1_v, a3)
            self._update_velocity(v, a, a_new, v_new, dt, gamma)

            u.copy(u_new)
            v.copy(v_new)
            a.copy(a_new)
            self.u = bc_applier.expand_solution(u_new)

            if self.solver_params["save_history"] and self.comm.rank == 0:
                self._store_results(t, bc_applier.expand_solution(u_new))

        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        return u, v, a

    def _update_acceleration(
        self,
        u_new: PETSc.Vec,
        u: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        a_new: PETSc.Vec,
        a0: float,
        a1_v: float,
        a3: float,
    ):
        """Update acceleration: a_new = a0*(u_new - u) - a1_v*v - a3*a."""
        delta_u = u_new.duplicate()
        delta_u.copy(u_new)
        delta_u.axpy(-1.0, u)

        a_new.set(0.0)
        a_new.axpy(a0, delta_u)
        a_new.axpy(-a1_v, v)
        a_new.axpy(-a3, a)

    def _update_velocity(
        self,
        v: PETSc.Vec,
        a: PETSc.Vec,
        a_new: PETSc.Vec,
        v_new: PETSc.Vec,
        dt: float,
        gamma: float,
    ):
        """Update velocity: v_new = v + dt*((1-γ)*a + γ*a_new)."""
        v_new.copy(v)
        v_new.axpy(dt * (1 - gamma), a)
        v_new.axpy(dt * gamma, a_new)

    def _store_results(self, t: float, u: PETSc.Vec):
        """Store full displacement at time t (MPI rank 0 only)."""
        scatter, u_full = PETSc.Scatter.toZero(u)
        scatter.scatter(u, u_full, False)
        if self.comm.rank == 0:
            self.time_history[t] = u_full.getArray().copy()

    def _create_damping_matrix(self, K: PETSc.Mat, M: PETSc.Mat) -> PETSc.Mat:
        """Build Rayleigh damping matrix C = η_k·K + η_m·M."""
        C = K.copy()
        C.scale(self.solver_params["eta_k"])
        C.axpy(self.solver_params["eta_m"], M)
        return C

    def _assemble_effective_matrix(
        self, K: PETSc.Mat, M: PETSc.Mat, C: PETSc.Mat, a0: float, a1_c: float
    ) -> PETSc.Mat:
        """Assemble K_eff = K + a0·M + a1_c·C."""
        K_eff = K.duplicate()
        K_eff.axpy(a0, M)
        K_eff.axpy(a1_c, C)
        return K_eff

    def _initialize_state_vectors(
        self, K: PETSc.Mat
    ) -> Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec]:
        """Initialize u, v, a state vectors to zero."""
        u = K.createVecRight()
        v = K.createVecRight()
        a = K.createVecRight()
        u.set(0.0)
        v.set(0.0)
        a.set(0.0)
        return u, v, a

    def _compute_effective_force(
        self,
        M: PETSc.Mat,
        C: PETSc.Mat,
        u: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        a0: float,
        a1_v: float,
        a3: float,
        t: float,
    ) -> PETSc.Vec:
        """Compute F_eff = F(t) + M*(a0*u + a1_v*v + a3*a)."""
        F_eff = self._time_dependent_load(t)

        temp = M.createVecRight()
        M.mult(u, temp)
        F_eff.axpy(a0, temp)
        M.mult(v, temp)
        F_eff.axpy(a1_v, temp)
        M.mult(a, temp)
        F_eff.axpy(a3, temp)

        return F_eff

    def _time_dependent_load(self, t: float) -> PETSc.Vec:
        if t <= 0.2:
            F = self.domain.assemble_load_vector(BodyForce(np.array([0.0, 1.0, 1.5]) * t / 0.2))
            return PETSc.Vec().createWithArray(F.array[self.free_dofs])
        else:
            arr = np.zeros(len(self.free_dofs))
            return PETSc.Vec().createWithArray(arr)

    def write_results(self, output_file: str | None = None) -> None:
        """Write time-history results to VTK/PVD files."""
        if output_file is None:
            output_file = os.path.join(self.solver_params["output_folder"], "results.vtk")
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vector_form = self.vector_form
        vector_components = [comp for vec in vector_form.values() for comp in vec]

        if not hasattr(self, "u"):
            raise AttributeError("self.u no está definido. Ejecute el solucionador primero.")

        multi_step = hasattr(self, "time_history") and self.time_history is not None

        cells_dict = {}
        for element in self.mesh_obj.element_map.values():
            if element:
                cell_type = element.element_type.name
                if cell_type not in cells_dict:
                    cells_dict[cell_type] = []
                cells_dict[cell_type].append(element.node_ids)
        cells = [(ctype, np.array(conns)) for ctype, conns in cells_dict.items()]

        base, _ = os.path.splitext(output_file)

        if multi_step:
            sorted_times = sorted(self.time_history.keys())
            vtk_files = []
            for t in sorted_times:
                U_current = self.time_history[t].reshape(-1, len(vector_components))
                self._validate_components(U_current, vector_components)
                point_data = self._generate_point_data(U_current, vector_form, vector_components)
                mesh = meshio.Mesh(self.mesh_obj.coords_array, cells, point_data=point_data)
                time_str = f"{t:.6f}".replace(".", "_")
                vtk_filename = f"{base}_{time_str}.vtu"
                mesh.write(vtk_filename, file_format="vtu")
                vtk_files.append(vtk_filename)
            self._write_pvd_file(base, sorted_times, vtk_files)
        else:
            U = self.u.reshape(-1, len(vector_components))
            self._validate_components(U, vector_components)
            point_data = self._generate_point_data(U, vector_form, vector_components)
            mesh = meshio.Mesh(self.mesh_obj.coords_array, cells, point_data=point_data)
            mesh.write(output_file)

    def _validate_components(self, data: np.ndarray, components: list) -> None:
        if data.shape[1] != len(components):
            raise ValueError(
                f"Número de componentes en datos ({data.shape[1]}) "
                f"no coincide con {len(components)} componentes definidas."
            )

    def _generate_point_data(
        self, solution: np.ndarray, vector_form: dict, components: list
    ) -> Dict[str, np.ndarray]:
        point_data = {}
        for i, comp in enumerate(components):
            point_data[comp] = solution[:, i]
        for vec_name, vec_comps in vector_form.items():
            vec_data = np.column_stack([solution[:, components.index(c)] for c in vec_comps])
            if vec_data.shape[1] == 2:
                vec_data = np.hstack([vec_data, np.zeros((vec_data.shape[0], 1))])
            point_data[vec_name] = vec_data
            point_data[f"{vec_name}_magnitude"] = np.linalg.norm(vec_data, axis=1)
        return point_data

    def _write_pvd_file(self, base: str, times: list, vtk_files: list) -> None:
        pvd_path = f"{base}.pvd"
        with open(pvd_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="1.0">\n')
            f.write("  <Collection>\n")
            for t, vtk_file in zip(times, vtk_files):
                rel_path = os.path.basename(vtk_file)
                f.write(f'    <DataSet timestep="{t}" part="0" file="{rel_path}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")
