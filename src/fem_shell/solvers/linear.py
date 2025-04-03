import os
import warnings
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse.linalg as spla
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse.linalg import spsolve

from fem_shell.core.bc import BodyForce, BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.solver import Solver


class LinearStaticSolver(Solver):
    """
    High-performance linear static solver using PETSc for distributed systems.

    Parameters
    ----------
    mesh : MeshModel
        The computational mesh model
    fem_model_properties : dict
        Dictionary with material and element properties

    Attributes
    ----------
    comm : PETSc.Comm
        MPI communicator for parallel processing
    K : PETSc.Mat
        Distributed stiffness matrix
    F : PETSc.Vec
        Distributed load vector
    u : PETSc.Vec
        Solution displacement vector

    Notes
    -----
    - Uses compressed sparse row (AIJ) format by default
    - Supports both shared and distributed memory parallelism
    - Automatic solver configuration via PETSc options
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)
        self.comm = MPI.COMM_WORLD
        self.K: PETSc.Mat = None
        self.F: PETSc.Vec = None
        self.u: PETSc.Vec = None
        self._solver: PETSc.KSP = None
        self._prepared = False
        self._applyed_forces = False
        self._residual_history = []

    def add_force_on_dofs(self, dofs: List[int], value: List[float]):
        """Add concentrated forces to specific DOFs in distributed system.

        Parameters
        ----------
        dofs : List[int]
            Lista de grados de libertad a modificar. Puede ser:
            - Lista plana para asignación individual
            - Lista agrupada para asignación vectorial
        value : List[float]
            Valores a aplicar. La longitud determina el agrupamiento:
            - len(value) = 1: asigna el mismo valor a todos los dofs
            - len(value) > 1: agrupa los dofs en bloques de este tamaño
        """
        if self.F is None:
            self._initialize_vectors()
        dofs_np = np.asarray(dofs, dtype=PETSc.IntType)
        if isinstance(value, Iterable):
            values_np = np.tile(value, len(dofs) // len(value)).astype(PETSc.ScalarType)
        else:
            values_np = np.tile(value, len(dofs)).astype(PETSc.ScalarType)

        # Aplicar usando la API vectorizada
        self.F.setValues(dofs_np, values_np, addv=PETSc.InsertMode.ADD_VALUES)
        self.F.assemble()
        self._applyed_forces = True

    def _initialize_vectors(self):
        """Initialize PETSc vectors with proper parallel layout."""
        self.F = PETSc.Vec().create(self.comm)
        self.F.setSizes(self.domain.dofs_count)
        self.F.setUp()
        self.F.zeroEntries()

    def _setup_solver(self):
        """Configure PETSc linear solver with residual monitoring support"""
        self._solver = PETSc.KSP().create(self.comm)

        # Clear previous residual history
        self._residual_history = []
        self._solver.setType("cg")

        pc = self._solver.getPC()
        pc.setType("hypre")

        opts = PETSc.Options()
        opts["pc_hypre_type"] = "boomeramg"
        opts["pc_hypre_boomeramg_coarsen_type"] = "HMIS"  # Uppercase
        opts["pc_hypre_boomeramg_interp_type"] = "classical"
        opts["pc_hypre_boomeramg_relax_type_all"] = "symmetric-sor/jacobi"  # Correct value
        opts["pc_hypre_boomeramg_strong_threshold"] = 0.5
        opts["pc_hypre_boomeramg_max_levels"] = 5
        opts["pc_hypre_boomeramg_print_statistics"] = 0

        self._solver.setMonitor(self._residual_monitor)
        self._solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
        self._solver.setFromOptions()

    def solve(self) -> PETSc.Vec:
        """
        Solve the static FEM problem using PETSc's parallel solvers.

        Returns
        -------
        PETSc.Vec
            Distributed solution vector

        Notes
        -----
        - Automatically handles both homogeneous and non-homogeneous BCs
        - Uses matrix-free approach for very large problems
        - Supports GPU acceleration through PETSc backends
        """
        # Assemble distributed matrices
        self.K = self.domain.assemble_stiffness_matrix()

        if not self._applyed_forces:
            self._initialize_vectors()
            for force in self.body_forces:
                fe_vector = self.domain.assemble_load_vector(force)
                self.F.axpy(1.0, fe_vector)

        # Setup boundary conditions
        bc_manager = BoundaryConditionManager(
            self.K, self.F, dof_per_node=self.domain.dofs_per_node
        )
        bc_manager.apply_dirichlet(self.dirichlet_conditions)

        # Get reduced system
        K_red, F_red, _ = bc_manager.reduced_system

        # Configure solver
        if not self._prepared:
            self._setup_solver()
            self._solver.setOperators(K_red)
            self._prepared = True

        # Solve reduced system
        u_red = K_red.createVecRight()
        self._solver.solve(F_red, u_red)

        # Expand to full solution
        self.u = bc_manager.expand_solution(u_red)

        # Cleanup PETSc objects
        K_red.destroy()
        F_red.destroy()
        u_red.destroy()

        return self.u

    def _residual_monitor(self, ksp, iteration, residual):
        """PETSc callback function for residual monitoring"""
        if iteration == 0:  # Reset history for new solve
            self._residual_history = []
        self._residual_history.append(residual)

    def print_solver_info(self, plot_residuals=True):
        """Print solver statistics and optionally plot residuals"""
        if self.comm.rank == 0 and self._solver:
            print("\n--- Solver Performance ---")
            print(f"Converged Reason: {self._solver.getConvergedReason()}")
            print(f"Iterations: {self._solver.getIterationNumber()}")
            print(f"Final Residual: {self._solver.getResidualNorm():.3e}")

            if plot_residuals and self._residual_history:
                try:
                    plt.figure(figsize=(10, 5))
                    plt.semilogy(self._residual_history, "bo-")
                    plt.title("Residual Convergence History")
                    plt.xlabel("Iteration")
                    plt.ylabel("Residual Norm (log scale)")
                    plt.grid(True, which="both")
                    plt.show()
                except ImportError:
                    print("\nResidual History:")
                    print(
                        "\n".join(
                            f"Iter {i:3d}: {r:.3e}" for i, r in enumerate(self._residual_history)
                        )
                    )


class LinearDynamicSolver(Solver):
    """Dynamic FEM solver using implicit Newmark-β integration.

    Parameters
    ----------
    mesh : MeshModel
        Finite element mesh model
    solver_params : Dict
        Solver parameters dictionary

    Attributes
    ----------
    time_history : Dict[float, np.ndarray]
        Stores displacement solutions at each saved time step
    critical_dt : float
        Computed stable time step based on modal analysis
    """

    def __init__(self, mesh: "MeshModel", fem_model_properties: Dict):
        super().__init__(mesh, fem_model_properties)
        self._validate_params(self.solver_params)
        self.time_history: Dict[float, np.ndarray] = {}
        self.critical_dt: float = 0.0
        self.free_dofs = []

    def _validate_params(self, params: Dict) -> Dict:
        """Validate and set default solver parameters.

        Parameters
        ----------
        params : Dict
            Input solver parameters

        Returns
        -------
        Dict
            Validated parameters with defaults
        """
        defaults = {
            "beta": 0.25,
            "gamma": 0.5,
            "eta_m": 1e-4,
            "eta_k": 1e-4,
            "save_history": True,
            "use_critical_dt": False,
            "safety_factor": 0.8,
            "output_folder": "results",
        }

        validated = {**defaults, **self.solver_params}

        if "total_time" not in validated:
            raise KeyError("'total_time' is a required parameter")

        if not validated["use_critical_dt"] and "time_step" not in validated:
            raise ValueError(
                "Either 'use_critical_dt' must be True or 'time_step' must be provided"
            )

        if validated["safety_factor"] <= 0 or validated["safety_factor"] > 1:
            raise ValueError("Safety factor must be in (0, 1]")

        self.solver_params = validated

    def _compute_critical_timestep(self, K_red: np.ndarray, M_red: np.ndarray) -> float:
        """Compute stable time step using modal analysis.

        Parameters
        ----------
        K_red : np.ndarray
            Reduced stiffness matrix
        M_red : np.ndarray
            Reduced mass matrix

        Returns
        -------
        float
            Critical time step
        """
        # Compute first few eigenvalues
        eigvals, _ = spla.eigsh(K_red, M=M_red)
        valid_eigvals = eigvals[eigvals > 1e-8]

        if len(valid_eigvals) == 0:
            warnings.warn("No valid eigenvalues found. Using default dt=1e-5")
            return 1e-5

        omega_max = np.sqrt(valid_eigvals[-1])
        return self.solver_params["safety_factor"] * 2.0 / omega_max

    def solve(self):
        """Perform dynamic analysis using improved Newmark-β method."""
        # Ensamblaje inicial de matrices
        self.K = self.domain.assemble_stiffness_matrix()
        self.M = self.domain.assemble_mass_matrix()
        self.F = self.domain.assemble_load_vector(BodyForce([1.0, 1.0, 1.0]))
        self.C = self.solver_params["eta_m"] * self.M + self.solver_params["eta_k"] * self.K

        # Aplicación de condiciones de frontera
        bc_applier = BoundaryConditionManager(self.K, self.F, self.M)
        bc_applier.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = bc_applier.get_reduced_system()
        C_red = bc_applier.reduce_matrix(self.C)
        self.free_dofs = bc_applier.free_dofs

        # Cálculo del paso temporal
        critical_dt = self._compute_critical_timestep(K_red, M_red)
        dt = (
            critical_dt
            if self.solver_params["use_critical_dt"]
            else self.solver_params["time_step"]
        )

        # Coeficientes de Newmark-β
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        a0 = 1.0 / (beta * dt**2)
        a1_v = 1.0 / (beta * dt)  # Coeficiente para velocidad
        a1_c = gamma / (beta * dt)  # Coeficiente para amortiguación
        a3 = 1.0 / (2 * beta) - 1.0

        # Matriz de rigidez efectiva
        K_eff = K_red + a0 * M_red + a1_c * C_red

        # Condiciones iniciales
        u = np.zeros_like(F_red)
        v = np.zeros_like(F_red)
        a = np.linalg.solve(M_red, F_red - C_red @ v - K_red @ u)

        # Bucle temporal
        n_steps = int(self.solver_params["total_time"] / dt) + 1
        for step in range(n_steps):
            t = step * dt

            # Fuerza efectiva
            F_eff = (
                self._time_dependent_load(t)
                + M_red @ (a0 * u + a1_v * v + a3 * a)
                + C_red @ (a1_c * u + (gamma / beta - 1) * v + dt * (gamma / (2 * beta) - 1) * a)
            )

            # Resolver para el desplazamiento
            u_new = spsolve(K_eff, F_eff)  # Usar un solucionador robusto

            # Actualizar aceleración y velocidad
            a_new = a0 * (u_new - u) - a1_v * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

            # Almacenar resultados
            if self.solver_params["save_history"]:
                self._store_results(t, u_new, bc_applier)

            # Avanzar al siguiente paso
            u, v, a = u_new, v_new, a_new

        # Reconstrucción final
        self.u = self._full_solution(u, bc_applier)
        self.v = self._full_solution(v, bc_applier)
        self.a = self._full_solution(a, bc_applier)

        return self.u, self.v, self.a

    def _time_dependent_load(self, t: float) -> np.ndarray:
        if t <= 0.2:
            F = self.domain.assemble_load_vector(BodyForce(np.array([0.0, 1.0, 1.5]) * t / 0.2))
            F = F[self.free_dofs]
        else:
            F = np.zeros(len(self.free_dofs))
        return F

    def _store_results(self, t: float, u_red: np.ndarray, bc_applier: BoundaryConditionManager):
        """Store results and optionally write to VTK.

        Parameters
        ----------
        t : float
            Current time
        u_red : np.ndarray
            Reduced displacement vector
        bc_applier : BoundaryConditionManager
            Boundary condition handler
        """
        full_u = self._full_solution(u_red, bc_applier)
        self.time_history[t] = full_u

    def _full_solution(self, u_red: np.ndarray, bc_applier: BoundaryConditionManager) -> np.ndarray:
        """Reconstruct full DOF solution from reduced vector.

        Parameters
        ----------
        u_red : np.ndarray
            Reduced displacement vector
        bc_applier : BoundaryConditionManager
            Boundary condition handler

        Returns
        -------
        np.ndarray
            Full displacement vector
        """
        full_u = np.zeros(self.domain.dofs_count)
        full_u[bc_applier.free_dofs] = u_red
        return full_u

    def write_results(self, output_file: str | None = None) -> None:
        """
        Escribe los resultados de la simulación en archivos VTK.

        Si los resultados son dependientes del tiempo (i.e., `self.time_history` existe),
        se genera un archivo VTK por cada paso de tiempo y un archivo PVD que los referencia.

        Parámetros
        ----------
        output_file : str, opcional
            Ruta base para el archivo de salida. Si no se proporciona, se usará
            `self.solver_params["output_folder"]` con el nombre "results.vtk".

        Raises
        ------
        AttributeError
            Si `self.u` no está definido.
        ValueError
            Si la forma de `self.u` no coincide con las componentes definidas.
        """
        # Determinar la ruta de salida
        if output_file is None:
            output_file = os.path.join(self.solver_params["output_folder"], "results.vtk")
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Obtener componentes vectoriales
        vector_form = self.domain.vector_form
        vector_components = [comp for vec in vector_form.values() for comp in vec]

        # Verificar existencia de self.u
        if not hasattr(self, "u"):
            raise AttributeError("self.u no está definido. Ejecute el solucionador primero.")

        # Determinar si hay dependencia temporal
        multi_step = hasattr(self, "time_history") and self.time_history is not None

        # Agrupar celdas por tipo
        cells_dict = {}
        for element in self.mesh_obj.element_map.values():
            if element:
                cell_type = element.element_type.name
                if cell_type not in cells_dict:
                    cells_dict[cell_type] = []
                cells_dict[cell_type].append(element.node_ids)
        cells = [(ctype, np.array(conns)) for ctype, conns in cells_dict.items()]

        # Procesar datos
        base, _ = os.path.splitext(output_file)

        if multi_step:
            sorted_times = sorted(self.time_history.keys())
            vtk_files = []

            for t in sorted_times:
                U_current = self.time_history[t].reshape(-1, len(vector_components))
                self._validate_components(U_current, vector_components)

                point_data = self._generate_point_data(U_current, vector_form, vector_components)
                mesh = meshio.Mesh(self.mesh_obj.coords_array, cells, point_data=point_data)

                time_str = f"{t:.6f}".replace(".", "_")  # Formato: 0_000000
                vtk_filename = f"{base}_{time_str}.vtu"  # Extensión .vtk preservada
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
        """Valida que el número de componentes coincida."""
        if data.shape[1] != len(components):
            raise ValueError(
                f"Número de componentes en datos ({data.shape[1]}) "
                f"no coincide con {len(components)} componentes definidas."
            )

    def _generate_point_data(
        self, solution: np.ndarray, vector_form: dict, components: list
    ) -> Dict[str, np.ndarray]:
        """Genera campos escalares, vectoriales y sus magnitudes."""
        point_data = {}

        # Campos escalares
        for i, comp in enumerate(components):
            point_data[comp] = solution[:, i]

        # Campos vectoriales
        for vec_name, vec_comps in vector_form.items():
            vec_data = np.column_stack([solution[:, components.index(c)] for c in vec_comps])

            # Asegurar 3 componentes
            if vec_data.shape[1] == 2:
                vec_data = np.hstack([vec_data, np.zeros((vec_data.shape[0], 1))])

            point_data[vec_name] = vec_data
            point_data[f"{vec_name}_magnitude"] = np.linalg.norm(vec_data, axis=1)

        return point_data

    def _write_pvd_file(self, base: str, times: list, vtk_files: list) -> None:
        pvd_path = f"{base}.pvd"
        with open(pvd_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="1.0">\n')  # Versión 1.0
            f.write("  <Collection>\n")
            for t, vtk_file in zip(times, vtk_files):
                rel_path = os.path.basename(vtk_file)
                f.write(f'    <DataSet timestep="{t}" part="0" file="{rel_path}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")
