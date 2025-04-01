import os
import warnings
from typing import Dict

import meshio
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import spsolve

from fem_shell.core.bc import BodyForce, BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.solver import Solver


class LinearStaticSolver(Solver):
    """
    A class to perform static analysis on a finite element model.

    Parameters
    ----------
    mesh : MeshModel
        The mesh model containing nodes, elements, and node sets.
    fem_model_properties : dict
        Dictionary containing the properties of the finite element model.
        Required keys:
        - "material": Material properties.
        - "element_family": Type of element family (e.g., SHELL, BEAM).
        - "thickness": Thickness of the element (required for SHELL elements).
    bcs : List[BoundaryCondition]
        List of boundary conditions to apply to the model.

    Raises
    ------
    KeyError
        If any of the required keys are missing in `fem_model_properties`.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)
        self.applyed_forces = False
        self.F = np.zeros(self.domain.dofs_count)

    def add_force_on_dofs(self, dofs, value):
        self.F[np.array(list(dofs)).reshape(-1, len(value))] = np.array(value)
        self.applyed_forces = True

    def solve(self) -> np.ndarray:
        """
        Solve the static finite element problem with both Dirichlet and Neumann conditions.

        Returns
        -------
        np.ndarray
            Displacement vector containing the solution for all DOFs.
        """
        # Assemble the global matrices and load vector.
        self.K = self.domain.assemble_stiffness_matrix()

        if not self.applyed_forces:
            self.F = np.zeros(self.domain.dofs_count)
            for neumann in self.body_forces:
                self.F += self.domain.assemble_load_vector(neumann)

        bc_manager = BoundaryConditionManager(
            self.K, self.F, dof_per_node=self.domain.dofs_per_node
        )
        bc_manager.apply_dirichlet(self.dirichlet_conditions)
        self.A, self.b, _ = bc_manager.reduced_system

        u_red = np.linalg.solve(self.A, self.b)
        self.u = bc_manager.expand_solution(u_red)
        return self.u


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
