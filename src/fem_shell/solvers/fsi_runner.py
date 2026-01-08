"""
Generic FSI Simulation Runner.

This module provides a generic runner that executes FSI simulations
based on YAML configuration files, without requiring any Python code editing.

Example usage:
    from fem_shell.solvers.fsi_runner import FSIRunner

    runner = FSIRunner("simulation.yaml")
    runner.run()

Or from command line:
    python -m fem_shell.cli.run_fsi simulation.yaml
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.bc import BodyForce, DirichletCondition
from ..core.config import (
    ElementFamily,
    FSISimulationConfig,
    MaterialType,
    MeshGeneratorType,
    MeshSource,
    SolverType,
)
from ..core.material import IsotropicMaterial, OrthotropicMaterial
from ..core.mesh import BoxSurfaceMesh, MeshModel, MultiFlapMesh, SquareShapeMesh
from ..elements import ElementFamily as ElemFamily

logger = logging.getLogger(__name__)


class FSIRunner:
    """
    Generic FSI simulation runner that executes simulations from YAML configuration.

    This class handles:
    - Loading mesh from H5 files or generating via mesh generators
    - Creating materials and element configurations
    - Setting up boundary conditions
    - Running the appropriate solver
    - Post-processing results

    Parameters
    ----------
    config : FSISimulationConfig or str or Path
        Configuration object or path to YAML configuration file.
    working_dir : str or Path, optional
        Working directory for the simulation. If None, uses current directory.

    Attributes
    ----------
    config : FSISimulationConfig
        The validated simulation configuration.
    mesh : MeshModel
        The loaded or generated mesh.
    solver : Solver
        The solver instance after setup.

    Examples
    --------
    >>> runner = FSIRunner("simulation.yaml")
    >>> runner.run()

    >>> # Or with configuration object
    >>> config = FSISimulationConfig.from_yaml("simulation.yaml")
    >>> runner = FSIRunner(config)
    >>> runner.run()
    """

    def __init__(
        self,
        config: Union[FSISimulationConfig, str, Path],
        working_dir: Optional[Union[str, Path]] = None,
    ):
        # Load configuration if path provided
        if isinstance(config, (str, Path)):
            self.config_path = Path(config)
            self.config = FSISimulationConfig.from_yaml(config)
        else:
            self.config_path = None
            self.config = config

        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.mesh: Optional[MeshModel] = None
        self.solver = None
        self._material = None

        # Auto-complete configuration from preCICE XML if available
        # This fills in total_time, time_step, and watchpoint_files
        if self.config.solver.type == "LinearDynamicFSI":
            self.config.auto_complete_from_precice()

    def run(self) -> Any:
        """
        Execute the complete FSI simulation pipeline.

        Returns
        -------
        Any
            The solver instance after running (for accessing results).

        Raises
        ------
        RuntimeError
            If simulation fails.
        """
        self._print_header()
        self._validate_config()

        logger.info("Starting FSI simulation...")

        # Step 1: Load or generate mesh
        self.mesh = self._setup_mesh()

        # Step 2: Create material
        self._material = self._create_material()

        # Step 3: Build model configuration
        model_config = self._build_model_config()

        # Step 4: Create and setup solver
        self.solver = self._create_solver(model_config)

        # Step 5: Apply boundary conditions
        self._apply_boundary_conditions()

        # Step 6: Run simulation
        logger.info("Starting solver...")
        self.solver.solve()

        # Step 7: Post-processing
        self._run_postprocessing()

        logger.info("Simulation completed successfully!")
        return self.solver

    def _print_header(self) -> None:
        """Print simulation header."""
        print("\n" + "=" * 70)
        print("  FEM-SHELL FSI SIMULATION RUNNER")
        print("=" * 70)
        print(f"  Configuration: {self.config_path or 'Provided object'}")
        print(f"  Solver: {self.config.solver.type}")
        print(f"  Mesh source: {self.config.mesh.source}")
        print("=" * 70 + "\n")

    def _validate_config(self) -> None:
        """Validate configuration before running."""
        warnings = self.config.validate()
        if warnings:
            for warning in warnings:
                logger.warning("Configuration warning: %s", warning)

    def _setup_mesh(self) -> MeshModel:
        """Load or generate the mesh based on configuration."""
        print("\n[1/6] Setting up mesh...", flush=True)

        mesh = None

        # Check for restart from checkpoint (latestTime)
        if self.config.output and self.config.output.start_from == "latestTime":
            mesh = self._try_load_checkpoint_mesh()

        if mesh is None:
            if self.config.mesh.source == MeshSource.FILE.value:
                mesh = self._load_mesh_from_file()
            else:
                mesh = self._generate_mesh()

        # Write VTK if configured
        if self.config.mesh.output_file:
            mesh.write_mesh(self.config.mesh.output_file)
            print(f"      Written: {self.config.mesh.output_file}")

        print(f"      Nodes: {len(mesh.nodes)}")
        print(f"      Elements: {len(mesh.elements)}")
        print(f"      Node sets: {list(mesh.node_sets.keys())}")

        return mesh

    def _try_load_checkpoint_mesh(self) -> Optional[MeshModel]:
        """Try to load deformed mesh from latest checkpoint."""
        from ..solvers.checkpoint import CheckpointManager

        output_folder = self.config.output.folder if self.config.output else "results"

        print("      Attempting to restore mesh from latest checkpoint...")

        # Create a minimal checkpoint manager to find latest
        cm = CheckpointManager(
            output_folder=output_folder,
            write_interval=0.0,
            mesh_obj=None,
            vector_form={},
            dofs_per_node=3,
            async_write=False,
        )

        latest = cm.find_latest()
        if latest is not None:
            deformed_path = Path(latest.path) / "deformed_mesh.h5"
            if deformed_path.exists():
                print(f"      ✓ Restored deformed mesh: {deformed_path}")
                return MeshModel.load(str(deformed_path))
            else:
                print("      ⚠️  No deformed mesh found in latest checkpoint")
        else:
            print("      ⚠️  No checkpoints found")

        return None

    def _load_mesh_from_file(self) -> MeshModel:
        """Load mesh from H5 or pickle file."""
        file_config = self.config.mesh.file
        if file_config is None:
            raise ValueError("Mesh file configuration is missing")

        file_path = Path(file_config.path)

        if not file_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {file_path}")

        print(f"      Loading mesh from: {file_path}")
        return MeshModel.load(str(file_path), format=file_config.format)

    def _generate_mesh(self) -> MeshModel:
        """Generate mesh using the configured generator."""
        gen_config = self.config.mesh.generator
        if gen_config is None:
            raise ValueError("Mesh generator configuration is missing")

        gen_type = gen_config.type
        params = gen_config.params

        print(f"      Generating mesh with: {gen_type}")

        if gen_type == MeshGeneratorType.SQUARE.value:
            mesh = SquareShapeMesh(
                width=params["width"],
                height=params["height"],
                nx=params["nx"],
                ny=params["ny"],
                quadratic=params.get("quadratic", False),
                triangular=params.get("triangular", False),
            ).generate()

        elif gen_type == MeshGeneratorType.BOX.value:
            mesh = BoxSurfaceMesh(
                center=tuple(params["center"]),
                dims=tuple(params["dims"]),
                nx=params["nx"],
                ny=params["ny"],
                nz=params["nz"],
                quadratic=params.get("quadratic", False),
                triangular=params.get("triangular", False),
            ).generate()

        elif gen_type == MeshGeneratorType.MULTIFLAP.value:
            mesh = MultiFlapMesh(
                n_flaps=params["n_flaps"],
                flap_width=params["flap_width"],
                flap_height=params["flap_height"],
                x_spacing=params["x_spacing"],
                base_height=params.get("base_height", 0.05),
                nx_flap=params.get("nx_flap", 4),
                ny_flap=params.get("ny_flap", 20),
                nx_base_segment=params.get("nx_base_segment", 10),
                ny_base=params.get("ny_base", 2),
                quadratic=params.get("quadratic", False),
            ).generate()

        else:
            raise ValueError(f"Unknown mesh generator type: {gen_type}")

        return mesh

    def _create_material(self):
        """Create material from configuration."""
        print("\n[2/6] Creating material...", flush=True)

        mat_config = self.config.material

        if mat_config.type == MaterialType.ISOTROPIC.value:
            if mat_config.E is None or mat_config.nu is None or mat_config.rho is None:
                raise ValueError("Isotropic material requires E, nu, and rho")
            e_val = float(mat_config.E) if not isinstance(mat_config.E, list) else mat_config.E[0]
            nu_val = (
                float(mat_config.nu) if not isinstance(mat_config.nu, list) else mat_config.nu[0]
            )
            material = IsotropicMaterial(
                name=mat_config.name,
                E=e_val,
                nu=nu_val,
                rho=float(mat_config.rho),
            )
        else:
            if (
                mat_config.E is None
                or mat_config.G is None
                or mat_config.nu is None
                or mat_config.rho is None
            ):
                raise ValueError("Orthotropic material requires E, G, nu, and rho")
            # Build proper 3-tuples for orthotropic material
            e_list = mat_config.E if isinstance(mat_config.E, list) else [mat_config.E] * 3
            g_list = (
                mat_config.G
                if len(mat_config.G) >= 3
                else mat_config.G + [mat_config.G[0]] * (3 - len(mat_config.G))
            )
            nu_list = mat_config.nu if isinstance(mat_config.nu, list) else [mat_config.nu] * 3
            material = OrthotropicMaterial(
                name=mat_config.name,
                E=(float(e_list[0]), float(e_list[1]), float(e_list[2])),
                G=(float(g_list[0]), float(g_list[1]), float(g_list[2])),
                nu=(float(nu_list[0]), float(nu_list[1]), float(nu_list[2])),
                rho=float(mat_config.rho),
            )

        print(f"      Material: {material.name}")
        print(f"      Type: {mat_config.type}")
        if mat_config.type == MaterialType.ISOTROPIC.value:
            print(f"      E={mat_config.E}, nu={mat_config.nu}, rho={mat_config.rho}")

        return material

    def _build_model_config(self) -> Dict[str, Any]:
        """Build the model configuration dictionary for the solver."""
        print("\n[3/6] Building model configuration...", flush=True)

        # Element family
        elem_family = (
            ElemFamily.PLANE
            if self.config.elements.family == ElementFamily.PLANE.value
            else ElemFamily.SHELL
        )

        # Base configuration
        model_config: Dict[str, Any] = {
            "solver": {
                "total_time": self.config.solver.total_time,
                "time_step": self.config.solver.time_step,
                "use_critical_dt": self.config.solver.use_critical_dt,
                "safety_factor": self.config.solver.safety_factor,
            },
            "elements": {
                "material": self._material,
                "element_family": elem_family,
            },
        }

        # Add thickness for shell elements
        if self.config.elements.thickness:
            model_config["elements"]["thickness"] = self.config.elements.thickness

        # Add Newmark parameters
        if self.config.solver.newmark:
            model_config["solver"]["beta"] = self.config.solver.newmark.beta
            model_config["solver"]["gamma"] = self.config.solver.newmark.gamma

        # Add damping parameters
        if self.config.solver.damping:
            model_config["solver"]["eta_m"] = self.config.solver.damping.eta_m
            model_config["solver"]["eta_k"] = self.config.solver.damping.eta_k

        # Add coupling configuration for FSI
        if self.config.coupling:
            if self.config.coupling.is_inline:
                # Pass inline configuration as dictionary
                model_config["solver"]["adapter_cfg"] = self.config.coupling.to_adapter_dict()
                # Store base path for resolving relative paths in adapter
                model_config["base_path"] = str(
                    self.config_path.parent if self.config_path else self.working_dir
                )
            else:
                # External file mode
                model_config["solver"]["adapter_cfg"] = self.config.coupling.adapter_config
            model_config["solver"]["coupling_boundaries"] = self.config.coupling.boundaries

        # Add output configuration
        if self.config.output:
            model_config["solver"]["output_folder"] = self.config.output.folder
            model_config["solver"]["write_interval"] = self.config.output.write_interval
            model_config["solver"]["start_from"] = self.config.output.start_from
            model_config["solver"]["start_time"] = self.config.output.start_time
            model_config["solver"]["save_deformed_mesh"] = self.config.output.save_deformed_mesh
            model_config["solver"]["deformed_mesh_scale"] = self.config.output.deformed_mesh_scale
            model_config["solver"]["write_initial_state"] = self.config.output.write_initial_state

        print(f"      Solver type: {self.config.solver.type}")
        print(f"      Element family: {self.config.elements.family}")

        return model_config

    def _create_solver(self, model_config: Dict[str, Any]):
        """Create the appropriate solver based on configuration."""
        print("\n[4/6] Creating solver...", flush=True)

        if self.mesh is None:
            raise RuntimeError("Mesh must be loaded before creating solver")

        solver_type = self.config.solver.type

        if solver_type == SolverType.LINEAR_STATIC.value:
            from ..solvers.linear import LinearStaticSolver

            solver = LinearStaticSolver(self.mesh, model_config)

        elif solver_type == SolverType.LINEAR_DYNAMIC.value:
            from ..solvers.linear import LinearDynamicSolver

            solver = LinearDynamicSolver(self.mesh, model_config)

        elif solver_type == SolverType.LINEAR_DYNAMIC_FSI.value:
            from ..solvers.fsi import LinearDynamicFSISolver

            solver = LinearDynamicFSISolver(self.mesh, model_config)

        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        print(f"      Created: {solver_type}")
        return solver

    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to the solver."""
        print("\n[5/6] Applying boundary conditions...", flush=True)

        if self.solver is None or self.mesh is None:
            raise RuntimeError("Solver and mesh must be initialized before applying BCs")

        # Apply Dirichlet conditions
        dirichlet_conditions = []
        for bc_config in self.config.boundary_conditions.dirichlet:
            try:
                dofs = self.solver.get_dofs_by_nodeset_name(bc_config.nodeset)
                dirichlet_conditions.append(DirichletCondition(dofs, bc_config.value))
                print(
                    f"      Dirichlet BC: '{bc_config.nodeset}' = {bc_config.value} ({len(dofs)} DOFs)"
                )
            except KeyError:
                raise ValueError(
                    f"Node set '{bc_config.nodeset}' not found in mesh. "
                    f"Available: {list(self.mesh.node_sets.keys())}"
                )

        if dirichlet_conditions:
            self.solver.add_dirichlet_conditions(dirichlet_conditions)

        # Apply body forces
        body_forces = []
        for bf_config in self.config.boundary_conditions.body_forces:
            body_forces.append(BodyForce(bf_config.value))
            print(f"      Body force: {bf_config.value}")

        if body_forces:
            self.solver.add_body_forces(body_forces)

        print(f"      Total Dirichlet BCs: {len(dirichlet_conditions)}")
        print(f"      Total body forces: {len(body_forces)}")

    def _run_postprocessing(self) -> None:
        """Run post-processing if configured."""
        if not self.config.postprocess:
            return

        print("\n[Post-processing]", flush=True)

        try:
            from ..postprocess.precice import FSIDataVisualizer

            if self.config.postprocess.watchpoint_file:
                visualizer = FSIDataVisualizer(self.config.postprocess.watchpoint_file)

                if self.config.postprocess.plots:
                    if "displacement" in self.config.postprocess.plots:
                        save_path = self.config.postprocess.plots["displacement"]
                        visualizer.plot_displacement(save_path=save_path)
                        print(f"      Saved: {save_path}")

                    if "force" in self.config.postprocess.plots:
                        save_path = self.config.postprocess.plots["force"]
                        visualizer.plot_force(save_path=save_path)
                        print(f"      Saved: {save_path}")

        except Exception as e:
            logger.warning("Post-processing failed: %s", e)
            print(f"      Warning: Could not generate plots: {e}")

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_available_nodesets(self) -> list:
        """Get list of available node sets in the mesh.

        Returns
        -------
        list
            Names of available node sets.

        Raises
        ------
        RuntimeError
            If mesh has not been loaded yet.
        """
        if self.mesh is None:
            raise RuntimeError("Mesh not loaded. Call run() or _setup_mesh() first.")
        return list(self.mesh.node_sets.keys())

    def preview_config(self) -> str:
        """Get a preview of the configuration.

        Returns
        -------
        str
            Human-readable configuration summary.
        """
        return str(self.config)


def run_from_yaml(yaml_path: Union[str, Path], working_dir: Optional[str] = None) -> Any:
    """
    Convenience function to run FSI simulation from YAML file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.
    working_dir : str, optional
        Working directory for the simulation.

    Returns
    -------
    Any
        The solver instance after running.

    Examples
    --------
    >>> from fem_shell.solvers.fsi_runner import run_from_yaml
    >>> solver = run_from_yaml("simulation.yaml")
    """
    runner = FSIRunner(yaml_path, working_dir)
    return runner.run()
