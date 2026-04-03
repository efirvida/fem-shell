"""
Generic FSI Simulation Runner.

This module provides a generic runner that executes FSI simulations
based on YAML configuration files, without requiring any Python code editing.

Example usage:
    from fem_shell.solvers.fsi import FSIRunner

    runner = FSIRunner("simulation.yaml")
    runner.run()

Or from command line:
    python -m fem_shell.cli.run_fsi simulation.yaml
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ...core.bc import BodyForce, DirichletCondition
from ...core.config import (
    ElementFamily,
    FSISimulationConfig,
    MaterialType,
    MeshGeneratorType,
    MeshSource,
    SolverType,
)
from ...core.material import IsotropicMaterial, OrthotropicMaterial
from ...core.mesh import (
    BoxSurfaceMesh,
    BoxVolumeMesh,
    MeshModel,
    MultiFlapMesh,
    RotorMesh,
    SquareShapeMesh,
    check_mesh_quality,
    verify_solid_element_orientations,
)
from ...elements import ElementFamily as ElemFamily

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
        fsi_types = (SolverType.LINEAR_DYNAMIC_FSI.value, SolverType.LINEAR_DYNAMIC_FSI_ROTOR.value)
        if self.config.solver.type in fsi_types:
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
        try:
            self.solver.solve()
        except RuntimeError as exc:
            if self._is_precice_peer_disconnect(exc):
                logger.error("preCICE peer disconnected: %s", exc)
                self._print_precice_disconnect_help(exc)
                raise
            raise

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

    @staticmethod
    def _is_precice_peer_disconnect(exc: BaseException) -> bool:
        """Detect common preCICE socket EOF when the peer participant crashes."""
        msg = str(exc)
        patterns = (
            "Receiving data from another participant",
            "End of file [asio.misc:2]",
            "other participant exited with an error",
        )
        return any(p in msg for p in patterns)

    @staticmethod
    def _print_precice_disconnect_help(exc: BaseException) -> None:
        """Print a user-friendly message for preCICE peer disconnects."""
        import os

        print("\n" + "=" * 80)
        print("PRECICE COUPLING STOPPED (PEER DISCONNECTED)")
        print("=" * 80)
        print("  Solid participant received EOF from preCICE socket.")
        print("  This usually means the fluid participant crashed or exited first.")
        print("\n  Next checks:")
        print("    1) Inspect the fluid participant log around the same timestamp")
        print("    2) Check preCICE adapter messages on the fluid side")
        print("    3) Verify both participants use matching preCICE XML and data names")
        if os.environ.get("FEM_SHELL_DEBUG_TRACEBACK") == "1":
            print(f"\n  Original error: {exc}")
        print("=" * 80)

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

        # Renumber mesh if configured
        if self.config.mesh.renumber:
            print(f"      Renumbering mesh (algorithm={self.config.mesh.renumber})...")
            mesh.renumber_mesh(algorithm=self.config.mesh.renumber, verbose=True)

        # Mesh quality checks for solid elements
        if self.config.elements.family == ElementFamily.SOLID.value:
            print("      Running solid element quality checks...")
            verify_solid_element_orientations(mesh, fix_inplace=True)
            check_mesh_quality(mesh)

        # Create node sets from geometric criteria
        if self.config.mesh.node_sets:
            self._create_node_sets(mesh)

        # Write VTK if configured
        if self.config.mesh.output_file:
            mesh.write_mesh(self.config.mesh.output_file)
            print(f"      Written: {self.config.mesh.output_file}")

        print(f"      Nodes: {len(mesh.nodes)}")
        print(f"      Elements: {len(mesh.elements)}")
        print(f"      Node sets: {list(mesh.node_sets.keys())}")

        # Mesh analysis for FSI coupling (RBF support-radius guidance)
        if self.config.coupling:
            self._log_mesh_analysis(mesh)

        return mesh

    def _try_load_checkpoint_mesh(self) -> Optional[MeshModel]:
        """Try to load deformed mesh from latest checkpoint."""
        from ..checkpoint import CheckpointManager

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

        elif gen_type == MeshGeneratorType.ROTOR.value:
            # Resolve relative path for blade YAML if needed
            yaml_file = params["yaml_file"]
            if self.config_path and not Path(yaml_file).is_absolute():
                # Path relative to config file
                yaml_file = str(self.config_path.parent / yaml_file)

            mesh = RotorMesh(
                yaml_file=yaml_file,
                n_blades=params.get("n_blades", 3),
                hub_radius=params.get("hub_radius"),
                element_size=params.get("element_size", 0.5),
                n_samples=params.get("n_samples", 300),
            ).generate(renumber="rcm")

        elif gen_type == MeshGeneratorType.BOX_VOLUME.value:
            mesh = BoxVolumeMesh(
                center=tuple(params["center"]),
                dims=tuple(params["dims"]),
                nx=params["nx"],
                ny=params["ny"],
                nz=params["nz"],
                element_type=params.get("element_type", "hex"),
                quadratic=params.get("quadratic", False),
            ).generate()

        else:
            raise ValueError(f"Unknown mesh generator type: {gen_type}")

        return mesh

    def _create_node_sets(self, mesh: MeshModel) -> None:
        """Create node sets from geometric criteria defined in configuration."""
        for ns_cfg in self.config.mesh.node_sets:
            kwargs = ns_cfg.params or {}
            nset = mesh.create_node_set_by_geometry(
                name=ns_cfg.name,
                criteria_type=ns_cfg.criteria_type,
                on_surface=ns_cfg.on_surface,
                **kwargs,
            )
            print(f"      Node set '{ns_cfg.name}': {nset.node_count} nodes "
                  f"(criteria={ns_cfg.criteria_type}, on_surface={ns_cfg.on_surface})")

    def _log_mesh_analysis(self, mesh: MeshModel) -> None:
        """Log mesh size statistics and RBF support-radius guidance."""
        import numpy as np
        from itertools import combinations

        print("\n" + "=" * 60)
        print("  MESH SIZE STATISTICS (for RBF support-radius)")
        print("=" * 60)

        # Volume element edge lengths
        edge_lengths = []
        for elem in mesh.elements:
            coords = np.array([n.coords for n in elem.nodes])
            for i, j in combinations(range(len(coords)), 2):
                edge_lengths.append(np.linalg.norm(coords[i] - coords[j]))
        edge_lengths = np.array(edge_lengths)

        print(f"  Volume element edge lengths:")
        print(f"    Min:    {edge_lengths.min():.6f} m")
        print(f"    Max:    {edge_lengths.max():.6f} m")
        print(f"    Mean:   {edge_lengths.mean():.6f} m")
        print(f"    Median: {np.median(edge_lengths):.6f} m")
        print(f"    Std:    {edge_lengths.std():.6f} m")

        # Coupling surface nearest-neighbor spacing
        if self.config.coupling and self.config.coupling.boundaries:
            for boundary_name in self.config.coupling.boundaries:
                ns = mesh.node_sets.get(boundary_name)
                if ns is None or ns.node_count == 0:
                    print(f"\n  WARNING: Coupling surface '{boundary_name}' "
                          f"not found for spacing analysis")
                    continue

                from scipy.spatial import cKDTree

                surface_coords = np.array(
                    [mesh.nodes[nid].coords for nid in ns.node_ids]
                )
                tree = cKDTree(surface_coords)
                dists, _ = tree.query(surface_coords, k=2)
                nn_dists = dists[:, 1]

                print(f"\n  Coupling surface '{boundary_name}' "
                      f"nearest-neighbor spacing:")
                print(f"    Nodes on surface: {ns.node_count}")
                print(f"    Min:    {nn_dists.min():.6f} m")
                print(f"    Max:    {nn_dists.max():.6f} m")
                print(f"    Mean:   {nn_dists.mean():.6f} m")
                print(f"    Std:    {nn_dists.std():.6f} m")
                print(f"\n  >>> Recommended RBF support-radius: "
                      f"{nn_dists.mean() * 4:.6f} m (4x mean spacing)")
                print(f"  >>> Conservative estimate:          "
                      f"{nn_dists.max() * 3:.6f} m (3x max spacing)")

                # Bounding box
                bb_min = surface_coords.min(axis=0)
                bb_max = surface_coords.max(axis=0)
                print(f"\n  Coupling surface bounding box:")
                print(f"    min = ({bb_min[0]:.6f}, {bb_min[1]:.6f}, "
                      f"{bb_min[2]:.6f})")
                print(f"    max = ({bb_max[0]:.6f}, {bb_max[1]:.6f}, "
                      f"{bb_max[2]:.6f})")

        print("=" * 60)

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

        # Element family mapping
        family_map = {
            ElementFamily.PLANE.value: ElemFamily.PLANE,
            ElementFamily.SHELL.value: ElemFamily.SHELL,
            ElementFamily.SOLID.value: ElemFamily.SOLID,
        }
        elem_family = family_map[self.config.elements.family]

        # Base configuration
        model_config: Dict[str, Any] = {
            "solver": {
                "total_time": self.config.solver.total_time,
                "time_step": self.config.solver.time_step,
                "use_critical_dt": self.config.solver.use_critical_dt,
                "safety_factor": self.config.solver.safety_factor,
                "solver_type": self.config.solver.solver_type,
                "debug_interface": self.config.solver.debug_interface,
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
            model_config["solver"]["coupling"] = {
                "participant": self.config.coupling.participant,
                "config_file": self.config.coupling.config_file,
                "coupling_mesh": self.config.coupling.coupling_mesh,
                "write_data": self.config.coupling.write_data,
                "read_data": self.config.coupling.read_data,
            }
            model_config["solver"]["coupling_boundaries"] = self.config.coupling.boundaries

            # Add force limiting parameters if specified
            if self.config.coupling.force_max_cap is not None:
                model_config["solver"]["force_max_cap"] = self.config.coupling.force_max_cap
            if self.config.coupling.force_ramp_time is not None:
                model_config["solver"]["force_ramp_time"] = self.config.coupling.force_ramp_time

        # Add output configuration
        if self.config.output:
            model_config["solver"]["output_folder"] = self.config.output.folder
            model_config["solver"]["write_interval"] = self.config.output.write_interval
            model_config["solver"]["start_from"] = self.config.output.start_from
            model_config["solver"]["start_time"] = self.config.output.start_time
            model_config["solver"]["save_deformed_mesh"] = self.config.output.save_deformed_mesh
            model_config["solver"]["deformed_mesh_scale"] = self.config.output.deformed_mesh_scale
            model_config["solver"]["write_initial_state"] = self.config.output.write_initial_state

        # Add rotor configuration (for LinearDynamicFSIRotor)
        if self.config.solver.rotor:
            model_config["solver"]["rotor"] = self.config.solver.rotor.to_dict()

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
            from ..linear import LinearStaticSolver

            solver = LinearStaticSolver(self.mesh, model_config)

        elif solver_type == SolverType.LINEAR_DYNAMIC.value:
            from ..linear import LinearDynamicSolver

            solver = LinearDynamicSolver(self.mesh, model_config)

        elif solver_type == SolverType.LINEAR_DYNAMIC_FSI.value:
            from . import LinearDynamicFSISolver

            solver = LinearDynamicFSISolver(self.mesh, model_config)

        elif solver_type == SolverType.LINEAR_DYNAMIC_FSI_ROTOR.value:
            from .rotor import LinearDynamicFSIRotorSolver

            solver = LinearDynamicFSIRotorSolver(self.mesh, model_config)

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
            from ...postprocess.precice import FSIDataVisualizer

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

    def visualize(self) -> None:
        """Visualize the model mesh.

        This loads the mesh if needed and opens the interactive viewer.
        """
        if self.mesh is None:
            self.mesh = self._setup_mesh()
        self.mesh.view()

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
    >>> from fem_shell.solvers.fsi import run_from_yaml
    >>> solver = run_from_yaml("simulation.yaml")
    """
    runner = FSIRunner(yaml_path, working_dir)
    return runner.run()
