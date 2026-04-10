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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    BladeMesh,
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
        self._console = Console()
        self._material = None
        self._blade_properties = None  # Composite properties from blade YAML
        self._mesh_generator = None  # Stored for property extraction
        self._precice_info = None

        # Auto-complete configuration from preCICE XML if available
        # This fills in total_time, time_step, and watchpoint_files
        fsi_types = (SolverType.LINEAR_DYNAMIC_FSI.value, SolverType.LINEAR_DYNAMIC_FSI_ROTOR.value)
        if self.config.solver.type in fsi_types:
            self._precice_info = self.config.load_precice_config()
            self.config.auto_complete_from_precice()

    @property
    def _is_modal(self) -> bool:
        return self.config.solver.type == SolverType.MODAL.value

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
        self._console.print("\n[bold cyan]\\[6/6] Running solver...[/bold cyan]")
        try:
            result = self.solver.solve()
        except RuntimeError as exc:
            if not self._is_modal and self._is_precice_peer_disconnect(exc):
                self._print_precice_disconnect_help(exc)
                raise
            raise

        # Step 7: Post-processing
        if self._is_modal:
            self._print_modal_results(result)
        else:
            self._run_postprocessing()

        self._console.print("\n[bold green]Simulation completed successfully.[/bold green]")
        return self.solver

    def export_calculix(
        self,
        output_path: str,
        num_modes: int = 10,
        span_direction: Optional[tuple] = None,
    ) -> None:
        """Export mesh with composite materials to CalculiX .inp format.

        Must be called after ``_setup_mesh()`` and ``_create_material()``
        (or after ``run()``).

        Parameters
        ----------
        output_path : str
            Path to the output .inp file.
        num_modes : int, optional
            Number of eigenvalues for the *FREQUENCY step. Default 10.
        span_direction : tuple of 3 floats, optional
            Global span axis of the structure.  For a wind turbine blade
            along Z pass ``(0., 0., 1.)``.  When given, all ply angles are
            referenced to this axis so that 0-degree plies align with the
            span rather than the (chordwise) first element edge.
        """
        from fem_shell.core.mesh.io.writers import write_ccx_mesh

        if self.mesh is None:
            self.mesh = self._setup_mesh()
        if self._blade_properties is None:
            self._material = self._create_material()

        # Determine boundary node set for clamping
        boundary_nodeset = None
        if self.config.boundary_conditions and self.config.boundary_conditions.dirichlet:
            boundary_nodeset = self.config.boundary_conditions.dirichlet[0].nodeset

        write_ccx_mesh(
            self.mesh,
            output_path,
            properties=self._blade_properties,
            boundary_nodeset=boundary_nodeset,
            num_modes=num_modes,
            span_direction=span_direction,
        )

    def _print_header(self) -> None:
        """Print simulation header."""
        content = Text()
        content.append("Configuration: ", style="bold")
        content.append(f"{self.config_path or 'Provided object'}\n")
        content.append("Solver:        ", style="bold")
        content.append(f"{self.config.solver.type}\n")
        content.append("Mesh source:   ", style="bold")
        content.append(f"{self.config.mesh.source}")
        self._console.print()
        self._console.print(Panel(content, title="FEM-SHELL FSI SIMULATION RUNNER",
                                  border_style="cyan", expand=False))

    def _validate_config(self) -> None:
        """Validate configuration before running."""
        warnings = self.config.validate()
        if warnings:
            for warning in warnings:
                logger.warning("Configuration warning: %s", warning)

        # Validate time parameters against preCICE XML
        self._validate_precice_time()

    def _validate_precice_time(self) -> None:
        """Check that YAML time params match preCICE XML values."""
        if not self._precice_info:
            return

        precice_time = self._precice_info.time
        solver_cfg = self.config.solver

        mismatches = []

        if precice_time.time_window_size and solver_cfg.time_step:
            yaml_dt = solver_cfg.time_step
            xml_dt = precice_time.time_window_size
            if abs(yaml_dt - xml_dt) / max(abs(xml_dt), 1e-30) > 1e-6:
                mismatches.append(
                    f"time_step: YAML={yaml_dt:.2e} vs preCICE={xml_dt:.2e}"
                )

        if precice_time.max_time and solver_cfg.total_time:
            yaml_t = solver_cfg.total_time
            xml_t = precice_time.max_time
            if abs(yaml_t - xml_t) / max(abs(xml_t), 1e-30) > 1e-6:
                mismatches.append(
                    f"total_time: YAML={yaml_t:.2e} vs preCICE={xml_t:.2e}"
                )

        if mismatches:
            lines = "\n".join(f"⚠ {m}" for m in mismatches)
            lines += "\n\nThe preCICE XML defines the coupling time window."
            lines += "\nMismatched values may cause unexpected behavior."
            self._console.print()
            self._console.print(Panel(lines, title="YAML ↔ preCICE time mismatch",
                                      border_style="bold yellow", expand=False))
            self._console.print()

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

        console = Console()
        lines = (
            "Solid participant received EOF from preCICE socket.\n"
            "This usually means the fluid participant crashed or exited first.\n\n"
            "[bold]Next checks:[/bold]\n"
            "  1) Inspect the fluid participant log around the same timestamp\n"
            "  2) Check preCICE adapter messages on the fluid side\n"
            "  3) Verify both participants use matching preCICE XML and data names"
        )
        if os.environ.get("FEM_SHELL_DEBUG_TRACEBACK") == "1":
            lines += f"\n\nOriginal error: {exc}"
        console.print()
        console.print(Panel(lines, title="PRECICE COUPLING STOPPED (PEER DISCONNECTED)",
                            border_style="bold red", expand=False))

    def _setup_mesh(self) -> MeshModel:
        """Load or generate the mesh based on configuration."""
        self._console.print("\n[bold cyan]\\[1/6] Setting up mesh...[/bold cyan]")

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
            self._console.print(
                f"      Renumbering mesh (algorithm={self.config.mesh.renumber})..."
            )
            mesh.renumber_mesh(algorithm=self.config.mesh.renumber, verbose=True)

        # Mesh quality checks for solid elements
        if self.config.elements.family == ElementFamily.SOLID.value:
            self._console.print("      Running solid element quality checks...")
            verify_solid_element_orientations(mesh, fix_inplace=True)
            check_mesh_quality(mesh)

        # Create node sets from geometric criteria
        if self.config.mesh.node_sets:
            self._create_node_sets(mesh)

        # Write VTK if configured
        if self.config.mesh.output_file:
            mesh.write_mesh(self.config.mesh.output_file)
            self._console.print(f"      Written: {self.config.mesh.output_file}")

        # Print mesh statistics
        self._print_mesh_statistics(mesh)

        # Mesh analysis for FSI coupling (RBF support-radius guidance)
        if self.config.coupling:
            self._log_mesh_analysis(mesh)

        return mesh

    def _print_mesh_statistics(self, mesh: MeshModel) -> None:
        """Print mesh topology and element size statistics."""
        import numpy as np
        from collections import Counter
        from itertools import combinations

        # --- Topology overview ---
        topo = Table(title="Mesh Topology", title_style="bold cyan",
                     show_lines=False, pad_edge=True, expand=False)
        topo.add_column("Property", style="bold")
        topo.add_column("Value", justify="right")
        topo.add_row("Nodes", f"{len(mesh.nodes):,}")
        topo.add_row("Elements", f"{len(mesh.elements):,}")
        topo.add_row("Node sets", ", ".join(mesh.node_sets.keys()) or "—")

        # Count element types
        type_counts = Counter(e.element_type.name for e in mesh.elements)
        for etype, cnt in sorted(type_counts.items()):
            topo.add_row(f"  {etype}", f"{cnt:,}")

        self._console.print()
        self._console.print(topo)

        # --- Element size statistics ---
        all_edge_lengths = []
        elem_max_sizes = []
        elem_min_sizes = []

        for elem in mesh.elements:
            coords = elem.node_coords
            n_corner = {3: 3, 4: 4, 6: 3, 8: 4, 9: 4, 16: 4,
                        10: 4, 20: 8}.get(len(coords), len(coords))
            corners = coords[:n_corner]
            edges = []
            for i, j in combinations(range(n_corner), 2):
                edges.append(np.linalg.norm(corners[i] - corners[j]))
            if edges:
                all_edge_lengths.extend(edges)
                elem_max_sizes.append(max(edges))
                elem_min_sizes.append(min(edges))

        if not all_edge_lengths:
            return

        all_edge_lengths = np.array(all_edge_lengths)
        elem_max_sizes = np.array(elem_max_sizes)
        elem_min_sizes = np.array(elem_min_sizes)

        stats = Table(title="Element Size Statistics",
                      title_style="bold cyan", show_lines=False,
                      pad_edge=True, expand=False)
        stats.add_column("Metric", style="bold")
        stats.add_column("Min", justify="right")
        stats.add_column("Mean", justify="right")
        stats.add_column("Max", justify="right")

        stats.add_row(
            "Edge length [m]",
            f"{all_edge_lengths.min():.4f}",
            f"{all_edge_lengths.mean():.4f}",
            f"{all_edge_lengths.max():.4f}",
        )
        stats.add_row(
            "Element max edge [m]",
            f"{elem_max_sizes.min():.4f}",
            f"{elem_max_sizes.mean():.4f}",
            f"{elem_max_sizes.max():.4f}",
        )
        stats.add_row(
            "Element min edge [m]",
            f"{elem_min_sizes.min():.4f}",
            f"{elem_min_sizes.mean():.4f}",
            f"{elem_min_sizes.max():.4f}",
        )

        # Aspect ratio (max_edge / min_edge per element)
        with np.errstate(divide="ignore", invalid="ignore"):
            aspect = np.where(elem_min_sizes > 0,
                              elem_max_sizes / elem_min_sizes, 0.0)
        if aspect.any():
            stats.add_row(
                "Aspect ratio",
                f"{aspect[aspect > 0].min():.2f}" if (aspect > 0).any() else "—",
                f"{aspect[aspect > 0].mean():.2f}" if (aspect > 0).any() else "—",
                f"{aspect.max():.2f}",
            )

        # Bounding box
        all_coords = np.array([n.coords for n in mesh.nodes])
        bb_min = all_coords.min(axis=0)
        bb_max = all_coords.max(axis=0)
        bb_size = bb_max - bb_min

        stats.add_section()
        stats.add_row(
            "Bounding box [m]",
            f"({bb_min[0]:.3f}, {bb_min[1]:.3f}, {bb_min[2]:.3f})",
            "",
            f"({bb_max[0]:.3f}, {bb_max[1]:.3f}, {bb_max[2]:.3f})",
        )
        stats.add_row(
            "Domain size [m]",
            "",
            "",
            f"({bb_size[0]:.3f}, {bb_size[1]:.3f}, {bb_size[2]:.3f})",
        )

        self._console.print(stats)

    def _try_load_checkpoint_mesh(self) -> Optional[MeshModel]:
        """Try to load deformed mesh from latest checkpoint."""
        from ..checkpoint import CheckpointManager

        output_folder = self.config.output.folder if self.config.output else "results"

        self._console.print("      Attempting to restore mesh from latest checkpoint...")

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
                self._console.print(
                    f"      [green]✓[/green] Restored deformed mesh: {deformed_path}"
                )
                return MeshModel.load(str(deformed_path))
            else:
                self._console.print(
                    "      [yellow]⚠[/yellow]  No deformed mesh found in latest checkpoint"
                )
        else:
            self._console.print(
                "      [yellow]⚠[/yellow]  No checkpoints found"
            )

        return None

    def _load_mesh_from_file(self) -> MeshModel:
        """Load mesh from H5 or pickle file."""
        file_config = self.config.mesh.file
        if file_config is None:
            raise ValueError("Mesh file configuration is missing")

        file_path = Path(file_config.path)

        if not file_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {file_path}")

        self._console.print(f"      Loading mesh from: {file_path}")
        return MeshModel.load(str(file_path), format=file_config.format)

    def _generate_mesh(self) -> MeshModel:
        """Generate mesh using the configured generator."""
        gen_config = self.config.mesh.generator
        if gen_config is None:
            raise ValueError("Mesh generator configuration is missing")

        gen_type = gen_config.type
        params = gen_config.params

        self._console.print(f"      Generating mesh with: [bold]{gen_type}[/bold]")

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

            generator = RotorMesh(
                yaml_file=yaml_file,
                n_blades=params.get("n_blades", 3),
                hub_radius=params.get("hub_radius"),
                element_size=params.get("element_size", 0.5),
                n_samples=params.get("n_samples", 300),
            )
            mesh = generator.generate(renumber="rcm")
            self._mesh_generator = generator

        elif gen_type == MeshGeneratorType.BLADE.value:
            # Single blade mesh from YAML or NuMAD Excel definition
            yaml_file = params.get("yaml_file")
            excel_file = params.get("excel_file")
            airfoil_dir = params.get("airfoil_dir")

            if yaml_file and self.config_path and not Path(yaml_file).is_absolute():
                yaml_file = str(self.config_path.parent / yaml_file)
            if excel_file and self.config_path and not Path(excel_file).is_absolute():
                excel_file = str(self.config_path.parent / excel_file)
            if airfoil_dir and self.config_path and not Path(airfoil_dir).is_absolute():
                airfoil_dir = str(self.config_path.parent / airfoil_dir)

            generator = BladeMesh(
                yaml_file=yaml_file,
                excel_file=excel_file,
                airfoil_dir=airfoil_dir,
                element_size=params.get("element_size", 0.15),
                n_samples=params.get("n_samples", 300),
                span_grading=params.get("span_grading", "chord"),
            )
            mesh = generator.generate(renumber=None)
            self._mesh_generator = generator

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
            if ns_cfg.name in mesh.node_sets:
                self._console.print(
                    f"      Node set [bold]'{ns_cfg.name}'[/bold]: already exists "
                    f"({mesh.node_sets[ns_cfg.name].node_count} nodes), skipping"
                )
                continue
            kwargs = ns_cfg.params or {}
            nset = mesh.create_node_set_by_geometry(
                name=ns_cfg.name,
                criteria_type=ns_cfg.criteria_type,
                on_surface=ns_cfg.on_surface,
                **kwargs,
            )
            self._console.print(
                f"      Node set [bold]'{ns_cfg.name}'[/bold]: {nset.node_count} nodes "
                f"(criteria={ns_cfg.criteria_type}, on_surface={ns_cfg.on_surface})"
            )

    def _log_mesh_analysis(self, mesh: MeshModel) -> None:
        """Log mesh size statistics and RBF support-radius guidance."""
        import numpy as np
        from itertools import combinations

        # Edge lengths for RBF radius estimation
        edge_lengths = []
        for elem in mesh.elements:
            coords = np.array([n.coords for n in elem.nodes])
            for i, j in combinations(range(len(coords)), 2):
                edge_lengths.append(np.linalg.norm(coords[i] - coords[j]))
        edge_lengths = np.array(edge_lengths)

        tbl = Table(title="RBF Support-Radius Guidance",
                    title_style="bold cyan", show_lines=False,
                    pad_edge=True, expand=False)
        tbl.add_column("Metric", style="bold")
        tbl.add_column("Value", justify="right")

        tbl.add_row("Edge length min", f"{edge_lengths.min():.6f} m")
        tbl.add_row("Edge length max", f"{edge_lengths.max():.6f} m")
        tbl.add_row("Edge length mean", f"{edge_lengths.mean():.6f} m")
        tbl.add_row("Edge length median", f"{np.median(edge_lengths):.6f} m")

        # Coupling surface nearest-neighbor spacing
        if self.config.coupling and self.config.coupling.boundaries:
            for boundary_name in self.config.coupling.boundaries:
                ns = mesh.node_sets.get(boundary_name)
                if ns is None or ns.node_count == 0:
                    self._console.print(
                        f"      [yellow]⚠[/yellow] Coupling surface "
                        f"'{boundary_name}' not found for spacing analysis"
                    )
                    continue

                from scipy.spatial import cKDTree

                surface_coords = np.array(
                    [mesh.nodes[nid].coords for nid in ns.node_ids]
                )
                tree = cKDTree(surface_coords)
                dists, _ = tree.query(surface_coords, k=2)
                nn_dists = dists[:, 1]

                tbl.add_section()
                tbl.add_row(f"Surface '{boundary_name}'", "")
                tbl.add_row("  Nodes", f"{ns.node_count:,}")
                tbl.add_row("  NN spacing min", f"{nn_dists.min():.6f} m")
                tbl.add_row("  NN spacing max", f"{nn_dists.max():.6f} m")
                tbl.add_row("  NN spacing mean", f"{nn_dists.mean():.6f} m")
                tbl.add_row(
                    "  Recommended radius (4×mean)",
                    f"[bold green]{nn_dists.mean() * 4:.6f} m[/bold green]",
                )
                tbl.add_row(
                    "  Conservative (3×max)",
                    f"[green]{nn_dists.max() * 3:.6f} m[/green]",
                )

                bb_min = surface_coords.min(axis=0)
                bb_max = surface_coords.max(axis=0)
                tbl.add_row(
                    "  Bounding box min",
                    f"({bb_min[0]:.4f}, {bb_min[1]:.4f}, {bb_min[2]:.4f})",
                )
                tbl.add_row(
                    "  Bounding box max",
                    f"({bb_max[0]:.4f}, {bb_max[1]:.4f}, {bb_max[2]:.4f})",
                )

        self._console.print()
        self._console.print(tbl)

        # Validate RBF support-radius from preCICE XML against mesh spacing
        self._validate_rbf_radius(mesh)

    def _validate_rbf_radius(self, mesh: MeshModel) -> None:
        """Compare preCICE RBF support-radius against coupling surface spacing."""
        import numpy as np

        if not self._precice_info or not self._precice_info.rbf_mappings:
            return
        if not self.config.coupling or not self.config.coupling.boundaries:
            return

        # Compute nearest-neighbor spacing on coupling surfaces
        from scipy.spatial import cKDTree

        boundary_spacings = {}
        for boundary_name in self.config.coupling.boundaries:
            ns = mesh.node_sets.get(boundary_name)
            if ns is None or ns.node_count < 2:
                continue
            coords = np.array([mesh.nodes[nid].coords for nid in ns.node_ids])
            tree = cKDTree(coords)
            dists, _ = tree.query(coords, k=2)
            nn_dists = dists[:, 1]
            boundary_spacings[boundary_name] = {
                "mean": float(nn_dists.mean()),
                "max": float(nn_dists.max()),
            }

        if not boundary_spacings:
            return

        # Use the first boundary for comparison (primary coupling surface)
        spacing = next(iter(boundary_spacings.values()))
        recommended_min = spacing["mean"] * 4
        recommended_max = spacing["max"] * 3

        # Check each RBF mapping
        coupling_mesh = self.config.coupling.coupling_mesh or ""
        warnings = []
        for rbf in self._precice_info.rbf_mappings:
            sr = rbf.support_radius
            label = f"{rbf.direction} {rbf.from_mesh}→{rbf.to_mesh}"

            if sr < spacing["max"]:
                warnings.append(
                    f"  ✗ {label}: support-radius={sr:.6f} < max_spacing={spacing['max']:.6f}\n"
                    f"    Too small — some nodes will have no neighbors in the support region.\n"
                    f"    Recommended: {recommended_min:.6f} (4×mean) or {recommended_max:.6f} (3×max)"
                )
            elif sr < recommended_max * 0.5:
                warnings.append(
                    f"  ⚠ {label}: support-radius={sr:.6f} may be marginal\n"
                    f"    Recommended: {recommended_min:.6f} (4×mean) or {recommended_max:.6f} (3×max)"
                )
            elif sr > spacing["mean"] * 10:
                warnings.append(
                    f"  ⚠ {label}: support-radius={sr:.6f} >> mean_spacing={spacing['mean']:.6f}\n"
                    f"    Very large radius — may cause expensive RBF evaluations and smoothing.\n"
                    f"    Recommended: {recommended_min:.6f} (4×mean) or {recommended_max:.6f} (3×max)"
                )

        if warnings:
            warn_text = "\n".join(warnings)
            self._console.print()
            self._console.print(Panel(
                warn_text,
                title="RBF support-radius may not be optimal",
                border_style="bold yellow",
                expand=False,
            ))
        else:
            for rbf in self._precice_info.rbf_mappings:
                self._console.print(
                    f"      [green]✓[/green] RBF {rbf.direction} "
                    f"{rbf.from_mesh}→{rbf.to_mesh}: "
                    f"support-radius={rbf.support_radius:.6f} OK"
                )

    def _create_material(self):
        """Create material from configuration.

        For blade/rotor generators without explicit material section,
        extracts composite properties from the blade YAML sections.
        """
        self._console.print("\n[bold cyan]\\[2/6] Creating material...[/bold cyan]")

        # Check if blade/rotor generator provides composite properties
        if self._mesh_generator is not None and self.config.material is None:
            self._blade_properties = self._extract_blade_properties()
            self._console.print(
                f"      Composite properties from blade YAML: "
                f"[bold]{len(self._blade_properties)}[/bold] element sets"
            )
            return None

        mat_config = self.config.material
        if mat_config is None:
            raise ValueError(
                "Material configuration is required unless using "
                "BladeMesh or RotorMesh generator."
            )

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

        self._console.print(f"      Material: [bold]{material.name}[/bold]")
        self._console.print(f"      Type: {mat_config.type}")
        if mat_config.type == MaterialType.ISOTROPIC.value:
            self._console.print(
                f"      E={mat_config.E}, nu={mat_config.nu}, rho={mat_config.rho}"
            )

        return material

    def _extract_blade_properties(self) -> Dict[str, Any]:
        """Extract per-element-set composite properties from blade YAML.

        Uses the same logic as ``Blade.get_element_properties()`` but
        operates on the ``BladeMesh`` or ``RotorMesh`` generator's
        numad mesh data.

        Returns
        -------
        dict[str, ShellPropertyType]
            Mapping of element-set name to composite shell property.
        """
        from ...core.laminate import Laminate, Ply
        from ...core.properties import (
            CompositeShellProperty,
            ShellProperty,
        )
        from ...models.blade.model import material_factory

        numad_data = self._mesh_generator.numad_mesh_data

        # Build material lookup from blade YAML materials
        mat_db = {}
        for mat_data in numad_data["materials"]:
            mat_obj = material_factory(mat_data)
            mat_db[mat_obj.name] = mat_obj

        properties = {}
        for section in numad_data["sections"]:
            set_name = section["elementSet"]
            layup = section["layup"]

            plies = []
            for mat_name, thickness, angle in layup:
                mat = mat_db.get(mat_name)
                if mat is None:
                    raise KeyError(
                        f"Material '{mat_name}' referenced in section "
                        f"'{set_name}' not found in blade material database."
                    )
                if isinstance(mat, OrthotropicMaterial):
                    ply_mat = mat
                else:
                    G = mat.E / (2.0 * (1.0 + mat.nu))
                    ply_mat = OrthotropicMaterial(
                        name=mat.name,
                        E=(mat.E, mat.E, mat.E),
                        G=(G, G, G),
                        nu=(mat.nu, mat.nu, mat.nu),
                        rho=mat.rho,
                    )
                plies.append(Ply(material=ply_mat, thickness=thickness, angle=angle))

            if len(plies) == 1 and plies[0].angle == 0.0:
                original_mat = mat_db[layup[0][0]]
                if isinstance(original_mat, IsotropicMaterial):
                    properties[set_name] = ShellProperty(
                        material=original_mat,
                        thickness=plies[0].thickness,
                    )
                    continue

            laminate = Laminate(plies=plies)
            properties[set_name] = CompositeShellProperty(laminate=laminate)

        return properties

    def _build_model_config(self) -> Dict[str, Any]:
        """Build the model configuration dictionary for the solver."""
        self._console.print("\n[bold cyan]\\[3/6] Building model configuration...[/bold cyan]")

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
                "num_modes": self.config.solver.num_modes,
            },
            "elements": {
                "element_family": elem_family,
            },
        }

        # Use composite properties from blade YAML when available,
        # otherwise use the single material
        if self._blade_properties is not None:
            model_config["elements"]["properties"] = self._blade_properties
        else:
            model_config["elements"]["material"] = self._material

        # Forward span_direction for composite fibre orientation correction
        span_dir = getattr(self.config.elements, "span_direction", None)
        if span_dir is not None:
            model_config["elements"]["span_direction"] = tuple(float(v) for v in span_dir)

        # Add thickness for shell elements
        if self.config.elements.thickness:
            model_config["elements"]["thickness"] = self.config.elements.thickness

        # Add Newmark parameters
        if self.config.solver.newmark:
            model_config["solver"]["beta"] = self.config.solver.newmark.beta
            model_config["solver"]["gamma"] = self.config.solver.newmark.gamma

        # Add damping parameters
        if self.config.solver.damping:
            cfg = self.config.solver.damping
            damping_dict = {
                "enabled": cfg.enabled,
                "eta_m": cfg.eta_m,
                "eta_k": cfg.eta_k,
                "zeta": cfg.zeta,
                "zeta_1": cfg.zeta_1,
                "zeta_2": cfg.zeta_2,
                "mode_i": cfg.mode_i,
                "mode_j": cfg.mode_j,
                "num_modes": cfg.num_modes,
            }
            model_config["solver"]["damping"] = damping_dict
            # Keep flat keys for backward compatibility with solvers that
            # still read eta_m / eta_k directly from solver_params
            if cfg.eta_m is not None:
                model_config["solver"]["eta_m"] = cfg.eta_m
            if cfg.eta_k is not None:
                model_config["solver"]["eta_k"] = cfg.eta_k

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

        # Forward postprocess performance parameters to solver
        if self.config.postprocess:
            perf = {}
            if self.config.postprocess.fluid_density is not None:
                perf["fluid_density"] = self.config.postprocess.fluid_density
            if self.config.postprocess.flow_velocity is not None:
                perf["flow_velocity"] = self.config.postprocess.flow_velocity
            if perf:
                model_config["solver"]["performance"] = perf

        self._console.print(f"      Solver type: [bold]{self.config.solver.type}[/bold]")
        self._console.print(f"      Element family: [bold]{self.config.elements.family}[/bold]")

        return model_config

    def _create_solver(self, model_config: Dict[str, Any]):
        """Create the appropriate solver based on configuration."""
        self._console.print("\n[bold cyan]\\[4/6] Creating solver...[/bold cyan]")

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

        elif solver_type == SolverType.MODAL.value:
            from ..modal import ModalSolver

            solver = ModalSolver(self.mesh, model_config)

        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        self._console.print(f"      Created: [bold]{solver_type}[/bold]")
        return solver

    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to the solver."""
        self._console.print("\n[bold cyan]\\[5/6] Applying boundary conditions...[/bold cyan]")

        if self.solver is None or self.mesh is None:
            raise RuntimeError("Solver and mesh must be initialized before applying BCs")

        # Apply Dirichlet conditions
        dirichlet_conditions = []
        for bc_config in self.config.boundary_conditions.dirichlet:
            try:
                dofs = self.solver.get_dofs_by_nodeset_name(bc_config.nodeset)
                dirichlet_conditions.append(DirichletCondition(dofs, bc_config.value))
                self._console.print(
                    f"      Dirichlet BC: [bold]'{bc_config.nodeset}'[/bold] = {bc_config.value} ({len(dofs)} DOFs)"
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
            self._console.print(f"      Body force: {bf_config.value}")

        if body_forces:
            self.solver.add_body_forces(body_forces)

        self._console.print(f"      Total Dirichlet BCs: {len(dirichlet_conditions)}")
        self._console.print(f"      Total body forces: {len(body_forces)}")

    def _print_modal_results(self, result) -> None:
        """Print modal analysis results (frequencies and mode shapes)."""
        import numpy as np

        frequencies, mode_shapes = result
        n = len(frequencies)

        # Participation factors from solver (may be None if not computed)
        pf = self.solver.participation_factors
        has_pf = pf is not None and pf.shape[0] >= n

        table = Table(
            title="MODAL ANALYSIS RESULTS",
            title_style="bold cyan",
            show_lines=False,
            pad_edge=True,
        )

        table.add_column("Mode", justify="right", style="bold")
        table.add_column("Frequency [Hz]", justify="right")
        table.add_column("Period [s]", justify="right")
        table.add_column("ω [rad/s]", justify="right")

        if has_pf:
            n_dirs = pf.shape[1]
            all_labels = ["Ux%", "Uy%", "Uz%", "Rx%", "Ry%", "Rz%"]
            for lb in all_labels[:n_dirs]:
                table.add_column(lb, justify="right")
            table.add_column("Type", justify="right", style="bold green")

        for i in range(n):
            f = frequencies[i]
            omega = 2.0 * np.pi * f
            period = 1.0 / f if f > 0 else float("inf")
            row: list[str] = [
                str(i + 1),
                f"{f:.6f}",
                f"{period:.6f}",
                f"{omega:.4f}",
            ]

            if has_pf:
                n_trans = min(n_dirs, 3)
                trans_max = pf[i, :n_trans].max()
                rot_max = pf[i, n_trans:n_dirs].max() if n_dirs > 3 else 0.0
                rb_labels = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"][:n_dirs]
                if trans_max < 1.0 and rot_max < 1.0:
                    mode_type = "Local"
                else:
                    dominant = int(np.argmax(pf[i, :n_dirs]))
                    mode_type = rb_labels[dominant]
                for d in range(n_dirs):
                    row.append(f"{pf[i, d]:.1f}")
                row.append(mode_type)

            table.add_row(*row)

        self._console.print()
        self._console.print(table)
        self._console.print()

        # Write results to CSV
        csv_path = self.working_dir / "modal_frequencies.csv"
        with open(csv_path, "w") as fh:
            csv_dir_names = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"]
            header = "mode,frequency_hz,period_s,omega_rad_s"
            if has_pf:
                for d in range(pf.shape[1]):
                    lbl = csv_dir_names[d] if d < len(csv_dir_names) else f"dir{d}"
                    header += f",eff_mass_{lbl}_pct"
            fh.write(header + "\n")
            for i, f in enumerate(frequencies):
                omega = 2.0 * np.pi * f
                period = 1.0 / f if f > 0 else float("inf")
                row = f"{i + 1},{f:.8e},{period:.8e},{omega:.8e}"
                if has_pf:
                    for d in range(pf.shape[1]):
                        row += f",{pf[i, d]:.4f}"
                fh.write(row + "\n")

        # Write mode shapes
        np_path = self.working_dir / "modal_modes.npy"
        np.save(str(np_path), mode_shapes)

        # Write VTU files for ParaView visualization
        output_dir = self.working_dir / "modal_results"
        self.solver.write_modal_results(str(output_dir), frequencies, mode_shapes)

        out_tbl = Table(title="Output Files", title_style="bold cyan",
                        show_lines=False, pad_edge=True, expand=False)
        out_tbl.add_column("File", style="bold")
        out_tbl.add_column("Path", style="dim")
        out_tbl.add_row("Frequencies (CSV)", str(csv_path))
        out_tbl.add_row("Mode shapes (NPY)", str(np_path))
        out_tbl.add_row("Mode shapes (VTU)", str(output_dir) + "/")
        out_tbl.add_row("ParaView collection", str(output_dir / "modal_results.pvd"))
        self._console.print()
        self._console.print(out_tbl)

    def _run_postprocessing(self) -> None:
        """Run post-processing if configured."""
        if not self.config.postprocess:
            return

        self._console.print("\n[bold cyan]\\[Post-processing][/bold cyan]")

        try:
            from ...postprocess.precice import FSIDataVisualizer

            if self.config.postprocess.watchpoint_file:
                visualizer = FSIDataVisualizer(self.config.postprocess.watchpoint_file)

                if self.config.postprocess.plots:
                    if "displacement" in self.config.postprocess.plots:
                        save_path = self.config.postprocess.plots["displacement"]
                        visualizer.plot_displacement(save_path=save_path)
                        self._console.print(f"      Saved: {save_path}")

                    if "force" in self.config.postprocess.plots:
                        save_path = self.config.postprocess.plots["force"]
                        visualizer.plot_force(save_path=save_path)
                        self._console.print(f"      Saved: {save_path}")

        except Exception as e:
            logger.warning("Post-processing failed: %s", e)
            self._console.print(f"      [yellow]Warning:[/yellow] Could not generate plots: {e}")

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
        For blade/rotor generators, shows element data fields (thickness,
        stiffness, plies, etc.) and colors by element sets.
        """
        if self.mesh is None:
            self.mesh = self._setup_mesh()

        element_data = None
        if self._mesh_generator is not None and self.config.material is None:
            # Extract composite properties and build visualization fields
            if self._blade_properties is None:
                self._blade_properties = self._extract_blade_properties()
            from ...core.properties import build_element_data

            element_data = build_element_data(self._blade_properties, self.mesh)

        self.mesh.view(color_by_sets=True, element_data=element_data)

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
