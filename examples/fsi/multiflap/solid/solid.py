#!/usr/bin/env python3
"""
Multi-flap FEM solid solver for FSI simulations.

Reads configuration from case_config.json in the case root directory.
Uses MultiFlapMesh to generate a single connected mesh with a base and multiple flaps.
The base receives fixed (Dirichlet) boundary conditions.
"""

import json
from pathlib import Path

from fem_shell.core.bc import DirichletCondition
from fem_shell.core.material import IsotropicMaterial as Material
from fem_shell.core.mesh import MeshModel, MultiFlapMesh
from fem_shell.elements import ElementFamily
from fem_shell.postprocess.precice import FSIDataVisualizer
from fem_shell.solvers.checkpoint import CheckpointManager
from fem_shell.solvers.fsi import LinearDynamicFSISolver


def load_config():
    """Load configuration from case_config.json"""
    base = Path(__file__).parent
    case_root = base.parent  # solid -> case_root
    config_path = case_root / "case_config.json"

    with config_path.open() as f:
        return json.load(f)


def create_multiflap_mesh(config):
    """Create the multi-flap mesh from configuration"""

    # Flap geometry
    n_flaps = config["flaps"]["number"]
    flap_width = config["flaps"]["width"]
    flap_height = config["flaps"]["height"]
    x_spacing = config["flaps"]["x_spacing"]

    # Mesh parameters
    mesh_cfg = config["solid"]["mesh"]
    nx_flap = mesh_cfg.get("nx", 4)
    ny_flap = mesh_cfg.get("ny", 20)
    quadratic = mesh_cfg.get("quadratic", True)

    # Base parameters (can be added to config if needed)
    base_height = mesh_cfg.get("base_height", 0.05)
    ny_base = mesh_cfg.get("ny_base", 2)
    nx_base_segment = mesh_cfg.get("nx_base_segment", 10)

    # Create the multi-flap mesh
    mesh = MultiFlapMesh(
        n_flaps=n_flaps,
        flap_width=flap_width,
        flap_height=flap_height,
        x_spacing=x_spacing,
        base_height=base_height,
        nx_flap=nx_flap,
        ny_flap=ny_flap,
        nx_base_segment=nx_base_segment,
        ny_base=ny_base,
        quadratic=quadratic,
    ).generate()
    mesh.write_mesh("mesh_multiflap.vtk")

    return mesh


def run_multiflap_simulation(config):
    """Run FSI simulation with unified multi-flap mesh.

    If `simulation.start_from == 'latestTime'`, attempt to load the last
    deformed mesh from checkpoints; otherwise generate the mesh.
    """

    sim_cfg = config["simulation"]
    solid_cfg = config["solid"]

    mesh = None
    restored_mesh = False
    if str(sim_cfg.get("start_from", "startTime")) == "latestTime":
        print("\nAttempting to restore mesh from latest checkpoint...")
        # Use CheckpointManager only to find latest checkpoint folder
        cm = CheckpointManager(
            output_folder=solid_cfg.get("output_folder", "results"),
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
                print(f"  ✓ Restored deformed mesh: {deformed_path}")
                mesh = MeshModel.load(str(deformed_path))
                restored_mesh = True
            else:
                print("  ⚠️  No deformed mesh found in latest checkpoint. Generating mesh...")
        else:
            print("  ⚠️  No checkpoints found. Generating mesh...")

    if mesh is None:
        # Create mesh
        print("\nGenerating multi-flap mesh...")
        mesh = create_multiflap_mesh(config)

        print(f"  Nodes: {len(mesh.nodes)}")
        print(f"  Elements: {len(mesh.elements)}")
        print(f"  Node sets: {list(mesh.node_sets.keys())}")

        # Write mesh for visualization
        mesh.write_mesh("mesh_multiflap.vtk")
        print("  Written: mesh_multiflap.vtk")

    # Material properties
    mat_cfg = config["solid"]["material"]
    material = Material(
        name="Flap_Material",
        E=mat_cfg["E"],
        nu=mat_cfg["nu"],
        rho=mat_cfg["rho"],
    )

    # Solver configuration from simulation section
    sim_cfg = config["simulation"]
    solid_cfg = config["solid"]

    # Keep OpenFOAM-like semantics: support startFrom/startTime keywords.
    solver_start_from = sim_cfg.get("start_from", sim_cfg.get("startFrom", "startTime"))
    start_time = sim_cfg.get("start_time", sim_cfg.get("startTime", 0.0))
    reset_state_on_restart = False
    if restored_mesh:
        solver_start_from = "latestTime"
        reset_state_on_restart = True
        print(
            "  ↳ Restored geometry: startFrom=latestTime; keeping time from checkpoint and resetting state vectors"
        )

    # Coupling boundaries - all flap surfaces (left, top, right for all flaps)
    # These are exposed to the fluid
    coupling_boundaries = ["flaps_left", "flaps_top", "flaps_right"]

    model_config = {
        "solver": {
            "total_time": sim_cfg["total_time"],
            "time_step": sim_cfg["time_step"],
            "adapter_cfg": solid_cfg.get("adapter_cfg", "precice-adapter.yaml"),
            "coupling_boundaries": coupling_boundaries,
            # Checkpoint/restart configuration
            "output_folder": solid_cfg.get("output_folder", "results"),
            "write_interval": sim_cfg.get("write_interval", 0),  # 0 = disabled
            "start_from": solver_start_from,
            "start_time": start_time,
            # Deformed mesh export configuration
            "save_deformed_mesh": solid_cfg.get("save_deformed_mesh", True),
            "deformed_mesh_scale": solid_cfg.get("deformed_mesh_scale", 1.0),
            # Preserve time from checkpoint but drop state vectors to avoid double-loading
            "reset_state_on_restart": reset_state_on_restart,
        },
        "elements": {
            "material": material,
            "element_family": ElementFamily.PLANE,
        },
    }

    # Create solver
    problem = LinearDynamicFSISolver(mesh, model_config)

    # Apply boundary conditions (fixed at base bottom)
    # The "bottom" node set is an alias for "base_bottom" created by MultiFlapMesh
    bottom_dofs = problem.get_dofs_by_nodeset_name("bottom")
    problem.add_dirichlet_conditions([DirichletCondition(bottom_dofs, 0.0)])

    print(f"\nApplied Dirichlet BC to {len(bottom_dofs)} DOFs at base")

    print("\nStarting FSI simulation...")
    problem.solve()

    return problem


def main():
    """Main entry point"""
    print("=" * 60)
    print("Multi-Flap FSI Solid Solver")
    print("=" * 60)

    # Load configuration
    config = load_config()

    print("\nConfiguration loaded:")
    print(f"  Flaps: {config['flaps']['number']}")
    print(f"  Flap size: {config['flaps']['width']} x {config['flaps']['height']}")
    print(f"  Spacing: {config['flaps']['x_spacing']}")
    print(
        f"  Material: E={config['solid']['material']['E']}, "
        f"nu={config['solid']['material']['nu']}, "
        f"rho={config['solid']['material']['rho']}"
    )
    print(f"  Mesh: nx={config['solid']['mesh']['nx']}, ny={config['solid']['mesh']['ny']}")

    # Run simulation
    problem = run_multiflap_simulation(config)

    # Post-processing
    print("\nGenerating visualizations...")
    try:
        visualizer = FSIDataVisualizer("precice-Solid-watchpoint-Flap-Tip.log")
        visualizer.plot_displacement(save_path="desplazamientos.png")
        visualizer.plot_force(save_path="fuerzas.png")
        print("  Saved: desplazamientos.png, fuerzas.png")
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
