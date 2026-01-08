#!/usr/bin/env python3
"""
FSI Simulation CLI Runner.

This script provides a command-line interface for running FSI simulations
from YAML configuration files.

Usage:
    python -m fem_shell.cli.run_fsi config.yaml [options]

Examples:
    # Run simulation from YAML
    python -m fem_shell.cli.run_fsi simulation.yaml

    # Run with custom working directory
    python -m fem_shell.cli.run_fsi simulation.yaml --workdir /path/to/case

    # Preview configuration without running
    python -m fem_shell.cli.run_fsi simulation.yaml --preview

    # Generate template configuration
    python -m fem_shell.cli.run_fsi --template > my_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Template YAML configuration
TEMPLATE_CONFIG = """# FSI Simulation Configuration
# ============================
# This file defines a complete FSI simulation for fem-shell.

#============================================================================
# MESH CONFIGURATION
#============================================================================
mesh:
  # Option 1: Load from file (recommended for production)
  source: "file"  # "file" or "generator"
  file:
    path: "mesh.h5"  # Supports .h5, .hdf5, .pkl, .pickle
    format: "auto"   # "auto", "hdf5", or "pickle"

  # Option 2: Generate programmatically
  # source: "generator"
  # generator:
  #   # Available types: "SquareShapeMesh", "BoxSurfaceMesh", "MultiFlapMesh"
  #   type: "SquareShapeMesh"
  #   params:
  #     width: 0.1
  #     height: 1.0
  #     nx: 4
  #     ny: 26
  #     quadratic: true
  #     triangular: false

  # Optional: Write mesh to file for visualization
  # Supported formats: .vtk, .vtu, .msh, .inp (CalculiX), .h5/.hdf5, .obj, .stl
  output_file: "mesh.vtk"

#============================================================================
# MATERIAL DEFINITION
#============================================================================
material:
  type: "isotropic"  # "isotropic" or "orthotropic"
  name: "Steel"
  
  # Isotropic material properties
  E: 4.0e6       # Young's modulus [Pa]
  nu: 0.3        # Poisson's ratio [-]
  rho: 3000.0    # Density [kg/m³]
  
  # Orthotropic properties (uncomment if type: "orthotropic")
  # E: [E1, E2, E3]        # Young's modulus in 3 directions
  # G: [G12, G23, G31]     # Shear modulus in 3 planes
  # nu: [nu12, nu23, nu31] # Poisson's ratios in 3 planes

#============================================================================
# ELEMENT CONFIGURATION
#============================================================================
elements:
  family: "PLANE"    # "PLANE" for 2D, "SHELL" for 3D shell
  # thickness: 0.1   # Required only for SHELL elements

#============================================================================
# SOLVER CONFIGURATION
#============================================================================
solver:
  # Solver type: "LinearStatic", "LinearDynamic", or "LinearDynamicFSI"
  type: "LinearDynamicFSI"
  
  # Time parameters
  total_time: 5.0    # Total simulation time [s]
  time_step: 0.001   # Time step size [s]
  
  # Newmark-β integration parameters (optional)
  newmark:
    beta: 0.25   # Newmark beta (0.25 = constant average acceleration)
    gamma: 0.5   # Newmark gamma (0.5 = no numerical damping)
  
  # Rayleigh damping parameters (optional)
  damping:
    eta_m: 1.0e-4  # Mass proportional damping
    eta_k: 1.0e-4  # Stiffness proportional damping
  
  # Advanced options
  use_critical_dt: false  # Auto-calculate critical time step
  safety_factor: 0.8      # Safety factor for critical dt

#============================================================================
# BOUNDARY CONDITIONS
#============================================================================
boundary_conditions:
  # Dirichlet (fixed displacement) conditions
  dirichlet:
    - nodeset: "bottom"   # Name of node set in mesh
      value: 0.0          # Prescribed displacement value
    # - nodeset: "left"
    #   value: 0.0
    #   components: [0, 1]  # Optional: fix only specific DOFs

  # Body forces (optional)
  body_forces: []
    # - value: [0, -9810, 0]  # Gravity in -Y direction

#============================================================================
# FSI COUPLING (preCICE) - Required for LinearDynamicFSI solver
#============================================================================
coupling:
  # Option 1: External adapter config file
  # adapter_config: "precice-adapter.yaml"  # Path to preCICE adapter config
  
  # Option 2: Inline configuration (recommended - single config file)
  participant: "Solid"              # preCICE participant name
  config_file: "../precice-config.xml"  # Path to preCICE config (relative to this file)
  interface:
    coupling_mesh: "Solid-Mesh"     # Name of coupling mesh in preCICE config
    write_data: "Displacement"      # Data to write to preCICE
    read_data: "Force"              # Data to read from preCICE
  
  boundaries:
    - "left"    # Coupling boundary names (from mesh node sets)
    - "top"
    - "right"

#============================================================================
# OUTPUT & CHECKPOINT CONFIGURATION
#============================================================================
output:
  folder: "results"         # Output folder for checkpoints
  write_interval: 0.1       # Checkpoint interval [s] (0 = disabled)
  
  # Restart configuration (OpenFOAM-style)
  start_from: "startTime"   # "startTime" or "latestTime"
  start_time: 0.0           # Initial time if start_from="startTime"
  
  # Deformed mesh output
  save_deformed_mesh: true  # Save deformed mesh at checkpoints
  deformed_mesh_scale: 1.0  # Scale factor for displacements
  
  # VTK output
  write_vtk: true
  vtk_file: "mesh.vtk"
  
  # Initial state
  write_initial_state: true  # Write t=0 state

#============================================================================
# POST-PROCESSING (optional)
#============================================================================
postprocess:
  # preCICE watchpoint file for plotting
  watchpoint_file: "precice-Solid-watchpoint-Flap-Tip.log"
  
  # Automatic plot generation
  plots:
    displacement: "displacement.png"
    force: "force.png"
"""

GENERATOR_TEMPLATES = {
    "SquareShapeMesh": """  generator:
    type: "SquareShapeMesh"
    params:
      width: 0.1       # Width in X direction [m]
      height: 1.0      # Height in Y direction [m]
      nx: 4            # Elements in X direction
      ny: 26           # Elements in Y direction
      quadratic: true  # Use quadratic elements
      triangular: false
""",
    "BoxSurfaceMesh": """  generator:
    type: "BoxSurfaceMesh"
    params:
      center: [0, 5.0, 0]    # Center coordinates [x, y, z]
      dims: [1.0, 10.0, 1.0] # Box dimensions [dx, dy, dz]
      nx: 5                   # Elements in X direction
      ny: 20                  # Elements in Y direction
      nz: 5                   # Elements in Z direction
      quadratic: false
      triangular: false
""",
    "MultiFlapMesh": """  generator:
    type: "MultiFlapMesh"
    params:
      n_flaps: 5           # Number of flaps
      flap_width: 0.1      # Width of each flap [m]
      flap_height: 1.0     # Height of each flap [m]
      x_spacing: 2.0       # Spacing between flaps [m]
      base_height: 0.05    # Base strip height [m]
      nx_flap: 4           # Elements per flap width
      ny_flap: 26          # Elements per flap height
      nx_base_segment: 10  # Elements in base segments
      ny_base: 2           # Elements in base height
      quadratic: true
""",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_template(generator_type: str = None) -> None:
    """Print template configuration to stdout."""
    print(TEMPLATE_CONFIG)

    if generator_type and generator_type in GENERATOR_TEMPLATES:
        print("\n# Example generator configuration for", generator_type)
        print(GENERATOR_TEMPLATES[generator_type])


def list_generators() -> None:
    """List available mesh generators with descriptions."""
    print("\nAvailable mesh generators:")
    print("=" * 50)
    print("\n1. SquareShapeMesh")
    print("   2D rectangular mesh for plane stress/strain")
    print("   Node sets: left, right, top, bottom, corners")

    print("\n2. BoxSurfaceMesh")
    print("   3D box surface mesh for shell analysis")
    print("   Node sets: left, right, top, bottom, front, back")

    print("\n3. MultiFlapMesh")
    print("   Multiple flaps on a common base for FSI")
    print("   Node sets: bottom, flaps_left, flaps_right, flaps_top")

    print("\nUse --template --generator <name> for example configuration")


def validate_config(config_path: str) -> bool:
    """Validate configuration file without running."""
    from fem_shell.core.config import FSISimulationConfig

    try:
        config = FSISimulationConfig.from_yaml(config_path)
        warnings = config.validate()

        print("Configuration validation:")
        print("=" * 50)
        print(config)

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠️  {w}")
            return False
        else:
            print("\n✓ Configuration is valid")
            return True

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        return False


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run FSI simulations from YAML configuration files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml                    Run simulation
  %(prog)s config.yaml --preview          Preview configuration
  %(prog)s --template > config.yaml       Generate template
  %(prog)s --template --generator MultiFlapMesh
  %(prog)s --list-generators              List available generators
        """,
    )

    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--workdir",
        "-w",
        help="Working directory for simulation",
    )

    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="Preview configuration without running",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration file",
    )

    parser.add_argument(
        "--template",
        "-t",
        action="store_true",
        help="Print template configuration to stdout",
    )

    parser.add_argument(
        "--generator",
        "-g",
        choices=list(GENERATOR_TEMPLATES.keys()),
        help="Include specific generator template",
    )

    parser.add_argument(
        "--list-generators",
        action="store_true",
        help="List available mesh generators",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Handle special commands first
    if args.template:
        print_template(args.generator)
        return 0

    if args.list_generators:
        list_generators()
        return 0

    # Require config file for other operations
    if not args.config:
        parser.print_help()
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    setup_logging(args.verbose)

    # Validate only
    if args.validate:
        return 0 if validate_config(str(config_path)) else 1

    # Preview configuration
    if args.preview:
        from fem_shell.core.config import FSISimulationConfig

        config = FSISimulationConfig.from_yaml(str(config_path))
        print(config)
        return 0

    # Run simulation
    try:
        from fem_shell.solvers.fsi_runner import FSIRunner

        runner = FSIRunner(str(config_path), args.workdir)
        runner.run()
        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 130

    except Exception as e:
        logging.exception("Simulation failed")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
