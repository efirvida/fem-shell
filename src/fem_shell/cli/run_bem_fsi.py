#!/usr/bin/env python3
"""CLI entry point for the BEM-FSI fluid participant.

Runs a BEM-based preCICE fluid participant as a standalone process,
equivalent to launching OpenFOAM on the fluid side.  The structural
solver (``fem-shell-fsi``) must be started in a separate process before
or concurrently with this command.

Usage
-----
::

    fem-shell-bem-fsi simulation.yaml
    fem-shell-bem-fsi simulation.yaml --workdir /path/to/case
    fem-shell-bem-fsi --template

Configuration file schema
-------------------------
The YAML file must contain at minimum::

    participant: "Fluid"            # preCICE participant name
    config_file: "precice-config.xml"
    coupling_mesh: "Fluid-Mesh"     # mesh name this participant provides
    blade_file: "path/to/blade.yaml"

    mesh:
      source: "generator"           # "generator" or "file"
      generator:
        type: "BladeMesh"
        params:
          yaml_file: "path/to/blade.yaml"
          element_size: 0.5

    bem:
      wind_speed: 10.59             # v_inf [m/s]
      omega: 7.56                   # angular velocity [rad/s]
      pitch: 0.0                    # collective pitch [deg]
      air_density: 1.225
      dynamic_viscosity: 1.81e-5
      hub_height: 150.0
      shear_exp: 0.2
      n_blades: 3
      hub_radius: 3.0
      default_re: 1.0e7
      neuralfoil_model: "large"
      precone: 0.0
      tilt: 0.0
      span_direction: [0.0, 0.0, 1.0]
      normal_direction: [1.0, 0.0, 0.0]
      tangential_direction: [0.0, 1.0, 0.0]

    output:
      folder: "bem_fsi_results"
      log_interval: 10              # windows between diagnostic lines

Notes
-----
The mesh section accepts any generator type supported by ``FSIRunner``
(``BladeMesh``, ``RotorMesh``, etc.) or ``source: file`` to load an
existing ``.h5`` mesh.  The user decides the mesh parameters;
the BEM-FSI participant does **not** impose any constraints.  preCICE
performs the spatial mapping between the fluid and solid meshes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

_TEMPLATE = """\
# BEM-FSI fluid participant configuration
# ========================================
# Launch with: fem-shell-bem-fsi simulation.yaml

participant: "Fluid"
config_file: "precice-config.xml"
coupling_mesh: "Fluid-Mesh"
blade_file: "path/to/blade.yaml"

mesh:
  source: "generator"
  generator:
    type: "BladeMesh"
    params:
      yaml_file: "path/to/blade.yaml"
      element_size: 0.5
  # Optional: write mesh to file for inspection / preCICE support-radius tuning
  # output_file: "fluid_mesh.vtk"

bem:
  wind_speed: 10.59           # m/s
  omega: 7.56                 # rad/s (7.23 rpm for IEA-15)
  pitch: 0.0                  # deg
  air_density: 1.225          # kg/m³
  dynamic_viscosity: 1.81e-5  # Pa·s
  hub_height: 150.0           # m
  shear_exp: 0.2
  n_blades: 3
  hub_radius: 3.0             # m
  default_re: 1.0e7
  neuralfoil_model: "large"
  precone: 0.0                # deg
  tilt: 0.0                   # deg
  span_direction: [0.0, 0.0, 1.0]
  normal_direction: [1.0, 0.0, 0.0]
  tangential_direction: [0.0, 1.0, 0.0]

output:
  folder: "bem_fsi_results"
  log_interval: 10
"""


def _build_mesh(cfg: dict, config_path: Path):
    """Build or load the coupling mesh from the YAML ``mesh`` section."""
    from fem_shell.core.config import MeshGeneratorType, MeshSource
    from fem_shell.core.mesh import (
        BladeMesh,
        BoxSurfaceMesh,
        BoxVolumeMesh,
        MeshModel,
        MultiFlapMesh,
        RotorMesh,
        SquareShapeMesh,
    )

    mesh_cfg = cfg.get("mesh", {})
    source = mesh_cfg.get("source", "generator")

    def _resolve(path_str: str) -> str:
        p = Path(path_str)
        if not p.is_absolute():
            return str(config_path.parent / p)
        return path_str

    if source == MeshSource.FILE.value:
        file_cfg = mesh_cfg.get("file", {})
        file_path = _resolve(file_cfg["path"])
        fmt = file_cfg.get("format", "auto")
        logging.info("[BEM-FSI] Loading mesh from %s", file_path)
        mesh = MeshModel.load(file_path, format=fmt)

    else:
        gen_cfg = mesh_cfg.get("generator", {})
        gen_type = gen_cfg.get("type", MeshGeneratorType.BLADE.value)
        params = gen_cfg.get("params", {})

        logging.info("[BEM-FSI] Generating mesh with %s", gen_type)

        if gen_type == MeshGeneratorType.BLADE.value:
            yaml_file = params.get("yaml_file")
            excel_file = params.get("excel_file")
            airfoil_dir = params.get("airfoil_dir")
            if yaml_file:
                yaml_file = _resolve(yaml_file)
            if excel_file:
                excel_file = _resolve(excel_file)
            if airfoil_dir:
                airfoil_dir = _resolve(airfoil_dir)
            generator = BladeMesh(
                yaml_file=yaml_file,
                excel_file=excel_file,
                airfoil_dir=airfoil_dir,
                element_size=params.get("element_size", 0.5),
                n_samples=params.get("n_samples", 300),
                span_grading=params.get("span_grading", "chord"),
            )
            mesh = generator.generate(renumber=None)

        elif gen_type == MeshGeneratorType.ROTOR.value:
            yaml_file = _resolve(params["yaml_file"])
            generator = RotorMesh(
                yaml_file=yaml_file,
                n_blades=params.get("n_blades", 3),
                hub_radius=params.get("hub_radius"),
                element_size=params.get("element_size", 0.5),
                n_samples=params.get("n_samples", 300),
            )
            mesh = generator.generate(renumber="rcm")

        elif gen_type == MeshGeneratorType.SQUARE.value:
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
            raise ValueError(f"Unknown mesh generator type: '{gen_type}'")

    # Optional: write full mesh for inspection (before any node-set filter)
    output_file = mesh_cfg.get("output_file")
    if output_file:
        mesh.write_mesh(_resolve(output_file))
        logging.info("[BEM-FSI] Mesh written to %s", output_file)

    # Optional: filter to a specific node set (e.g. "allOuterShellNods" to
    # exclude shear-web nodes from the aerodynamic coupling mesh).
    # The BEM participant only needs node coordinates for preCICE registration
    # and force projection — elements are not required after this point.
    # The full mesh (with elements) is kept as viz_mesh for VTU surface output.
    viz_mesh = None
    coupling_node_set = gen_cfg.get("coupling_node_set") if source != MeshSource.FILE.value else None
    if coupling_node_set:
        try:
            ns = mesh.get_node_set(coupling_node_set)
        except ValueError:
            available = list(mesh.node_sets.keys())
            raise ValueError(
                f"coupling_node_set '{coupling_node_set}' not found. "
                f"Available node sets: {available}"
            )
        viz_mesh = mesh  # full mesh with elements, for surface VTU
        filtered_nodes = list(ns.nodes.values())
        mesh = MeshModel(nodes=filtered_nodes)
        logging.info(
            "[BEM-FSI] Filtered to node set '%s': %d nodes",
            coupling_node_set,
            len(filtered_nodes),
        )

    logging.info(
        "[BEM-FSI] Mesh ready: %d nodes, %d elements",
        len(mesh.nodes),
        len(mesh.elements),
    )
    return mesh, viz_mesh


def main(argv=None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fem-shell-bem-fsi",
        description=(
            "BEM-based preCICE fluid participant for FSI coupling. "
            "Runs as an independent process alongside the structural solver."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  fem-shell-bem-fsi simulation.yaml\n"
            "  fem-shell-bem-fsi simulation.yaml --workdir /path/to/case\n"
            "  fem-shell-bem-fsi --template > simulation.yaml\n"
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        metavar="CONFIG_FILE",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--workdir",
        metavar="DIR",
        help="Working directory (default: directory of CONFIG_FILE).",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Print a template YAML configuration and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.template:
        print(_TEMPLATE, end="")
        return 0

    if args.config is None:
        parser.error("CONFIG_FILE is required (or use --template to generate one).")

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logging.error("Configuration file not found: %s", config_path)
        return 1

    # Change working directory
    workdir = Path(args.workdir).resolve() if args.workdir else config_path.parent
    os.chdir(workdir)
    logging.info("[BEM-FSI] Working directory: %s", workdir)

    # Load YAML
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve config_file path (preCICE XML) relative to workdir
    if "config_file" in cfg and not Path(cfg["config_file"]).is_absolute():
        cfg["config_file"] = str(workdir / cfg["config_file"])

    # Build the mesh (user's responsibility to configure)
    try:
        mesh, viz_mesh = _build_mesh(cfg, config_path)
    except Exception as exc:
        logging.error("Failed to build mesh: %s", exc)
        return 1

    # Build and run the BEM-FSI participant
    from fem_shell.solvers.bem.fsi_participant import build_from_config

    try:
        participant = build_from_config(mesh, cfg, viz_mesh=viz_mesh)
        participant.run()
    except Exception as exc:
        logging.error("BEM-FSI participant failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
