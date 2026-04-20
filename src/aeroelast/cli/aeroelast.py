#!/usr/bin/env python3
"""
aeroelast — Structural FEM solver CLI.

Runs modal, static and dynamic FEM analyses from YAML configuration files.
Assembly and element kernels execute in Rust; mesh generation and
post-processing remain in Python.

Usage:
    aeroelast modal.yaml
    aeroelast modal.yaml --workdir /path/to/case
    aeroelast modal.yaml --preview
    aeroelast --template > config.yaml

Supported solver types (solver.type in YAML):
    Modal           — natural frequencies and mode shapes (SLEPc)
    LinearStatic    — static linear analysis
    LinearDynamic   — transient Newmark-β integration
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aeroelast",
        description="Structural FEM solver (Rust-accelerated assembly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aeroelast modal.yaml                  Run modal analysis
  aeroelast modal.yaml --workdir /tmp   Set working directory
  aeroelast modal.yaml --preview        Preview config without running
  aeroelast modal.yaml --validate       Validate config and exit
  aeroelast --template > config.yaml    Generate template config
        """,
    )

    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--workdir", "-w",
        help="Working directory (default: directory of the config file)",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Print parsed configuration and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration file and exit",
    )
    parser.add_argument(
        "--template", "-t",
        action="store_true",
        help="Print a template YAML configuration to stdout and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--export-ccx",
        metavar="FILE",
        help="Export model to CalculiX .inp format and exit",
    )

    args = parser.parse_args()

    # --template: no config needed
    if args.template:
        # Reuse the template from run_fsi to keep them in sync
        from aeroelast.cli.run_fsi import TEMPLATE_CONFIG
        print(TEMPLATE_CONFIG)
        return 0

    if not args.config:
        parser.print_help()
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: configuration file not found: {config_path}")
        return 1

    setup_logging(args.verbose)

    # --preview
    if args.preview:
        from aeroelast.core.config import FSISimulationConfig
        cfg = FSISimulationConfig.from_yaml(str(config_path))
        print(cfg)
        return 0

    # --validate
    if args.validate:
        from aeroelast.cli.run_fsi import validate_config
        return 0 if validate_config(str(config_path)) else 1

    # --export-ccx
    if args.export_ccx:
        try:
            from aeroelast.solvers.fsi.runner import FSIRunner
            runner = FSIRunner(str(config_path), args.workdir)
            num_modes = 10
            if runner.config.solver.num_modes:
                num_modes = runner.config.solver.num_modes
            span_direction = None
            elem_cfg = getattr(runner.config, "elements", None)
            if elem_cfg is not None and getattr(elem_cfg, "span_direction", None) is not None:
                span_direction = tuple(float(v) for v in elem_cfg.span_direction)
            runner.export_calculix(args.export_ccx, num_modes=num_modes,
                                   span_direction=span_direction)
            return 0
        except Exception as e:
            logging.exception("CalculiX export failed")
            print(f"\nError: {e}")
            return 1

    # Run simulation
    try:
        from aeroelast.solvers.fsi.runner import FSIRunner
        runner = FSIRunner(str(config_path), args.workdir)
        runner.run()
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    except Exception as e:
        logging.exception("Simulation failed")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
