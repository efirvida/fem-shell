#!/usr/bin/env python3
"""
fem-shell-monitor — CLI entry point for the FSI Monitor TUI.

Monitors a running (or completed) FSI simulation in real time.
Reads rotor_performance.csv and optional checkpoints — independent of solver.

Usage:
    fem-shell-monitor /path/to/sim/workdir
    fem-shell-monitor /path/to/sim/workdir --csv-name my_output.csv
    fem-shell-monitor /path/to/sim/workdir --refresh-interval 1.0
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Entry point for fem-shell-monitor CLI."""
    parser = argparse.ArgumentParser(
        prog="fem-shell-monitor",
        description=(
            "Real-time TUI dashboard for FSI simulations. "
            "Reads rotor_performance.csv and checkpoints; no solver connection required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Key bindings (inside the TUI):
  q / ctrl+c      Quit
  l               Toggle Y-axis: linear ↔ logarithmic
  r               Reset plot to default series
  n               Next series page in legend
  1–5             Toggle series #1–5 in current legend page
  ?               Toggle help overlay
""",
    )

    parser.add_argument(
        "workdir",
        type=Path,
        help="Simulation working directory (must contain the CSV)",
    )
    parser.add_argument(
        "--csv-name",
        default="rotor_performance.csv",
        metavar="FILENAME",
        help="CSV filename within workdir (default: rotor_performance.csv)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.5,
        metavar="SECS",
        help="File polling interval in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    workdir: Path = args.workdir.resolve()
    if not workdir.is_dir():
        print(f"Error: workdir not found: {workdir}", file=sys.stderr)
        sys.exit(1)

    csv_file = workdir / args.csv_name
    if not csv_file.exists():
        print(
            f"Error: CSV file not found: {csv_file}\n"
            "Make sure the simulation has started and is writing output.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Lazy import — avoids pulling Textual at import time when not needed
    from fem_shell.fsi_monitor.tui.app import FSIMonitorApp

    app = FSIMonitorApp(
        csv_file=csv_file,
        refresh_interval=args.refresh_interval,
    )
    app.run()


if __name__ == "__main__":
    main()
