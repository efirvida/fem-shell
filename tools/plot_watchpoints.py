#!/usr/bin/env python3
"""
Quick watchpoint visualization script for preCICE FSI simulations.

Usage:
    python plot_watchpoints.py <watchpoint_file> [options]

Examples:
    python plot_watchpoints.py precice-Solid-watchpoint-Flap-Tip.log
    python plot_watchpoints.py precice-Fluid-watchpoint-Center.log --displacement --force
    python plot_watchpoints.py precice-Solid-watchpoint-*.log --all
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl


class WatchpointPlotter:
    """Simple watchpoint data plotter for preCICE simulations."""

    def __init__(self, file_path: str):
        """Load and parse watchpoint file."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Watchpoint file not found: {file_path}")

        # Parse watchpoint file
        with open(self.file_path, "r") as f:
            content = f.read()

        # Convert space-separated format to CSV
        csv_content = (
            content.replace("   ", ",")
            .replace("  ", ",")
            .replace(" ", ",")
            .replace("\n,", "\n")
            .lstrip()  # remove any leading whitespace/newlines
        )
        if csv_content.startswith(","):
            csv_content = csv_content[1:]

        self.df = pl.read_csv(io.StringIO(csv_content), separator=",", has_header=True)
        self.case_name = self._extract_case_name()
        self._detect_variables()

    def _extract_case_name(self) -> str:
        """Extract case name from filename."""
        name = self.file_path.stem.replace("precice-", "").replace("-watchpoint-", " | ")
        return name

    def _detect_variables(self):
        """Detect available variables in the data."""
        self.vars = {"Displacement": [], "Force": [], "Velocity": [], "Pressure": []}
        for col in self.df.columns:
            for var in list(self.vars.keys()):
                if var in col:
                    idx = col.replace(var, "").strip()
                    if idx.isdigit():
                        self.vars[var].append(int(idx))
        # Drop missing groups
        for var in list(self.vars.keys()):
            if not self.vars[var]:
                del self.vars[var]

    def plot_displacement(self, save_path: Optional[str] = None):
        """Plot displacement over time."""
        if "Displacement" not in self.vars:
            print("No displacement data found")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        n_dims = len(self.vars["Displacement"])
        labels = ["X", "Y", "Z"]

        for i in range(n_dims):
            col_name = f"Displacement{i}"
            if col_name in self.df.columns:
                ax.plot(self.df["Time"], self.df[col_name], label=f"u_{labels[i]}", linewidth=2)

        ax.set_title(f"Displacement - {self.case_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Displacement [m]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()

    def plot_force(self, save_path: Optional[str] = None):
        """Plot force over time."""
        if "Force" not in self.vars:
            print("No force data found")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        n_dims = len(self.vars["Force"])
        labels = ["X", "Y", "Z"]

        for i in range(n_dims):
            col_name = f"Force{i}"
            if col_name in self.df.columns:
                ax.plot(self.df["Time"], self.df[col_name], label=f"F_{labels[i]}", linewidth=2)

        ax.set_title(f"Force - {self.case_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force [N]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()

    def plot_velocity(self, save_path: Optional[str] = None):
        """Plot velocity over time."""
        if "Velocity" not in self.vars:
            print("No velocity data found")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        n_dims = len(self.vars["Velocity"])
        labels = ["X", "Y", "Z"]

        for i in range(n_dims):
            col_name = f"Velocity{i}"
            if col_name in self.df.columns:
                ax.plot(self.df["Time"], self.df[col_name], label=f"v_{labels[i]}", linewidth=2)

        ax.set_title(f"Velocity - {self.case_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()

    def plot_pressure(self, save_path: Optional[str] = None):
        """Plot pressure over time."""
        if "Pressure" not in self.vars:
            print("No pressure data found")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        col_name = "Pressure0"
        if col_name in self.df.columns:
            ax.plot(self.df["Time"], self.df[col_name], linewidth=2, color="orange")

        ax.set_title(f"Pressure - {self.case_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [Pa]")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()

    def plot_all(self, output_dir: Optional[str] = None):
        """Plot all available variables."""
        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)

        case_base = self.file_path.stem

        if "Displacement" in self.vars:
            save = f"{output_dir}/{case_base}_displacement.png" if output_dir else None
            self.plot_displacement(save_path=save)

        if "Force" in self.vars:
            save = f"{output_dir}/{case_base}_force.png" if output_dir else None
            self.plot_force(save_path=save)

        if "Velocity" in self.vars:
            save = f"{output_dir}/{case_base}_velocity.png" if output_dir else None
            self.plot_velocity(save_path=save)

        if "Pressure" in self.vars:
            save = f"{output_dir}/{case_base}_pressure.png" if output_dir else None
            self.plot_pressure(save_path=save)

    def print_info(self):
        """Print data summary."""
        print(f"\nWatchpoint Data Summary")
        print(f"   File: {self.file_path.name}")
        print(f"   Case: {self.case_name}")
        print(f"   Time steps: {len(self.df)}")
        print(f"   Available variables: {', '.join(self.vars.keys())}")
        if "Time" in self.df.columns:
            t_start = self.df["Time"][0]
            t_end = self.df["Time"][-1]
            print(f"   Time range: {t_start:.6f} → {t_end:.6f} s")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Quick visualization of preCICE watchpoint data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_watchpoints.py precice-Solid-watchpoint-Flap-Tip.log
  python plot_watchpoints.py precice-Fluid-watchpoint-Center.log --displacement --force
  python plot_watchpoints.py precice-Solid-watchpoint-*.log --all --output plots/
        """,
    )

    parser.add_argument("file", help="Watchpoint log file to plot")
    parser.add_argument("--displacement", action="store_true", help="Plot displacement")
    parser.add_argument("--force", action="store_true", help="Plot force")
    parser.add_argument("--velocity", action="store_true", help="Plot velocity")
    parser.add_argument("--pressure", action="store_true", help="Plot pressure")
    parser.add_argument("--all", action="store_true", help="Plot all available variables")
    parser.add_argument("--output", "-o", help="Output directory for saving plots")
    parser.add_argument("--info", action="store_true", help="Print data info and exit")

    args = parser.parse_args()

    # Default: plot all if no specific plot requested
    if not any([args.displacement, args.force, args.velocity, args.pressure, args.all, args.info]):
        args.all = True

    try:
        plotter = WatchpointPlotter(args.file)

        if args.info:
            plotter.print_info()
            return

        print(f"Processing: {args.file}")
        plotter.print_info()

        if args.all:
            plotter.plot_all(output_dir=args.output)
        else:
            if args.displacement:
                save = f"{args.output}/displacement.png" if args.output else None
                plotter.plot_displacement(save_path=save)

            if args.force:
                save = f"{args.output}/force.png" if args.output else None
                plotter.plot_force(save_path=save)

            if args.velocity:
                save = f"{args.output}/velocity.png" if args.output else None
                plotter.plot_velocity(save_path=save)

            if args.pressure:
                save = f"{args.output}/pressure.png" if args.output else None
                plotter.plot_pressure(save_path=save)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
