#!/usr/bin/env python3
"""
Plot X- and Y-displacement of all flap watchpoints in separate figures.

Looks for files named: precice-Solid-watchpoint-Flap-*-Tip.log
in the current directory (this solid folder).

Outputs:
    - flaps_displacement_x.png
    - flaps_displacement_y.png
"""

import io
import re
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import polars as pl


def _read_watchpoint(path: Path) -> pl.DataFrame:
    """Read a preCICE watchpoint .log with whitespace columns into a Polars DataFrame."""
    with path.open("r") as f:
        content = f.read()
    # Convert variable whitespace to CSV-like
    csv_content = (
        content.replace("   ", ",")
        .replace("  ", ",")
        .replace(" ", ",")
        .replace("\n,", "\n")
        .lstrip()
    )
    if csv_content.startswith(","):
        csv_content = csv_content[1:]
    return pl.read_csv(io.StringIO(csv_content), separator=",", has_header=True)


def _flap_index_from_name(stem: str) -> int:
    """Extract flap index from filename stem '...-Flap-<i>-Tip' or return a large number."""
    m = re.search(r"Flap[-_](\d+)[-_]Tip", stem)
    return int(m.group(1)) if m else 10**9


def find_watchpoints(base: Path) -> List[Path]:
    # Support both 'Flap-<i>-Tip' and 'Flap<i>-Tip' naming
    matches = list(base.glob("precice-Solid-watchpoint-Flap-*-Tip.log"))
    matches += list(base.glob("precice-Solid-watchpoint-Flap*-Tip.log"))
    # Unique paths
    unique = {p.resolve(): p for p in matches}.values()
    return sorted(unique, key=lambda p: _flap_index_from_name(p.stem))


def _plot_component(files: List[Path], comp: int, title: str, ylabel: str, outfile: Path) -> bool:
    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    colname = f"Displacement{comp}"
    for fp in files:
        try:
            df = _read_watchpoint(fp)
        except Exception as e:
            print(f"Skipping {fp.name}: read error: {e}")
            continue

        if "Time" not in df.columns or colname not in df.columns:
            print(f"Skipping {fp.name}: missing columns 'Time' or '{colname}'")
            continue

        time = df["Time"].to_numpy()
        disp = df[colname].to_numpy()
        flap_label = fp.stem.split("-watchpoint-")[-1]
        ax.plot(time, disp, linewidth=2, label=flap_label)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.axhline(y=0, color="black", linewidth=1.5, linestyle="-")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    return True


def _plot_resultant(files: List[Path], outfile: Path) -> bool:
    """Plot the signed total displacement: sign(X) * sqrt(X^2 + Y^2) for each flap."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False
    for fp in files:
        try:
            df = _read_watchpoint(fp)
        except Exception as e:
            print(f"Skipping {fp.name}: read error: {e}")
            continue
        if (
            "Time" not in df.columns
            or "Displacement0" not in df.columns
            or "Displacement1" not in df.columns
        ):
            print(f"Skipping {fp.name}: missing columns 'Time', 'Displacement0' or 'Displacement1'")
            continue
        time = df["Time"].to_numpy()
        disp_x = df["Displacement0"].to_numpy()
        disp_y = df["Displacement1"].to_numpy()
        # Signed total displacement: magnitude with sign of X component
        magnitude = np.sqrt(disp_x**2 + disp_y**2)
        resultant = np.sign(disp_x) * magnitude
        flap_label = fp.stem.split("-watchpoint-")[-1]
        ax.plot(time, resultant, linewidth=2, label=flap_label)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return False
    ax.set_title("Flap tip displacement (Total)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total Displacement [m]")
    ax.axhline(y=0, color="black", linewidth=1.5, linestyle="-")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    return True


def main() -> int:
    here = Path(__file__).parent
    files = find_watchpoints(here)
    if not files:
        print("No watchpoint files found in this folder.")
        print("Expected pattern: precice-Solid-watchpoint-Flap-*-Tip.log")
        return 2

    # Plot X (component 0)
    out_x = here / "flaps_displacement_x.png"
    ok_x = _plot_component(
        files,
        comp=0,
        title="Flap tip displacement (X)",
        ylabel="Displacement X [m]",
        outfile=out_x,
    )

    # Plot Y (component 1)
    out_y = here / "flaps_displacement_y.png"
    ok_y = _plot_component(
        files,
        comp=1,
        title="Flap tip displacement (Y)",
        ylabel="Displacement Y [m]",
        outfile=out_y,
    )

    # Plot Resultant
    out_res = here / "flaps_displacement_resultant.png"
    ok_res = _plot_resultant(
        files,
        outfile=out_res,
    )

    if not (ok_x or ok_y or ok_res):
        print("No valid time/displacement data to plot for X, Y, or Resultant.")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
