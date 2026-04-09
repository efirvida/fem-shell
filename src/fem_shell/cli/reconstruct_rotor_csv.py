#!/usr/bin/env python3
"""
Reconstruct rotor_performance.csv from checkpoint state.npz files and any
existing rotor_performance.csv.

Strategy
--------
1. Load all rows from the existing ``rotor_performance.csv`` (full data).
2. Scan ``<results_dir>/<t>/state.npz`` files for kinematic coverage.
   For each checkpoint time NOT already covered by rotor_performance.csv
   (within a small tolerance), add a kinematics-only row with blank force
   columns.
3. Sort all rows by time and write the merged result.

This correctly handles the typical restart scenario where:
  - rotor_performance.csv only covers the time range of the most recent run.
  - Checkpoints from earlier runs cover the full simulation history.
  - The two datasets may have different time resolutions.

Usage
-----
    fem-shell-reconstruct-csv results/

    # Read output_folder from YAML config
    fem-shell-reconstruct-csv results/ --config simulation.yaml

    # Override output file
    fem-shell-reconstruct-csv results/ --output recovered.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_MIN_DENOM = 1e-12
_TIME_TOL  = 5e-5   # seconds — tolerance for deduplication between sources

# CSV header — must match _log_rotor_performance in rotor.py
_CSV_HEADER = (
    "Time [s],Angle [deg],Speed [RPM],Omega [rad/s],Alpha [rad/s2],"
    "Aero Thrust [N],"
    "Aero Torque [Nm],Non-Aero Torque [Nm],Inertial Torque [Nm],"
    "Gravity Torque [Nm],Total Torque [Nm],"
    "Aero Power [W],Total Power [W],Structural Efficiency,"
    "Cp,Cq,Ct,TSR,"
    "Aero Torque X [Nm],Aero Torque Y [Nm],Aero Torque Z [Nm],"
    "Total Torque X [Nm],Total Torque Y [Nm],Total Torque Z [Nm],"
    "Max Displacement [m],Deformed Radius [m]\n"
)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_checkpoints(results_dir: str) -> List[Dict]:
    """
    Scan time-stamped checkpoint folders and load ``state.npz``.

    Returns a list of dicts (sorted by time) with keys:
    ``t``, ``theta`` [rad], ``omega`` [rad/s], ``alpha`` (None until filled).
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    pattern = re.compile(r"^(\d+(?:\.\d+)?)$")
    entries: List[Tuple[float, str]] = []

    for entry in os.listdir(results_dir):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        m = pattern.match(entry)
        if not m:
            continue
        npz = os.path.join(path, "state.npz")
        if not os.path.exists(npz):
            continue
        entries.append((float(m.group(1)), path))

    if not entries:
        raise RuntimeError(f"No checkpoints (state.npz) found in: {results_dir}")

    entries.sort(key=lambda x: x[0])

    rows: List[Dict] = []
    for t_folder, path in entries:
        npz = os.path.join(path, "state.npz")
        try:
            with np.load(npz) as data:
                t     = float(data["t"])     if "t"     in data.files else t_folder
                theta = float(data["theta"]) if "theta" in data.files else 0.0
                omega = float(data["omega"]) if "omega" in data.files else 0.0
                alpha = float(data["alpha"]) if "alpha" in data.files else None
        except Exception as e:
            _logger.warning("Skipping corrupt checkpoint %s: %s", path, e)
            continue
        rows.append({"t": t, "theta": theta, "omega": omega, "alpha": alpha})

    # Fill missing alpha via central finite differences
    n = len(rows)
    for i, row in enumerate(rows):
        if row["alpha"] is not None:
            continue
        if n < 2:
            row["alpha"] = float("nan")
        elif i == 0:
            dt = rows[1]["t"] - rows[0]["t"]
            row["alpha"] = ((rows[1]["omega"] - rows[0]["omega"]) / dt
                            if dt > _MIN_DENOM else float("nan"))
        elif i == n - 1:
            dt = rows[-1]["t"] - rows[-2]["t"]
            row["alpha"] = ((rows[-1]["omega"] - rows[-2]["omega"]) / dt
                            if dt > _MIN_DENOM else float("nan"))
        else:
            dt = rows[i + 1]["t"] - rows[i - 1]["t"]
            row["alpha"] = ((rows[i + 1]["omega"] - rows[i - 1]["omega"]) / dt
                            if dt > _MIN_DENOM else float("nan"))

    return rows


def _load_performance_csv(results_dir: str) -> List[Dict]:
    """
    Load ``rotor_performance.csv`` and return a list of row dicts, each with:
    ``t`` (float), ``raw_line`` (str, verbatim CSV line without newline).

    Returns an empty list if the file does not exist.
    """
    csv_path = os.path.join(results_dir, "rotor_performance.csv")
    if not os.path.exists(csv_path):
        return []

    rows: List[Dict] = []
    with open(csv_path) as fh:
        next(fh, None)  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                t = float(line.split(",", 1)[0])
                rows.append({"t": t, "raw_line": line, "source": "perf"})
            except ValueError:
                pass

    print(f"Loaded {len(rows)} rows from rotor_performance.csv")
    return rows


# ---------------------------------------------------------------------------
# Main reconstruction logic
# ---------------------------------------------------------------------------

def reconstruct(
    results_dir: str,
    output: Optional[str] = None,
) -> str:
    """
    Reconstruct ``rotor_performance.csv`` from checkpoint data.

    Combines:
    - All rows from existing ``rotor_performance.csv`` (full data).
    - Kinematic rows from ``state.npz`` checkpoints not already covered.

    Parameters
    ----------
    results_dir : str
        Directory containing time-stamped checkpoint folders and optionally
        an existing ``rotor_performance.csv``.
    output : str, optional
        Output CSV path.  Defaults to
        ``<results_dir>/rotor_performance_reconstructed.csv``.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    if output is None:
        output = os.path.join(results_dir, "rotor_performance_reconstructed.csv")

    # Load both sources
    perf_rows  = _load_performance_csv(results_dir)
    ckpt_rows  = _load_checkpoints(results_dir)
    print(f"Found {len(ckpt_rows)} checkpoint(s) in '{results_dir}'.")

    # Build set of times already covered by rotor_performance.csv
    perf_times = {r["t"] for r in perf_rows}

    # Number of blank force columns = total CSV columns - 5 kinematic columns
    n_total_cols = _CSV_HEADER.rstrip("\n").count(",") + 1
    blank = "," * (n_total_cols - 5 - 1)  # commas needed after the 5th field
    extra_rows: List[Dict] = []

    for row in ckpt_rows:
        t = row["t"]
        # Skip if this time is already covered (within tolerance)
        if any(abs(t - tp) <= _TIME_TOL for tp in perf_times):
            continue

        theta = row["theta"]
        omega = row["omega"]
        alpha = row.get("alpha", float("nan"))

        angle_deg = float(np.degrees(theta))
        omega_rpm = omega * 30.0 / np.pi
        alpha_str = f"{alpha:.6e}" if (alpha is not None and not np.isnan(alpha)) else ""

        raw_line = (
            f"{t:.6f},{angle_deg:.4f},{omega_rpm:.4f},{omega:.6e},{alpha_str},"
            f"{blank}"
        )
        extra_rows.append({"t": t, "raw_line": raw_line, "source": "ckpt"})

    # Merge and sort by time
    all_rows = sorted(perf_rows + extra_rows, key=lambda r: r["t"])

    n_full = sum(1 for r in all_rows if r["source"] == "perf")
    n_kin  = sum(1 for r in all_rows if r["source"] == "ckpt")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        fh.write(_CSV_HEADER)
        for row in all_rows:
            fh.write(row["raw_line"] + "\n")

    print(
        f"Wrote {len(all_rows)} rows to '{output}' "
        f"({n_full} full from rotor_performance.csv, {n_kin} kinematics-only from checkpoints)."
    )
    return output


# ---------------------------------------------------------------------------
# YAML config reader
# ---------------------------------------------------------------------------

def _output_from_yaml(yaml_path: str) -> Optional[str]:
    """Return ``output.folder`` from an FSI YAML config, or None."""
    try:
        import yaml
    except ImportError:
        _logger.warning("PyYAML not available — ignoring --config.")
        return None

    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    return (
        cfg.get("output", {}).get("folder")
        or cfg.get("solver", {}).get("output_folder")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fem-shell-reconstruct-csv",
        description=(
            "Reconstruct rotor_performance.csv merging checkpoint kinematics\n"
            "with any existing rotor_performance.csv rows."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "results_dir",
        help="Results directory (contains <t>/state.npz and optionally rotor_performance.csv).",
    )
    p.add_argument(
        "--config", "-c",
        metavar="YAML",
        help="FSI YAML config file (reads output.folder to locate results_dir).",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output CSV path (default: <results_dir>/rotor_performance_reconstructed.csv).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    results_dir = args.results_dir

    # If a YAML is given and results_dir is not explicitly set, infer it
    if args.config and not os.path.isdir(results_dir):
        folder = _output_from_yaml(args.config)
        if folder:
            results_dir = folder

    params: Dict = {}
    if args.output is not None:
        params["output"] = args.output

    try:
        out = reconstruct(results_dir, **params)
        print(f"Done: {out}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
