#!/usr/bin/env python3
"""
OpenFOAM postProcessing and residuals plotting tool.

Usage:
    # List available data sources
    python plot_openfoam.py /path/to/case --list

    # Plot residuals
    python plot_openfoam.py /path/to/case residuals Ux Uy Uz p
    python plot_openfoam.py /path/to/case residuals --all

    # Plot any postProcessing function object
    python plot_openfoam.py /path/to/case blade01Forces          # auto-detect and plot forces
    python plot_openfoam.py /path/to/case blade01Forces force    # plot force.dat
    python plot_openfoam.py /path/to/case blade01Forces moment   # plot moment.dat
    python plot_openfoam.py /path/to/case fieldMinMax            # plot field min/max

Examples:
    python plot_openfoam.py examples/fsi/rotor/fluid residuals Ux Uy Uz
    python plot_openfoam.py examples/fsi/rotor/fluid blade01Forces --components x y z
    python plot_openfoam.py examples/fsi/rotor/fluid_steady fieldMinMax
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Residuals parsing (from log files and solverInfo.dat)
# =============================================================================


def parse_residuals_from_log(
    log_file: Path, variables: list[str], use_initial: bool = True
) -> dict:
    """Parse residuals from OpenFOAM log file."""
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    data = {"time": []}
    for var in variables:
        data[var] = []

    time_pattern = re.compile(r"^Time\s*=\s*([0-9.eE+-]+)")
    residual_type = "Initial" if use_initial else "Final"

    var_patterns = {}
    for var in variables:
        pattern = re.compile(
            rf"Solving for {re.escape(var)},.*{residual_type} residual\s*=\s*([0-9.eE+-]+)"
        )
        var_patterns[var] = pattern

    current_time = None
    current_residuals = {var: [] for var in variables}

    with open(log_file, "r") as f:
        for line in f:
            time_match = time_pattern.match(line)
            if time_match:
                if current_time is not None:
                    data["time"].append(current_time)
                    for var in variables:
                        if current_residuals[var]:
                            data[var].append(current_residuals[var][-1])
                        else:
                            data[var].append(np.nan)

                current_time = float(time_match.group(1))
                current_residuals = {var: [] for var in variables}
                continue

            for var, pattern in var_patterns.items():
                match = pattern.search(line)
                if match:
                    current_residuals[var].append(float(match.group(1)))

    if current_time is not None:
        data["time"].append(current_time)
        for var in variables:
            if current_residuals[var]:
                data[var].append(current_residuals[var][-1])
            else:
                data[var].append(np.nan)

    return data


def find_log_file(case_path: Path) -> Optional[Path]:
    """Find the solver log file in the case directory."""
    priority_solvers = [
        "log.simpleFoam",
        "log.pimpleFoam",
        "log.pisoFoam",
        "log.interFoam",
        "log.rhoPimpleFoam",
        "log.rhoSimpleFoam",
        "log.sonicFoam",
        "log.buoyantPimpleFoam",
        "log.buoyantSimpleFoam",
    ]

    for solver in priority_solvers:
        log_file = case_path / solver
        if log_file.exists():
            return log_file

    exclude_patterns = [
        "potentialFoam",
        "blockMesh",
        "snappyHexMesh",
        "decomposePar",
        "reconstructPar",
        "checkMesh",
        "renumberMesh",
        "topoSet",
        "createPatch",
        "surfaceFeatureExtract",
    ]

    for log_file in case_path.glob("log.*Foam"):
        if not any(excl in log_file.name for excl in exclude_patterns):
            return log_file

    for log_file in case_path.glob("log.*"):
        if any(excl in log_file.name for excl in exclude_patterns):
            continue
        try:
            with open(log_file, "r") as f:
                content = f.read(5000)
                if "Solving for" in content:
                    return log_file
        except:
            pass

    return None


def get_available_variables(log_file: Path) -> list[str]:
    """Scan log file to find all available variables."""
    variables = set()
    pattern = re.compile(r"Solving for (\w+),")

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                variables.add(match.group(1))

    return sorted(variables)


def find_solver_info_file(case_path: Path) -> Optional[Path]:
    """Find solverInfo.dat in postProcessing directory."""
    residuals_dir = case_path / "postProcessing" / "residuals"
    if not residuals_dir.exists():
        return None

    time_dirs = sorted([d for d in residuals_dir.iterdir() if d.is_dir()])
    if not time_dirs:
        return None

    solver_info = time_dirs[0] / "solverInfo.dat"
    if solver_info.exists():
        return solver_info

    return None


def parse_solver_info(
    solver_info_file: Path, variables: list[str], use_initial: bool = True
) -> dict:
    """Parse residuals from solverInfo.dat file."""
    with open(solver_info_file, "r") as f:
        lines = f.readlines()

    header_line = None
    for line in lines:
        if line.startswith("# Time"):
            header_line = line
            break

    if not header_line:
        raise ValueError(f"Could not find header in {solver_info_file}")

    columns = header_line.replace("#", "").split()

    col_indices = {}
    residual_type = "initial" if use_initial else "final"

    for var in variables:
        col_name = f"{var}_{residual_type}"
        if col_name in columns:
            col_indices[var] = columns.index(col_name)

    data = {"time": []}
    for var in variables:
        data[var] = []

    for line in lines:
        if line.startswith("#") or not line.strip():
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            time_val = float(parts[0])
            data["time"].append(time_val)

            for var in variables:
                if var in col_indices:
                    idx = col_indices[var]
                    if idx < len(parts):
                        data[var].append(float(parts[idx]))
                    else:
                        data[var].append(np.nan)
                else:
                    data[var].append(np.nan)
        except (ValueError, IndexError):
            continue

    return data


def get_available_variables_from_solver_info(solver_info_file: Path) -> list[str]:
    """Get available variables from solverInfo.dat header."""
    variables = set()

    with open(solver_info_file, "r") as f:
        for line in f:
            if line.startswith("# Time"):
                columns = line.replace("#", "").split()
                for col in columns:
                    if col.endswith("_initial"):
                        var = col.replace("_initial", "")
                        variables.add(var)
                break

    return sorted(variables)


# =============================================================================
# PostProcessing parsing (forces, moments, fieldMinMax, etc.)
# =============================================================================


def get_postprocessing_dirs(case_path: Path) -> list[str]:
    """Get list of postProcessing function objects."""
    post_dir = case_path / "postProcessing"
    if not post_dir.exists():
        return []

    return sorted([d.name for d in post_dir.iterdir() if d.is_dir()])


def get_postprocessing_files(case_path: Path, func_name: str) -> list[str]:
    """Get list of data files in a postProcessing function object."""
    func_dir = case_path / "postProcessing" / func_name
    if not func_dir.exists():
        return []

    # Find the time directory
    time_dirs = sorted([d for d in func_dir.iterdir() if d.is_dir()])
    if not time_dirs:
        return []

    # Get .dat files
    dat_files = []
    for f in time_dirs[0].iterdir():
        if f.suffix == ".dat":
            dat_files.append(f.stem)

    return sorted(dat_files)


def detect_postprocessing_type(case_path: Path, func_name: str) -> str:
    """Detect the type of postProcessing function object."""
    files = get_postprocessing_files(case_path, func_name)

    if "force" in files or "moment" in files:
        return "forces"
    elif "solverInfo" in files:
        return "residuals"
    elif "fieldMinMax" in files:
        return "fieldMinMax"
    elif "coefficient" in files:
        return "forceCoeffs"
    else:
        return "generic"


def parse_forces(case_path: Path, func_name: str, data_type: str = "force") -> dict:
    """Parse forces or moments from postProcessing directory."""
    post_dir = case_path / "postProcessing" / func_name

    if not post_dir.exists():
        raise FileNotFoundError(f"PostProcessing directory not found: {post_dir}")

    time_dirs = sorted([d for d in post_dir.iterdir() if d.is_dir()])
    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in: {post_dir}")

    data = {
        "time": [],
        "total_x": [],
        "total_y": [],
        "total_z": [],
        "pressure_x": [],
        "pressure_y": [],
        "pressure_z": [],
        "viscous_x": [],
        "viscous_y": [],
        "viscous_z": [],
    }

    for time_dir in time_dirs:
        data_file = time_dir / f"{data_type}.dat"
        if not data_file.exists():
            continue

        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                parts = line.split()
                if len(parts) >= 10:
                    data["time"].append(float(parts[0]))
                    data["total_x"].append(float(parts[1]))
                    data["total_y"].append(float(parts[2]))
                    data["total_z"].append(float(parts[3]))
                    data["pressure_x"].append(float(parts[4]))
                    data["pressure_y"].append(float(parts[5]))
                    data["pressure_z"].append(float(parts[6]))
                    data["viscous_x"].append(float(parts[7]))
                    data["viscous_y"].append(float(parts[8]))
                    data["viscous_z"].append(float(parts[9]))

    return data


def parse_field_minmax(case_path: Path, func_name: str) -> dict:
    """Parse fieldMinMax data from postProcessing directory."""
    post_dir = case_path / "postProcessing" / func_name

    if not post_dir.exists():
        raise FileNotFoundError(f"PostProcessing directory not found: {post_dir}")

    time_dirs = sorted([d for d in post_dir.iterdir() if d.is_dir()])
    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in: {post_dir}")

    # First pass: find all fields
    fields = set()
    for time_dir in time_dirs:
        data_file = time_dir / "fieldMinMax.dat"
        if not data_file.exists():
            continue

        with open(data_file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    fields.add(parts[1])

    # Initialize data structure
    data = {"time": []}
    for field in fields:
        data[f"{field}_min"] = []
        data[f"{field}_max"] = []

    # Second pass: collect data
    current_time = None
    current_values = {}

    for time_dir in time_dirs:
        data_file = time_dir / "fieldMinMax.dat"
        if not data_file.exists():
            continue

        with open(data_file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 4:
                    time_val = float(parts[0])
                    field = parts[1]
                    min_val = float(parts[2])
                    # Max value position varies due to location string
                    # Find it by looking for the next numeric value after min
                    max_val = None
                    for i in range(3, len(parts)):
                        try:
                            # Skip location tuples
                            if parts[i].startswith("("):
                                continue
                            max_val = float(parts[i])
                            break
                        except ValueError:
                            continue

                    if max_val is None:
                        continue

                    if time_val != current_time:
                        if current_time is not None:
                            data["time"].append(current_time)
                            for f in fields:
                                data[f"{f}_min"].append(current_values.get(f"{f}_min", np.nan))
                                data[f"{f}_max"].append(current_values.get(f"{f}_max", np.nan))
                        current_time = time_val
                        current_values = {}

                    current_values[f"{field}_min"] = min_val
                    current_values[f"{field}_max"] = max_val

    # Last entry
    if current_time is not None:
        data["time"].append(current_time)
        for f in fields:
            data[f"{f}_min"].append(current_values.get(f"{f}_min", np.nan))
            data[f"{f}_max"].append(current_values.get(f"{f}_max", np.nan))

    return data, list(fields)


def parse_generic_dat(case_path: Path, func_name: str, dat_file: str) -> dict:
    """Parse a generic .dat file from postProcessing."""
    post_dir = case_path / "postProcessing" / func_name

    if not post_dir.exists():
        raise FileNotFoundError(f"PostProcessing directory not found: {post_dir}")

    time_dirs = sorted([d for d in post_dir.iterdir() if d.is_dir()])
    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in: {post_dir}")

    # Read header to get column names
    data_file = time_dirs[0] / f"{dat_file}.dat"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    columns = []
    data = {}

    with open(data_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                # Try to parse header
                if "Time" in line:
                    columns = line.replace("#", "").split()
                continue
            break

    if not columns:
        columns = ["time"] + [f"col{i}" for i in range(20)]

    for col in columns:
        data[col] = []

    # Parse all time directories
    for time_dir in time_dirs:
        data_file = time_dir / f"{dat_file}.dat"
        if not data_file.exists():
            continue

        with open(data_file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.split()
                for i, col in enumerate(columns):
                    if i < len(parts):
                        try:
                            data[col].append(float(parts[i]))
                        except ValueError:
                            data[col].append(np.nan)

    return data, columns


# =============================================================================
# Plotting functions
# =============================================================================


def plot_residuals(
    data: dict,
    variables: list[str],
    title: str = "Residuals",
    save_path: Optional[str] = None,
    use_initial: bool = True,
):
    """Plot residuals vs time."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for var in variables:
        if var in data and data[var]:
            ax.semilogy(data["time"], data[var], label=var, linewidth=1.5)

    residual_type = "Initial" if use_initial else "Final"
    ax.set_title(f"{title} - {residual_type} Residuals", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_forces(
    data: dict,
    components: list[str] = None,
    force_type: str = "total",
    title: str = "Forces",
    save_path: Optional[str] = None,
    data_label: str = "Force",
    unit: str = "N",
):
    """Plot forces/moments vs time."""
    if components is None:
        components = ["x", "y", "z"]

    fig, ax = plt.subplots(figsize=(12, 7))

    labels = {"x": "X", "y": "Y", "z": "Z"}
    colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}

    for comp in components:
        key = f"{force_type}_{comp}"
        if key in data and data[key]:
            ax.plot(
                data["time"],
                data[key],
                label=f"{data_label}_{labels[comp]}",
                linewidth=1.5,
                color=colors.get(comp),
            )

    ax.set_title(f"{title} - {force_type.capitalize()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel(f"{data_label} [{unit}]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_field_minmax(
    data: dict,
    fields: list[str],
    title: str = "Field Min/Max",
    save_path: Optional[str] = None,
    plot_type: str = "both",
):
    """Plot field min/max values."""
    n_fields = len(fields)

    if n_fields == 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_fields, 1, figsize=(12, 4 * n_fields), sharex=True)
        if n_fields == 1:
            axes = [axes]

    for i, field in enumerate(fields):
        ax = axes[i]

        if plot_type in ["both", "min"]:
            key = f"{field}_min"
            if key in data and data[key]:
                ax.plot(
                    data["time"], data[key], label=f"{field} min", linewidth=1.5, color="tab:blue"
                )

        if plot_type in ["both", "max"]:
            key = f"{field}_max"
            if key in data and data[key]:
                ax.plot(
                    data["time"], data[key], label=f"{field} max", linewidth=1.5, color="tab:red"
                )

        ax.set_ylabel(field, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=10)

    axes[-1].set_xlabel("Time [s]", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_generic(
    data: dict, columns: list[str], title: str = "Data", save_path: Optional[str] = None
):
    """Plot generic data."""
    fig, ax = plt.subplots(figsize=(12, 7))

    time_col = columns[0] if columns else "time"

    for col in columns[1:]:
        if col in data and data[col]:
            ax.plot(data[time_col], data[col], label=col, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(time_col, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================


def list_available(case_path: Path):
    """List available data sources."""
    print(f"\n{'=' * 60}")
    print(f"Case: {case_path}")
    print(f"{'=' * 60}")

    # Residuals sources
    print("\n[Residuals]")
    solver_info = find_solver_info_file(case_path)
    if solver_info:
        variables = get_available_variables_from_solver_info(solver_info)
        print(f"  solverInfo.dat: {', '.join(variables)}")

    log_file = find_log_file(case_path)
    if log_file:
        variables = get_available_variables(log_file)
        print(f"  {log_file.name}: {', '.join(variables)}")

    if not solver_info and not log_file:
        print("  No residual data found")

    # PostProcessing
    post_dirs = get_postprocessing_dirs(case_path)
    if post_dirs:
        print("\n[PostProcessing]")
        for func_name in post_dirs:
            if func_name == "residuals":
                continue
            func_type = detect_postprocessing_type(case_path, func_name)
            files = get_postprocessing_files(case_path, func_name)
            print(f"  {func_name} ({func_type}): {', '.join(files)}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot OpenFOAM residuals and postProcessing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available data sources
  %(prog)s /path/to/case --list
  
  # Plot residuals
  %(prog)s /path/to/case residuals Ux Uy Uz p
  %(prog)s /path/to/case residuals --all
  %(prog)s /path/to/case residuals --all --final
  
  # Plot forces (auto-detects force.dat)
  %(prog)s /path/to/case blade01Forces
  %(prog)s /path/to/case blade01Forces force
  %(prog)s /path/to/case blade01Forces moment
  %(prog)s /path/to/case blade01Forces --components x z --type pressure
  
  # Plot fieldMinMax
  %(prog)s /path/to/case fieldMinMax
  
  # Save to file
  %(prog)s /path/to/case residuals --all --save residuals.png
        """,
    )

    parser.add_argument("case", type=str, help="Path to OpenFOAM case directory")
    parser.add_argument(
        "source",
        type=str,
        nargs="?",
        help="Data source: 'residuals' or postProcessing function name",
    )
    parser.add_argument("series", type=str, nargs="*", help="Variables/file to plot")

    parser.add_argument("--all", action="store_true", help="Plot all available variables")
    parser.add_argument(
        "--initial", action="store_true", default=True, help="Use initial residuals (default)"
    )
    parser.add_argument(
        "--final", action="store_true", help="Use final residuals instead of initial"
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        choices=["x", "y", "z"],
        help="Force/moment components to plot (default: all)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["total", "pressure", "viscous"],
        default="total",
        help="Type of force/moment to plot (default: total)",
    )
    parser.add_argument(
        "--save", type=str, metavar="FILE", help="Save plot to file instead of displaying"
    )
    parser.add_argument("--list", action="store_true", help="List available data sources")
    parser.add_argument("--log", type=str, metavar="FILE", help="Specify log file explicitly")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["auto", "log", "solverInfo"],
        default="auto",
        dest="data_source",
        help="Residuals data source (default: auto)",
    )

    args = parser.parse_args()

    case_path = Path(args.case).resolve()
    if not case_path.exists():
        print(f"Error: Case directory not found: {case_path}", file=sys.stderr)
        sys.exit(1)

    # List mode
    if args.list:
        list_available(case_path)
        return

    if not args.source:
        parser.print_help()
        print("\nUse --list to see available data sources")
        return

    use_initial = not args.final

    # Handle residuals
    if args.source == "residuals":
        solver_info = find_solver_info_file(case_path)
        log_file = None

        if args.log:
            log_file = Path(args.log)
            use_solver_info = False
        elif args.data_source == "solverInfo":
            use_solver_info = True
        elif args.data_source == "log":
            use_solver_info = False
            log_file = find_log_file(case_path)
        else:
            use_solver_info = solver_info is not None
            if not use_solver_info:
                log_file = find_log_file(case_path)

        if use_solver_info and solver_info:
            if args.all:
                variables = get_available_variables_from_solver_info(solver_info)
            elif args.series:
                variables = args.series
            else:
                available = get_available_variables_from_solver_info(solver_info)
                print("Error: Specify variables to plot or use --all", file=sys.stderr)
                print(f"Available: {', '.join(available)}")
                sys.exit(1)

            print(f"Parsing residuals from: {solver_info}")
            print(f"Variables: {', '.join(variables)}")
            data = parse_solver_info(solver_info, variables, use_initial)
        else:
            if not log_file:
                log_file = find_log_file(case_path)

            if not log_file:
                print("Error: No solver log file found. Use --log to specify.", file=sys.stderr)
                sys.exit(1)

            if args.all:
                variables = get_available_variables(log_file)
            elif args.series:
                variables = args.series
            else:
                print("Error: Specify variables to plot or use --all", file=sys.stderr)
                print(f"Available: {', '.join(get_available_variables(log_file))}")
                sys.exit(1)

            print(f"Parsing residuals from: {log_file}")
            print(f"Variables: {', '.join(variables)}")
            data = parse_residuals_from_log(log_file, variables, use_initial)

        if not data["time"]:
            print("Error: No residual data found", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(data['time'])} time steps")
        plot_residuals(
            data, variables, title=case_path.name, save_path=args.save, use_initial=use_initial
        )

    # Handle postProcessing function objects
    else:
        func_name = args.source
        post_dir = case_path / "postProcessing" / func_name

        if not post_dir.exists():
            print(f"Error: PostProcessing directory not found: {post_dir}", file=sys.stderr)
            print(f"Available: {', '.join(get_postprocessing_dirs(case_path))}")
            sys.exit(1)

        func_type = detect_postprocessing_type(case_path, func_name)
        available_files = get_postprocessing_files(case_path, func_name)

        print(f"Function object: {func_name} ({func_type})")
        print(f"Available files: {', '.join(available_files)}")

        if func_type == "forces":
            # Determine which file to plot
            if args.series and args.series[0] in ["force", "moment"]:
                data_type = args.series[0]
            elif "force" in available_files:
                data_type = "force"
            elif "moment" in available_files:
                data_type = "moment"
            else:
                data_type = available_files[0] if available_files else "force"

            print(f"Plotting: {data_type}.dat")
            data = parse_forces(case_path, func_name, data_type)

            if not data["time"]:
                print("Error: No data found", file=sys.stderr)
                sys.exit(1)

            print(f"Found {len(data['time'])} time steps")

            label = "Force" if data_type == "force" else "Moment"
            unit = "N" if data_type == "force" else "N·m"

            plot_forces(
                data,
                components=args.components,
                force_type=args.type,
                title=f"{case_path.name} - {func_name}",
                save_path=args.save,
                data_label=label,
                unit=unit,
            )

        elif func_type == "fieldMinMax":
            data, fields = parse_field_minmax(case_path, func_name)

            if not data["time"]:
                print("Error: No data found", file=sys.stderr)
                sys.exit(1)

            print(f"Found {len(data['time'])} time steps")
            print(f"Fields: {', '.join(fields)}")

            plot_field_minmax(
                data, fields, title=f"{case_path.name} - {func_name}", save_path=args.save
            )

        else:
            # Generic plotting
            if args.series:
                dat_file = args.series[0]
            elif available_files:
                dat_file = available_files[0]
            else:
                print("Error: No data files found", file=sys.stderr)
                sys.exit(1)

            print(f"Plotting: {dat_file}.dat")
            data, columns = parse_generic_dat(case_path, func_name, dat_file)

            if not data.get(columns[0], []):
                print("Error: No data found", file=sys.stderr)
                sys.exit(1)

            print(f"Found {len(data[columns[0]])} data points")
            plot_generic(
                data, columns, title=f"{case_path.name} - {func_name}", save_path=args.save
            )


if __name__ == "__main__":
    main()
