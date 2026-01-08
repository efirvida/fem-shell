#!/usr/bin/env python3
"""
Utility script to estimate FSI force clipping parameters from CFD data.

Usage:
    python estimate_force_cap.py --force-file forces.dat [--config-file config.yaml] [--safety-factor 3.0]

This script reads force data from a CFD simulation and estimates appropriate
values for `force_max_cap` and `force_ramp_time` parameters in the FSI coupling.

It can output recommendations or directly update the configuration YAML file.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_force_file(force_file: Path) -> np.ndarray:
    """
    Parse force data from text file.

    Expected format: One force vector per line (Fx, Fy, Fz)
    or flat array (n_nodes × 3).
    """
    try:
        forces = np.loadtxt(force_file, ndmin=2)
        if forces.shape[1] < 3:
            logger.warning("Force file has %d columns; expected ≥3 (Fx, Fy, Fz)", forces.shape[1])
        return forces
    except Exception as e:
        logger.error(f"Failed to parse force file: {e}")
        sys.exit(1)


def compute_force_statistics(forces: np.ndarray) -> dict:
    """
    Compute statistics on force magnitudes.
    """
    # Compute per-node magnitude
    if forces.ndim == 1:
        # Single force vector
        mag = np.linalg.norm(forces[:3])
        return {
            "n_samples": 1,
            "n_components": forces.shape[0],
            "mean_mag": float(mag),
            "max_mag": float(mag),
            "min_mag": float(mag),
            "std_mag": 0.0,
            "percentile_75": float(mag),
            "percentile_90": float(mag),
            "percentile_95": float(mag),
            "percentile_99": float(mag),
        }
    else:
        # Multiple force vectors (n_samples, 3)
        mags = np.linalg.norm(forces, axis=1)
        return {
            "n_samples": forces.shape[0],
            "n_components": forces.shape[1],
            "mean_mag": float(np.mean(mags)),
            "max_mag": float(np.max(mags)),
            "min_mag": float(np.min(mags)),
            "std_mag": float(np.std(mags)),
            "percentile_75": float(np.percentile(mags, 75)),
            "percentile_90": float(np.percentile(mags, 90)),
            "percentile_95": float(np.percentile(mags, 95)),
            "percentile_99": float(np.percentile(mags, 99)),
        }


def recommend_force_parameters(stats: dict, safety_factor: float = 3.0) -> dict:
    """
    Recommend force_max_cap and force_ramp_time based on statistics.

    Strategy for force_max_cap:
    - Conservative: max_force × safety_factor
    - Moderate: 99th percentile × 1.5 (recommended)
    - Aggressive: 95th percentile × 1.5

    Strategy for force_ramp_time (based on variability σ/μ):
    - High variability (σ/μ > 2.0): 0.05 s
    - Medium variability (σ/μ > 1.0): 0.03 s
    - Low variability (σ/μ > 0.5): 0.02 s
    - Very low variability (σ/μ ≤ 0.5): 0.01 s
    """
    mean_f = stats["mean_mag"]
    std_f = stats["std_mag"]
    p99 = stats["percentile_99"]

    # Calculate variability
    variability = std_f / mean_f if mean_f > 0 else 0.0

    # Force cap recommendations
    conservative = stats["max_mag"] * safety_factor
    moderate = p99 * 1.5
    sigma_based = mean_f + 3 * std_f

    # Ramp time based on variability
    if variability > 2.0:
        ramp_time = 0.05
    elif variability > 1.0:
        ramp_time = 0.03
    elif variability > 0.5:
        ramp_time = 0.02
    else:
        ramp_time = 0.01

    return {
        "force_max_cap_options": {
            "conservative": conservative,
            "moderate": moderate,
            "sigma_based": sigma_based,
            "max_observed": stats["max_mag"],
        },
        "force_max_cap_recommended": moderate,
        "force_ramp_time": ramp_time,
        "variability": variability,
    }


def update_config_file(config_file: Path, force_max_cap: float, force_ramp_time: float) -> None:
    """
    Update the fsi_config.yaml file with new parameters.
    """
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)

    # Read current config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        logger.warning(f"Config file is empty or malformed: {config_file}")
        config = {}

    # Update solver section
    if "solver" not in config:
        config["solver"] = {}

    config["solver"]["force_max_cap"] = force_max_cap
    config["solver"]["force_ramp_time"] = force_ramp_time

    # Write back
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated {config_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Estimate FSI force clipping parameters from CFD data"
    )
    parser.add_argument(
        "--force-file",
        required=True,
        type=Path,
        help="Input file with force data (text, one force vector per line)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Config YAML file to update (optional)",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        help="Output YAML config file for recommendations only (legacy, use --config-file)",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=3.0,
        help="Safety factor for force cap (default: 3.0)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update config file in place (requires --config-file)",
    )
    args = parser.parse_args()

    # Validate input file
    if not args.force_file.exists():
        logger.error(f"Force file not found: {args.force_file}")
        sys.exit(1)

    logger.info(f"Reading force data from: {args.force_file}")
    forces = parse_force_file(args.force_file)

    logger.info(f"Parsed {forces.shape[0]} force samples with {forces.shape[1]} components")

    # Compute statistics
    logger.info("Computing force statistics...")
    stats = compute_force_statistics(forces)

    # Display statistics
    print("\n" + "=" * 70)
    print("FORCE STATISTICS")
    print("=" * 70)
    print(f"Number of samples:  {stats['n_samples']}")
    print(f"Components/sample:  {stats['n_components']}")
    print("\nMagnitude Statistics (N):")
    print(f"  Mean:             {stats['mean_mag']:.2e}")
    print(f"  Min:              {stats['min_mag']:.2e}")
    print(f"  Max:              {stats['max_mag']:.2e}")
    print(f"  Std Dev:          {stats['std_mag']:.2e}")
    print(f"  75th percentile:  {stats['percentile_75']:.2e}")
    print(f"  90th percentile:  {stats['percentile_90']:.2e}")
    print(f"  95th percentile:  {stats['percentile_95']:.2e}")
    print(f"  99th percentile:  {stats['percentile_99']:.2e}")

    # Recommend force cap
    logger.info("Computing force cap recommendations...")
    recommendations = recommend_force_parameters(stats, safety_factor=args.safety_factor)

    print("\n" + "=" * 70)
    print("FORCE CAP RECOMMENDATIONS (N)")
    print("=" * 70)
    print(
        f"Conservative (max × {args.safety_factor}):     {recommendations['force_max_cap_options']['conservative']:.2e}"
    )
    print(
        f"Moderate (99%ile × 1.5):           {recommendations['force_max_cap_options']['moderate']:.2e}"
    )
    print(
        f"Statistical (μ + 3σ):              {recommendations['force_max_cap_options']['sigma_based']:.2e}"
    )
    print(
        f"Max observed (no safety factor):   {recommendations['force_max_cap_options']['max_observed']:.2e}"
    )

    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print(f"force_max_cap:    {recommendations['force_max_cap_recommended']:.2e} N")
    print(f"force_ramp_time:  {recommendations['force_ramp_time']:.3f} s")
    print(f"Variability σ/μ:  {recommendations['variability']:.2f}")

    # Update config file if requested
    if args.update and args.config_file:
        print(f"\nUpdating {args.config_file}...")
        update_config_file(
            args.config_file,
            recommendations["force_max_cap_recommended"],
            recommendations["force_ramp_time"],
        )
        print("✓ Configuration updated successfully")

    # Legacy: output config for recommendations
    if args.output_config:
        config = {
            "force_clipping": {
                "force_max_cap": recommendations["force_max_cap_recommended"],
                "force_ramp_time": recommendations["force_ramp_time"],
                "statistics": stats,
                "recommendations": recommendations["force_max_cap_options"],
            }
        }
        logger.info(f"Writing config to: {args.output_config}")
        with open(args.output_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nConfig saved to: {args.output_config}")

    print("\n" + "=" * 70)
    print("TUNING NOTES")
    print("=" * 70)
    print("1. Copy the recommended values into fsi_config.yaml")
    print("2. Monitor clipping diagnostics in the first 20 FSI coupling steps")
    print("3. Adjust if needed:")
    print("   - Too many clips (>5% of nodes): increase cap by 20%")
    print("   - Almost no clips after step 30: decrease cap by 20%")
    print("4. Optimal: ~1-2% nodes clipped in first 10 steps, then stabilize")
    print()


if __name__ == "__main__":
    main()
