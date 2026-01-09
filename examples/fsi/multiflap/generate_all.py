#!/usr/bin/env python3
"""
Generate all configuration files from case_config.json

This script runs all generators to create:
- fluid/system/blockMeshDict
- fluid/system/controlDict
- precice-config.xml

Run this after modifying case_config.json to update all dependent files.
"""

import subprocess
import sys
from pathlib import Path


def run_generator(script_path, name):
    """Run a generator script and report result"""
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=script_path.parent, capture_output=False
    )

    if result.returncode != 0:
        print(f"ERROR: {name} failed with code {result.returncode}")
        return False
    return True


def main():
    case_root = Path(__file__).parent

    print("=" * 60)
    print("Generating all configuration files from case_config.json")
    print("=" * 60)

    generators = [
        (case_root / "fluid/system/generate_blockMeshDict.py", "blockMeshDict generator"),
        (case_root / "fluid/system/generate_controlDict.py", "controlDict generator"),
        (case_root / "fluid/system/generate_preciceDict.py", "preciceDict generator"),
        (case_root / "generate_precice_config.py", "precice-config.xml generator"),
    ]

    success = True
    for script_path, name in generators:
        if script_path.exists():
            if not run_generator(script_path, name):
                success = False
        else:
            print(f"\nWARNING: {name} not found at {script_path}")

    print("\n" + "=" * 60)
    if success:
        print("All generators completed successfully!")
    else:
        print("Some generators failed. Check output above.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
