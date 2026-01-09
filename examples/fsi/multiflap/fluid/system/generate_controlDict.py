#!/usr/bin/env python3
"""
Generate OpenFOAM controlDict from case_config.json

This script reads the case configuration and generates the controlDict
file with proper time stepping and simulation duration.
"""

import json
from pathlib import Path


def load_config():
    """Load configuration from case_config.json"""
    # Navigate from fluid/system to case root
    config_path = Path(__file__).parent.parent.parent / "case_config.json"
    with config_path.open() as f:
        return json.load(f)


def calculate_flap_patches(config):
    """Generate the list of flap patch names"""
    n_flaps = config["flaps"]["number"]
    return [f"flap{i + 1}" for i in range(n_flaps)]


def generate_control_dict(config):
    """Generate controlDict content"""

    sim_cfg = config["simulation"]
    total_time = sim_cfg["total_time"]
    time_step = sim_cfg["time_step"]
    write_interval = sim_cfg.get("write_interval", 0.05)
    start_from = sim_cfg.get("start_from", "startTime")

    # Map start_from to OpenFOAM format
    start_from_openfoam = start_from  # "startTime" or "latestTime"
    purge_write = 0

    flap_patches = calculate_flap_patches(config)
    patches_str = " ".join(flap_patches)

    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     pimpleFoam;

startFrom       {start_from_openfoam};

startTime       0;

stopAt          endTime;

endTime         {total_time};

deltaT          {time_step};

writeControl    adjustableRunTime;

writeInterval   {write_interval};

purgeWrite      {purge_write};

writeFormat     binary;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

adjustTimeStep  no;

maxCo           0.9;  // Max Courant number

maxDeltaT       0.01;  // Maximum time step

functions
{{
    preCICE_Adapter
    {{
        type preciceAdapterFunctionObject;
        libs ("libpreciceAdapterFunctionObject.so");
        errors strict;
    }}
}}
"""
    return content


def main():
    print("Generating controlDict from case_config.json...")

    config = load_config()
    content = generate_control_dict(config)

    output_path = Path(__file__).parent / "controlDict"
    with output_path.open("w") as f:
        f.write(content)

    sim_cfg = config["simulation"]
    print("\nSimulation parameters:")
    print(f"  Total time: {sim_cfg['total_time']} s")
    print(f"  Time step: {sim_cfg['time_step']} s")
    print(f"  Write interval: {sim_cfg.get('write_interval', 0.05)} s")
    print(f"  Start from: {sim_cfg.get('start_from', 'startTime')}")
    print(f"  Flap patches: {calculate_flap_patches(config)}")

    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
