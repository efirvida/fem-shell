#!/usr/bin/env python3
"""
Generate precice-config.xml from case_config.json

This script reads the case configuration and generates the preCICE
configuration file with watch-points at the top center of each flap.
"""

import json
from pathlib import Path


def load_config():
    """Load configuration from case_config.json"""
    config_path = Path(__file__).parent / "case_config.json"
    with config_path.open() as f:
        return json.load(f)


def calculate_flap_positions(config):
    """Calculate X positions of all flaps (centered around x=0)"""
    n_flaps = config["flaps"]["number"]
    flap_w = config["flaps"]["width"]
    x_spacing = config["flaps"]["x_spacing"]

    # Total width of flap array
    total_width = n_flaps * flap_w + (n_flaps - 1) * x_spacing

    # Starting X position (leftmost flap)
    x_start = -total_width / 2

    positions = []
    for i in range(n_flaps):
        x_left = x_start + i * (flap_w + x_spacing)
        x_center = x_left + flap_w / 2
        # Round to avoid floating point errors
        x_center = round(x_center, 10)
        if abs(x_center) < 1e-10:
            x_center = 0.0
        positions.append({
            "index": i + 1,
            "x_center": x_center,
        })

    return positions


def generate_precice_config(config):
    """Generate precice-config.xml content"""

    flap_positions = calculate_flap_positions(config)
    flap_height = config["flaps"]["height"]
    sim_cfg = config["simulation"]
    total_time = sim_cfg["total_time"]
    time_step = sim_cfg["time_step"]

    # Generate watch-point lines
    watch_points = []
    for fp in flap_positions:
        watch_points.append(
            f'    <watch-point mesh="Solid-Mesh" name="Flap{fp["index"]}-Tip" '
            f'coordinate="{fp["x_center"]};{flap_height}" />'
        )
    watch_points_str = "\n".join(watch_points)

    xml_content = f'''<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  <profiling mode="off" />

  <log>
      <sink type="stream" output="stdout"  filter= "(%Severity% > debug) or (%Severity% >= trace and %Module% contains ParticipantImpl)"  enabled="true" />   
  </log> 

  <data:vector name="Force" />
  <data:vector name="Displacement" />

  <mesh name="Fluid-Mesh" dimensions="2">
    <use-data name="Force" />
    <use-data name="Displacement" />
  </mesh>

  <mesh name="Solid-Mesh" dimensions="2">
    <use-data name="Displacement" />
    <use-data name="Force" />
  </mesh>

  <participant name="Fluid">
    <provide-mesh name="Fluid-Mesh" />
    <receive-mesh name="Solid-Mesh" from="Solid" />
    <write-data name="Force" mesh="Fluid-Mesh" />
    <read-data name="Displacement" mesh="Fluid-Mesh" />
    <mapping:rbf direction="write" from="Fluid-Mesh" to="Solid-Mesh" constraint="conservative">
      <basis-function:compact-polynomial-c6 support-radius="0.25" />
    </mapping:rbf>
    <mapping:rbf direction="read" from="Solid-Mesh" to="Fluid-Mesh" constraint="consistent">
      <basis-function:compact-polynomial-c6 support-radius="0.25" />
    </mapping:rbf>
  </participant>

  <participant name="Solid">
    <provide-mesh name="Solid-Mesh" />
    <write-data name="Displacement" mesh="Solid-Mesh" />
    <read-data name="Force" mesh="Solid-Mesh" />
    <!-- Watch-points at top center of each flap -->
{watch_points_str}
  </participant>

  <m2n:sockets acceptor="Fluid" connector="Solid" exchange-directory=".." />

  <coupling-scheme:parallel-implicit>
    <time-window-size value="{time_step}" />
    <max-time value="{total_time}" />
    <participants first="Fluid" second="Solid" />
    <exchange data="Force" mesh="Solid-Mesh" from="Fluid" to="Solid" />
    <exchange data="Displacement" mesh="Solid-Mesh" from="Solid" to="Fluid" />
    <max-iterations value="50" />
    <relative-convergence-measure limit="5e-3" data="Displacement" mesh="Solid-Mesh" />
    <relative-convergence-measure limit="5e-3" data="Force" mesh="Solid-Mesh" />
    <acceleration:IQN-ILS>
      <data name="Displacement" mesh="Solid-Mesh" />
      <data name="Force" mesh="Solid-Mesh" />
      <preconditioner type="residual-sum" />
      <filter type="QR2" limit="1e-2" />
      <initial-relaxation value="0.5" />
      <max-used-iterations value="100" />
      <time-windows-reused value="15" />
    </acceleration:IQN-ILS>
  </coupling-scheme:parallel-implicit>
</precice-configuration>
'''
    return xml_content


def main():
    print("Generating precice-config.xml from case_config.json...")

    config = load_config()
    xml_content = generate_precice_config(config)

    output_path = Path(__file__).parent / "precice-config.xml"
    with output_path.open("w") as f:
        f.write(xml_content)

    # Print summary
    flap_positions = calculate_flap_positions(config)
    flap_height = config["flaps"]["height"]

    print(f"\nGenerated {len(flap_positions)} watch-points:")
    for fp in flap_positions:
        print(f"  Flap{fp['index']}-Tip: ({fp['x_center']}, {flap_height})")

    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
