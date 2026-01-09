#!/usr/bin/env python3
"""
Generate OpenFOAM preciceDict from case_config.json

This script reads the case configuration and generates the preciceDict
file with explicit patch names for FSI coupling.
"""

import json
from pathlib import Path


def load_config():
    """Load configuration from case_config.json"""
    # Navigate from fluid/system to case root
    config_path = Path(__file__).parent.parent.parent / "case_config.json"
    with config_path.open() as f:
        return json.load(f)


def generate_precice_dict(config):
    """Generate preciceDict content with explicit patch names"""

    n_flaps = config["flaps"]["number"]

    # Generate explicit list of patch names
    patches = [f"flap{i + 1}" for i in range(n_flaps)]
    patches_str = " ".join(patches)

    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      preciceDict;
}}

preciceConfig "../precice-config.xml";

participant Fluid;

modules (FSI);

interfaces
{{
  Interface1
  {{
    mesh              Fluid-Mesh;
    patches           ({patches_str});
    locations         faceCenters;
    
    readData
    (
      Displacement
    );
    
    writeData
    (
      Force
    );
  }};
}};

FSI
{{
  rho rho [1 -3 0 0 0 0 0] 1;
}}
"""
    return content


def main():
    print("Generating preciceDict from case_config.json...")

    config = load_config()
    content = generate_precice_dict(config)

    output_path = Path(__file__).parent / "preciceDict"
    with output_path.open("w") as f:
        f.write(content)

    n_flaps = config["flaps"]["number"]
    patches = [f"flap{i + 1}" for i in range(n_flaps)]

    print(f"Generated {output_path}")
    print(f"  Number of flaps: {n_flaps}")
    print(f"  Patches: {', '.join(patches)}")


if __name__ == "__main__":
    main()
