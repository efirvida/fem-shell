#!/usr/bin/env python3
"""
Generate solid/solid_config.yaml from case_config.json

This script reads the case configuration and generates the YAML configuration
for the solid solver, ensuring all parameters stay synchronized.
"""

import json
import yaml
from pathlib import Path


def load_config():
    """Load configuration from case_config.json"""
    config_path = Path(__file__).parent.parent / "case_config.json"
    with config_path.open() as f:
        return json.load(f)


def generate_solid_config(config):
    """Generate solid_config.yaml content from case_config.json"""
    
    # Extract relevant parameters
    sim = config["simulation"]
    flaps = config["flaps"]
    solid = config["solid"]
    
    solid_config = {
        "# FSI Simulation Configuration - Multi-Flap Case": None,
        "# ===============================================": None,
        "# This YAML configuration is auto-generated from case_config.json": None,
        "# Time parameters (total_time, time_step) and watchpoints are automatically": None,
        "# read from the preCICE configuration file (precice-config.xml).": None,
        "": None,
        
        "#============================================================================": None,
        "# MESH CONFIGURATION": None,
        "#============================================================================": None,
        "mesh": {
            "# Generate multi-flap mesh programmatically": None,
            "source": "generator",
            "generator": {
                "type": "MultiFlapMesh",
                "params": {
                    "n_flaps": flaps["number"],
                    "flap_width": flaps["width"],
                    "flap_height": flaps["height"],
                    "x_spacing": flaps["x_spacing"],
                    "base_height": solid["mesh"]["base_height"],
                    "nx_flap": solid["mesh"]["nx"],
                    "ny_flap": solid["mesh"]["ny"],
                    "nx_base_segment": solid["mesh"]["nx_base_segment"],
                    "ny_base": solid["mesh"]["ny_base"],
                    "quadratic": solid["mesh"]["quadratic"]
                }
            },
            "# Output mesh file (format inferred from extension)": None,
            "# Supported: .vtk, .vtu, .msh, .inp (CalculiX), .h5/.hdf5, .obj, .stl": None,
            "output_file": "mesh_multiflap.vtk"
        },
        
        " ": None,
        "#============================================================================ ": None,
        "# MATERIAL DEFINITION": None,
        "#============================================================================ ": None,
        "material": {
            "type": "isotropic",
            "name": "Flap_Material",
            "E": float(solid["material"]["E"]),
            "nu": solid["material"]["nu"],
            "rho": float(solid["material"]["rho"])
        },
        
        "  ": None,
        "#============================================================================  ": None,
        "# ELEMENT CONFIGURATION": None,
        "#============================================================================  ": None,
        "elements": {
            "family": "PLANE"
        },
        
        "   ": None,
        "#============================================================================   ": None,
        "# SOLVER CONFIGURATION": None,
        "# Note: total_time and time_step are read automatically from precice-config.xml": None,
        "#       (<max-time> and <time-window-size> respectively)": None,
        "#============================================================================   ": None,
        "solver": {
            "type": "LinearDynamicFSI",
            f"# total_time: auto from <max-time value=\"{sim['total_time']}\"/>": None,
            f"# time_step: auto from <time-window-size value=\"{sim['time_step']}\"/>": None,
            " ": None,
            "newmark": {
                "beta": 0.25,
                "gamma": 0.5
            }
        },
        
        "    ": None,
        "#============================================================================    ": None,
        "# BOUNDARY CONDITIONS": None,
        "#============================================================================    ": None,
        "boundary_conditions": {
            "# Fixed at base bottom (the \"bottom\" node set is an alias for \"base_bottom\")": None,
            "dirichlet": [
                {
                    "nodeset": "bottom",
                    "value": 0.0
                }
            ],
            " ": None,
            "body_forces": []
        },
        
        "     ": None,
        "#============================================================================     ": None,
        "# FSI COUPLING (preCICE) - Inline configuration": None,
        "#============================================================================     ": None,
        "coupling": {
            "# preCICE participant name": None,
            "participant": "Solid",
            "# Path to preCICE configuration file (relative to this config file)": None,
            "config_file": "../precice-config.xml",
            " ": None,
            "# Interface configuration": None,
            "interface": {
                "coupling_mesh": "Solid-Mesh",
                "write_data": "Displacement",
                "read_data": "Force"
            },
            "  ": None,
            "# All flap surfaces exposed to fluid": None,
            "boundaries": [
                "flaps_left",
                "flaps_top", 
                "flaps_right"
            ]
        },
        
        "      ": None,
        "#============================================================================      ": None,
        "# OUTPUT & CHECKPOINT CONFIGURATION": None,
        "#============================================================================      ": None,
        "output": {
            "folder": solid["output_folder"],
            "write_interval": sim["write_interval"],
            "start_from": sim["start_from"],
            "start_time": 0.0,
            "save_deformed_mesh": solid["save_deformed_mesh"],
            "deformed_mesh_scale": solid["deformed_mesh_scale"],
            "write_initial_state": True
        },
        
        "       ": None,
        "#============================================================================       ": None,
        "# POST-PROCESSING": None,
        "# Note: watchpoint_files are auto-detected from <watch-point> in precice-config.xml": None,
        "#============================================================================       ": None,
        "postprocess": {
            "# watchpoint_files: auto-detected from preCICE config": None,
            f"# Will find: precice-Solid-watchpoint-Flap1-Tip.log, etc.": None,
            "plots": {
                "displacement": "desplazamientos.png",
                "force": "fuerzas.png"
            }
        }
    }
    
    return solid_config


def format_yaml_with_comments(data):
    """Format YAML preserving comments and structure"""
    lines = []
    lines.append("# FSI Simulation Configuration - Multi-Flap Case")
    lines.append("# ===============================================")
    lines.append("# This YAML configuration is auto-generated from case_config.json")
    lines.append("# Time parameters (total_time, time_step) and watchpoints are automatically")
    lines.append("# read from the preCICE configuration file (precice-config.xml).")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# MESH CONFIGURATION")
    lines.append("#============================================================================")
    lines.append("mesh:")
    lines.append("  # Generate multi-flap mesh programmatically")
    lines.append("  source: \"generator\"")
    lines.append("  generator:")
    lines.append("    type: \"MultiFlapMesh\"")
    lines.append("    params:")
    mesh_params = data["mesh"]["generator"]["params"]
    lines.append(f"      n_flaps: {mesh_params['n_flaps']}           # Number of flaps")
    lines.append(f"      flap_width: {mesh_params['flap_width']}      # Width of each flap [m]")
    lines.append(f"      flap_height: {mesh_params['flap_height']}     # Height of each flap [m]")
    lines.append(f"      x_spacing: {mesh_params['x_spacing']}       # Spacing between flaps [m]")
    lines.append(f"      base_height: {mesh_params['base_height']}    # Base strip height [m]")
    lines.append(f"      nx_flap: {mesh_params['nx_flap']}           # Elements per flap width")
    lines.append(f"      ny_flap: {mesh_params['ny_flap']}          # Elements per flap height")
    lines.append(f"      nx_base_segment: {mesh_params['nx_base_segment']}  # Elements in base segments")
    lines.append(f"      ny_base: {mesh_params['ny_base']}           # Elements in base height")
    lines.append(f"      quadratic: {str(mesh_params['quadratic']).lower()}      # Use quadratic elements")
    lines.append("")
    lines.append("  # Output mesh file (format inferred from extension)")
    lines.append("  # Supported: .vtk, .vtu, .msh, .inp (CalculiX), .h5/.hdf5, .obj, .stl")
    lines.append("  output_file: \"mesh_multiflap.vtk\"")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# MATERIAL DEFINITION") 
    lines.append("#============================================================================")
    lines.append("material:")
    lines.append("  type: \"isotropic\"")
    lines.append("  name: \"Flap_Material\"")
    mat = data["material"]
    lines.append(f"  E: {mat['E']:.1e}       # Young's modulus [Pa]")
    lines.append(f"  nu: {mat['nu']}        # Poisson's ratio [-]")
    lines.append(f"  rho: {mat['rho']:.1f}    # Density [kg/m³]")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# ELEMENT CONFIGURATION")
    lines.append("#============================================================================")
    lines.append("elements:")
    lines.append("  family: \"PLANE\"  # 2D plane stress/strain elements")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# SOLVER CONFIGURATION")
    lines.append("# Note: total_time and time_step are read automatically from precice-config.xml")
    lines.append("#       (<max-time> and <time-window-size> respectively)")
    lines.append("#============================================================================")
    lines.append("solver:")
    lines.append("  type: \"LinearDynamicFSI\"")
    solver = data["solver"]
    total_time = solver.get("total_time_comment", "120")
    time_step = solver.get("time_step_comment", "0.001")
    lines.append(f"  # total_time: auto from <max-time value=\"{total_time}\"/>")
    lines.append(f"  # time_step: auto from <time-window-size value=\"{time_step}\"/>")
    lines.append("  ")
    lines.append("  newmark:")
    lines.append("    beta: 0.25")
    lines.append("    gamma: 0.5")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# BOUNDARY CONDITIONS")
    lines.append("#============================================================================")
    lines.append("boundary_conditions:")
    lines.append("  # Fixed at base bottom (the \"bottom\" node set is an alias for \"base_bottom\")")
    lines.append("  dirichlet:")
    lines.append("    - nodeset: \"bottom\"")
    lines.append("      value: 0.0")
    lines.append("")
    lines.append("  body_forces: []")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# FSI COUPLING (preCICE) - Inline configuration")
    lines.append("#============================================================================")
    lines.append("coupling:")
    lines.append("  # preCICE participant name")
    lines.append("  participant: \"Solid\"")
    lines.append("  # Path to preCICE configuration file (relative to this config file)")
    lines.append("  config_file: \"../precice-config.xml\"")
    lines.append("  ")
    lines.append("  # Interface configuration")
    lines.append("  interface:")
    lines.append("    coupling_mesh: \"Solid-Mesh\"")
    lines.append("    write_data: \"Displacement\"")
    lines.append("    read_data: \"Force\"")
    lines.append("  ")
    lines.append("  # All flap surfaces exposed to fluid")
    lines.append("  boundaries:")
    lines.append("    - \"flaps_left\"")
    lines.append("    - \"flaps_top\"")
    lines.append("    - \"flaps_right\"")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# OUTPUT & CHECKPOINT CONFIGURATION")
    lines.append("#============================================================================")
    lines.append("output:")
    output = data["output"]
    lines.append(f"  folder: \"{output['folder']}\"")
    lines.append(f"  write_interval: {output['write_interval']}    # Checkpoint every {output['write_interval']}s")
    lines.append(f"  start_from: \"{output['start_from']}\"  # Resume from last checkpoint if available")
    lines.append(f"  start_time: {output['start_time']}")
    lines.append(f"  save_deformed_mesh: {str(output['save_deformed_mesh']).lower()}")
    lines.append(f"  deformed_mesh_scale: {output['deformed_mesh_scale']}")
    lines.append(f"  write_initial_state: {str(output['write_initial_state']).lower()}")
    lines.append("")
    
    lines.append("#============================================================================")
    lines.append("# POST-PROCESSING")
    lines.append("# Note: watchpoint_files are auto-detected from <watch-point> in precice-config.xml")
    lines.append("#============================================================================")
    lines.append("postprocess:")
    lines.append("  # watchpoint_files: auto-detected from preCICE config")
    lines.append("  # Will find: precice-Solid-watchpoint-Flap1-Tip.log, etc.")
    lines.append("  plots:")
    lines.append("    displacement: \"desplazamientos.png\"")
    lines.append("    force: \"fuerzas.png\"")
    
    return "\n".join(lines)


def main():
    print("Generating solid_config.yaml from case_config.json...")
    
    # Load configuration
    config = load_config()
    
    # Generate solid config structure
    solid_config_data = {
        "mesh": {
            "generator": {
                "params": {
                    "n_flaps": config["flaps"]["number"],
                    "flap_width": config["flaps"]["width"],
                    "flap_height": config["flaps"]["height"],
                    "x_spacing": config["flaps"]["x_spacing"],
                    "base_height": config["solid"]["mesh"]["base_height"],
                    "nx_flap": config["solid"]["mesh"]["nx"],
                    "ny_flap": config["solid"]["mesh"]["ny"],
                    "nx_base_segment": config["solid"]["mesh"]["nx_base_segment"],
                    "ny_base": config["solid"]["mesh"]["ny_base"],
                    "quadratic": config["solid"]["mesh"]["quadratic"]
                }
            }
        },
        "material": {
            "E": float(config["solid"]["material"]["E"]),
            "nu": config["solid"]["material"]["nu"],
            "rho": float(config["solid"]["material"]["rho"])
        },
        "solver": {
            "total_time_comment": config["simulation"]["total_time"],
            "time_step_comment": config["simulation"]["time_step"]
        },
        "output": {
            "folder": config["solid"]["output_folder"],
            "write_interval": config["simulation"]["write_interval"],
            "start_from": config["simulation"]["start_from"],
            "start_time": 0.0,
            "save_deformed_mesh": config["solid"]["save_deformed_mesh"],
            "deformed_mesh_scale": config["solid"]["deformed_mesh_scale"],
            "write_initial_state": True
        }
    }
    
    # Format as YAML with comments
    yaml_content = format_yaml_with_comments(solid_config_data)
    
    # Write to output file
    output_path = Path(__file__).parent.parent / "solid/solid_config.yaml"
    output_path.parent.mkdir(exist_ok=True)
    
    with output_path.open("w") as f:
        f.write(yaml_content)
    
    print(f"Generated: {output_path}")
    print("Configuration synchronized with case_config.json:")
    print(f"  Flaps: {config['flaps']['number']} x ({config['flaps']['width']}m x {config['flaps']['height']}m)")
    print(f"  Material: E={config['solid']['material']['E']}, ν={config['solid']['material']['nu']}, ρ={config['solid']['material']['rho']}")
    print(f"  Mesh: {config['solid']['mesh']['nx']}x{config['solid']['mesh']['ny']} elements")
    print(f"  Simulation: {config['simulation']['total_time']}s, dt={config['simulation']['time_step']}s")


if __name__ == "__main__":
    main()