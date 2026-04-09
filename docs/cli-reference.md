# fem-shell-fsi CLI Reference

Command-line interface for running FEM shell/solid FSI simulations coupled with
OpenFOAM via preCICE. All simulation parameters are defined in a single YAML file.

## Installation

```bash
cd fem-shell
pip install -e .
```

This registers the `fem-shell-fsi` command globally.

---

## Quick Start

```bash
# 1. Generate a template configuration
fem-shell-fsi --template > simulation.yaml

# 2. Edit simulation.yaml with your parameters

# 3. Validate the configuration
fem-shell-fsi simulation.yaml --validate

# 4. Preview what will run (no execution)
fem-shell-fsi simulation.yaml --preview

# 5. Run the simulation
fem-shell-fsi simulation.yaml
```

---

## Command Reference

### Synopsis

```
fem-shell-fsi [config.yaml] [OPTIONS]
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `config` | | Path to YAML configuration file (positional) |
| `--template` | `-t` | Print a complete template YAML to stdout |
| `--generator NAME` | `-g NAME` | Include a mesh generator template (`SquareShapeMesh`, `BoxSurfaceMesh`, `BoxVolumeMesh`, `MultiFlapMesh`, `RotorMesh`) |
| `--list-generators` | | List available mesh generators with node set names |
| `--validate` | | Validate config syntax and semantics without running |
| `--preview` | `-p` | Print the parsed configuration tree and exit |
| `--view` | | Open interactive 3D mesh viewer |
| `--workdir DIR` | `-w DIR` | Set working directory (default: current directory) |
| `--verbose` | `-v` | Enable DEBUG-level logging |

### Usage Examples

```bash
# Print template with BoxVolumeMesh generator example
fem-shell-fsi --template --generator BoxVolumeMesh > simulation.yaml

# List generators and their default node sets
fem-shell-fsi --list-generators

# Validate config (exits 0 = valid, 1 = errors)
fem-shell-fsi simulation.yaml --validate

# Preview parsed config without running
fem-shell-fsi simulation.yaml --preview

# Run from a different working directory
fem-shell-fsi simulation.yaml --workdir /path/to/case/solid

# Verbose logging (DEBUG level)
fem-shell-fsi simulation.yaml -v

# Visualize mesh interactively
fem-shell-fsi simulation.yaml --view
```

---

## Configuration File Reference

The YAML configuration has 8 top-level sections. Required sections are marked
with **(R)**, optional with **(O)**.

```yaml
mesh:                  # (R) Mesh definition
material:              # (R) Material properties
elements:              # (R) Element type
solver:                # (R) Solver and time integration
boundary_conditions:   # (R) Dirichlet BCs
coupling:              # (O) preCICE FSI coupling — required for FSI solvers
output:                # (O) Checkpointing and restart
postprocess:           # (O) Automatic plot generation
```

---

### `mesh` — Mesh Configuration **(R)**

Loads a mesh from file or generates one programmatically.

#### From file (recommended for production)

```yaml
mesh:
  source: "file"
  file:
    path: "propeller_volumetric.h5"   # .h5, .hdf5, .pkl, .pickle
    format: "auto"                     # "auto" | "hdf5" | "pickle"
```

#### From generator (for testing / parametric studies)

```yaml
mesh:
  source: "generator"
  generator:
    type: "BoxVolumeMesh"          # See generators table below
    params:
      center: [0, 5.0, 0]
      dims: [1.0, 10.0, 1.0]
      nx: 4
      ny: 20
      nz: 4
      element_type: "hex"          # "hex" | "tet" | "wedge" | "mixed"
      quadratic: false
```

#### Common mesh options

```yaml
mesh:
  # ...source config above...

  renumber: "rcm"             # (O) Reverse Cuthill-McKee reordering — reduces solver bandwidth
  output_file: "mesh.vtk"    # (O) Write mesh to file (.vtk, .vtu, .msh, .inp, .h5, .obj, .stl)

  # (O) Create node sets from geometric criteria
  node_sets:
    - name: "coupling_surface"
      criteria_type: "all"
      on_surface: true

    - name: "fixed_base"
      criteria_type: "direction"
      params:
        point: [0, 0, 0]
        direction: [0, 1, 0]
        method: "radial"
        distance: 0.03
        mode: "inside"
```

#### Available Mesh Generators

| Generator | Description | Default Node Sets |
|-----------|-------------|-------------------|
| `SquareShapeMesh` | 2D rectangular mesh | `left`, `right`, `top`, `bottom`, `corners` |
| `BoxSurfaceMesh` | 3D box surface (shell) | `left`, `right`, `top`, `bottom`, `front`, `back` |
| `BoxVolumeMesh` | 3D solid volume (hex/tet/wedge) | `left`, `right`, `top`, `bottom`, `front`, `back` |
| `MultiFlapMesh` | Multiple flaps on a base strip | `bottom`, `flaps_left`, `flaps_right`, `flaps_top` |
| `RotorMesh` | Wind turbine rotor from blade YAML | `RootNodes_blade_N`, `allOuterShellNods_blade_N` |

#### Generator Parameters

<details>
<summary><b>SquareShapeMesh</b></summary>

```yaml
params:
  width: 0.1          # Width in X [m]
  height: 1.0         # Height in Y [m]
  nx: 4               # Elements in X
  ny: 26              # Elements in Y
  quadratic: true     # Quadratic elements (8/9-node quads)
  triangular: false   # Triangulate into triangles
```
</details>

<details>
<summary><b>BoxSurfaceMesh</b></summary>

```yaml
params:
  center: [0, 5.0, 0]       # Center [x, y, z]
  dims: [1.0, 10.0, 1.0]    # Dimensions [dx, dy, dz]
  nx: 5
  ny: 20
  nz: 5
  quadratic: false
  triangular: false
```
</details>

<details>
<summary><b>BoxVolumeMesh</b></summary>

```yaml
params:
  center: [0, 5.0, 0]
  dims: [1.0, 10.0, 1.0]
  nx: 4
  ny: 20
  nz: 4
  element_type: "hex"        # "hex" | "tet" | "wedge" | "mixed"
  quadratic: false
```
</details>

<details>
<summary><b>MultiFlapMesh</b></summary>

```yaml
params:
  n_flaps: 5
  flap_width: 0.1
  flap_height: 1.0
  x_spacing: 2.0
  base_height: 0.05
  nx_flap: 4
  ny_flap: 26
  nx_base_segment: 10
  ny_base: 2
  quadratic: true
```
</details>

<details>
<summary><b>RotorMesh</b></summary>

```yaml
params:
  yaml_file: "IEA-15-240-RWT.yaml"   # Path to blade YAML (relative to config)
  n_blades: 3
  hub_radius: null                     # null = use blade definition
  element_size: 0.5                    # Target element size [m]
  n_samples: 300                       # Airfoil discretization samples
```
</details>

#### Node Set Criteria Types

| Type | Description | Required `params` |
|------|-------------|-------------------|
| `"all"` | All nodes (use `on_surface: true` for surface only) | None |
| `"coordinate"` | Nodes matching coordinate criteria | `axis`, `value`, `tolerance` |
| `"box"` | Nodes inside a bounding box | `min`, `max` (3D points) |
| `"distance"` | Nodes within distance of a point | `point`, `distance` |
| `"direction"` | Nodes relative to a direction from a point | `point`, `direction`, `method`, `distance`, `mode` |

---

### `material` — Material Definition **(R)**

#### Isotropic material

```yaml
material:
  type: "isotropic"
  name: "Steel"
  E: 2.1e11        # Young's modulus [Pa]
  nu: 0.3          # Poisson's ratio [-]   (valid range: -1 < ν < 0.5)
  rho: 7850.0      # Density [kg/m³]
```

#### Orthotropic material

```yaml
material:
  type: "orthotropic"
  name: "CFRP"
  E: [1.4e11, 1.0e10, 1.0e10]          # [E1, E2, E3] [Pa]
  G: [5.0e9, 3.8e9, 5.0e9]             # [G12, G23, G31] [Pa]
  nu: [0.28, 0.40, 0.28]               # [ν12, ν23, ν31]
  rho: 1600.0
```

---

### `elements` — Element Configuration **(R)**

```yaml
elements:
  family: "SOLID"          # "PLANE" | "SHELL" | "SOLID"
  # thickness: 0.001       # Required only for "SHELL"
```

| Family | Description | Mesh Type |
|--------|-------------|-----------|
| `PLANE` | 2D plane stress/strain (quad/tri) | 2D |
| `SHELL` | 3D shell elements (requires `thickness`) | 3D surface |
| `SOLID` | 3D solid elements (hex/tet/wedge) | 3D volume |

---

### `solver` — Solver Configuration **(R)**

#### Solver types

| Type | Class | Description |
|------|-------|-------------|
| `LinearStatic` | `LinearStaticSolver` | Static analysis |
| `LinearDynamic` | `LinearDynamicSolver` | Transient dynamics (no coupling) |
| `LinearDynamicFSI` | `LinearDynamicFSISolver` | FSI with preCICE coupling |
| `LinearDynamicFSIRotor` | `LinearDynamicFSIRotorSolver` | FSI in rotating frame with inertial forces |

#### Basic solver configuration

```yaml
solver:
  type: "LinearDynamicFSI"
  total_time: 5.0           # Total simulation time [s]
  time_step: 0.001          # Time step [s]
  solver_type: "auto"       # "auto" | "direct" | "iterative"

  newmark:                  # (O) Newmark-β time integration
    beta: 0.25              # 0.25 = constant average acceleration (default)
    gamma: 0.5              # 0.5 = no numerical damping (default)

  damping:                  # (O) Rayleigh damping
    eta_m: 1.0e-4           # Mass proportional
    eta_k: 1.0e-4           # Stiffness proportional

  use_critical_dt: false    # (O) Auto-calculate critical ∆t from mesh
  safety_factor: 0.8        # (O) Multiplier for critical ∆t
  debug_interface: false    # (O) Verbose preCICE data exchange logging
```

> **Note:** If `total_time` and `time_step` are omitted for FSI solvers, they are
> auto-read from the preCICE XML config (`<max-time>` and `<time-window-size>`).

#### Numerical damping (Newmark HHT-α)

For simulations with high-frequency oscillations, increase numerical damping:

```yaml
  newmark:
    beta: 0.3025     # (0.5 + γ)² / 4 — unconditionally stable
    gamma: 0.6       # > 0.5 introduces numerical dissipation
```

#### Rotor configuration

Required when `type: "LinearDynamicFSIRotor"`. Controls rotating reference frame
physics and omega dynamics.

```yaml
solver:
  type: "LinearDynamicFSIRotor"
  rotor:
    # --- Rotation kinematics ---
    omega: 158                  # Initial angular velocity [rad/s]
    rotation_axis: [0, 1, 0]   # Unit vector for rotation axis
    rotation_center: [0, 0, 0] # Center of rotation [m]

    # --- Ramp-up ---
    omega_ramp_time: 1.0e-3    # Linear ramp from 0 → omega [s] (0 = instantaneous)
    force_ramp_time: 1.0e-3    # Ramp for aerodynamic forces [s]

    # --- Inertial forces ---
    include_centrifugal: true          # Centrifugal stiffening
    include_coriolis: true             # Coriolis forces
    include_euler: true                # Euler forces (angular acceleration)
    include_geometric_stiffness: true  # Geometric stiffness (stress stiffening)
    include_spin_softening: true       # Spin softening effect
    gravity: [0.0, 0.0, -9.81]        # Gravity in global frame [m/s²]

    # --- Dynamic omega (computed from aerodynamic torque) ---
    moment_of_inertia: null   # null = constant omega
                              # "auto" = compute I from mesh, then update ω each step
                              # <float> = use given I [kg·m²]
    resistive_torque: 0.0     # Generator/brake torque [N·m]
                              # DEPRECATED: use shaft_torque instead.
                              # If both are set, shaft_torque takes precedence.
    shaft_torque: 0.0         # External shaft torque [N·m]
                              # Positive = drives rotation (e.g. motor)
                              # Negative = resists rotation (e.g. generator)

    # --- preCICE omega feedback ---
    send_omega_to_precice: true  # Write ω as "AngularVelocity" on GlobalSolidMesh
                                 # so CFD can update mesh rotation speed

    # --- Displacement transform ---
    transform_displacement_to_inertial: true  # Convert rotating-frame displacements
                                              # to inertial frame for preCICE

    # --- Force limiting (stability) ---
    force_max_magnitude: null    # null = disabled. Max force per node [N]
    force_jump_factor: 1000.0    # Reject forces > factor × previous step

    # --- Aerodynamic estimates (initial force cap) ---
    fluid_density: 1.225     # [kg/m³]
    flow_velocity: 10.0      # [m/s]
    radius: null              # Rotor radius [m], null = auto from mesh
```

##### Omega Modes

The solver selects an omega provider based on the combination of `moment_of_inertia`
and `omega_ramp_time`:

| `moment_of_inertia` | `omega_ramp_time` | Behavior |
|----------------------|-------------------|----------|
| `null` | `0` | **ConstantOmega** — fixed ω throughout |
| `null` | `> 0` | **RampedOmega** — linear ramp 0 → ω, then constant |
| `"auto"` or `<float>` | `0` | **ComputedOmega** — ω updated from torque balance |
| `"auto"` or `<float>` | `> 0` | **RampedComputedOmega** — ramp, then torque-driven |

When `moment_of_inertia: "auto"`, the solver computes $I$ from the mesh mass
distribution around the rotation axis. Then at each time step:

$$\alpha = \frac{\tau_{aero} - \tau_{gen}}{I}, \quad \omega^{n+1} = \omega^n + \Delta t \cdot \alpha$$

When `send_omega_to_precice: true`, the computed $\omega$ is written as a scalar
`"AngularVelocity"` on a single-vertex mesh `"GlobalSolidMesh"` at the rotation
center. The CFD solver reads this to update overset mesh rotation. This is handled
internally — it does **not** need to be listed in `coupling.write_data`.

---

### `boundary_conditions` — Boundary Conditions **(R)**

```yaml
boundary_conditions:
  dirichlet:
    - nodeset: "base_nodes"     # Mesh node set name
      value: 0.0                # Prescribed displacement

    - nodeset: "symmetry_plane"
      value: 0.0
      components: [2]           # (O) Fix only Z-direction (DOF index 2)

  # (O) Distributed body forces
  body_forces:
    - value: [0, -9810, 0]      # Force per unit volume [N/m³]
```

The `nodeset` name must match either:
- A built-in node set from the mesh generator, or
- A node set created in `mesh.node_sets`

The `components` field selects specific DOFs:
- 2D (PLANE): `[0]` = X, `[1]` = Y
- 3D (SHELL/SOLID): `[0]` = X, `[1]` = Y, `[2]` = Z

---

### `coupling` — preCICE FSI Coupling **(O)**

Required for `LinearDynamicFSI` and `LinearDynamicFSIRotor` solvers.

```yaml
coupling:
  participant: "Solid"                   # Participant name in precice-config.xml
  config_file: "../precice-config.xml"   # Path (relative to YAML file)
  coupling_mesh: "Solid-Mesh"            # Mesh name in preCICE config

  write_data:
    - "Displacement"
  read_data:
    - "Force"

  boundaries:                            # Node sets that form the coupling surface
    - "turbine"

  # (O) Force limiting for stability
  force_max_cap: 1.0e6           # Max force per node [N]
  force_ramp_time: 0.01          # Ramp time for forces [s]
```

> **Note:** The `AngularVelocity` data exchange (rotor → CFD) is handled internally
> by the solver when `rotor.send_omega_to_precice: true`. It does not appear here.

---

### `output` — Output & Checkpointing **(O)**

```yaml
output:
  folder: "results"              # Output directory
  write_interval: 1.0e-3        # Checkpoint interval [s] (0 = disabled)
  start_from: "startTime"       # "startTime" | "latestTime" (for restart)
  start_time: 0.0               # Initial time when start_from="startTime"
  deformed_mesh_scale: 1.0       # Scale factor for displacements in output
  save_deformed_mesh: true       # Save deformed mesh at each checkpoint
  write_vtk: true                # Write VTK files
  vtk_file: "mesh.vtk"           # VTK filename
  write_initial_state: true      # Write state at t=0
```

#### Restart from Checkpoint

To resume a simulation that was interrupted:

```yaml
output:
  start_from: "latestTime"      # Auto-find latest checkpoint in folder
  folder: "results"
```

The runner searches `output.folder` for the latest checkpoint and resumes from it.

---

### `postprocess` — Post-Processing **(O)**

```yaml
postprocess:
  watchpoint_file: "precice-Solid-watchpoint-Flap-Tip.log"

  plots:
    displacement: "displacement.png"
    force: "force.png"
```

If `watchpoint_file` is omitted but preCICE config has `<watch-point>` entries,
the runner auto-discovers watchpoint log files.

---

## Complete Examples

### Example 1: FSI shell flap (2D plate in 3D flow)

```yaml
mesh:
  source: "generator"
  generator:
    type: "BoxSurfaceMesh"
    params:
      center: [0, 0.5, 0]
      dims: [0.1, 1.0, 0.001]
      nx: 4
      ny: 20
      nz: 1

material:
  type: "isotropic"
  name: "Rubber"
  E: 4.0e6
  nu: 0.3
  rho: 3000.0

elements:
  family: "SHELL"
  thickness: 0.001

solver:
  type: "LinearDynamicFSI"
  # total_time and time_step auto-read from preCICE config

  newmark:
    beta: 0.25
    gamma: 0.5

  damping:
    eta_m: 1.0e-4
    eta_k: 1.0e-4

boundary_conditions:
  dirichlet:
    - nodeset: "bottom"
      value: 0.0

coupling:
  participant: "Solid"
  config_file: "../precice-config.xml"
  coupling_mesh: "Solid-Mesh"
  write_data: ["Displacement"]
  read_data: ["Force"]
  boundaries: ["left", "right", "top", "bottom", "front", "back"]

output:
  folder: "results"
  write_interval: 0.1
  start_from: "startTime"
```

### Example 2: Rotor with fixed omega

```yaml
mesh:
  source: "file"
  file:
    path: "propeller_volumetric.h5"
  renumber: "rcm"
  node_sets:
    - name: "turbine"
      criteria_type: "all"
      on_surface: true
    - name: "base_nodes"
      criteria_type: "direction"
      params:
        point: [0, 0, 0]
        direction: [0, 1, 0]
        method: "radial"
        distance: 0.0265
        mode: "inside"

material:
  type: "isotropic"
  name: "FSI_Test_LightSemiFlexible"
  E: 1.0e10
  nu: 0.30
  rho: 600.0

elements:
  family: "SOLID"

solver:
  type: "LinearDynamicFSIRotor"
  total_time: 0.15
  time_step: 1.0e-6
  solver_type: "auto"
  newmark:
    beta: 0.3025
    gamma: 0.6
  damping:
    eta_m: 30.0
    eta_k: 5.0e-5
  rotor:
    omega: 158
    send_omega_to_precice: false
    transform_displacement_to_inertial: true
    rotation_axis: [0, 1, 0]
    rotation_center: [0, 0, 0]
    include_geometric_stiffness: true
    include_centrifugal: true
    include_euler: true
    include_coriolis: true
    gravity: [0.0, 0.0, -9.81]
    force_ramp_time: 5.0e-4

boundary_conditions:
  dirichlet:
    - nodeset: "base_nodes"
      value: 0.0

coupling:
  participant: "Solid"
  config_file: "../precice-config.xml"
  coupling_mesh: "Solid-Mesh"
  write_data: ["Displacement"]
  read_data: ["Force"]
  boundaries: ["turbine"]

output:
  folder: "results"
  write_interval: 1.0e-3
  start_from: "latestTime"
```

### Example 3: Rotor with dynamic omega (computed + sent to CFD)

```yaml
mesh:
  source: "file"
  file:
    path: "propeller_volumetric.h5"
  renumber: "rcm"
  node_sets:
    - name: "turbine"
      criteria_type: "all"
      on_surface: true
    - name: "base_nodes"
      criteria_type: "direction"
      params:
        point: [0, 0, 0]
        direction: [0, 1, 0]
        method: "radial"
        distance: 0.0265
        mode: "inside"

material:
  type: "isotropic"
  name: "FSI_Test_LightSemiFlexible"
  E: 1.0e10
  nu: 0.30
  rho: 600.0

elements:
  family: "SOLID"

solver:
  type: "LinearDynamicFSIRotor"
  total_time: 0.15
  time_step: 1.0e-5
  solver_type: "auto"
  newmark:
    beta: 0.3025
    gamma: 0.6
  damping:
    eta_m: 30.0
    eta_k: 5.0e-5
  rotor:
    omega: 158
    omega_ramp_time: 1.0e-3
    force_ramp_time: 1.0e-3
    moment_of_inertia: "auto"       # Compute I from mesh → dynamic ω
    send_omega_to_precice: true     # Write ω to GlobalSolidMesh → CFD
    shaft_torque: 0                 # External shaft torque [N·m] (+drives, -resists)
    transform_displacement_to_inertial: true
    rotation_axis: [0, 1, 0]
    rotation_center: [0, 0, 0]
    include_geometric_stiffness: true
    include_centrifugal: true
    include_euler: true
    include_coriolis: true
    gravity: [0.0, 0.0, -9.81]

boundary_conditions:
  dirichlet:
    - nodeset: "base_nodes"
      value: 0.0

coupling:
  participant: "Solid"
  config_file: "../precice-config.xml"
  coupling_mesh: "Solid-Mesh"
  write_data: ["Displacement"]
  read_data: ["Force"]
  boundaries: ["turbine"]

output:
  folder: "results"
  write_interval: 1.0e-5
  start_from: "startTime"

postprocess:
  watchpoint_file: "precice-Solid-watchpoint-Flap-Tip.log"
  plots:
    displacement: "desplazamientos_rotor.png"
    force: "fuerzas_rotor.png"
```

---

## Validation and Debugging

### Validation checks (`--validate`)

The validator performs these checks:

- YAML syntax is valid
- All required sections are present
- Material properties are physically valid (E > 0, -1 < ν < 0.5, ρ > 0)
- Element family is recognized
- SHELL elements have `thickness` defined
- Solver type is recognized
- Rotor config is present when solver type is `LinearDynamicFSIRotor`
- preCICE config file exists and is parseable
- `total_time` and `time_step` are defined (or available from preCICE config)
- All referenced node sets exist in the mesh

### Mesh analysis output

When running with SOLID elements, the runner automatically logs:

1. **Element orientation check** — fixes inverted elements in-place
2. **Mesh quality report** — Jacobian ratios, aspect ratios
3. **Edge length statistics** — min/max/mean/median/std for volume elements
4. **RBF support radius guidance** — for each coupling boundary:
   - Nearest-neighbor spacing (min/max/mean)
   - Bounding box
   - Recommended support radius: `4 × mean_spacing` or `3 × max_spacing`

### preCICE peer disconnect

If the CFD side crashes or disconnects, the runner catches the error and prints:

```
preCICE peer (Fluid) disconnected unexpectedly.
Possible causes:
  - The fluid solver crashed
  - The fluid solver finished first
Check the fluid solver logs for errors.
```

Set `FEM_SHELL_DEBUG_TRACEBACK=1` for the full traceback.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FEM_SHELL_DEBUG_TRACEBACK` | Set to `1` to print full tracebacks on preCICE errors |

---

## Workflow: From Python Script to YAML

If you have an existing `solid.py` simulation script, convert it to YAML by
mapping each section:

| Python | YAML |
|--------|------|
| `load_mesh("file.h5")` | `mesh.source: "file"`, `mesh.file.path: "file.h5"` |
| `mesh.renumber_mesh(algorithm="rcm")` | `mesh.renumber: "rcm"` |
| `mesh.create_node_set_by_geometry(...)` | `mesh.node_sets: [...]` |
| `Material(name, E, nu, rho)` | `material: {type: isotropic, ...}` |
| `ElementFamily.SOLID` | `elements.family: "SOLID"` |
| `model_config["solver"]["rotor"]` | `solver.rotor: {...}` |
| `DirichletCondition(dofs, 0.0)` | `boundary_conditions.dirichlet: [{nodeset: "...", value: 0.0}]` |
| `coupling: {...}` in model_config | `coupling: {...}` top-level section |
| `problem = Solver(mesh, model_config)` | Automatic from `solver.type` |
| `problem.solve()` | Automatic |

Run `fem-shell-fsi --validate` after conversion to verify correctness.
