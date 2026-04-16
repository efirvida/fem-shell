# BEM Standalone Example

Standalone BEM (Blade Element Momentum) aerodynamic analysis of the IEA 15 MW
reference wind turbine blade parked at 45 m/s extreme wind.

## Usage

```bash
cd examples/blade/bem
fem-shell-fsi simulation.yaml
```

## What it does

1. Generates a shell mesh from the WindIO YAML blade definition
2. Loads airfoil polar data (Cl, Cd, Cm) from the YAML — falls back to
   NeuralFoil if polars are missing
3. Runs CCBlade BEM analysis at the specified wind conditions
4. Projects aerodynamic loads onto shell mesh nodes (force + moment
   conservative)
5. Exports results

## Outputs

| File | Description |
|------|-------------|
| `results/bem_sectional_loads.csv` | Per-station: r, Np, Tp, alpha, cl, cd, a, ap |
| `results/bem_global_loads.csv` | Integrated thrust, torque, power |
| `results/bem_nodal_forces.vtu` | Shell mesh with projected force vectors (ParaView) |

## Dependencies

CCBlade requires a Fortran compiler and Meson to build its `_bem` extension.
Install the build tools, then CCBlade, and finally the BEM extras:

```bash
# 1. Build tools (if not already installed)
pip install meson meson-python ninja setuptools numpy

# 2. CCBlade (Fortran extension — needs --no-build-isolation)
pip install git+https://github.com/WISDEM/CCBlade.git --no-build-isolation

# 3. NeuralFoil and other BEM extras
pip install -e ".[bem]"
```

On Fedora/RHEL, install gfortran with `sudo dnf install gcc-gfortran`.
On Ubuntu/Debian, use `sudo apt install gfortran`.
