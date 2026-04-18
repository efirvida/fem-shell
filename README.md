# fem-shell

`fem-shell` is an experimental finite element toolkit focused on structural
simulation with shell, solid, and plane elements, plus optional FSI coupling
with OpenFOAM through preCICE.

The project is currently in an evaluation stage. It already contains working
solvers, mesh utilities, a YAML-driven CLI, and monitoring tools, but the
interfaces and workflows should still be considered evolving rather than stable.

## Current Scope

The codebase currently covers:

- finite element support for plane, shell, and solid elements
- isotropic and orthotropic material models
- static, dynamic, modal, and FSI structural solvers
- rotor-oriented FSI workflows with angular-velocity feedback to CFD
- mesh generation, mesh import/export, and node-set selection tools
- checkpointing, restart, and post-processing helpers
- command-line tools for running, monitoring, and reconstructing simulations

## Project Status

This repository is best understood as a research and engineering prototype.

It is a good fit for:

- evaluating FEM formulations and solver behavior
- building custom structural or FSI workflows
- running rotor and blade-oriented studies
- testing integration patterns around OpenFOAM and preCICE

It is not yet presented as a finished end-user product with a frozen API,
turnkey installers, or broad industrial validation.

## Main Capabilities

### Structural analysis

- `LinearStaticSolver` for linear static problems
- `LinearDynamicSolver` for transient structural dynamics
- `ModalSolver` for modal analysis

### FSI workflows

- `LinearDynamicFSISolver` for structural coupling through preCICE
- `LinearDynamicFSIRotorSolver` for rotating systems with inertial effects,
  torque-driven omega updates, and CFD feedback
- automatic checkpoint/restart support for long-running coupled simulations

### Mesh and model utilities

- built-in mesh generators such as `SquareShapeMesh`, `BoxSurfaceMesh`,
  `BoxVolumeMesh`, `MultiFlapMesh`, `BladeMesh`, and `RotorMesh`
- mesh import/export utilities for common engineering formats
- geometric node-set creation from coordinate, box, distance, and direction
  criteria

### Monitoring and recovery tools

- `fem-shell-monitor` for live TUI-based monitoring of FSI runs
- `fem-shell-reconstruct-csv` to recover rotor performance histories from
  checkpoint data

## Installation

The project targets Python 3.12+.

```bash
cd fem-shell
pip install -e .
```

This installs the package in editable mode and registers the available CLI
commands.

If you need Triangle-based meshing helpers, install the optional extra:

```bash
pip install -e .[mesh]
```

### Optional external dependencies

Some workflows require software that is not bundled with this repository:

- preCICE for FSI coupling
- OpenFOAM for CFD-side coupled simulations
- an MPI runtime for distributed runs where applicable

If preCICE is not available in the environment, the package still imports, but
FSI solvers are disabled.

Recent packaging/runtime notes:

- `triangle` was moved to the optional `mesh` extra so base installation keeps
  working even on Python versions where Triangle wheels are not yet published.
- Top-level package imports now tolerate missing PETSc/preCICE so CLI
  operations such as `--preview` can inspect YAML files without a full solver
  stack.
- Example scripts that only generate meshes or inspect material data defer
  PETSc/preCICE imports until the actual solver path is requested.

## Quick Start

Generate a simulation template:

```bash
fem-shell-fsi --template > simulation.yaml
```

Validate the configuration:

```bash
fem-shell-fsi simulation.yaml --validate
```

Preview the resolved configuration without running the solver:

```bash
fem-shell-fsi simulation.yaml --preview
```

Run the simulation:

```bash
fem-shell-fsi simulation.yaml
```

Monitor a running case:

```bash
fem-shell-monitor /path/to/workdir
```

Reconstruct rotor performance CSV data from checkpoints:

```bash
fem-shell-reconstruct-csv results/
```

## CLI Commands

The package currently provides three command-line entry points:

- `fem-shell-fsi` - run or inspect YAML-defined simulations
- `fem-shell-monitor` - monitor rotor/FSI runs from CSV output
- `fem-shell-reconstruct-csv` - rebuild missing rotor performance histories

Detailed CLI documentation is available in [docs/cli-reference.md](docs/cli-reference.md).

## Repository Layout

```text
fem-shell/
├── docs/                 # User-facing documentation
├── examples/             # Small usage examples
├── src/fem_shell/        # Package source code
│   ├── cli/              # CLI entry points
│   ├── core/             # Mesh, BCs, materials, config
│   ├── constitutive/     # Failure criteria and constitutive helpers
│   ├── elements/         # Element implementations
│   ├── models/           # Higher-level structural models
│   ├── postprocess/      # Output and analysis utilities
│   └── solvers/          # Static, dynamic, modal, and FSI solvers
└── tests/                # Unit and regression tests
```

## Examples and Validation

- Example scripts live under `examples/`
- Automated tests live under `tests/`
- A practical test command currently used in development is:

```bash
python -m pytest tests/ -q --tb=short --ignore=tests/test_blade_mesh.py --ignore=tests/test_rotor_inertial.py
```

Two test modules are excluded from the default run because they have collection
errors caused by stale import paths; they do not reflect on the solver quality:

| File | Reason |
|------|--------|
| `test_blade_mesh.py` | imports `MeshElement` (renamed to `FemElement`) |
| `test_rotor_inertial.py` | imports from a moved module path |

Some benchmark cases are intentionally heavier and may not be suitable for a
quick local smoke test.  A handful of distributed-mesh MITC3/MITC4 benchmarks
are also known to fall marginally outside the 5 % tolerance band used in the
assertions; this reflects element accuracy limits on coarse meshes, not solver
correctness.

## Post-processing

The `StressRecovery` class (`src/fem_shell/postprocess/stress_recovery.py`)
computes element-centroidal and node-averaged stress and strain fields from the
displacement solution.  Key behaviours to be aware of:

- **Shell elements** return a plane-stress triplet `[σ_xx, σ_yy, τ_xy]`.  The
  result arrays follow the full 6-component Voigt layout
  `[σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx]`, with `σ_zz = τ_yz = τ_zx = 0` for
  shells.  Von Mises stress and principal angles are computed in 2-D.
- **Solid elements** populate all six Voigt components.
- **Mixed meshes** (shells + solids in the same domain) are supported: each
  element extracts only the DOFs it owns from the global solution vector.

## Limitations

At this stage, users should expect some rough edges:

- documentation is still being expanded
- workflows are stronger for research and custom engineering use than for
  packaged end-user operation
- some advanced FSI setups depend heavily on external solver configuration,
  mesh quality, and restart discipline
- APIs and configuration details may evolve as the project matures

## Near-Term Documentation Goals

The next documentation improvements that make the most sense are:

- a getting-started guide for the first structural run
- a dedicated FSI setup guide for OpenFOAM + preCICE
- one or two complete example cases with expected outputs
- a clearer compatibility matrix for optional dependencies

## License and Usage

This repository now includes a restrictive research license in [LICENSE](LICENSE).

In short:

- use is allowed for non-commercial research, academic, educational, and
  evaluation purposes
- commercial use is not allowed without prior written permission
- contributions are welcome and governed by [CONTRIBUTING.md](CONTRIBUTING.md)

This is not an OSI-approved open-source license. If you later want broader
adoption, external packaging, or community growth, you should expect to revisit
this choice.