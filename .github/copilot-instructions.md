# fem-shell — Workspace Instructions

Finite element library for shell, solid and plane elements, with FSI coupling
via preCICE for rotating structures (wind turbine blades, propellers).
All simulations are configured via YAML and executed through the `fem-shell-fsi` CLI.

## Architecture

```
src/fem_shell/
  cli/           # YAML-driven CLI: fem-shell-fsi command
  core/
    config.py    # FSISimulationConfig — 30+ dataclasses, preCICE XML parsing
    mesh/        # MeshModel, entities, 13 generators (Gmsh), I/O, quality checks
    material.py  # IsotropicMaterial, OrthotropicMaterial
    bc.py        # DirichletCondition, BodyForce
    laminate.py  # Classical Laminate Theory (CLT)
    assembler.py # PETSc sparse assembly from element K/M
  elements/      # MITC3/4 (shell), HEXA/TETRA/WEDGE/PYRAMID (solid), QUAD (plane)
  solvers/
    linear.py    # LinearStaticSolver, LinearDynamicSolver
    modal.py     # ModalSolver (SLEPc eigenvalue)
    fsi/
      linear_dynamic.py  # LinearDynamicFSISolver (preCICE)
      rotor.py           # LinearDynamicFSIRotorSolver (co-rotational + inertial)
      corotational.py    # OmegaProviders, InertialForcesCalculator, CoordinateTransforms
      runner.py          # FSIRunner — YAML-to-simulation executor
      base.py            # Adapter, ForceClipper, SolverState
  postprocess/   # Stress recovery (shell + solid), watchpoint plots
  constitutive/  # Composite failure criteria (Tsai-Wu, Hashin, Max Stress)
  models/        # Domain models (Blade, NuMAD integration)
  cfd_openfoam/  # OpenFOAM blockMeshDict generator from blade geometry
```

### Solver Hierarchy

```
Solver (ABC)
├── LinearStaticSolver
├── LinearDynamicSolver (Newmark-β)
├── ModalSolver (SLEPc)
└── LinearDynamicFSISolver (preCICE)
    └── LinearDynamicFSIRotorSolver (co-rotational + inertial forces)
```

### Data Flow

```
YAML config → FSISimulationConfig.from_yaml()
            → FSIRunner._setup_mesh()         # load/generate + renumber + node sets
            → FSIRunner._create_material()
            → FSIRunner._build_model_config()  # dict for solver constructor
            → FSIRunner._create_solver()       # dispatches by SolverType
            → FSIRunner._apply_boundary_conditions()
            → solver.solve()                   # PETSc KSP + preCICE coupling loop
            → FSIRunner._run_postprocessing()  # watchpoint plots
```

Assembly pipeline: `MeshModel` → generators (Gmsh) → `MeshAssembler`
→ PETSc sparse matrices → solver → checkpoint (VTU/PVD via meshio).

## CLI: `fem-shell-fsi`

Registered via `pyproject.toml` as `fem-shell-fsi = "fem_shell.cli.run_fsi:main"`.

```sh
fem-shell-fsi --template > simulation.yaml  # generate template
fem-shell-fsi simulation.yaml --validate    # check config
fem-shell-fsi simulation.yaml --preview     # print parsed config tree
fem-shell-fsi simulation.yaml               # run simulation
fem-shell-fsi simulation.yaml --view        # interactive mesh viewer
fem-shell-fsi --list-generators             # available mesh generators
fem-shell-fsi --template -g BoxVolumeMesh   # template with generator example
```

Full reference: [docs/cli-reference.md](../docs/cli-reference.md)

## Build and Test

```sh
pip install -e .                         # editable install
python -m pytest tests/ -q --tb=short    # run tests (~294 pass)
ruff check                               # lint (line-length=100)
```

Known test exclusions (pre-existing import issues):
```sh
--ignore=tests/test_blade_mesh.py --ignore=tests/test_rotor_inertial.py
```

Benchmark tests under `tests/benchmarks/` have tight tolerances and may show
8 pre-existing failures in `test_ko2017_performance.py` — these are NOT
regressions.

## Conventions

### DOF Ordering
- Shell (MITC3/4): 6 DOFs/node `[u, v, w, θx, θy, θz]`
- Solid (HEXA, TETRA, …): 3 DOFs/node `[u, v, w]`
- Plane (QUAD): 2 DOFs/node `[u, v]`
- Mixed meshes use the **max stride** (6) to avoid DOF aliasing

### Voigt Notation
- Solid: `[σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx]` — 6 components
- Shell: `[σ_xx, σ_yy, τ_xy]` — 3 components (plane stress)

### Element Families (`ElementFamily` enum)
| Family | DOFs/node | Elements | Mesh type |
|--------|-----------|----------|-----------|
| `PLANE` | 2 | QUAD4/8 | 2D |
| `SHELL` | 6 | MITC3, MITC4, MITC3Composite, MITC4Composite | 3D surface |
| `SOLID` | 3 | HEXA8/20/27, TETRA4/10, WEDGE6/15, PYRAMID5/13 | 3D volume |

### Naming
- Element files: UPPERCASE (`MITC3.py`, `SOLID.py`, `QUAD.py`)
- Element classes: UPPERCASE + node count (`HEXA8`, `TETRA10`, `MITC3`)
- Core/solver files: lowercase snake_case
- `@cached_property` for expensive element computations (K, M)
- `@dataclass` for data-oriented types (materials, plies, configs)

### Shell Stress Recovery
`element.Cm()` returns the *integrated* membrane stiffness D = C·h (N/m).
To get actual stress (Pa), divide by thickness: `C_mat = element.Cm() / h`.

### Configuration System
- `FSISimulationConfig` is the top-level YAML config dataclass
- `FSISimulationConfig.from_yaml(path)` loads and validates
- `config.auto_complete_from_precice()` reads `total_time`, `time_step`,
  and watchpoint files from the preCICE XML config
- `RotorConfig.to_dict()` serializes to the dict format `LinearDynamicFSIRotorSolver` expects
- `moment_of_inertia: "auto"` triggers automatic I computation from mesh

### Omega Modes (Rotor Solver)
| `moment_of_inertia` | `omega_ramp_time` | Provider |
|----------------------|-------------------|----------|
| `null` | `0` | ConstantOmega |
| `null` | `> 0` | RampedOmega |
| `"auto"` / float | `0` | ComputedOmega |
| `"auto"` / float | `> 0` | RampedComputedOmega |

When `send_omega_to_precice: true`, the solver writes `AngularVelocity` on
`GlobalSolidMesh` (single vertex at rotation center) — handled internally,
not configured in `coupling.write_data`.

## Mesh Generators

| Generator | Description | Default Node Sets |
|-----------|-------------|-------------------|
| `SquareShapeMesh` | 2D rectangular | left, right, top, bottom, corners |
| `BoxSurfaceMesh` | 3D shell box | left, right, top, bottom, front, back |
| `BoxVolumeMesh` | 3D solid (hex/tet/wedge) | left, right, top, bottom, front, back |
| `MultiFlapMesh` | Flaps on base strip | bottom, flaps_left, flaps_right, flaps_top |
| `RotorMesh` | Turbine from blade YAML | RootNodes_blade_N, allOuterShellNods_blade_N |
| `BladeMesh` | Single blade from YAML | root, tip, surface |
| `CylindricalSurfaceMesh` | Cylindrical shell | — |
| `HyperbolicParaboloidMesh` | Saddle surface | — |
| `RaaschHookMesh` | Curved hook benchmark | — |
| `SphericalSurfaceMesh` | Spherical shell | — |
| `CylinderVolumeMesh` | Cylindrical solid | — |
| `PyramidTransitionMesh` | Hex-to-tet transition | — |

## External Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| PETSc (`petsc4py`) | Sparse assembly, KSP solvers | Yes |
| SLEPc (`slepc4py`) | Eigenvalue problems (modal) | For modal analysis |
| preCICE (`precice`) | FSI coupling | Optional (guarded import) |
| MPI (`mpi4py`) | Parallel computing | Yes |
| Gmsh (`gmsh`) | Mesh generation | Yes |
| meshio | VTU/PVD checkpoint export | Yes |

## Simulations Workspace

The `simulations/` folder contains OpenFOAM + preCICE FSI cases. Each case has:
- `precice-config.xml` — coupling scheme, data mappings, convergence
- `solid/solid.py` — legacy Python script (being replaced by `simulation.yaml`)
- `solid/simulation.yaml` — YAML config for `fem-shell-fsi` (new approach)
- `fluid/` — OpenFOAM case (system/, constant/, 0/)
- `runAll.srm` — SLURM submission script

### YAML-Driven Examples
| Case | Omega Mode | Key Feature |
|------|------------|-------------|
| `LinearDynamicFSIRotorSolver/` | Fixed (158 rad/s) | Basic rotor FSI, `start_from: latestTime` |
| `overset-fsi-dyn/` | Dynamic (computed) | `moment_of_inertia: "auto"`, `send_omega_to_precice: true` |
| `overset-fsi-fixed/` | Fixed (158 rad/s) | Overset mesh, `send_omega_to_precice: false` |

### Related Customizations
- [instructions/openfoam-fsi.instructions.md](instructions/openfoam-fsi.instructions.md) — OpenFOAM coupling pitfalls (applies to `simulations/**`)
- [skills/run-fsi-simulation/SKILL.md](skills/run-fsi-simulation/SKILL.md) — SLURM job lifecycle (submit, diagnose, restart, clean)
- [agents/fem-reviewer.agent.md](agents/fem-reviewer.agent.md) — FEA numerical correctness reviewer
