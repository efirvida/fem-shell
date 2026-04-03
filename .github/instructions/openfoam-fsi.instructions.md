---
description: "Use when editing OpenFOAM FSI simulation cases, preCICE configs, pointDisplacement BCs, overset mesh setups, SLURM scripts, or diagnosing FSI divergence. Covers critical coupling pitfalls."
applyTo: "simulations/**"
---
# OpenFOAM + preCICE FSI Conventions

## pointDisplacement Boundary Conditions

- On **preCICE-coupled patches** (where the adapter reads `Displacement`):
  `pointDisplacement` MUST be `type fixedValue` with an initial `value uniform (0 0 0)`.
  Using `zeroGradient` causes: *"Attempt to cast type zeroGradient to type Field"*.

- On **overset patches**: do NOT use `type overset;` (invalid `pointPatchField` in v2506).
  Use a valid field type (e.g. `zeroGradient`) and add `patchType overset;` separately.

## preCICE Coupling Scheme

- When using **IQN-ILS** (implicit quasi-Newton): accelerate only `Displacement`
  first.  Accelerating both `Displacement` and `Force` can cause intermediate
  iterates to overshoot by orders of magnitude even when the window converges.
- To diagnose force mismatches, compare values at the **same time-window AND
  iteration number** — preferably at converged window endpoints.

## Overset FSI with Dynamic Rotation

- `tetIndices` mesh warnings indicate marginal mesh quality but are NOT the root
  cause of divergence.
- The real culprit is **mesh deformation rate**: when angular velocity (omega)
  changes rapidly from the solid solver, unpredictable mesh strain creates
  degenerate cells → NaN/Inf in pressure coupling → residual explosion.
- Mitigations: (1) filter/smooth the omega time-history from solid, (2) improve
  mesh quality in the rotor-tip region, (3) use implicit time-interpolation.

## MPI / InfiniBand Issues

- Random segfaults in `setFields` or OpenFOAM startup with backtraces through
  `openib` / `udcm_component_query` are OpenMPI+IB runtime issues, not code bugs.
- Workaround:
  ```sh
  export OMPI_MCA_btl=self,vader,tcp
  export OMPI_MCA_btl_openib_allow_ib=0
  ```

## Case Structure

Each FSI case follows this layout:
```
case-name/
  precice-config.xml    # coupling scheme, data mappings, time windows
  run-tutorial.sh       # launches both participants in background
  runAll.srm            # SLURM submission wrapper
  solid/
    solid.py            # fem_shell structural solver
    run.sh              # activates venv, runs solid.py
  fluid/
    system/             # OpenFOAM dictionaries
    constant/           # mesh, transportProperties
    0/                  # initial conditions
    run.sh              # sources OpenFOAM, runs solver
```

## Debugging Checklist

1. Check SLURM `.err` / `.out` files for stack traces
2. Check `solid.log` and `fluid.log` for preCICE convergence info
3. Look for `precice-run/` directory — contains coupling iteration data
4. Verify `precice-config.xml` data names match adapter `preciceDict`
5. If residuals explode: check omega rate-of-change and mesh quality first
