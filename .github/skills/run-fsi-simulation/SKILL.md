---
name: run-fsi-simulation
description: 'Launch, monitor, and diagnose OpenFOAM + preCICE FSI simulations on SLURM. Use when: submitting FSI jobs, checking job status, diagnosing divergence, reading SLURM logs, checking preCICE convergence, restarting failed runs.'
argument-hint: 'Describe the action: submit, status, diagnose, restart, or clean'
---

# Run FSI Simulation

Manage the lifecycle of OpenFOAM + preCICE FSI cases on the SDumont HPC cluster.

## When to Use

- Submit a new FSI simulation to SLURM
- Check running job status and convergence
- Diagnose why a simulation diverged or crashed
- Restart a simulation from the last checkpoint
- Clean a case directory for a fresh run

## Procedure

### 1. Submit a Job

```sh
cd simulations/<case-name>
sbatch runAll.srm
```

Verify submission: `squeue -u $USER`

### 2. Check Status

```sh
# SLURM queue
squeue -u $USER

# Live residuals (fluid side)
tail -f fluid.log | grep -E "Time =|Solving for|GAMG"

# preCICE convergence (look for "Convergence" lines)
grep -i "convergence\|relative\|time window" solid.log | tail -20
```

### 3. Diagnose Failure

Follow this checklist in order:

1. **SLURM error log**: `cat <jobid>.err` — look for segfaults, MPI errors
2. **Fluid log**: `tail -100 fluid.log` — check for NaN, residual explosion
3. **Solid log**: `tail -100 solid.log` — check for preCICE errors, solver exceptions
4. **preCICE iterations**: `grep "it " solid.log | tail -20` — excessive inner iterations indicate instability

Common failure patterns:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Attempt to cast type zeroGradient to type Field` | Wrong pointDisplacement BC | Use `fixedValue` on coupled patch |
| Residuals jump 10+ orders of magnitude | Rapid omega change → mesh deformation | Smooth omega, improve mesh |
| Segfault in `openib`/`udcm_component_query` | OpenMPI+InfiniBand issue | `export OMPI_MCA_btl=self,vader,tcp` |
| preCICE never converges (100+ iterations) | IQN-ILS accelerating both data sets | Accelerate only `Displacement` |
| `tetIndices` warnings + divergence | Mesh quality + deformation rate | Improve tip-region mesh |

### 4. Restart from Checkpoint

```sh
# Find latest time directory in fluid/
ls -d fluid/[0-9]* | sort -n | tail -1

# Ensure controlDict startFrom is set to latestTime
grep -A1 "startFrom" fluid/system/controlDict

# Re-submit
sbatch runAll.srm
```

### 5. Clean for Fresh Run

```sh
chmod +x clean-tutorial.sh
./clean-tutorial.sh
# Verify precice-run/ is removed
ls precice-run/ 2>/dev/null && echo "WARNING: precice-run/ still exists"
```

## Key Files to Check

| File | Contains |
|------|----------|
| `<jobid>.err` | SLURM stderr — segfaults, assertion failures |
| `<jobid>.out` | SLURM stdout — job start/end timestamps |
| `fluid.log` | OpenFOAM solver residuals and time steps |
| `solid.log` | fem_shell solver output + preCICE coupling info |
| `precice-config.xml` | Coupling scheme, data mappings, convergence criteria |
| `precice-run/` | preCICE event logs and iteration data |
