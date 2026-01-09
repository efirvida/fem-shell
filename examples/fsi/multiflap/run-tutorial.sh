#!/bin/bash
# ==============================================
#           FSI TUTORIAL: PERPENDICULAR FLAP
# ==============================================

. /opt/OpenFOAM-v2406/etc/bashrc

# Configuration
SOLID_DIR="solid"
FLUID_DIR="fluid"
SOLID_LOG="solid.log"
FLUID_LOG="fluid.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Initialize log files
> "$SOLID_LOG"
> "$FLUID_LOG"

# Function to check and run a solver
run_solver() {
    local solver_dir=$1
    local solver_name=$2
    local log_file=$3
    
    echo "▶ [${solver_name^^}] Starting in directory: ${solver_dir}"
    echo "   Logging to: ${log_file}"
    
    if ! cd "${solver_dir}"; then
        echo "✖ ERROR: Failed to enter ${solver_dir} directory" | tee -a "../${log_file}"
        return 1
    fi
    
    if [ ! -f "./run.sh" ]; then
        echo "✖ ERROR: run.sh not found in ${solver_dir}" | tee -a "../${log_file}"
        return 1
    fi
    
    # Clear previous log and start new one
    > "../${log_file}"
    ./run.sh >> "../${log_file}" 2>&1 &
    local pid=$!
    echo "✔ [${solver_name^^}] Started with PID: ${pid}" | tee -a "../${log_file}"
    cd ..
    return 0
}

# Clean previous runs
echo "🔄 [CLEANUP] Cleaning previous tutorial runs..."
./clean-tutorial.sh || {
    echo "✖ ERROR: Failed to clean previous runs" | tee -a "$SOLID_LOG" "$FLUID_LOG"
    exit 1
}
echo "✔ [CLEANUP] Previous runs cleaned successfully" | tee -a "$SOLID_LOG" "$FLUID_LOG"

# Generate configuration files from case_config.json
echo ""
echo "⚙️  [CONFIG] Generating configuration files from case_config.json..."
python3 generate_all.py || {
    echo "✖ ERROR: Failed to generate configuration files" | tee -a "$SOLID_LOG" "$FLUID_LOG"
    exit 1
}
echo "✔ [CONFIG] Configuration files generated successfully" | tee -a "$SOLID_LOG" "$FLUID_LOG"

# Run solvers in parallel
echo ""
echo "🚀 [FSI START] Launching coupled simulation..." | tee -a "$SOLID_LOG" "$FLUID_LOG"
run_solver "${SOLID_DIR}" "solid" "$SOLID_LOG" || exit 1
run_solver "${FLUID_DIR}" "fluid" "$FLUID_LOG" || exit 1

# Wait for both processes to finish
echo ""
echo "⏳ [MONITOR] Waiting for simulations to complete..." | tee -a "$SOLID_LOG" "$FLUID_LOG"
wait

# Post-processing
echo ""
echo "✔ [COMPLETION] Both solvers have finished" | tee -a "$SOLID_LOG" "$FLUID_LOG"
echo "📄 Log files created:" | tee -a "$SOLID_LOG" "$FLUID_LOG"
echo "   - Solid solver: ${SOLID_LOG}" | tee -a "$SOLID_LOG"
echo "   - Fluid solver: ${FLUID_LOG}" | tee -a "$FLUID_LOG"

# Clean precice-run directory
rm -rf precice-run 2>/dev/null

echo ""
echo "==============================================" | tee -a "$SOLID_LOG" "$FLUID_LOG"
echo "            TUTORIAL EXECUTION COMPLETE       " | tee -a "$SOLID_LOG" "$FLUID_LOG"
echo "==============================================" | tee -a "$SOLID_LOG" "$FLUID_LOG"

# Optional: Archive logs
mkdir -p "logs_archive"
cp "$SOLID_LOG" "logs_archive/solid_${TIMESTAMP}.log"
cp "$FLUID_LOG" "logs_archive/fluid_${TIMESTAMP}.log"

exit 0