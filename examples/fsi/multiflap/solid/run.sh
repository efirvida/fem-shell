#!/bin/bash
# ==============================================
#  Multi-Flap FSI Solid Solver Runner
# ==============================================
# 
# This script runs the structural solver for the multi-flap FSI simulation
# using the YAML configuration system.
#
# Usage:
#   ./run.sh              # Run simulation
#   ./run.sh --validate   # Validate configuration only
#   ./run.sh --preview    # Preview configuration
#   ./run.sh --legacy     # Run legacy Python script (solid.py)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/solid_config.yaml"

# Parse arguments
ACTION="run"
if [[ "$1" == "--validate" ]]; then
    ACTION="validate"
elif [[ "$1" == "--preview" ]]; then
    ACTION="preview"
elif [[ "$1" == "--legacy" ]]; then
    ACTION="legacy"
elif [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Multi-Flap FSI Solid Solver"
    echo ""
    echo "Usage: ./run.sh [option]"
    echo ""
    echo "Options:"
    echo "  (none)       Run simulation using YAML config"
    echo "  --validate   Validate configuration file"
    echo "  --preview    Preview configuration without running"
    echo "  --help, -h   Show this help message"
    echo ""
    echo "Configuration: ${CONFIG_FILE}"
    exit 0
fi

cd "${SCRIPT_DIR}"

case "${ACTION}" in
    "validate")
        echo "Validating configuration..."
        python -m fem_shell.cli.run_fsi "${CONFIG_FILE}" --validate
        ;;
    "preview")
        echo "Configuration preview:"
        python -m fem_shell.cli.run_fsi "${CONFIG_FILE}" --preview
        ;;
    "run")
        echo "Starting Multi-Flap FSI Solid Solver..."
        echo "Configuration: ${CONFIG_FILE}"
        echo ""
        python -m fem_shell.cli.run_fsi "${CONFIG_FILE}"
        ;;
esac
