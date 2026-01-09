#!/bin/bash

# Clean OpenFOAM case directories
rm -rf 0
rm -rf 0.[0-9]* [1-9]* processor* postProcessing
rm -f *.foam
rm -rf constant/polyMesh
rm -rf examples
rm -f log.*

# Clean preCICE files
rm -rf precice-*.log
rm -rf precice-profiling*
