#!/bin/bash

# Restore initial conditions
if [ -d "0.orig" ]; then
    rm -rf 0
    cp -r 0.orig 0
fi

# Generate parametric blockMeshDict for multiple flaps
python3 ../scripts/generate_blockMeshDict.py || { echo "Failed to generate blockMeshDict"; exit 1; }

blockMesh      > log.blockMesh
decomposePar   > log.decomposePar
mpirun -np 3 pimpleFoam -parallel     > log.pimpleFoam 
touch case.foam

./clean_empty_results.sh --force .