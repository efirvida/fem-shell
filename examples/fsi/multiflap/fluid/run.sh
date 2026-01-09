#!/bin/bash

# Restore initial conditions
if [ -d "0.orig" ]; then
    rm -rf 0
    cp -r 0.orig 0
fi

# Generate parametric blockMeshDict for multiple flaps
python3 system/generate_blockMeshDict.py || { echo "Failed to generate blockMeshDict"; exit 1; }

openfoam blockMesh      > log.blockMesh
openfoam decomposePar   > log.decomposePar
openfoam mpirun -np 6 pimpleFoam -parallel     > log.pimpleFoam 
touch case.foam

./clean_empty_results.sh --force .