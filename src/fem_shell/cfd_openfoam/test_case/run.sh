#!/bin/bash


./clean.sh
. RunFunctions                                                              
runApplication blockMesh
sed -i 's/defaultFaces/overset/g' constant/polyMesh/boundary
sed -i 's/type.*empty/type            patch/g' constant/polyMesh/boundary
sed -i 's/inGroups.*(empty)/inGroups        1(patch)/g' constant/polyMesh/boundary
runApplication checkMesh -allGeometry -allTopology -writeChecks json