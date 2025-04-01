#!/bin/bash

clean_openfoam_dirs() {
    local target_dirs=()
    
    while IFS= read -r -d $'\0' dir; do
        if [ -d "$dir/uniform" ] && 
           [ -d "$dir/uniform/functionObjects" ] && 
           [ -f "$dir/uniform/functionObjects/functionObjectProperties" ] &&
           [ $(find "$dir" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ] &&
           [ $(find "$dir/uniform" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ] &&
           [ $(find "$dir/uniform/functionObjects" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ]; then
            target_dirs+=("$dir")
        fi
    done < <(find . -type d -name "[0-9]*" -print0)

    if [ ${#target_dirs[@]} -gt 0 ]; then
        rm -rf "${target_dirs[@]}"
    fi
}

./clean.sh
. RunFunctions                                                              
restore0Dir

runApplication blockMesh
runApplication decomposePar
runParallel    $(getApplication)
clean_openfoam_dirs
touch case.foam
