#!/bin/bash

rm -rf precice-run *.log

cd solid && ./clean.sh; &> /dev/null
cd ..
cd fluid && ./clean.sh  &> /dev/null
