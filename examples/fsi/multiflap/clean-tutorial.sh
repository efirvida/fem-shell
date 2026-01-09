#!/bin/bash

rm -rf precice-run
rm -rf *.log
rm -rf logs_archive

cd solid && ./clean.sh; &> /dev/null
cd ..
cd fluid && ./clean.sh  &> /dev/null
