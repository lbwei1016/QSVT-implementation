#!/bin/bash

for N in {7..15} 
do
    echo "Now running N = ${N}."
    python3 qsvt-linear-solver.py -N ${N} > "${N}q-nAA-121.txt"
    echo "N = ${N} done!"
done
