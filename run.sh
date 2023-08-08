#!/bin/bash

START=$1
END=$2

for N in $(seq $START $END)
do
    for d in {0..1}
    do
        # d = 0: deg = 121; d = 1: deg = 601; d = 2: deg = 2501
        echo "Now running N = ${N}, set_deg = ${d}."
        file="./results_tmp/${N}q-nAA-${d}.txt"
        time python3 -m cProfile -o "${N}q-nAA-${d}.prof" qsvt-linear-solver.py -N ${N} -d ${d} > ${file}

        if [ ! -s ${file} ];
        then
            echo "N = ${N} does not finish!" 
            break
        fi

        echo "set_deg = ${d} done!"
    done
    echo "N = ${N} done!"
done
