#!/bin/bash

START=$1
END=$2

for d in {1..4}
do
    flag="1"
    for N in $(seq $START $END)
    do
        # d = 0: deg = 121; d = 1: deg = 601; d = 2: deg = 1501; d = 3: deg = 2001; d = 4: deg = 5001
        echo "Now running N = ${N}, set_deg = ${d}."

        filename="${N}q-nAA-${d}"
        record="./experiments/results_self_sample/${filename}.txt"
        # profile="./profiles_run/${filename}.prof"

        # time python3 -m cProfile -o "${profile}" qsvt-linear-solver-run.py -N ${N} -d ${d} > ${record}
        time python3 qsvt-linear-solver-new.py -N ${N} -d ${d} > ${record}

        if [ ! -s "${record}" ];
        then
            echo "N = ${N} does not finish!" 
            flag="0"
            break
        fi

        echo "N = ${N}, set_deg = ${d} done!"
    done

    if [ ${flag} -eq "0" ]; then 
        break
    fi
    # echo "N = ${N} done!"
done
