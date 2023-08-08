#!/bin/bash
# on nchc twnia3
# not up-to-date (check run.sh)

START=$1
END=15
for N in $(seq $START $END)
do
    echo "Now running N = ${N}."
    file="./results_tmp/${N}q-nAA-121.txt"
    time srun python3 -m cProfile -o "${N}q.prof" qsvt-linear-solver.py -N ${N} > ${file}

    if [ ! -s ${file} ];
    then
	echo "N = ${N} does not finish!" 
	break
    fi
    echo "N = ${N} done!"
done
