#!/bin/bash
mv results.log results.log.bak
. ../myenv/bin/activate

num_qubits="$(seq 1 15)"
for q in ${num_qubits}; do
  echo "Now solving ${q}..."
  python3 qsvt-linear-solver.py -N ${q} -s >> results.log
done