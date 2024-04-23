#!/bin/bash
#SBATCH -A GOV113006 -p betatest
#SBATCH -J qsvt_16 -t 24:00:00                                    ## <~~ better to roughly estimate your walltime (-t)
#SBATCH --nodes=16 --ntasks-per-node=1 --cpus-per-task=112    ## <~~ change nodes (ntasks-per-node * cpus-per-task is usually 112 for T4)
#SBATCH --mem-bind=no
#SBATCH -o /home/u6945379/QSVT-implementation/job_results/16_qubit10-15.out          	# Path to the standard output file
#SBATCH -e /home/u6945379/QSVT-implementation/job_results/16_qubit10-15.err 				# Path to the standard error ouput file

##export OMP_PROC_BIND=true
##export OMP_PLACES=cores
##export OMP_DYNAMIC=false
VARS="SLURM_JOB_ID SLURM_NTASKS SLURM_NNODES SLURM_NTASKS_PER_NODE SLURM_CPUS_PER_TASK SLURM_TASK_PID SLURM_NODELIST SLURM_SUBMIT_DIR  SLURM_MEM_BIND SLURM_MPI_TYPE OMP_PROC_BIND OMP_PLACES OMP_DYNAMIC"
for v in ${VARS}; do echo " $v = ${!v}"; done

export KMP_AFFINITY=compact,1,0,granularity=fine

ml purge
module use /home/qusim/intel-qs/modulefiles

. /pkg/x86/apps/intel_oneapi/2024/setvars.sh

# ml sys/intelmpi
ml openmpi
ml intel-qs/oneAPI23/20240201
#ml python

conda init
conda activate qiskit_1.0

T0=0
num_qubits="$(seq 10 15)"
### Enable debug messages ###
#export I_MPI_DEBUG=30
#export FI_LOG_LEVEL=debug

NUM_NODES=16
echo "Number of nodes: ${NUM_NODES}"
for q in ${num_qubits}; do
  mpirun -np ${NUM_NODES} python qsvt-linear-solver.py -N ${q} -m
  echo "-- loop wall time: $((SECONDS - T0)) sec."
  echo ""
  echo ""
  T0=${SECONDS}
done

echo "---- Walltime: ${SECONDS} sec."
