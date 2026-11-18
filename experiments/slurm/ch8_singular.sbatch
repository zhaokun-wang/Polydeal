#!/bin/bash
#SBATCH --job-name=ch8_singular
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log
# Chapter 8, point singularity: the four combinations of quadrature-free
# integration (q) and the ParMETIS copy (p), one node of 32 ranks.
source $SLURM_SUBMIT_DIR/common.sh
for t in q1p1 q0p1 q1p0 q0p0; do
  run ch8/singular_$t 32 adaptive_polytopal_timing $PRM/singular_$t.prm
done
