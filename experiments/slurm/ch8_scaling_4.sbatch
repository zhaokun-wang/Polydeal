#!/bin/bash
#SBATCH --job-name=ch8_scaling_4
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# Chapter 8, strong scaling: the moving peak (q1p1) on 4 nodes of 32 ranks
# each. One node is ch8_peaks_q1p1.
source $SLURM_SUBMIT_DIR/common.sh
run ch8/scaling_n4 128 adaptive_polytopal_timing $PRM/peaks4_q1p1.prm
