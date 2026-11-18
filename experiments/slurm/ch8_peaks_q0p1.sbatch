#!/bin/bash
#SBATCH --job-name=ch8_peaks_q0p1
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log
# Chapter 8, moving peak (four peaks per direction): combination q0p1 of
# quadrature-free integration (q) and the ParMETIS copy (p), one node of 32 ranks.
# This run is also the one-node point of the strong scaling (ch8_scaling_*).
source $SLURM_SUBMIT_DIR/common.sh
run ch8/peaks4_q0p1 32 adaptive_polytopal_timing $PRM/peaks4_q0p1.prm
