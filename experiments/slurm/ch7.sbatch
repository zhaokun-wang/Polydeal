#!/bin/bash
#SBATCH --job-name=ch7
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# Chapter 7: the moving peak, initial mesh 256 x 256, p = 4, 32 ranks.
source $SLURM_SUBMIT_DIR/common.sh
run ch7/np32 32 adaptive_heat_test
