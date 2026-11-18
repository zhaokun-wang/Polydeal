#!/bin/bash
#SBATCH --job-name=ch8_3d
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# Chapter 8 in three dimensions: polynomial solution and point singularity on
# the unit cube, 32^3 initial cells, p = 3, the four combinations, 32 ranks.
source $SLURM_SUBMIT_DIR/common.sh
for prob in polynomial3d singular3d; do
  for t in q1p1 q0p1 q1p0 q0p0; do
    run ch8/${prob}_$t 32 adaptive_polytopal_timing $PRM/${prob}_$t.prm
  done
done
