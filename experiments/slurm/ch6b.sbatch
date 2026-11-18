#!/bin/bash
#SBATCH --job-name=ch6b
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log
# Chapter 6: as ch6a, on 8, 16 and 32 ranks.
source $SLURM_SUBMIT_DIR/common.sh
for np in 8 16 32; do
  for c in 1 0; do run ch6/np${np}_parmetis$c $np repartitioned_poisson_test parmetis=$c; done
done
