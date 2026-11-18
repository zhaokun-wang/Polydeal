#!/bin/bash
#SBATCH --job-name=ch6a
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log
# Chapter 6: the copy partitioned by ParMETIS (parmetis=1) against the copy
# without ParMETIS (parmetis=0), on 2 and 4 ranks, point singularity at (1/4, 1/4).
source $SLURM_SUBMIT_DIR/common.sh
for np in 2 4; do
  for c in 1 0; do run ch6/np${np}_parmetis$c $np repartitioned_poisson_test parmetis=$c; done
done
