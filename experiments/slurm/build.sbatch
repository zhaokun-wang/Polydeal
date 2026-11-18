#!/bin/bash
#SBATCH --job-name=build
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# Builds polyDEAL from this repository (the library and the examples the
# experiments use) and the three test copies of experiments/, on a compute node.
# Every program is one large translation unit of about 2.5 GB at its peak, so
# make runs half as many jobs as there are cores.
source $SLURM_SUBMIT_DIR/common.sh
JOBS=$(( ${SLURM_CPUS_PER_TASK:-4} / 2 ))
set -e
mkdir -p $BUILD && cd $BUILD
cmake -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR \
      -DPARMETIS_DIR=$PARMETIS_DIR $ROOT
make -j$JOBS polydeal adaptive_polytopal polydeal_baseline
mkdir -p $EXP && cd $EXP
cmake -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR -DPOLYDEAL_BUILD=$BUILD \
      -DPARMETIS_DIR=$PARMETIS_DIR $ROOT/experiments
make -j$JOBS
ls -l $BUILD/examples/polydeal_baseline $EXP/repartitioned_poisson_test \
      $EXP/adaptive_heat_test $EXP/adaptive_polytopal_timing
echo "build finished"
