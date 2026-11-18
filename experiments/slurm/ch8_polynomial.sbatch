#!/bin/bash
#SBATCH --job-name=ch8_polynomial
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# Chapter 8, polynomial solution: three problem sizes (initial mesh n x n,
# tolerance) with the four combinations, one node of 32 ranks; each run is cut
# after 20 minutes. One size is chosen afterwards.
source $SLURM_SUBMIT_DIR/common.sh
RUN_TIMEOUT=1200
for s in "256 1e-9" "512 1e-8" "1024 1e-8"; do
  set -- $s; n=$1; tol=$2
  for t in q1p1 q0p1 q1p0 q0p0; do
    prm=$RUNS/polynomial_n${n}_tol${tol}_$t.prm
    sed -e "s/set Number of subdivisions = 256/set Number of subdivisions = $n/" \
        -e "s/set Tolerance              = 1e-7/set Tolerance              = $tol/" \
        $PRM/polynomial_$t.prm > $prm
    grep -q "subdivisions = $n" $prm && grep -q "Tolerance              = $tol" $prm || { echo "sed failed: $prm"; exit 1; }
    run ch8/polynomial_n${n}_tol${tol}_$t 32 adaptive_polytopal_timing $prm
  done
done
