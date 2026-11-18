#!/bin/bash
#SBATCH --job-name=unstructured
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log
# Point singularity on the unstructured mesh meshes/t3.msh (91 quadrilaterals
# of the unit square, 4 global refinements, 23296 initial cells). (1) The
# pipeline: the four combinations, 32 ranks. (2) The baseline, the polyDEAL
# Poisson example on the same fixed mesh (examples/polydeal_baseline), serial
# with a direct solver: R-tree levels 4 to 7, every cell a polytope, and METIS
# with the same numbers of polytopes.
source $SLURM_SUBMIT_DIR/common.sh
for t in q1p1 q0p1 q1p0 q0p0; do
  # The setting of ch8_singular, on the mesh file instead of the unit square.
  prm=$RUNS/singular_t3_$t.prm
  sed "s#^\(  subsection Problem\)#\1\n    set Mesh file              = $ROOT/meshes/t3.msh\n    set Mesh file refinements  = 4#" \
      $PRM/singular_$t.prm > $prm
  grep -q "Mesh file refinements  = 4" $prm || { echo "sed failed: $prm"; exit 1; }
  run unstructured/singular_t3_$t 32 adaptive_polytopal_timing $prm
done
B=$BUILD/examples
for part in 4 5 6 7 cells metis91 metis364 metis1456 metis5824; do
  run unstructured/baseline_$part 1 polydeal_baseline $ROOT/meshes/t3.msh 4 $part
done
