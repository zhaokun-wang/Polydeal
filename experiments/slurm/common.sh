# Sourced by every job script. Edit the settings below for your installation;
# nothing else in the job scripts needs to change.

# This repository, its build directory, and the build directory of experiments/.
ROOT=$HOME/Polydeal
BUILD=$ROOT/build
EXP=$ROOT/build-experiments
# Where the runs write their output, one directory per run.
RUNS=$HOME/polydeal_runs
# The modules (or environment) that provide MPI, deal.II, Trilinos, p4est and
# ParMETIS, and the installation directories of deal.II and ParMETIS (only
# build.sbatch uses these two).
# module load ...
DEAL_II_DIR=${DEAL_II_DIR:-CHANGE_ME}
PARMETIS_DIR=${PARMETIS_DIR:-CHANGE_ME}

B=$EXP
PRM=$ROOT/experiments/prm
mkdir -p $RUNS

# run <tag> <ranks> <binary> [arguments...]
# Runs one configuration in $RUNS/<tag>, writing log.txt. A run that has
# finished before (its directory holds DONE) is skipped, so a job that was cut
# by its time limit can simply be resubmitted. The ranks are spread over the
# sockets of a node in turn (block:cyclic), so that a run on fewer ranks than
# cores uses the memory bandwidth of both sockets, and each rank is bound to
# one core. With RUN_TIMEOUT=<seconds> set, the run is cut after that time.
run() {
  local tag=$1 np=$2 bin=$3; shift 3
  local dir=$RUNS/$tag
  [ -f $dir/DONE ] && { echo "skip $tag"; return; }
  mkdir -p $dir && cd $dir
  echo "start $(date -Is) host $(hostname) np $np args $*" > log.txt
  ${RUN_TIMEOUT:+timeout $RUN_TIMEOUT} \
    srun -n $np --distribution=block:cyclic --cpu-bind=cores $B/$bin "$@" >> log.txt 2>&1
  local rc=$?
  echo "end $(date -Is) rc=$rc" >> log.txt
  [ $rc -eq 0 ] && touch DONE
  cd $RUNS
}
