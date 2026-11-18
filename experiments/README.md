# Experiments

Job scripts for the parallel experiments of the thesis *Efficient Adaptive
Discontinuous Galerkin Methods over Agglomerated Polytopal Meshes*: the
repartitioning of Chapter 6, the adaptive time stepping of Chapter 7, the
complete pipeline of Chapter 8, and a comparison on an unstructured mesh with
the polyDEAL Poisson example on a fixed mesh. The scripts are written for the
general-purpose partition of MareNostrum 5 and use plain Slurm, so they carry
over to other clusters by editing `slurm/common.sh` and the `#SBATCH` lines.

## Contents

| path | |
|---|---|
| `test_copies/` | Three copies of example programs with extra output (phase timers, per-step statistics). Each states at its top which example it copies and what it adds. |
| `prm/` | Parameter files of `adaptive_polytopal_timing`. `q1`/`q0`: with or without quadrature-free integration; `p1`/`p0`: the copy of the mesh partitioned by ParMETIS, or kept on the ranks of the R-trees. |
| `slurm/` | `common.sh` (settings, to be edited), `build.sbatch`, and one job script per experiment. |
| `CMakeLists.txt` | Builds the three test copies. |

The example `examples/polydeal_baseline.cc` (the polyDEAL Poisson example on a
gmsh mesh with the point singularity) is built with the other examples.

## Requirements

An existing installation of deal.II (9.7 or later) configured with MPI, p4est,
Trilinos (with ML and Ifpack) and METIS, and ParMETIS. Nothing is downloaded or
built besides this repository. The thesis used deal.II at commit `a1ead589`,
Trilinos 14.4, p4est 2.8.7 and ParMETIS 4.0.3. Other versions should work; the
numbers of polytopes and degrees of freedom may then differ slightly, since
the partitions of p4est and METIS and the aggregation of ML can change between
versions.

### ParMETIS

ParMETIS is required: the copy of the mesh partitioned on the graph of the
polytopes calls it directly. Without it, the build of the examples skips
`adaptive_polytopal` and the other examples that need it, with only a status
message, and configuring `experiments/` stops. Look first for an existing
installation (`module avail parmetis`; an installation of PETSc configured
with `--download-parmetis` also contains one) and pass its directory as
`PARMETIS_DIR`. Otherwise ParMETIS builds in a few minutes in a user
directory, for instance version 4.0.3:

```bash
make config cc=mpicc prefix=$HOME/sw/parmetis && make -j4 install
cd metis && make config prefix=$HOME/sw/parmetis && make -j4 install
```

## Build

polyDEAL has to be built from this repository even where another version of it
is installed: the experiments rely on changes to the library itself (hanging
nodes in `source/agglomeration_handler.cc`). The libraries it depends on are
used as installed.

1. Put the repository on the cluster, for instance with
   `git clone -b thesis-experiments https://github.com/zhaokun-wang/Polydeal.git`;
   if the login nodes cannot reach GitHub, clone it elsewhere and copy it over.
2. Edit the settings at the top of `experiments/slurm/common.sh`: the paths,
   the output directory, the modules, and the installation directories of
   deal.II and ParMETIS.
3. Replace `CHANGE_ME` in the `#SBATCH --account` and `#SBATCH --qos` lines of
   the job scripts in `experiments/slurm/`.
4. Build on a compute node, submitting from `experiments/slurm/` (the scripts
   find `common.sh` there):

   ```bash
   cd experiments/slurm
   sbatch build.sbatch
   ```

   The job ends with `build finished` in `build_<jobid>.log`. It runs, with the
   paths of `common.sh`:

   ```bash
   cd $BUILD
   cmake -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR -DPARMETIS_DIR=$PARMETIS_DIR $ROOT
   make polydeal adaptive_polytopal polydeal_baseline
   cd $EXP
   cmake -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR -DPOLYDEAL_BUILD=$BUILD -DPARMETIS_DIR=$PARMETIS_DIR $ROOT/experiments
   make
   ```

Each program is a single large translation unit of about 2.5 GB of memory at
its peak, which is why the build runs as a job rather than on a login node.
The `make` line names its targets, so the build stops with an error if
ParMETIS was not found and `adaptive_polytopal` was skipped, and configuring
`experiments/` stops with a message if deal.II lacks one of the required
features.

## Running

Once the build has finished, submit the experiments from `experiments/slurm/`:

```bash
for f in ch*.sbatch unstructured.sbatch; do sbatch $f; done
```

Every run writes `log.txt` in its own directory under the output directory,
and a finished run leaves a file `DONE` there. A job that is cut by its time
limit can be resubmitted unchanged: the runs that finished are skipped.

| job script | nodes | runs |
|---|---|---|
| `ch6a`, `ch6b` | 1 | Chapter 6: the ParMETIS copy against the copy without ParMETIS, on 2, 4, 8, 16 and 32 ranks |
| `ch7` | 1 | Chapter 7: the moving peak, 32 ranks |
| `ch8_singular` | 1 | Chapter 8: the point singularity, four combinations, 32 ranks |
| `ch8_peaks_q?p?` | 1 | Chapter 8: the moving peak with four peaks per direction, one combination each, 32 ranks |
| `ch8_polynomial` | 1 | Chapter 8: the polynomial solution at three sizes, four combinations, 32 ranks |
| `ch8_3d` | 1 | Chapter 8: polynomial solution and point singularity on the unit cube, four combinations, 32 ranks |
| `ch8_scaling_2`, `_4`, `_8`, `_16` | 2 to 16 | Chapter 8: strong scaling of the moving peak; one node is `ch8_peaks_q1p1` |
| `unstructured` | 1 | The point singularity on `meshes/t3.msh`: the pipeline (four combinations, 32 ranks) and the fixed-mesh baseline (serial) |

### Ranks per node

Every job uses 32 ranks per node on whole nodes (`--exclusive`), although a
node of MareNostrum 5 has 112 cores. The experiments were designed for nodes
of 32 cores: on 16 nodes the moving peak already has too few degrees of
freedom per rank to scale further, and 112 ranks per node would leave even
fewer. The ranks are spread over the two sockets in turn and bound to cores
(`srun --distribution=block:cyclic --cpu-bind=cores`), so that runs on fewer
ranks than cores still use the memory bandwidth of both sockets.

### Time limits

The limits in the job scripts are about twice the times measured on 32-core
Intel Xeon E5-2683 v4 nodes. The longest single runs are the moving peak (about
45 minutes each) and the moving peak on two nodes (about 30 minutes).

## Output

Please send back the whole output directory (`RUNS` in `slurm/common.sh`),
for instance as `tar czf runs.tar.gz -C $(dirname $RUNS) $(basename $RUNS)`,
together with the Slurm log files `*_<jobid>.log` and the versions of the
modules used. The lines that the analysis reads are, in each `log.txt`:

- `[phase]`: the wall-clock time of each phase, cumulative, as
  `name:wall/max/mean` (max over mean is the load imbalance);
- `[count]` and `[cg]`: solver statistics;
- the cycle or step tables printed by the programs.
