// One coarse cell (hyper_cube) refined globally 6 times: 64x64 cells.
// polyDEAL's R-tree agglomeration only; no AMR, no solve.
//
//   ./rstar_tree pack|insert [level]   (level default: ~64 cells per polytope)
#include <deal.II/grid/grid_generator.h>

#include "test.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  Triangulation<2>                 tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(6);
  agglomerate_and_write(tria,
                        argc > 1 && std::string(argv[1]) == "insert" ? TreeBuild::insert : TreeBuild::pack,
                        argc > 2 ? std::atoi(argv[2]) : -1);
}
