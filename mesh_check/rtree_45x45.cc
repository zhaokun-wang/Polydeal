// 45x45 coarse cells (subdivided_hyper_cube) refined globally once: 90x90
// cells. polyDEAL's R-tree agglomeration only; no AMR, no solve.
//
//   ./rtree_45x45 pack|insert [level]   (level default: ~64 cells per polytope)
#include <deal.II/grid/grid_generator.h>

#include "test.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  Triangulation<2>                 tria;
  GridGenerator::subdivided_hyper_cube(tria, 45, 0., 1.);
  tria.refine_global(1);
  agglomerate_and_write(tria,
                        argc > 1 && std::string(argv[1]) == "insert" ? TreeBuild::insert : TreeBuild::pack,
                        argc > 2 ? std::atoi(argv[2]) : -1);
}
