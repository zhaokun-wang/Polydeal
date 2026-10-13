// Shared by rtree_45x45.cc and rstar_tree.cc. Needs only deal.II and polyDEAL
// (github.com/fdrmrc/Polydeal).
//
// Either driver: ./<driver> pack|insert [level]
//
// Builds polyDEAL's own R-tree agglomeration on a given mesh, exactly as
// examples/poisson.cc does it (pack_rtree<rstar<2^dim>> over the cells'
// bounding boxes, CellsAgglomerator::extract_agglomerates at one level,
// AgglomerationHandler::define_agglomerate, distribute_agglomerated_dofs),
// and writes, <name> being pack or insert:
//
//   <name>.svg   deal.II's GridOut::write_svg, cells coloured by material_id
//                (= scrambled polytope index)
//   <name>.vtu   for ParaView: cell data polytope, polytope_colour and
//                polytope_cells
//
// The polytope of each cell is read back from the handler
// (polytope->get_agglomerate()), not from the agglomerator's output, so the
// pictures show what the library holds.

#include <deal.II/base/mpi.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/rtree.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <cmath>
#include <fstream>
#include <iostream>

using namespace dealii;
namespace bgi = boost::geometry::index;

// How the R-tree is filled.
//   pack:   pack_rtree, polyDEAL's bulk loading (Boost's packing algorithm;
//           the rstar<4> tag only sets the node capacity, the R* insertion
//           heuristics never run).
//   insert: the same tree type filled by inserting the cells one at a time
//           in active-cell order, so the R* split/reinsert algorithm builds it.
enum class TreeBuild
{
  pack,
  insert
};

// Writes pack.svg/.vtu or insert.svg/.vtu into the current directory; run
// each driver in a directory of its own. level < 0: the level whose mean
// agglomerate is closest to 64 cells.
inline void
agglomerate_and_write(Triangulation<2> &tria, const TreeBuild build, const int requested_level = -1)
{
  constexpr int     dim  = 2;
  const std::string name = build == TreeBuild::insert ? "insert" : "pack";
  MappingQ1<dim>    mapping;

  std::vector<std::pair<BoundingBox<dim>, Triangulation<dim>::active_cell_iterator>> boxes;
  for (const auto &cell : tria.active_cell_iterators())
    boxes.emplace_back(mapping.get_bounding_box(cell), cell);
  using Tag  = bgi::rstar<PolyUtils::constexpr_pow(2, dim)>;
  using Tree = decltype(pack_rtree<Tag>(boxes));
  Tree tree;
  if (build == TreeBuild::insert)
    for (const auto &box : boxes)
      tree.insert(box);
  else
    tree = pack_rtree<Tag>(boxes);

  std::cout << name << ": " << tria.n_active_cells()
            << " cells, R-tree levels " << n_levels(tree) << std::endl;
  unsigned int level     = 0;
  double       best_miss = 1e300;
  for (unsigned int l = 1; l <= n_levels(tree); ++l)
    {
      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree, l};
      const double mean = double(tria.n_active_cells()) / agglomerator.extract_agglomerates().size();
      std::cout << "  level " << l << ": mean " << mean << " cells per agglomerate" << std::endl;
      if (std::abs(std::log(mean / 64.)) < best_miss)
        {
          best_miss = std::abs(std::log(mean / 64.));
          level     = l;
        }
    }
  if (requested_level >= 0)
    level = requested_level;
  std::cout << "  using level " << level << std::endl;

  GridTools::Cache<dim>     cached_tria(tria, mapping);
  AgglomerationHandler<dim> ah(cached_tria);
  CellsAgglomerator<dim, decltype(tree)> agglomerator{tree, level};
  for (const auto &agglomerate : agglomerator.extract_agglomerates())
    ah.define_agglomerate(agglomerate);
  FE_AggloDGP<dim> fe(1);
  ah.distribute_agglomerated_dofs(fe);

  Vector<float> polytope(tria.n_active_cells()), colour(tria.n_active_cells()), size(tria.n_active_cells());
  polytope = -1.f;
  for (const auto &p : ah.polytope_iterators())
    for (const auto &cell : p->get_agglomerate())
      {
        const unsigned int i = cell->active_cell_index();
        polytope[i]          = float(p->index());
        colour[i]            = float((p->index() * 7919ul) % 64ul);
        size[i]              = float(p->get_agglomerate().size());
      }
  unsigned int min_size = tria.n_active_cells(), max_size = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      AssertThrow(polytope[cell->active_cell_index()] >= 0, ExcMessage("cell in no polytope"));
      min_size = std::min<unsigned int>(min_size, size[cell->active_cell_index()]);
      max_size = std::max<unsigned int>(max_size, size[cell->active_cell_index()]);
    }
  std::cout << "  handler: " << ah.n_agglomerates() << " polytopes, " << min_size << "-" << max_size
            << " cells each" << std::endl;

  // deal.II's SVG writer, coloured by material_id.
  for (const auto &cell : tria.active_cell_iterators())
    cell->set_material_id(types::material_id(colour[cell->active_cell_index()]));
  {
    GridOut           grid_out;
    GridOutFlags::Svg flags;
    flags.coloring                = GridOutFlags::Svg::material_id;
    flags.line_thickness          = 0;
    flags.boundary_line_thickness = 1;
    flags.background              = GridOutFlags::Svg::Background::white;
    flags.draw_legend             = false;
    flags.draw_colorbar           = false;
    grid_out.set_flags(flags);
    std::ofstream out(name + ".svg");
    grid_out.write_svg(tria, out);
  }

  {
    DataOut<dim> data_out;
    data_out.attach_triangulation(tria);
    data_out.add_data_vector(polytope, "polytope", DataOut<dim>::type_cell_data);
    data_out.add_data_vector(colour, "polytope_colour", DataOut<dim>::type_cell_data);
    data_out.add_data_vector(size, "polytope_cells", DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    std::ofstream out(name + ".vtu");
    data_out.write_vtu(out);
  }
}
