// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------


#ifndef polytope_adaptivity_h
#define polytope_adaptivity_h


#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/cell_data_transfer.h>
#include <deal.II/distributed/p4est_wrappers.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/adaptation_strategies.h>
#include <deal.II/numerics/rtree.h>

#include <boost/geometry/index/rtree.hpp>

#include <Teuchos_ParameterList.hpp>
#include <agglomeration_handler.h>
#include <ml_MultiLevelPreconditioner.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace dealii::PolytopeAdaptivity
{
  /**
   * By how many levels a polytope may be refined or coarsened in one call of
   * AdaptivePolytopalMesh::refine_and_coarsen(), that is, in one cycle. A
   * level is one step down or up the R-tree, or, below a single initial cell,
   * one refinement or coarsening of a cell by p4est.
   */
  enum class LevelRule
  {
    /**
     * One level: a polytope over the tolerance is refined one level down the
     * R-tree (a polytope of one cell is refined once in p4est), and siblings
     * are merged one level up.
     */
    single_level,
    /**
     * As many levels as the error indicator predicts, see predicted_levels().
     * This is the metric-based adaptation of Africa, de Falco and Perotto
     * (2023, Alg. 2). The paper refines and coarsens the elements of a mesh;
     * here a level is a step in the R-tree or a refinement level of p4est.
     */
    multiple_levels
  };



  /**
   * Which rank builds the R-tree over an initial cell.
   */
  enum class InitialPartition
  {
    /**
     * The rank that owns the initial cell on the p4est triangulation. The
     * initial cells of a rank are a stretch of the space-filling curve of
     * p4est. On a mesh of $2^n \times 2^n$ cells and a power of two ranks it
     * is a rectangle, otherwise in general a staircase, along which the
     * R-tree cuts long, thin polytopes.
     */
    p4est,
    /**
     * The rank that METIS assigns the initial cell to, by
     * GridTools::partition_triangulation() on the graph of the faces between
     * the initial cells. A polytope may then hold initial cells of several
     * ranks of the p4est triangulation, and the system must be solved on a
     * copy of the mesh on which every polytope lies on one rank, see
     * PolytopeRepartition::PartitionedCopy.
     */
    metis
  };



  /**
   * Parameters of AdaptivePolytopalMesh.
   */
  struct AdaptivityParameters
  {
    /**
     * The largest number of initial cells in a polytope. It bounds the
     * initial polytopes and every merge.
     */
    unsigned int max_cells_per_polytope = 64;

    /**
     * For LevelRule::single_level: polytopes are merged, and active descendants
     * coarsened, when their error indicator is at or under this fraction of the
     * polytope tolerance. A merge of $2^{d}$ children raises the error
     * indicator by about $2^{d/2} 2^{p}$ for a smooth solution, so the fraction
     * must stay well under $2^{-(d/2 + p)}$. LevelRule::multiple_levels does
     * not use it, see coarsening_damping.
     */
    double coarsening_fraction = 0.02;

    /**
     * One level per call, or as many as the error indicator predicts.
     */
    LevelRule level_rule = LevelRule::multiple_levels;

    /**
     * For LevelRule::multiple_levels: the assumed reduction rate of the error
     * indicator per level, $\eta \sim 2^{-\text{rate}\,\ell}$. Zero selects
     * $p + d/2$, the rate of a smooth solution with elements of degree $p$; the
     * rate $d/2$ of the paper overestimates the levels for higher degrees.
     */
    double level_rate = 0.;

    /**
     * For LevelRule::multiple_levels: the predicted number of refinement
     * levels is reduced by this many.
     */
    unsigned int refinement_damping = 0;

    /**
     * For LevelRule::multiple_levels: the predicted number of coarsening levels
     * is reduced by this many. A damping of one keeps a polytope from being
     * refined and merged back in alternate calls: each child of a polytope
     * may predict one level of coarsening,
     * while their parent, holding the error indicator of all of them, is over
     * the tolerance again. With one, children are merged when each of them is
     * at or under $2^{-2\,\text{rate}}$ times the tolerance.
     */
    unsigned int coarsening_damping = 1;

    /**
     * Which rank builds the R-tree over an initial cell.
     */
    InitialPartition initial_partition = InitialPartition::p4est;
  };



  /**
   * For LevelRule::multiple_levels: the number of levels by which a polytope
   * is to be refined (positive) or coarsened (negative). The polytope has the
   * error indicator @p error, the polytope tolerance is @p tolerance and the
   * degree of the elements is @p degree. This number of levels is
   * $\ell = \log_2(\eta / \text{tol}) / \text{rate}$, rounded up for
   * refinement, so that any error indicator over the tolerance refines at
   * least once, and towards zero for coarsening. It is then reduced by the
   * damping of @p parameters.
   */
  template <int dim>
  int
  predicted_levels(const double                error,
                   const double                tolerance,
                   const unsigned int          degree,
                   const AdaptivityParameters &parameters)
  {
    Assert(tolerance > 0., ExcMessage("The tolerance must be positive."));
    // The rate at which the error indicator falls per level.
    const double rate = (parameters.level_rate == 0.) ? degree + 0.5 * dim :
                                                        parameters.level_rate;
    // x is the number of levels as a real number. It is rounded as above and
    // capped at max_levels. A zero error indicator gives the largest
    // coarsening.
    constexpr int max_levels = 64;
    int           levels     = -max_levels;
    if (error > 0.)
      {
        const double exact_levels = std::log2(error / tolerance) / rate;
        levels                    = (exact_levels > 0.) ?
                                      static_cast<int>(
                     std::ceil(std::min(exact_levels, double(max_levels)))) :
                                      -static_cast<int>(
                     std::floor(std::min(-exact_levels, double(max_levels))));
      }
    // The damping takes levels off in either direction, never past zero.
    if (levels >= 0)
      return std::max(0,
                      levels - static_cast<int>(parameters.refinement_damping));
    else
      return std::min(0,
                      levels + static_cast<int>(parameters.coarsening_damping));
  }



  /**
   * Split every group of cells into its face-connected components. The
   * cells of a polytope of an R-tree are close in space, but need not be
   * connected. A breadth-first search from every cell not yet reached follows
   * the faces between cells of the same group. Neighbours across a hanging
   * face are not followed: a group of more than one cell consists of initial
   * cells that are not refined, whose finer neighbours are polytopes of
   * their own.
   */
  template <int dim>
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
  split_into_connected_components(
    const std::vector<
      std::vector<typename Triangulation<dim>::active_cell_iterator>> &groups)
  {
    using CellIterator = typename Triangulation<dim>::active_cell_iterator;
    std::vector<std::vector<CellIterator>> components;
    components.reserve(groups.size());

    for (const auto &group : groups)
      {
        // A group of one cell is connected.
        if (group.size() <= 1)
          {
            components.push_back(group);
            continue;
          }

        // The cells of the group by their active cell index, the cells
        // reached so far, and a function that queues a neighbour if it
        // belongs to the group and has not been reached.
        std::unordered_map<unsigned int, CellIterator> cell_of_index;
        for (const auto &cell : group)
          cell_of_index[cell->active_cell_index()] = cell;

        std::unordered_set<unsigned int> visited;
        const auto                       visit =
          [&](const typename Triangulation<dim>::cell_iterator &neighbor,
              std::queue<CellIterator>                         &queue) {
            const unsigned int index = neighbor->active_cell_index();
            if (cell_of_index.count(index) > 0 && visited.count(index) == 0)
              {
                visited.insert(index);
                queue.push(cell_of_index.at(index));
              }
          };

        // Every breadth-first search from a cell not yet reached gives one
        // component.
        for (const auto &start : group)
          {
            if (visited.count(start->active_cell_index()) > 0)
              continue;
            std::vector<CellIterator> component;
            std::queue<CellIterator>  queue;
            visited.insert(start->active_cell_index());
            queue.push(start);
            while (!queue.empty())
              {
                const CellIterator cell = queue.front();
                queue.pop();
                component.push_back(cell);
                for (const unsigned int f : cell->face_indices())
                  {
                    if (!cell->at_boundary(f) && cell->neighbor(f)->is_active())
                      visit(cell->neighbor(f), queue);
                  }
              }
            components.push_back(std::move(component));
          }
      }
    return components;
  }



  namespace internal
  {
    /**
     * A polytope of an R-tree over the initial cells: its parent, its
     * children, and its initial cells, which are one range of
     * PolytopeTree::order.
     */
    struct TreePolytope
    {
      /**
       * Index of the parent, -1 for the root.
       */
      int parent = -1;

      /**
       * Indices of the children, none for a polytope of one initial cell.
       */
      std::vector<int> children;

      /**
       * The initial cells of the polytope are PolytopeTree::order[begin],
       * ..., PolytopeTree::order[end - 1].
       */
      unsigned int begin = 0;
      unsigned int end   = 0;

      /**
       * Number of initial cells of the polytope.
       */
      unsigned int
      size() const
      {
        return end - begin;
      }
    };



    /**
     * Copies a Boost R-tree into PolytopeTree::polytopes. Every node of the
     * Boost tree becomes a polytope. A leaf of the Boost tree holds several
     * initial cells, and below it each of them becomes a polytope of its own,
     * so that the tree can be refined down to single initial cells. The tree
     * is visited depth first: a parent is recorded before its children, and
     * the initial cells of every polytope are one contiguous range of
     * @p order.
     */
    template <typename Tree, typename Value>
    struct RTreeRecordVisitor
      : public boost::geometry::index::detail::rtree::visitor<
          Value,
          typename boost::geometry::index::detail::rtree::utilities::view<
            Tree>::options_type::parameters_type,
          typename boost::geometry::index::detail::rtree::utilities::view<
            Tree>::box_type,
          typename boost::geometry::index::detail::rtree::utilities::view<
            Tree>::allocators_type,
          typename boost::geometry::index::detail::rtree::utilities::view<
            Tree>::options_type::node_tag,
          true>::type
    {
      using View = boost::geometry::index::detail::rtree::utilities::view<Tree>;
      using InternalNode =
        typename boost::geometry::index::detail::rtree::internal_node<
          Value,
          typename View::options_type::parameters_type,
          typename View::box_type,
          typename View::allocators_type,
          typename View::options_type::node_tag>::type;
      using Leaf = typename boost::geometry::index::detail::rtree::leaf<
        Value,
        typename View::options_type::parameters_type,
        typename View::box_type,
        typename View::allocators_type,
        typename View::options_type::node_tag>::type;

      /**
       * Record the tree into @p polytopes and @p order.
       */
      RTreeRecordVisitor(std::vector<TreePolytope> &polytopes,
                         std::vector<unsigned int> &order)
        : polytopes(polytopes)
        , order(order)
      {}

      /**
       * Append a polytope below the current parent and return its index.
       */
      int
      open_polytope()
      {
        const int polytope = polytopes.size();
        polytopes.emplace_back();
        polytopes[polytope].parent = parent;
        polytopes[polytope].begin  = order.size();
        if (parent >= 0)
          polytopes[parent].children.push_back(polytope);
        return polytope;
      }

      /**
       * An internal node of the Boost tree: append a polytope, record the
       * subtrees below it as its descendants, and set the end of its range
       * of initial cells.
       */
      void
      operator()(const InternalNode &internal_node)
      {
        const int polytope = open_polytope(), outer = parent;
        parent = polytope;
        for (const auto &child :
             boost::geometry::index::detail::rtree::elements(internal_node))
          boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                               *child.second);
        parent                  = outer;
        polytopes[polytope].end = order.size();
      }

      /**
       * A leaf of the Boost tree: append a polytope and, below it, a polytope
       * for each initial cell of the leaf.
       */
      void
      operator()(const Leaf &leaf)
      {
        const int polytope = open_polytope(), outer = parent;
        parent = polytope;
        for (const auto &value :
             boost::geometry::index::detail::rtree::elements(leaf))
          {
            const int single = open_polytope();
            order.push_back(value.second);
            polytopes[single].end = order.size();
          }
        parent                  = outer;
        polytopes[polytope].end = order.size();
      }

      // Where the tree is recorded, and the polytope whose children are being
      // recorded.
      std::vector<TreePolytope> &polytopes;
      std::vector<unsigned int> &order;
      int                        parent = -1;
    };



    /**
     * An R-tree over the initial cells of one rank, built once. Every node of
     * the tree is a polytope. The current polytopes are the polytopes the mesh
     * consists of at present: every initial cell lies below exactly one of
     * them.
     *
     * The tree refers to an initial cell by its tree index, its position
     * 0, 1, ..., n-1 in the list given to build(). All vectors over initial
     * cells, such as current_polytope_of, are indexed by it.
     */
    struct PolytopeTree
    {
      /**
       * Build the tree from @p boxes, the bounding box of every initial cell
       * paired with its tree index, and set the current polytopes at the
       * start: down every branch, the first polytope of at most @p max_cells
       * initial cells.
       */
      template <int dim>
      void
      build(const std::vector<std::pair<BoundingBox<dim>, unsigned int>> &boxes,
            const unsigned int max_cells)
      {
        polytopes.clear();
        order.clear();
        current_polytope_of.assign(boxes.size(), -1);
        if (boxes.empty())
          return;

        // Pack the Boost R-tree and record it as polytopes, depth first.
        auto tree   = pack_rtree<boost::geometry::index::rstar<
          GeometryInfo<dim>::max_children_per_cell>>(boxes);
        using Tree  = decltype(tree);
        using Value = std::pair<BoundingBox<dim>, unsigned int>;
        boost::geometry::index::detail::rtree::utilities::view<Tree> view(tree);
        RTreeRecordVisitor<Tree, Value> visitor(polytopes, order);
        view.apply_visitor(visitor);
        AssertThrow(order.size() == boxes.size(), ExcInternalError());

        // The depth of every polytope. A parent is recorded before its
        // children, so a single loop in this order suffices.
        depth.assign(polytopes.size(), 0);
        for (unsigned int polytope = 0; polytope < polytopes.size(); ++polytope)
          if (polytopes[polytope].parent >= 0)
            depth[polytope] = depth[polytopes[polytope].parent] + 1;
        // The highest polytope of one initial cell: depth first, it is met
        // before the ones below it.
        singleton.assign(order.size(), -1);
        for (unsigned int polytope = 0; polytope < polytopes.size(); ++polytope)
          if (polytopes[polytope].size() == 1 &&
              singleton[order[polytopes[polytope].begin]] < 0)
            singleton[order[polytopes[polytope].begin]] = polytope;

        // The current polytopes at the start.
        set_initial_current_polytopes(0, max_cells);
      }

      /**
       * Number of initial cells of the current polytope that holds the initial
       * cell with tree index @p initial.
       */
      unsigned int
      current_polytope_size(const unsigned int initial) const
      {
        return polytopes[current_polytope_of[initial]].size();
      }

      /**
       * One step of LevelRule::single_level. A current polytope of two or more
       * cells is replaced by its children if its error indicator is at or over
       * @p tolerance or it holds a refined initial cell. Then children are
       * merged into their parent if all of them are current, none is refined in
       * this step, every child's error indicator is at or under
       * @p coarsening_line, and the parent holds at most @p max_cells cells,
       * none of them refined. The error indicator of a polytope is the root of
       * the sum of its cells' squared error indicators. @p error_sq and
       * @p is_refined are indexed by tree index.
       */
      void
      refine_and_merge_single_level(const std::vector<double>       &error_sq,
                                    const std::vector<unsigned int> &is_refined,
                                    const double                     tolerance,
                                    const double       coarsening_line,
                                    const unsigned int max_cells,
                                    const bool         allow_refinement,
                                    const bool         allow_merge)
      {
        // The error indicator of every current polytope, and the number of
        // refined initial cells below every polytope.
        const std::vector<double> polytope_error_sq =
          compute_polytope_error_sq(error_sq);
        const std::vector<unsigned int> n_refined = count_refined(is_refined);

        const auto polytope_error = [&](const int polytope) {
          return std::sqrt(polytope_error_sq[polytope]);
        };

        // The current polytopes before this step.
        std::vector<char> is_current(polytopes.size(), 0);
        for (const int polytope : current_polytope_of)
          is_current[polytope] = 1;

        // Refinement: a current polytope is replaced by its children.
        std::vector<char> refined_now(polytopes.size(), 0);
        if (allow_refinement)
          for (unsigned int polytope = 0; polytope < polytopes.size();
               ++polytope)
            if (is_current[polytope] && polytopes[polytope].size() >= 2 &&
                (polytope_error(polytope) >= tolerance ||
                 n_refined[polytope] > 0))
              {
                for (const int child : polytopes[polytope].children)
                  make_current(child);
                refined_now[polytope] = 1;
                is_current[polytope]  = 0;
              }
        if (!allow_merge)
          return;

        // Coarsening: current polytopes that are all the children of one
        // parent are replaced by the parent. Refinement runs before, so a
        // parent merged here is not refined in this step, and is_current
        // holds the current polytopes from before this step, so it is not
        // merged further up either.
        for (unsigned int parent = 0; parent < polytopes.size(); ++parent)
          {
            // The parent must not be current, nor refined in this step, and
            // must hold at most max_cells cells, none of them refined.
            if (polytopes[parent].children.empty() || is_current[parent] ||
                refined_now[parent] || polytopes[parent].size() > max_cells ||
                n_refined[parent] > 0)
              continue;
            // Every child must be current, not refined in this step, and at
            // or under the coarsening line.
            bool mergeable = true;
            for (const int child : polytopes[parent].children)
              if (!is_current[child] || refined_now[child] ||
                  polytope_error(child) > coarsening_line)
                mergeable = false;
            if (mergeable)
              make_current(parent);
          }
      }

      /**
       * One step of LevelRule::multiple_levels; @p levels(error) is the number
       * of levels a polytope with this error indicator is to be refined
       * (positive) or coarsened (negative) by. A current polytope of two or
       * more cells predicting $\ell > 0$ levels is replaced by its
       * descendants $\ell$ levels down (a branch stops at a single cell); one
       * holding a refined initial cell is refined on until no polytope of two
       * or more cells holds one. Then the current polytopes below an ancestor
       * are merged into it if none was refined in this step, every one of
       * them predicts at least as many coarsening levels as it lies below the
       * ancestor, and the ancestor holds at most @p max_cells cells, none of
       * them refined; of nested ancestors the highest is taken.
       */
      template <typename LevelFunction>
      void
      refine_and_merge_multiple_levels(
        const std::vector<double>       &error_sq,
        const std::vector<unsigned int> &is_refined,
        const LevelFunction             &levels,
        const unsigned int               max_cells,
        const bool                       allow_refinement,
        const bool                       allow_merge)
      {
        // The error indicator of every current polytope, and the number of
        // refined initial cells below every polytope.
        const std::vector<double> polytope_error_sq =
          compute_polytope_error_sq(error_sq);
        const std::vector<unsigned int> n_refined = count_refined(is_refined);

        // The current polytopes before this step, and their predicted levels.
        std::vector<char> is_current(polytopes.size(), 0);
        for (const int polytope : current_polytope_of)
          is_current[polytope] = 1;
        std::vector<int> levels_of(polytopes.size(), 0);
        for (unsigned int polytope = 0; polytope < polytopes.size(); ++polytope)
          if (is_current[polytope])
            levels_of[polytope] =
              levels(std::sqrt(polytope_error_sq[polytope]));

        // Refinement: a current polytope is replaced by its descendants
        // @p remaining levels down. A child of two or more cells goes on down
        // while levels remain, or while it holds a refined initial cell.
        // Refinement runs before coarsening, so a polytope merged in this step
        // is not refined in it.
        if (allow_refinement)
          {
            const std::function<void(int, int)> refine_down =
              [&](const int polytope, const int remaining) {
                for (const int child : polytopes[polytope].children)
                  make_current(child);
                for (const int child : polytopes[polytope].children)
                  if (polytopes[child].size() >= 2 &&
                      (remaining > 1 || n_refined[child] > 0))
                    refine_down(child, remaining - 1);
              };
            for (unsigned int polytope = 0; polytope < polytopes.size();
                 ++polytope)
              if (is_current[polytope] && polytopes[polytope].size() >= 2 &&
                  (levels_of[polytope] >= 1 || n_refined[polytope] > 0))
                refine_down(polytope, levels_of[polytope]);
          }
        if (!allow_merge)
          return;

        // Coarsening: the current polytopes below an ancestor are replaced by
        // the ancestor. The ancestor must hold at most max_cells cells, none of
        // them refined. Every initial cell below it must lie in a current
        // polytope strictly below the ancestor that predicts at least as many
        // coarsening levels as it lies below the ancestor. A polytope made
        // current by refinement in this step was not current before it, so
        // levels_of holds zero for it and it is never merged in this step.
        // Ancestors are visited root first. Once an ancestor is merged, its
        // initial cells lie in a polytope above every lower ancestor, so of
        // nested candidates only the highest is merged.
        for (unsigned int ancestor = 0; ancestor < polytopes.size(); ++ancestor)
          {
            if (polytopes[ancestor].children.empty() ||
                polytopes[ancestor].size() > max_cells ||
                n_refined[ancestor] > 0)
              continue;
            bool mergeable = true;
            for (unsigned int position = polytopes[ancestor].begin;
                 position < polytopes[ancestor].end && mergeable;
                 ++position)
              {
                const unsigned int initial  = order[position];
                const int          polytope = current_polytope_of[initial];
                if (depth[polytope] <= depth[ancestor] ||
                    -levels_of[polytope] < depth[polytope] - depth[ancestor])
                  mergeable = false;
              }
            if (mergeable)
              make_current(ancestor);
          }
      }

      /**
       * For LevelRule::multiple_levels: the number of levels by which every
       * initial cell is to be refined in this step.
       *
       * A current polytope of two or more cells that predicts $\ell$ levels
       * of refinement is replaced by its descendants $\ell$ levels down. The
       * tree ends where a polytope holds a single initial cell. If that
       * polytope is $d < \ell$ levels below the current polytope, the
       * remaining $\ell - d$ levels can only come from refining the initial
       * cell in p4est. This function returns $\max(0, \ell - d)$ for every
       * initial cell, so that the cell is flagged in the same step. It
       * returns zero for the cells of current polytopes of one cell, which
       * are flagged by their own error indicator.
       */
      template <typename LevelFunction>
      std::vector<int>
      cell_refinement_levels(const std::vector<double> &error_sq,
                             const LevelFunction       &levels) const
      {
        const std::vector<double> polytope_error_sq =
          compute_polytope_error_sq(error_sq);
        // Once for every current polytope of two or more cells: its predicted
        // levels minus the depth from it down to the single-cell polytope of
        // each of its initial cells.
        std::vector<int>  remaining_levels(current_polytope_of.size(), 0);
        std::vector<char> done(polytopes.size(), 0);
        for (const int polytope : current_polytope_of)
          {
            if (done[polytope] || polytopes[polytope].size() < 2)
              continue;
            done[polytope]     = 1;
            const int n_levels = levels(std::sqrt(polytope_error_sq[polytope]));
            for (unsigned int position = polytopes[polytope].begin;
                 position < polytopes[polytope].end;
                 ++position)
              remaining_levels[order[position]] =
                std::max(0,
                         n_levels - (depth[singleton[order[position]]] -
                                     depth[polytope]));
          }
        return remaining_levels;
      }

      /**
       * Every polytope of the R-tree, the root first and every parent before
       * its children. Below every leaf of the Boost tree there is one
       * polytope per initial cell, so that every initial cell is a polytope
       * of its own.
       */
      std::vector<TreePolytope> polytopes;

      /**
       * The initial cells in depth-first order. The initial cells of polytope
       * n are order[polytopes[n].begin], ..., order[polytopes[n].end - 1].
       */
      std::vector<unsigned int> order;

      /**
       * For every tree index, the current polytope that holds this initial
       * cell. The initial cells with the same entry form one current
       * polytope.
       */
      std::vector<int> current_polytope_of;

      /**
       * For every polytope, its distance from the root.
       */
      std::vector<int> depth;

      /**
       * For every tree index, the highest polytope that holds this initial
       * cell alone.
       */
      std::vector<int> singleton;

    private:
      /**
       * Make current, down every branch below @p polytope, the first polytope
       * of at most @p max_cells initial cells.
       */
      void
      set_initial_current_polytopes(const int          polytope,
                                    const unsigned int max_cells)
      {
        if (polytopes[polytope].size() <= max_cells ||
            polytopes[polytope].children.empty())
          make_current(polytope);
        else
          for (const int child : polytopes[polytope].children)
            set_initial_current_polytopes(child, max_cells);
      }

      /**
       * Make @p polytope current: every initial cell below it now belongs to
       * it. The current polytopes that held these cells before are current
       * no longer. Refinement calls this for every child of a current
       * polytope, which replaces the polytope by its children. Coarsening
       * calls it for a parent or an ancestor, which replaces the current
       * polytopes below it by the one polytope.
       */
      void
      make_current(const int polytope)
      {
        for (unsigned int position = polytopes[polytope].begin;
             position < polytopes[polytope].end;
             ++position)
          current_polytope_of[order[position]] = polytope;
      }

      /**
       * Squared error indicator of every current polytope, the sum over its
       * initial cells. Zero for the other polytopes.
       */
      std::vector<double>
      compute_polytope_error_sq(const std::vector<double> &error_sq) const
      {
        std::vector<double> polytope_error_sq(polytopes.size(), 0.);
        for (unsigned int i = 0; i < current_polytope_of.size(); ++i)
          polytope_error_sq[current_polytope_of[i]] += error_sq[i];
        return polytope_error_sq;
      }

      /**
       * Number of refined initial cells below every polytope. It is computed
       * bottom up, which works because children have larger indices than
       * their parent.
       */
      std::vector<unsigned int>
      count_refined(const std::vector<unsigned int> &is_refined) const
      {
        std::vector<unsigned int> n_refined(polytopes.size(), 0);
        for (int polytope = static_cast<int>(polytopes.size()) - 1;
             polytope >= 0;
             --polytope)
          if (polytopes[polytope].children.empty())
            for (unsigned int position = polytopes[polytope].begin;
                 position < polytopes[polytope].end;
                 ++position)
              n_refined[polytope] += is_refined[order[position]];
          else
            for (const int child : polytopes[polytope].children)
              n_refined[polytope] += n_refined[child];
        return n_refined;
      }
    };
  } // namespace internal



  /**
   * An adaptive mesh of polytopes on a p4est triangulation, refined and
   * coarsened in both directions.
   *
   * The initial cells are the coarse cells of the triangulation, the cells of
   * level zero. Every rank builds one R-tree over its initial cells, and every
   * node of the tree is a polytope. At the start the current polytopes are,
   * down every branch, the first polytope of at most
   * AdaptivityParameters::max_cells_per_polytope initial cells. Polytopes
   * are refined down the tree and merged back up it. Below a single initial
   * cell, p4est refines and coarsens the cell, and every active descendant
   * of a refined initial cell is a polytope of its own.
   *
   * The triangulation must be created with the setting
   * parallel::distributed::Triangulation::no_automatic_repartitioning: cells
   * then never change their rank.
   *
   * The initial cells of a rank's tree are, by
   * AdaptivityParameters::initial_partition, the ones it owns on the p4est
   * triangulation, or the ones METIS assigns to it. In the first case every
   * polytope lies on one rank and all decisions are local. In the second, the
   * rank that holds an initial cell in its tree, its tree rank, is in general
   * not the rank that owns it on the p4est triangulation, its p4est rank.
   * Both stay fixed, so every rank exchanges data with the same ranks about
   * the same initial cells in every adaptation: the p4est rank sends the data
   * the tree needs, the error indicator of the initial cell and whether p4est
   * has refined it, and the tree rank sends back the decisions of the tree,
   * the polytope of the initial cell and the levels by which p4est is to
   * refine it.
   *
   * Every polytope has a global polytope index. Rank r holds the indices
   * from get_polytope_index_offsets()[r] to get_polytope_index_offsets()[r+1]
   * - 1: first the active descendants of the refined initial cells it owns on
   * the p4est triangulation, then the polytopes of its tree.
   *
   * Marking is against a tolerance for the error estimator, shared equally
   * by the $N$ polytopes: $\text{tol}_P = \text{TOL} / \sqrt{N}$. When every
   * polytope error indicator is at or under $\text{tol}_P$, the total is at or
   * under TOL.
   */
  template <int dim>
  class AdaptivePolytopalMesh
  {
  public:
    using CellIterator = typename Triangulation<dim>::active_cell_iterator;

    /**
     * Constructor. @p fe_degree is the degree of the elements on the
     * polytopes, used by LevelRule::multiple_levels.
     */
    AdaptivePolytopalMesh(
      parallel::distributed::Triangulation<dim> &tria,
      const unsigned int                         fe_degree,
      const AdaptivityParameters &parameters = AdaptivityParameters());

    /**
     * Build the R-tree over the initial cells of this rank, and the current
     * polytopes at the start. The triangulation must not be refined yet.
     */
    void
    initialize();

    /**
     * The polytopes of this rank, each a face-connected list of locally
     * owned active cells. With InitialPartition::metis a polytope may hold
     * cells of several ranks, and every rank then lists the part it owns,
     * possibly in several face-connected pieces. The pieces of one polytope
     * have the same global polytope index, see get_polytope_indices().
     */
    const std::vector<std::vector<CellIterator>> &
    get_polytopes() const;

    /**
     * The global polytope index of every polytope, or piece of a polytope, of
     * get_polytopes().
     */
    const std::vector<types::global_cell_index> &
    get_polytope_indices() const;

    /**
     * For every rank r, the first global polytope index it holds, and at the
     * end the number of polytopes over all ranks.
     */
    const std::vector<types::global_cell_index> &
    get_polytope_index_offsets() const;

    /**
     * The tree rank of every polytope, or piece of a polytope, of
     * get_polytopes(): the rank whose tree holds the initial cells of the
     * polytope, or the initial cell an active descendant lies in. With
     * InitialPartition::p4est it is this rank.
     */
    std::vector<unsigned int>
    get_tree_ranks() const;

    /**
     * Number of polytopes over all ranks.
     */
    types::global_cell_index
    n_global_polytopes() const;

    /**
     * The polytope tolerance $\text{TOL} / \sqrt{N}$ for the total tolerance
     * @p tolerance over the current polytopes.
     */
    double
    polytope_tolerance(const double tolerance) const;

    /**
     * Refine and coarsen the polytopes against the polytope tolerance
     * @p tolerance (for a total tolerance TOL, polytope_tolerance(TOL)), from
     * the error indicators @p cell_indicators of the locally owned active
     * cells; the error indicator of a polytope is the root of the sum of its
     * cells' squared error indicators.
     *
     * A polytope over the tolerance is refined down the tree, a one-cell
     * polytope over it is refined by p4est. Siblings are merged, and the
     * active descendants of a refined cell coarsened, when all of them are
     * far enough under the tolerance: with LevelRule::single_level at or
     * under AdaptivityParameters::coarsening_fraction times the tolerance,
     * with LevelRule::multiple_levels when each of them predicts, after
     * AdaptivityParameters::coarsening_damping, at least as many levels of
     * coarsening as the merge goes up.
     *
     * Under LevelRule::multiple_levels a cell may have to be refined or
     * coarsened by several levels, while execute_coarsening_and_refinement()
     * changes every cell by one level. The function is therefore called once
     * per level, as often as the largest number of levels any cell on any
     * rank needs, and the levels a cell still owes are carried to its new
     * cells. Under LevelRule::single_level it is called once. The tree
     * changes once, after the last call.
     *
     * The caller can carry its own data through every call: @p prepare is
     * called after the flags are set and before
     * execute_coarsening_and_refinement(), the place for
     * CellDataTransfer::prepare_for_coarsening_and_refinement(), and
     * @p unpack after it, for CellDataTransfer::unpack().
     *
     * @p allow_refinement and @p allow_coarsening restrict the adaptation to
     * one direction. Return false, and change nothing, if a cell flagged by
     * its error indicator is at the finest level p4est allows. This is checked
     * before the first call only: a level owed beyond the finest level in a
     * later call is dropped.
     */
    bool
    refine_and_coarsen(const Vector<float>         &cell_indicators,
                       const double                 tolerance,
                       const std::function<void()> &prepare          = {},
                       const std::function<void()> &unpack           = {},
                       const bool                   allow_refinement = true,
                       const bool                   allow_coarsening = true);

  private:
    /**
     * Group the locally owned active cells into the current polytopes of the
     * tree.
     */
    void
    update_polytopes();

    /**
     * The data the tree needs per initial cell, from the error indicators
     * @p cell_indicators of the active cells, computed on the p4est rank and
     * sent to the tree rank, indexed by tree index. For the initial cell with
     * tree index i, error_sq[i] is the sum of the squared error indicators of
     * the active cells inside it: its own if p4est has not refined it, else
     * those of its active descendants. is_refined[i] is one if p4est has
     * refined it.
     */
    void
    initial_cell_data(const Vector<float>       &cell_indicators,
                      std::vector<double>       &error_sq,
                      std::vector<unsigned int> &is_refined) const;

    /**
     * The index, among the initial cells this rank owns on the p4est
     * triangulation, of the initial cell that contains @p cell, which is
     * @p cell itself or the initial cell it descends from. With
     * InitialPartition::p4est it is the tree index.
     */
    unsigned int
    initial_cell(const CellIterator &cell) const;

    /**
     * Send @p n_values values per initial cell from the p4est ranks to the
     * tree ranks: @p owned_values holds them for the initial cells this rank
     * owns, the result for the initial cells of its tree, in the order of
     * their tree index. With InitialPartition::p4est the two are the same.
     */
    std::vector<double>
    to_tree_ranks(const std::vector<double> &owned_values,
                  const unsigned int         n_values) const;

    /**
     * Send @p n_values values per initial cell from the tree ranks back to
     * the p4est ranks, the reverse of to_tree_ranks().
     */
    std::vector<double>
    to_p4est_ranks(const std::vector<double> &tree_values,
                   const unsigned int         n_values) const;

    /**
     * The polytopes of the tree with InitialPartition::metis, computed on the
     * tree rank: every current polytope of the tree, without its initial
     * cells that p4est has refined, split into face-connected components,
     * numbered from zero. @p n_components is set to their number. For every
     * initial cell of the tree, three values are returned: the number of its
     * component (-1 if p4est has refined it), the number of initial cells of
     * the component, and the number of initial cells of its current polytope
     * in the tree.
     */
    std::vector<double>
    tree_components(unsigned int &n_components) const;

    /**
     * The triangulation the polytopes are made of.
     */
    parallel::distributed::Triangulation<dim> &tria;

    /**
     * The degree of the elements on the polytopes.
     */
    const unsigned int fe_degree;

    /**
     * The parameters of the adaptation.
     */
    const AdaptivityParameters parameters;

    /**
     * This rank's tree over its initial cells.
     */
    internal::PolytopeTree tree;

    /**
     * For every initial cell this rank owns on the p4est triangulation, its
     * index among them, keyed by its coarse cell id. deal.II calls the cells
     * of the initial mesh coarse cells, and the CellId of every cell starts
     * with the coarse cell id of the initial cell it descends from. This map
     * therefore finds the initial cell of any active cell, see
     * initial_cell().
     */
    std::unordered_map<types::coarse_cell_id, unsigned int>
      initial_of_coarse_cell_id;

    /**
     * With InitialPartition::metis, for every rank r: the indices, among the
     * initial cells this rank owns, of those in the tree of r, and the tree
     * indices of the initial cells of this rank's tree that r owns. Both in
     * the order of the coarse cell ids, so that the two ranks agree on the
     * order and only values are sent.
     */
    std::vector<std::vector<unsigned int>> owned_for_tree_rank;
    std::vector<std::vector<unsigned int>> tree_from_p4est_rank;

    /**
     * The tree rank of every initial cell this rank owns.
     */
    std::vector<unsigned int> tree_rank_of_owned;

    /**
     * With InitialPartition::metis, the initial cells of the tree by tree
     * index, to find the faces between them, and for each whether p4est has
     * refined it, as of the last adaptation.
     */
    std::vector<typename Triangulation<dim>::cell_iterator> tree_cells;
    std::vector<unsigned int>                               tree_is_refined;

    /**
     * For every initial cell this rank owns, the number of initial cells of
     * its current polytope in the tree.
     */
    std::vector<unsigned int> tree_polytope_size;

    /**
     * The current polytopes of this rank, each as the list of its active
     * cells, and their global polytope indices.
     */
    std::vector<std::vector<CellIterator>> current_polytopes;
    std::vector<types::global_cell_index>  polytope_indices;

    /**
     * The first global polytope index of every rank, and the number of
     * polytopes over all ranks.
     */
    std::vector<types::global_cell_index> polytope_index_offsets;

    /**
     * For every locally owned active cell, the number of cells of its
     * current polytope.
     */
    Vector<float> polytope_size;

    /**
     * Number of polytopes over all ranks.
     */
    types::global_cell_index n_polytopes_global;
  };



  template <int dim>
  AdaptivePolytopalMesh<dim>::AdaptivePolytopalMesh(
    parallel::distributed::Triangulation<dim> &tria,
    const unsigned int                         fe_degree,
    const AdaptivityParameters                &parameters)
    : tria(tria)
    , fe_degree(fe_degree)
    , parameters(parameters)
    , n_polytopes_global(0)
  {}



  template <int dim>
  void
  AdaptivePolytopalMesh<dim>::initialize()
  {
    const MPI_Comm     comm    = tria.get_mpi_communicator();
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
    const bool metis = parameters.initial_partition == InitialPartition::metis;
    AssertThrow(tria.n_levels() == 1,
                ExcMessage("The polytopes are built on the coarse mesh, "
                           "before any refinement."));

    // Every rank holds all initial cells, the ones it does not own as
    // artificial cells. With METIS, rank 0 partitions a serial copy of them
    // and sends the tree rank of every initial cell, by cell index, to all
    // ranks.
    std::vector<unsigned int> tree_rank(tria.n_cells(0), my_rank);
    if (metis && n_ranks > 1)
      {
        if (my_rank == 0)
          {
            std::vector<CellData<dim>> cells(tria.n_cells(0));
            for (const auto &cell : tria.cell_iterators_on_level(0))
              for (const unsigned int v : cell->vertex_indices())
                cells[cell->index()].vertices[v] = cell->vertex_index(v);
            Triangulation<dim> serial;
            serial.create_triangulation(tria.get_vertices(),
                                        cells,
                                        SubCellData());
            GridTools::partition_triangulation(
              n_ranks, serial, SparsityTools::Partitioner::metis);
            for (const auto &cell : serial.active_cell_iterators())
              tree_rank[cell->index()] = cell->subdomain_id();
          }
        tree_rank = Utilities::MPI::broadcast(comm, tree_rank, 0);
      }

    // The initial cells of the tree and the ones this rank owns, in the order
    // of the cell indices. With InitialPartition::p4est they are the same.
    std::vector<typename Triangulation<dim>::cell_iterator> owned_cells;
    tree_cells.clear();
    for (const auto &cell : tria.cell_iterators_on_level(0))
      {
        if (cell->is_locally_owned())
          owned_cells.push_back(cell);
        if (metis ? tree_rank[cell->index()] == my_rank :
                    cell->is_locally_owned())
          tree_cells.push_back(cell);
      }

    // The tree index of an initial cell is its position among the initial
    // cells of the tree.
    std::vector<std::pair<BoundingBox<dim>, unsigned int>> boxes;
    std::map<unsigned int, unsigned int>                   tree_index_of_cell;
    for (unsigned int tree_index = 0; tree_index < tree_cells.size();
         ++tree_index)
      {
        boxes.emplace_back(tree_cells[tree_index]->bounding_box(), tree_index);
        tree_index_of_cell[tree_cells[tree_index]->index()] = tree_index;
      }
    tree.build(boxes, parameters.max_cells_per_polytope);
    tree_is_refined.assign(tree_cells.size(), 0);

    // The index among the owned initial cells by coarse cell id. Every
    // descendant of an initial cell has the same coarse cell id.
    initial_of_coarse_cell_id.clear();
    tree_rank_of_owned.resize(owned_cells.size());
    for (unsigned int owned = 0; owned < owned_cells.size(); ++owned)
      {
        initial_of_coarse_cell_id
          [owned_cells[owned]->id().get_coarse_cell_id()] = owned;
        tree_rank_of_owned[owned] = tree_rank[owned_cells[owned]->index()];
      }

    // With METIS, every rank tells the tree rank of each of its initial
    // cells that it owns it. Both sides list the initial cells they share in
    // the order of the coarse cell ids.
    owned_for_tree_rank.assign(metis ? n_ranks : 0, {});
    tree_from_p4est_rank.assign(metis ? n_ranks : 0, {});
    if (metis)
      {
        std::vector<types::coarse_cell_id> coarse_cell_id(tria.n_cells(0));
        for (const auto &cell : tria.cell_iterators_on_level(0))
          coarse_cell_id[cell->index()] = cell->id().get_coarse_cell_id();

        // The cell indices of the owned initial cells, by tree rank.
        std::map<unsigned int, std::vector<unsigned int>> cells_to_send;
        for (const auto &cell : owned_cells)
          cells_to_send[tree_rank[cell->index()]].push_back(cell->index());
        for (auto &[rank, cell_indices] : cells_to_send)
          {
            std::sort(cell_indices.begin(),
                      cell_indices.end(),
                      [&](const unsigned int a, const unsigned int b) {
                        return coarse_cell_id[a] < coarse_cell_id[b];
                      });
            for (const unsigned int index : cell_indices)
              owned_for_tree_rank[rank].push_back(
                initial_of_coarse_cell_id.at(coarse_cell_id[index]));
          }
        for (const auto &[rank, cell_indices] :
             Utilities::MPI::some_to_some(comm, cells_to_send))
          for (const unsigned int index : cell_indices)
            tree_from_p4est_rank[rank].push_back(tree_index_of_cell.at(index));
      }

    update_polytopes();
  }



  template <int dim>
  const std::vector<
    std::vector<typename AdaptivePolytopalMesh<dim>::CellIterator>> &
  AdaptivePolytopalMesh<dim>::get_polytopes() const
  {
    return current_polytopes;
  }



  template <int dim>
  const std::vector<types::global_cell_index> &
  AdaptivePolytopalMesh<dim>::get_polytope_indices() const
  {
    return polytope_indices;
  }



  template <int dim>
  const std::vector<types::global_cell_index> &
  AdaptivePolytopalMesh<dim>::get_polytope_index_offsets() const
  {
    return polytope_index_offsets;
  }



  template <int dim>
  std::vector<unsigned int>
  AdaptivePolytopalMesh<dim>::get_tree_ranks() const
  {
    // All cells of a polytope lie in initial cells of one tree.
    std::vector<unsigned int> tree_ranks;
    tree_ranks.reserve(current_polytopes.size());
    for (const auto &polytope : current_polytopes)
      tree_ranks.push_back(tree_rank_of_owned[initial_cell(polytope.front())]);
    return tree_ranks;
  }



  template <int dim>
  types::global_cell_index
  AdaptivePolytopalMesh<dim>::n_global_polytopes() const
  {
    return n_polytopes_global;
  }



  template <int dim>
  double
  AdaptivePolytopalMesh<dim>::polytope_tolerance(const double tolerance) const
  {
    return tolerance / std::sqrt(static_cast<double>(n_polytopes_global));
  }



  template <int dim>
  unsigned int
  AdaptivePolytopalMesh<dim>::initial_cell(const CellIterator &cell) const
  {
    return initial_of_coarse_cell_id.at(cell->id().get_coarse_cell_id());
  }



  template <int dim>
  void
  AdaptivePolytopalMesh<dim>::initial_cell_data(
    const Vector<float>       &cell_indicators,
    std::vector<double>       &error_sq,
    std::vector<unsigned int> &is_refined) const
  {
    // An active cell of a level above zero is an active descendant of a
    // refined initial cell. Two values per owned initial cell: the sum of
    // the squared error indicators, and one if it is refined.
    std::vector<double> owned_values(2 * initial_of_coarse_cell_id.size(), 0.);
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int initial = initial_cell(cell);
          const double       error = cell_indicators[cell->active_cell_index()];
          owned_values[2 * initial] += error * error;
          if (cell->level() > 0)
            owned_values[2 * initial + 1] = 1.;
        }
    const std::vector<double> tree_values = to_tree_ranks(owned_values, 2);
    error_sq.resize(tree_cells.size());
    is_refined.resize(tree_cells.size());
    for (unsigned int i = 0; i < tree_cells.size(); ++i)
      {
        error_sq[i]   = tree_values[2 * i];
        is_refined[i] = static_cast<unsigned int>(tree_values[2 * i + 1]);
      }
  }



  template <int dim>
  std::vector<double>
  AdaptivePolytopalMesh<dim>::to_tree_ranks(
    const std::vector<double> &owned_values,
    const unsigned int         n_values) const
  {
    if (parameters.initial_partition == InitialPartition::p4est)
      return owned_values;
    std::map<unsigned int, std::vector<double>> values_to_send;
    for (unsigned int rank = 0; rank < owned_for_tree_rank.size(); ++rank)
      for (const unsigned int owned : owned_for_tree_rank[rank])
        values_to_send[rank].insert(values_to_send[rank].end(),
                                    owned_values.begin() + n_values * owned,
                                    owned_values.begin() +
                                      n_values * (owned + 1));
    std::vector<double> tree_values(n_values * tree_cells.size());
    for (const auto &[rank, values] :
         Utilities::MPI::some_to_some(tria.get_mpi_communicator(),
                                      values_to_send))
      for (unsigned int k = 0; k < tree_from_p4est_rank[rank].size(); ++k)
        std::copy(values.begin() + n_values * k,
                  values.begin() + n_values * (k + 1),
                  tree_values.begin() +
                    n_values * tree_from_p4est_rank[rank][k]);
    return tree_values;
  }



  template <int dim>
  std::vector<double>
  AdaptivePolytopalMesh<dim>::to_p4est_ranks(
    const std::vector<double> &tree_values,
    const unsigned int         n_values) const
  {
    if (parameters.initial_partition == InitialPartition::p4est)
      return tree_values;
    std::map<unsigned int, std::vector<double>> values_to_send;
    for (unsigned int rank = 0; rank < tree_from_p4est_rank.size(); ++rank)
      for (const unsigned int tree_index : tree_from_p4est_rank[rank])
        values_to_send[rank].insert(values_to_send[rank].end(),
                                    tree_values.begin() + n_values * tree_index,
                                    tree_values.begin() +
                                      n_values * (tree_index + 1));
    std::vector<double> owned_values(n_values *
                                     initial_of_coarse_cell_id.size());
    for (const auto &[rank, values] :
         Utilities::MPI::some_to_some(tria.get_mpi_communicator(),
                                      values_to_send))
      for (unsigned int k = 0; k < owned_for_tree_rank[rank].size(); ++k)
        std::copy(values.begin() + n_values * k,
                  values.begin() + n_values * (k + 1),
                  owned_values.begin() +
                    n_values * owned_for_tree_rank[rank][k]);
    return owned_values;
  }



  template <int dim>
  std::vector<double>
  AdaptivePolytopalMesh<dim>::tree_components(unsigned int &n_components) const
  {
    // The initial cells of every current polytope of the tree that p4est has
    // not refined, in the order of their CellId, as on the p4est triangulation
    // with InitialPartition::p4est.
    std::map<int, std::vector<unsigned int>> cells_of_polytope;
    for (unsigned int i = 0; i < tree_cells.size(); ++i)
      if (tree_is_refined[i] == 0)
        cells_of_polytope[tree.current_polytope_of[i]].push_back(i);
    std::map<unsigned int, unsigned int> tree_index_of_cell;
    for (unsigned int i = 0; i < tree_cells.size(); ++i)
      tree_index_of_cell[tree_cells[i]->index()] = i;

    std::vector<double> values(3 * tree_cells.size(), -1.);
    for (unsigned int i = 0; i < tree_cells.size(); ++i)
      {
        values[3 * i + 1] = 1.;
        values[3 * i + 2] = tree.current_polytope_size(i);
      }

    // Every breadth-first search from a cell not yet reached gives one
    // component, as in split_into_connected_components().
    n_components = 0;
    for (auto &[polytope, cells] : cells_of_polytope)
      {
        std::sort(cells.begin(),
                  cells.end(),
                  [&](const unsigned int a, const unsigned int b) {
                    return tree_cells[a]->id() < tree_cells[b]->id();
                  });
        const std::set<unsigned int> in_polytope(cells.begin(), cells.end());
        std::set<unsigned int>       visited;
        for (const unsigned int start : cells)
          {
            if (visited.count(start) > 0)
              continue;
            std::vector<unsigned int> component;
            std::queue<unsigned int>  queue;
            visited.insert(start);
            queue.push(start);
            while (!queue.empty())
              {
                const unsigned int i = queue.front();
                queue.pop();
                component.push_back(i);
                for (const unsigned int f : tree_cells[i]->face_indices())
                  if (!tree_cells[i]->at_boundary(f))
                    {
                      const auto neighbor = tree_cells[i]->neighbor(f);
                      const auto found =
                        tree_index_of_cell.find(neighbor->index());
                      if (found != tree_index_of_cell.end() &&
                          in_polytope.count(found->second) > 0 &&
                          visited.count(found->second) == 0)
                        {
                          visited.insert(found->second);
                          queue.push(found->second);
                        }
                    }
              }
            for (const unsigned int i : component)
              {
                values[3 * i]     = n_components;
                values[3 * i + 1] = component.size();
              }
            ++n_components;
          }
      }
    return values;
  }



  template <int dim>
  void
  AdaptivePolytopalMesh<dim>::update_polytopes()
  {
    const MPI_Comm     comm    = tria.get_mpi_communicator();
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
    const bool metis = parameters.initial_partition == InitialPartition::metis;

    // Every active descendant is a polytope of its own.
    current_polytopes.clear();
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && cell->level() > 0)
        current_polytopes.push_back({cell});
    const unsigned int n_descendants = current_polytopes.size();

    // The initial cells that are not refined, grouped by their polytope:
    // with InitialPartition::p4est by their current polytope in the tree,
    // with InitialPartition::metis by the component of it the tree rank
    // computed. The cells of a polytope are listed in the order of their
    // CellId. The AgglomerationHandler makes the first cell of the list the
    // master cell and orders the quadrature points of the polytope by this
    // list. PolytopeRepartition::PartitionedCopy, the copy of the mesh on
    // which the system can be solved instead, lists the cells in the same
    // order, so that the two meshes give the same discrete problem.
    // The global polytope indices: rank r holds its descendants, then the
    // polytopes of its tree, which with InitialPartition::metis its tree rank
    // counts.
    const auto set_index_offsets = [&](const types::global_cell_index count) {
      polytope_index_offsets = {0};
      for (const auto n : Utilities::MPI::all_gather(comm, count))
        polytope_index_offsets.push_back(polytope_index_offsets.back() + n);
      n_polytopes_global = polytope_index_offsets.back();
    };
    tree_polytope_size.resize(initial_of_coarse_cell_id.size());
    std::vector<double> owned_values;
    std::map<types::global_cell_index, std::vector<CellIterator>>
      cells_of_polytope;
    if (!metis)
      {
        for (unsigned int owned = 0; owned < tree_polytope_size.size(); ++owned)
          tree_polytope_size[owned] = tree.current_polytope_size(owned);
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() && cell->level() == 0)
            cells_of_polytope[tree.current_polytope_of[initial_cell(cell)]]
              .push_back(cell);
      }
    else
      {
        unsigned int        n_components = 0;
        std::vector<double> tree_values  = tree_components(n_components);
        set_index_offsets(n_descendants + n_components);
        const types::global_cell_index first_component =
          polytope_index_offsets[my_rank] + n_descendants;
        for (unsigned int i = 0; i < tree_cells.size(); ++i)
          if (tree_values[3 * i] >= 0.)
            tree_values[3 * i] += first_component;
        owned_values = to_p4est_ranks(tree_values, 3);
        for (unsigned int owned = 0; owned < tree_polytope_size.size(); ++owned)
          tree_polytope_size[owned] =
            static_cast<unsigned int>(owned_values[3 * owned + 2]);
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() && cell->level() == 0)
            cells_of_polytope[static_cast<types::global_cell_index>(
                                owned_values[3 * initial_cell(cell)])]
              .push_back(cell);
      }

    // A polytope of the tree need not be connected through faces, nor need
    // the part of a polytope on one rank. With InitialPartition::metis the
    // pieces of one polytope keep its index; with InitialPartition::p4est
    // every piece is a polytope, numbered in order.
    std::vector<types::global_cell_index> local_index(n_descendants);
    for (unsigned int i = 0; i < n_descendants; ++i)
      local_index[i] = i;
    for (auto &[polytope, cells] : cells_of_polytope)
      {
        std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
          return a->id() < b->id();
        });
        for (auto &piece : split_into_connected_components<dim>({cells}))
          {
            current_polytopes.push_back(std::move(piece));
            local_index.push_back(metis ? polytope : local_index.size());
          }
      }
    if (!metis)
      set_index_offsets(current_polytopes.size());
    polytope_indices.resize(current_polytopes.size());
    for (unsigned int i = 0; i < current_polytopes.size(); ++i)
      polytope_indices[i] = (metis && i >= n_descendants) ?
                              local_index[i] :
                              polytope_index_offsets[my_rank] + local_index[i];

    // The size of the polytope of every cell: with InitialPartition::metis
    // the size the tree rank computed, which counts the cells on all ranks.
    polytope_size.reinit(tria.n_active_cells());
    for (const auto &polytope : current_polytopes)
      for (const auto &cell : polytope)
        polytope_size[cell->active_cell_index()] =
          (!metis || cell->level() > 0) ?
            polytope.size() :
            owned_values[3 * initial_cell(cell) + 1];
  }



  template <int dim>
  bool
  AdaptivePolytopalMesh<dim>::refine_and_coarsen(
    const Vector<float>         &cell_indicators,
    const double                 tolerance,
    const std::function<void()> &prepare,
    const std::function<void()> &unpack,
    const bool                   allow_refinement,
    const bool                   allow_coarsening)
  {
    const MPI_Comm comm            = tria.get_mpi_communicator();
    const double   coarsening_line = parameters.coarsening_fraction * tolerance;
    const bool multiple = parameters.level_rule == LevelRule::multiple_levels;
    const auto levels   = [&](const double error) {
      return predicted_levels<dim>(error, tolerance, fe_degree, parameters);
    };
    const int max_level = dealii::internal::p4est::functions<dim>::max_level;

    // Levels of refinement (positive) or coarsening (negative) a cell
    // still owes after the next call of execute_coarsening_and_refinement().
    Vector<float> pending(tria.n_active_cells());

    // For LevelRule::multiple_levels: the number of levels by which p4est is
    // to refine every owned initial cell in this call, see
    // PolytopeTree::cell_refinement_levels(). The tree rank computes them.
    std::vector<int> remaining_levels;
    if (multiple && allow_refinement)
      {
        std::vector<double>       error_sq;
        std::vector<unsigned int> is_refined;
        initial_cell_data(cell_indicators, error_sq, is_refined);
        const std::vector<int> tree_levels =
          tree.cell_refinement_levels(error_sq, levels);
        const std::vector<double> owned_levels = to_p4est_ranks(
          std::vector<double>(tree_levels.begin(), tree_levels.end()), 1);
        remaining_levels.assign(owned_levels.begin(), owned_levels.end());
      }

    // Refinement flags. A cell is refined by p4est when it is a current
    // polytope of its own that the tree cannot refine further: an active
    // descendant, or an initial cell whose current polytope is the cell
    // alone. Under LevelRule::multiple_levels, the initial cells of a larger
    // polytope are flagged as well when the tree cannot provide all the
    // levels it predicts.
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->clear_refine_flag();
          cell->clear_coarsen_flag();
          if (!allow_refinement)
            continue;
          const unsigned int cell_index = cell->active_cell_index();
          const unsigned int initial    = initial_cell(cell);
          const bool         alone =
            cell->level() > 0 || tree_polytope_size[initial] == 1;
          const int n_levels =
            multiple ? levels(cell_indicators[cell_index]) : 0;
          const bool over_tolerance =
            multiple ? n_levels >= 1 : cell_indicators[cell_index] > tolerance;
          if (polytope_size[cell_index] == 1 && alone && over_tolerance)
            {
              cell->set_refine_flag();
              if (multiple)
                pending[cell_index] = n_levels - 1;
            }
          else if (multiple && cell->level() == 0 && !alone &&
                   remaining_levels[initial] >= 1)
            {
              cell->set_refine_flag();
              pending[cell_index] = remaining_levels[initial] - 1;
            }
        }

    // Coarsening flags. An active descendant is coarsened by p4est when its
    // error indicator is far enough under the tolerance. p4est coarsens only
    // when all the cells of one refinement are flagged.
    if (allow_coarsening)
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned() && !cell->refine_flag_set() &&
            cell->level() > 0)
          {
            const unsigned int cell_index = cell->active_cell_index();
            const int          n_levels =
              multiple ? levels(cell_indicators[cell_index]) : 0;
            if (multiple ? n_levels <= -1 :
                           cell_indicators[cell_index] <= coarsening_line)
              {
                cell->set_coarsen_flag();
                if (multiple)
                  pending[cell_index] = n_levels + 1;
              }
          }

    // If a flagged cell is at the finest level p4est allows, no rank changes
    // anything.
    bool at_max_level = false;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && cell->refine_flag_set() &&
          cell->level() + 1 >= max_level)
        at_max_level = true;
    if (Utilities::MPI::logical_or(at_max_level, comm))
      {
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->clear_refine_flag();
              cell->clear_coarsen_flag();
            }
        return false;
      }

    // How data moves to the new cells. Under LevelRule::multiple_levels the
    // error indicator of a refined cell is divided among its children so
    // that the sum of the squares is kept, and the tree step below does not
    // count it $2^d$ times. A cell that replaces coarsened cells takes the
    // largest of their error indicators. The levels a cell still owes are
    // copied to all its children.
    using Transfer =
      parallel::distributed::CellDataTransfer<dim, dim, dealii::Vector<float>>;
    const auto split_indicator =
      [](const typename Triangulation<dim>::cell_iterator &parent,
         const float                                       value) {
        return std::vector<float>(parent->n_children(),
                                  value / std::sqrt(static_cast<float>(
                                            parent->n_children())));
      };
    const auto largest = [](const typename Triangulation<dim>::cell_iterator &,
                            const std::vector<float> &children_values) {
      return children_values.empty() ?
               0.f :
               *std::max_element(children_values.begin(),
                                 children_values.end());
    };
    const auto preserve =
      &AdaptationStrategies::Refinement::preserve<dim, dim, float>;
    Vector<float> indicators = cell_indicators;

    // One call of execute_coarsening_and_refinement() per level: after the
    // first, the cells that still owe a level are flagged again, until no
    // cell on any rank does.
    for (unsigned int call = 0;; ++call)
      {
        // After the first call: flag the cells that still owe a level.
        if (call > 0)
          {
            if (!multiple)
              break;
            unsigned int n_flagged = 0;
            for (const auto &cell : tria.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  cell->clear_refine_flag();
                  cell->clear_coarsen_flag();
                  float &owed = pending[cell->active_cell_index()];
                  if (owed > 0.5f && cell->level() + 1 < max_level)
                    {
                      cell->set_refine_flag();
                      owed -= 1.f;
                      ++n_flagged;
                    }
                  else if (owed < -0.5f && cell->level() > 0)
                    {
                      cell->set_coarsen_flag();
                      owed += 1.f;
                      ++n_flagged;
                    }
                  else
                    owed = 0.f;
                }
            if (Utilities::MPI::sum(n_flagged, comm) == 0)
              break;
          }

        // Carry the error indicators, the owed levels and the data of the
        // caller through this call.
        Transfer indicator_transfer =
          multiple ? Transfer(tria, false, split_indicator, largest) :
                     Transfer(tria, false, preserve, largest);
        Transfer pending_transfer(tria, false, preserve, largest);
        indicator_transfer.prepare_for_coarsening_and_refinement(indicators);
        pending_transfer.prepare_for_coarsening_and_refinement(pending);
        if (prepare)
          prepare();

        tria.execute_coarsening_and_refinement();

        indicators.reinit(tria.n_active_cells());
        pending.reinit(tria.n_active_cells());
        indicator_transfer.unpack(indicators);
        pending_transfer.unpack(pending);
        if (unpack)
          unpack();
      }

    // The tree step, from the error indicators carried to the new cells, and
    // the new list of current polytopes.
    std::vector<double>       error_sq;
    std::vector<unsigned int> is_refined;
    initial_cell_data(indicators, error_sq, is_refined);
    if (multiple)
      tree.refine_and_merge_multiple_levels(error_sq,
                                            is_refined,
                                            levels,
                                            parameters.max_cells_per_polytope,
                                            allow_refinement,
                                            allow_coarsening);
    else
      tree.refine_and_merge_single_level(error_sq,
                                         is_refined,
                                         tolerance,
                                         coarsening_line,
                                         parameters.max_cells_per_polytope,
                                         allow_refinement,
                                         allow_coarsening);
    tree_is_refined = is_refined;
    update_polytopes();
    return true;
  }



  /**
   * The locally relevant degrees of freedom of the polytopes of @p ah: those
   * of the locally owned polytopes, and those of every polytope on another
   * rank that shares a face with one of them. A solution vector with these
   * ghost entries can evaluate the jumps across the faces between ranks.
   *
   * DoFTools::extract_locally_relevant_dofs() on the handler's DoFHandler
   * returns the degrees of freedom of the locally owned and the ghost cells.
   * The degrees of freedom of a polytope live on its master cell only, and
   * the master cell of a neighbouring polytope on another rank is in general
   * not a ghost cell of this rank, so most of them are missing. The handler
   * receives the degree of freedom indices of these neighbours from their
   * ranks when it distributes the degrees of freedom, and returns them for a
   * neighbour that is not locally owned through get_dof_indices(). This
   * function collects them.
   */
  template <int dim>
  IndexSet
  locally_relevant_polytope_dofs(const AgglomerationHandler<dim> &ah)
  {
    // The locally owned degrees of freedom, and those of every neighbour
    // across a face to another rank, as the handler received them.
    IndexSet relevant = ah.agglo_dh.locally_owned_dofs();
    std::vector<types::global_dof_index> dof_indices(ah.get_fe().dofs_per_cell);
    for (const auto &polytope : ah.polytope_iterators())
      if (polytope->is_locally_owned())
        for (unsigned int f = 0; f < polytope->n_faces(); ++f)
          if (!polytope->at_boundary(f) &&
              !polytope->neighbor(f)->is_locally_owned())
            {
              polytope->neighbor(f)->get_dof_indices(dof_indices);
              std::sort(dof_indices.begin(), dof_indices.end());
              relevant.add_indices(dof_indices.begin(), dof_indices.end());
            }
    relevant.compress();
    return relevant;
  }



  /**
   * Initializes @p amg for @p matrix, the matrix of a discretization with
   * the polynomials of @p ah on its polytopes: ML's algebraic multigrid by
   * smoothed aggregation, to precondition the conjugate gradients.
   *
   * The basis functions of a polytope are polynomials on its bounding box. On
   * a polytope that fills only part of its box they are nearly linearly
   * dependent there, so the diagonal block of the matrix that couples the
   * degrees of freedom of one polytope is badly conditioned, the more so the
   * higher the degree. A pointwise smoother cannot correct this. On the
   * finest level the smoother is therefore block Jacobi with one block per
   * polytope, which inverts each of these diagonal blocks exactly. The
   * degrees of freedom of a polytope are numbered consecutively, so the
   * blocks are consecutive ranges of rows of the same size. The rows of the
   * coarser levels no longer belong to single polytopes, and these levels
   * keep a Chebyshev smoother.
   *
   * The basis is modal, so the constant function is the first basis function
   * of every polytope, not the sum of all of them. ML builds its coarse levels
   * from the near-null space, so the degrees of freedom of a polytope form one
   * node of ML ("PDE equations" = the degrees of freedom per polytope), and
   * the near-null space is the single vector that is 1 on the first degree of
   * freedom of every polytope and 0 elsewhere, i.e. the coefficient vector of
   * the constant. With one equation per node ML would take the vector of all
   * ones, and the coarse levels would not represent the smooth error. Without
   * an explicit near-null space ML takes one vector per equation, one for each
   * basis function; only the first of them is close to the kernel of the
   * operator, and the others make every coarse level as many times larger as
   * there are basis functions.
   */
  template <int dim>
  void
  initialize_polytope_amg(TrilinosWrappers::PreconditionAMG    &amg,
                          const TrilinosWrappers::SparseMatrix &matrix,
                          const AgglomerationHandler<dim>      &ah)
  {
    Teuchos::ParameterList parameters;
    ML_Epetra::SetDefaults("SA", parameters);
    parameters.set("coarse: type", "Amesos-KLU");
    parameters.set("coarse: max size", 2000);
    parameters.set("initialize random seed", true);
    parameters.set("prec type", "MGV");
    parameters.set("cycle applications", 1);
    parameters.set("aggregation: threshold", 0.02);
    parameters.set("smoother: type", "Chebyshev");
    parameters.set("smoother: Chebyshev alpha", 10.);
    parameters.set("smoother: sweeps", 2);
    parameters.set("ML output", 0);

    // One block per locally owned polytope.
    const auto         range = matrix.local_range();
    const unsigned int n_local_rows =
      static_cast<unsigned int>(range.second - range.first);
    const unsigned int dofs_per_polytope = ah.get_fe().dofs_per_cell;
    AssertThrow(n_local_rows % dofs_per_polytope == 0,
                ExcMessage("The locally owned rows are not whole polytopes."));
    parameters.set("PDE equations", static_cast<int>(dofs_per_polytope));
    // The near-null space: the constant. ML reads it during initialize().
    std::vector<double> constant_mode(n_local_rows, 0.);
    for (unsigned int i = 0; i < n_local_rows; i += dofs_per_polytope)
      constant_mode[i] = 1.;
    parameters.set("null space: type", "pre-computed");
    parameters.set("null space: dimension", 1);
    parameters.set("null space: vectors", constant_mode.data());
    Teuchos::ParameterList &finest =
      parameters.sublist("smoother: list (level 0)");
    finest.set("smoother: type", "IFPACK");
    finest.set("smoother: ifpack type", "block relaxation stand-alone");
    finest.set("smoother: sweeps", 1);
    Teuchos::ParameterList &blocks = finest.sublist("smoother: ifpack list");
    blocks.set("relaxation: type", "Jacobi");
    blocks.set("relaxation: sweeps", 1);
    blocks.set("partitioner: type", "linear");
    blocks.set("partitioner: local parts",
               static_cast<int>(n_local_rows / dofs_per_polytope));

    amg.initialize(matrix, parameters);
  }



  /**
   * The length of the shortest side of @p box.
   */
  template <int dim>
  double
  shortest_side(const BoundingBox<dim> &box)
  {
    double side = box.side_length(0);
    for (unsigned int d = 1; d < dim; ++d)
      side = std::min(side, box.side_length(d));
    return side;
  }



  /**
   * The SIPG penalty on an interior face between polytopes $P$ and $P'$,
   * @f[
   *   \sigma_F = \frac{2 C}{\min(s_P, s_{P'})},
   * @f]
   * with $C$ = @p penalty_constant, which contains $(p+1)(p+d)$, and the
   * shortest sides $s_P$, $s_{P'}$ of the bounding boxes @p box and
   * @p box_neighbor of the two polytopes. This is the penalty of Cangiani,
   * Dong and Georgoulis (2023, eq. 2.16), the larger of
   * $C_\sigma C_\mathrm{inv}/h_K$ of the two sides, with
   * $C_\mathrm{inv} = \tau (p+1)(p+d)$ of the inverse trace inequality (their
   * Lemma 2.5), in which $\tau / h_K$ is replaced by $2 / s_P$. On a square or
   * a cube $2 / s_P$ is the inverse of the radius of the inscribed ball, the
   * value of $\tau / h_K$ with $\tau = 2\sqrt{d}$. On a thin polytope the
   * diameter is set by the long side, and $\tau / h_K$ with a fixed $\tau$
   * would be too small; $2 / s_P$ grows as the polytope gets thinner.
   */
  template <int dim>
  double
  interior_penalty(const double            penalty_constant,
                   const BoundingBox<dim> &box,
                   const BoundingBox<dim> &box_neighbor)
  {
    return 2. * penalty_constant /
           std::min(shortest_side(box), shortest_side(box_neighbor));
  }



  /**
   * The SIPG penalty on a face of polytope $P$ on the boundary,
   * $\sigma_F = 2 C / s_P$, the one of interior_penalty() with $P$ alone.
   */
  template <int dim>
  double
  boundary_penalty(const double penalty_constant, const BoundingBox<dim> &box)
  {
    return 2. * penalty_constant / shortest_side(box);
  }



  /**
   * Residual error indicator of Cangiani, Dong and Georgoulis (2023, Thm 3.5,
   * (3.21)) for the SIPG discretization of $-\Delta u = f$ with $u = 0$ on
   * the boundary, written into @p cell_indicators for every locally owned
   * active cell. For a polytope $P$ of diameter $h_P$,
   * @f[
   *   \eta_P^2 = h_P^2 \| f + \Delta u_h \|_P^2
   *     + \sum_{F \subset \partial P} \left( h_P \| [\nabla u_h] \|_F^2
   *     + \sigma_F \| [u_h] \|_F^2 \right),
   * @f]
   * the full gradient jump comprising the normal and the tangential term of
   * the paper, with $u_h$ itself as the jump on the boundary. $\sigma_F$ is
   * the penalty of the assembly: interior_penalty() on an interior face and
   * boundary_penalty() on the boundary. The error indicators are stored on the
   * cells: each of the $n_P$ cells of $P$ gets $\eta_P / \sqrt{n_P}$, so
   * that the squares over the cells of $P$ sum to $\eta_P^2$.
   *
   * Every locally owned polytope adds its volume term and the terms of its
   * faces, of which there are three kinds. A face on the boundary adds the
   * penalty term of $u_h$ and the tangential part of $\nabla u_h$. A face
   * between two polytopes of this rank is visited once and adds its terms
   * to both. A face to a polytope of another rank is visited from both
   * ranks, and each adds the terms of its own polytope. The neighbour's side
   * is then evaluated from the values of its shape functions that its rank
   * sent and from its coefficients in @p solution.
   *
   * @p volume_source, if given, is added to $f$ in the volume term: a
   * function of the cell, the point and the value of $u_h$ there, for
   * instance $-(u_h - u^{n-1})/\Delta t$ for a time step.
   *
   * @p solution must hold the degrees of freedom of the neighbours on other
   * ranks, see locally_relevant_polytope_dofs(). The handler's FEValues are
   * initialized here with values, gradients and second derivatives, on
   * quadratures of $p+1$ points per direction; on a face to another rank the
   * points are those its rank sent.
   */
  template <int dim>
  void
  compute_residual_indicators(
    AgglomerationHandler<dim>           &ah,
    const TrilinosWrappers::MPI::Vector &solution,
    const Function<dim>                 &rhs_function,
    const double                         penalty_constant,
    Vector<float>                       &cell_indicators,
    const std::function<
      double(const typename Triangulation<dim>::active_cell_iterator &,
             const Point<dim> &,
             const double)> &volume_source = {})
  {
    const Triangulation<dim> &tria   = ah.get_triangulation();
    const unsigned int        degree = ah.get_fe().degree;
    const unsigned int        n_dofs = ah.get_fe().dofs_per_cell;
    const unsigned int        n_q_1d = degree + 1;
    // Values, gradients and second derivatives on the polytopes and their
    // faces, with p+1 Gauss points per direction.
    ah.initialize_fe_values(QGauss<dim>(n_q_1d),
                            update_values | update_gradients | update_hessians |
                              update_quadrature_points | update_JxW_values,
                            QGauss<dim - 1>(n_q_1d));
    const unsigned int n_q_cell = QGauss<dim>(n_q_1d).size();

    // Values at the points of a polytope or of a face: f, the Laplacian and
    // the value of u_h on the polytope, and u_h and its gradient on this side
    // (here) and on the neighbour's side (there) of a face.
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    std::vector<double>                  neighbor_coefficients(n_dofs);
    std::vector<double>                  rhs_values, laplacians, u_values;
    std::vector<double>                  values_here, values_there;
    std::vector<Tensor<1, dim>>          gradients_here, gradients_there;

    // Squared error indicator of every locally owned polytope.
    std::map<unsigned int, double> error_sq;

    const auto tangential_sq = [](const Tensor<1, dim> &v,
                                  const Tensor<1, dim> &normal) {
      return (v - (v * normal) * normal).norm_square();
    };

    for (const auto &polytope : ah.polytope_iterators())
      {
        if (!polytope->is_locally_owned())
          continue;
        const double h = polytope->diameter();

        // 1. The volume term $h_P^2 \| f + \Delta u_h \|_P^2$. The quadrature
        // of the handler on a polytope lists the points of its cells one cell
        // after the other, in the order of get_agglomerate(). The loop over the
        // cells is only there because @p volume_source needs the cell of every
        // point.
        const auto &fe_values = ah.reinit(polytope);
        const auto &points    = fe_values.get_quadrature_points();
        rhs_values.resize(points.size());
        rhs_function.value_list(points, rhs_values);
        laplacians.resize(points.size());
        fe_values.get_function_laplacians(solution, laplacians);
        if (volume_source)
          {
            u_values.resize(points.size());
            fe_values.get_function_values(solution, u_values);
          }
        double       volume_sq = 0.;
        unsigned int q = 0;
        for (const auto &cell : polytope->get_agglomerate())
          for (const unsigned int end = q + n_q_cell; q < end; ++q)
            {
              double residual = rhs_values[q] + laplacians[q];
              if (volume_source)
                residual += volume_source(cell, points[q], u_values[q]);
              volume_sq += residual * residual * fe_values.JxW(q);
            }
        AssertDimension(q, points.size());
        error_sq[polytope->index()] += h * h * volume_sq;

        for (unsigned int f = 0; f < polytope->n_faces(); ++f)
          {
            // 2. A face on the boundary. The jump of the value is $u_h$
            // itself, and the gradient jump is the tangential part of
            // $\nabla u_h$. The penalty is that of the assembly on the
            // boundary.
            if (polytope->at_boundary(f))
              {
                const auto &face_values = ah.reinit(polytope, f);
                const auto &normals     = face_values.get_normal_vectors();
                values_here.resize(face_values.n_quadrature_points);
                gradients_here.resize(face_values.n_quadrature_points);
                face_values.get_function_values(solution, values_here);
                face_values.get_function_gradients(solution, gradients_here);
                double value_sq = 0., gradient_sq = 0.;
                for (const unsigned int q :
                     face_values.quadrature_point_indices())
                  {
                    value_sq +=
                      values_here[q] * values_here[q] * face_values.JxW(q);
                    gradient_sq +=
                      tangential_sq(gradients_here[q], normals[q]) *
                      face_values.JxW(q);
                  }
                const double sigma =
                  boundary_penalty<dim>(penalty_constant,
                                        polytope->get_bounding_box());
                error_sq[polytope->index()] +=
                  sigma * value_sq + h * gradient_sq;
                continue;
              }

            // 3. An interior face. A face between two polytopes of this rank
            // is visited once, from the polytope with the smaller id, and adds
            // its terms to both. A face to a polytope of another rank is
            // visited from both ranks, and each adds the terms of its own
            // polytope.
            const auto        &neighbor = polytope->neighbor(f);
            const unsigned int neighbor_face =
              polytope->neighbor_of_agglomerated_neighbor(f);
            const bool ghost = !neighbor->is_locally_owned();
            if (!ghost && !(polytope->id() < neighbor->id()))
              continue;

            // u_h and its gradient on this side.
            const auto &interface_values =
              ah.reinit_interface(polytope, neighbor, f, neighbor_face);
            const auto        &values0 = interface_values.first;
            const unsigned int n_q     = values0.n_quadrature_points;
            values_here.resize(n_q);
            gradients_here.resize(n_q);
            values_there.assign(n_q, 0.);
            gradients_there.assign(n_q, Tensor<1, dim>());
            values0.get_function_values(solution, values_here);
            values0.get_function_gradients(solution, gradients_here);

            const std::vector<double> *jxw = nullptr;
            double                     h_neighbor;
            if (ghost)
              {
                // 3a. The neighbour lies on another rank. Its rank sent the
                // values and gradients of its shape functions and the weights
                // at the points of this face; with its coefficients from
                // @p solution they give u_h on its side. diameter() cannot be
                // evaluated on a polytope of another rank, so its size is the
                // diagonal of its bounding box.
                neighbor->get_dof_indices(dof_indices);
                for (unsigned int i = 0; i < n_dofs; ++i)
                  neighbor_coefficients[i] = solution(dof_indices[i]);
                const types::subdomain_id rank = neighbor->subdomain_id();
                const auto               &recv_values =
                  ah.recv_values.at(rank).at({neighbor->id(), neighbor_face});
                const auto &recv_gradients = ah.recv_gradients.at(rank).at(
                  {neighbor->id(), neighbor_face});
                jxw =
                  &ah.recv_jxws.at(rank).at({neighbor->id(), neighbor_face});
                for (unsigned int q = 0; q < n_q; ++q)
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    {
                      values_there[q] +=
                        neighbor_coefficients[i] * recv_values[i][q];
                      gradients_there[q] +=
                        neighbor_coefficients[i] * recv_gradients[i][q];
                    }
                const auto corners =
                  neighbor->get_bounding_box().get_boundary_points();
                h_neighbor = (corners.second - corners.first).norm();
              }
            else
              {
                // 3b. The neighbour lies on this rank: its side of the
                // interface values.
                interface_values.second.get_function_values(solution,
                                                            values_there);
                interface_values.second.get_function_gradients(solution,
                                                               gradients_there);
                h_neighbor = neighbor->diameter();
              }
            // The squared jumps of the gradient and of the value over the
            // face, with the penalty of the assembly on an interior face.
            // Each polytope weights the gradient jump by its own diameter.
            double gradient_sq = 0., value_sq = 0.;
            for (unsigned int q = 0; q < n_q; ++q)
              {
                const double weight = ghost ? (*jxw)[q] : values0.JxW(q);
                const Tensor<1, dim> gradient_jump =
                  gradients_here[q] - gradients_there[q];
                const double value_jump = values_here[q] - values_there[q];
                gradient_sq += gradient_jump.norm_square() * weight;
                value_sq += value_jump * value_jump * weight;
              }
            const double sigma =
              interior_penalty<dim>(penalty_constant,
                                    polytope->get_bounding_box(),
                                    neighbor->get_bounding_box());
            error_sq[polytope->index()] += h * gradient_sq + sigma * value_sq;
            if (!ghost)
              error_sq[neighbor->index()] +=
                h_neighbor * gradient_sq + sigma * value_sq;
          }
      }

    // 4. Store $\eta_P$ on the cells: each of the $n_P$ cells of a polytope
    // gets $\eta_P / \sqrt{n_P}$.
    cell_indicators.reinit(tria.n_active_cells());
    for (const auto &polytope : ah.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          const auto  &cells = polytope->get_agglomerate();
          const double share =
            std::sqrt(error_sq[polytope->index()] / cells.size());
          for (const auto &cell : cells)
            cell_indicators[cell->active_cell_index()] = share;
        }
  }
} // namespace dealii::PolytopeAdaptivity

#endif
