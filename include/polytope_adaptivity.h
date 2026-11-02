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

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/adaptation_strategies.h>
#include <deal.II/numerics/rtree.h>

#include <boost/geometry/index/rtree.hpp>

#include <agglomeration_handler.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
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
   * level zero. Every rank builds one R-tree over the initial cells it owns,
   * and every node of the tree is a polytope. At the start the current
   * polytopes are, down every branch, the first polytope of at most
   * AdaptivityParameters::max_cells_per_polytope initial cells. Polytopes
   * are refined down the tree and merged back up it. Below a single initial
   * cell, p4est refines and coarsens the cell, and every active descendant
   * of a refined initial cell is a polytope of its own.
   *
   * The triangulation must be created with the setting
   * parallel::distributed::Triangulation::no_automatic_repartitioning: cells
   * then never change their rank, so every polytope of a rank's tree stays
   * on one rank, and all decisions are local.
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
     * Build the R-tree over the initial cells this rank owns, and the current
     * polytopes at the start. The triangulation must not be refined yet.
     */
    void
    initialize();

    /**
     * The polytopes of this rank, each a face-connected list of locally
     * owned active cells.
     */
    const std::vector<std::vector<CellIterator>> &
    get_polytopes() const;

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
     * @p cell_indicators of the active cells, indexed by tree index. For the
     * initial cell with tree index i, error_sq[i] is the sum of the squared
     * error indicators of the active cells inside it: its own if p4est has not
     * refined it, else those of its active descendants. is_refined[i] is one
     * if p4est has refined it.
     */
    void
    initial_cell_data(const Vector<float>       &cell_indicators,
                      std::vector<double>       &error_sq,
                      std::vector<unsigned int> &is_refined) const;

    /**
     * The tree index of the initial cell that contains @p cell, which is
     * @p cell itself or the initial cell it descends from.
     */
    unsigned int
    initial_cell(const CellIterator &cell) const;

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
     * For every initial cell of this rank, its tree index, keyed by its coarse
     * cell id. deal.II calls the cells of the initial mesh coarse
     * cells, and the CellId of every cell starts with the coarse cell id of
     * the initial cell it descends from. This map therefore finds the
     * initial cell of any active cell, see initial_cell().
     */
    std::unordered_map<types::coarse_cell_id, unsigned int>
      initial_of_coarse_cell_id;

    /**
     * The current polytopes of this rank, each as the list of its active
     * cells.
     */
    std::vector<std::vector<CellIterator>> current_polytopes;

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
    // The bounding box of every initial cell of this rank, in the order of
    // the cell indices.
    std::vector<std::pair<BoundingBox<dim>, unsigned int>> boxes;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          AssertThrow(cell->level() == 0,
                      ExcMessage("The polytopes are built on the coarse "
                                 "mesh, before any refinement."));
          boxes.emplace_back(cell->bounding_box(), cell->index());
        }
    std::sort(boxes.begin(), boxes.end(), [](const auto &a, const auto &b) {
      return a.second < b.second;
    });

    // The tree index of an initial cell is its position in the order of the
    // cell indices.
    std::map<unsigned int, unsigned int> initial_of_index;
    for (unsigned int tree_index = 0; tree_index < boxes.size(); ++tree_index)
      {
        initial_of_index[boxes[tree_index].second] = tree_index;
        boxes[tree_index].second                   = tree_index;
      }
    tree.build(boxes, parameters.max_cells_per_polytope);

    // The tree index by coarse cell id. Every descendant of an initial cell
    // has the same coarse cell id.
    initial_of_coarse_cell_id.clear();
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        initial_of_coarse_cell_id[cell->id().get_coarse_cell_id()] =
          initial_of_index.at(cell->index());

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
    // refined initial cell.
    error_sq.assign(tree.current_polytope_of.size(), 0.);
    is_refined.assign(tree.current_polytope_of.size(), 0);
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int initial = initial_cell(cell);
          const double       error = cell_indicators[cell->active_cell_index()];
          error_sq[initial] += error * error;
          if (cell->level() > 0)
            is_refined[initial] = 1;
        }
  }



  template <int dim>
  void
  AdaptivePolytopalMesh<dim>::update_polytopes()
  {
    // Every active descendant is a polytope of its own. The initial cells
    // that are not refined are grouped by their current polytope in the
    // tree.
    current_polytopes.clear();
    std::map<int, std::vector<CellIterator>> cells_of_polytope;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (cell->level() > 0)
            current_polytopes.push_back({cell});
          else
            cells_of_polytope[tree.current_polytope_of[initial_cell(cell)]]
              .push_back(cell);
        }
    // The cells of a polytope are listed in the order of their CellId. The
    // AgglomerationHandler makes the first cell of the list the master cell
    // and orders the quadrature points of the polytope by this list.
    // PolytopeRepartition::PartitionedCopy, the copy of the mesh on which the
    // system can be solved instead, lists the cells in the same order, so
    // that the two meshes give the same discrete problem.
    for (auto &[polytope, cells] : cells_of_polytope)
      {
        std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
          return a->id() < b->id();
        });
        current_polytopes.push_back(std::move(cells));
      }
    // A polytope of the tree need not be connected through faces.
    current_polytopes = split_into_connected_components<dim>(current_polytopes);

    // The size of the polytope of every cell, and the number of polytopes
    // over all ranks.

    polytope_size.reinit(tria.n_active_cells());
    for (const auto &polytope : current_polytopes)
      for (const auto &cell : polytope)
        polytope_size[cell->active_cell_index()] = polytope.size();
    n_polytopes_global = Utilities::MPI::sum<types::global_cell_index>(
      current_polytopes.size(), tria.get_mpi_communicator());
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
    // to refine every initial cell in this call, see
    // PolytopeTree::cell_refinement_levels().
    std::vector<int> remaining_levels;
    if (multiple && allow_refinement)
      {
        std::vector<double>       error_sq;
        std::vector<unsigned int> is_refined;
        initial_cell_data(cell_indicators, error_sq, is_refined);
        remaining_levels = tree.cell_refinement_levels(error_sq, levels);
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
            cell->level() > 0 || tree.current_polytope_size(initial) == 1;
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
   * the penalty of the assembly: @p penalty_constant over the smaller
   * diameter on an interior face, @p penalty_constant times
   * $\max(1/h_P, |F|/|P|)$ on the boundary. The error indicators are stored
   * on the cells: each of the $n_P$ cells of $P$ gets $\eta_P / \sqrt{n_P}$,
   * so that the squares over the cells of $P$ sum to $\eta_P^2$.
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
   * quadratures of $2p+2$ points per direction; on a face to another rank the
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
    const unsigned int        n_q_1d = 2 * degree + 2;
    // Values, gradients and second derivatives on the polytopes and their
    // faces, with 2p+2 Gauss points per direction.
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

    // Squared error indicator and measure of every locally owned polytope.
    // The measure is a sum of quadrature weights: CellAccessor::measure()
    // loses all digits on cells much smaller than their distance to the
    // origin.
    std::map<unsigned int, double> error_sq, polytope_measure;

    const auto tangential_sq = [](const Tensor<1, dim> &v,
                                  const Tensor<1, dim> &normal) {
      return (v - (v * normal) * normal).norm_square();
    };

    for (const auto &polytope : ah.polytope_iterators())
      {
        if (!polytope->is_locally_owned())
          continue;
        const double h = polytope->diameter();

        // 1. The volume term $h_P^2 \| f + \Delta u_h \|_P^2$, and the
        // measure of the polytope. The quadrature of the handler on a polytope
        // lists the points of its cells one cell after the other, in the order
        // of get_agglomerate(). The loop over the cells is only there because
        // @p volume_source needs the cell of every point.
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
        double       volume_sq = 0., measure = 0.;
        unsigned int q = 0;
        for (const auto &cell : polytope->get_agglomerate())
          for (const unsigned int end = q + n_q_cell; q < end; ++q)
            {
              double residual = rhs_values[q] + laplacians[q];
              if (volume_source)
                residual += volume_source(cell, points[q], u_values[q]);
              volume_sq += residual * residual * fe_values.JxW(q);
              measure += fe_values.JxW(q);
            }
        AssertDimension(q, points.size());
        error_sq[polytope->index()] += h * h * volume_sq;
        polytope_measure[polytope->index()] = measure;

        for (unsigned int f = 0; f < polytope->n_faces(); ++f)
          {
            // 2. A face on the boundary. The jump of the value is $u_h$
            // itself, and the gradient jump is the tangential part of
            // $\nabla u_h$. The penalty is that of the assembly on the
            // boundary, which needs the measure of the polytope from step 1.
            if (polytope->at_boundary(f))
              {
                const auto &face_values = ah.reinit(polytope, f);
                const auto &normals     = face_values.get_normal_vectors();
                values_here.resize(face_values.n_quadrature_points);
                gradients_here.resize(face_values.n_quadrature_points);
                face_values.get_function_values(solution, values_here);
                face_values.get_function_gradients(solution, gradients_here);
                double value_sq = 0., gradient_sq = 0., face_measure = 0.;
                for (const unsigned int q :
                     face_values.quadrature_point_indices())
                  {
                    value_sq +=
                      values_here[q] * values_here[q] * face_values.JxW(q);
                    gradient_sq +=
                      tangential_sq(gradients_here[q], normals[q]) *
                      face_values.JxW(q);
                    face_measure += face_values.JxW(q);
                  }
                const double sigma =
                  penalty_constant *
                  std::max(1. / h,
                           face_measure / polytope_measure[polytope->index()]);
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
            const double sigma = penalty_constant / std::min(h, h_neighbor);
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
