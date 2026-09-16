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


#ifndef agglomerator_h
#define agglomerator_h


#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/strategies.hpp>

#include <optional>
#include <functional>
#include <limits>
#include <cmath>
#include <iostream>
#include <deal.II/base/timer.h>

template <int dim, int spacedim>
class AgglomerationHandler;

namespace dealii
{
  namespace internal
  {
    enum class Mode
    {
      collecting,
      refining,
      normal
    };

    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    struct RefineCoarsenRtreeVisitor
      : public boost::geometry::index::detail::rtree::visitor<
          Value,
          typename Options::parameters_type,
          Box,
          Allocators,
          typename Options::node_tag,
          true>::type
    {
      /**
       * Translator interface, required by the boost implementation of the
       * rtree.
       */
      const Translator &translator;

      /**
       * Store the level we are currently visiting.
       */
      size_t level;

      /**
       * Index used to keep track of the number of different visited nodes
       * during recursion/
       */
      // size_t node_counter;
      size_t agglomerate_counter;

      /**
       * The level where children are living.
       * Before: "we want to extract from the RTree object."
       */
      // const size_t target_level;

      /**
       * A reference to the input vector of vector of BoundingBox objects. This
       * vector v has the following property: v[i] = vector with all the mesh
       * iterators composing the i-th agglomerate.
       */
      std::vector<std::vector<typename Triangulation<
        boost::geometry::dimension<Box>::value>::active_cell_iterator>>
        &agglomerates;

      /**
       * Store the total number of nodes on each level.
       */
      std::vector<types::global_cell_index> &n_nodes_per_level;

      /**
       * Map that associates to a given node on level l its children, identified
       * by their integer index.
       */
      std::map<std::pair<types::global_cell_index, types::global_cell_index>,
               std::vector<types::global_cell_index>>
        &parent_node_to_children_nodes;

            /**
       * An alias that identifies an InternalNode of the tree.
       */
      using InternalNode =
        typename boost::geometry::index::detail::rtree::internal_node<
          Value,
          typename Options::parameters_type,
          Box,
          Allocators,
          typename Options::node_tag>::type;

      /**
       * An alias that identifies a Leaf of the tree.
       */
      using Leaf = typename boost::geometry::index::detail::rtree::leaf<
        Value,
        typename Options::parameters_type,
        Box,
        Allocators,
        typename Options::node_tag>::type;

      using ActiveCellIterator = typename Triangulation<boost::geometry::dimension<Box>::value>::active_cell_iterator;
      using CellIterator       = typename Triangulation<boost::geometry::dimension<Box>::value>::cell_iterator;

      using EstimatorFunction = std::function<Mode(size_t /*level*/, const InternalNode &)>;
      const EstimatorFunction &posterior_estimator;

      using LeafEstimatorFunction = std::function<Mode(size_t /*level*/, const ActiveCellIterator &)>;
      const LeafEstimatorFunction &leaf_estimator;
      
      Mode mode;

      /**
       * Hard cap on how many fine cells may be merged into a single
       * agglomerate while in Mode::collecting.
       *
       * IMPORTANT (corrected after A/B testing in adaptive.cc): this is
       * NOT just a cheap fallback for when the tree bottoms out -- it is
       * the control that actually bounds a polytope's ABSOLUTE physical
       * size, uniformly everywhere, regardless of whether its members
       * are all the same size. That is what keeps h_f = min(diam(K),
       * diam(K')) bounded BETWEEN two separately-formed NEIGHBORING
       * polytopes. extract_error_based_agglomerates()'s max_diameter_ratio
       * check only bounds size disparity WITHIN a single candidate
       * polytope (does it internally straddle a refinement-level
       * boundary) -- it does NOT bound how large a polytope can grow if
       * all its members happen to be the same size, so a big uniform
       * low-error region can still be merged into one unboundedly large
       * polytope that then mismatches a much smaller neighboring polytope
       * at the boundary of the region. An A/B test confirmed this
       * directly: disabling max_diameter_ratio (infinity) while keeping a
       * tight max_cells_per_agglomerate restored clean, monotone
       * convergence; the reverse (ratio-only, generous cell count) did
       * not. Both caps are needed together -- see max_diameter_ratio's
       * own documentation on extract_error_based_agglomerates() for the
       * failure mode it catches that this one doesn't. Defaults to "no
       * cap" (max value) so existing callers/behavior are unaffected
       * unless they explicitly opt in.
       */
      size_t max_cells_per_agglomerate;

      inline RefineCoarsenRtreeVisitor(
        const Translator  &translator,
        std::vector<std::vector<typename Triangulation<
          boost::geometry::dimension<Box>::value>::active_cell_iterator>>
                                                        &agglomerates_,
        std::vector<types::global_cell_index>           &n_nodes_per_level,
        std::map<std::pair<types::global_cell_index, types::global_cell_index>,
                 std::vector<types::global_cell_index>> &parent_to_children,
        const typename RefineCoarsenRtreeVisitor<Value, Options, Translator, Box, Allocators>::EstimatorFunction &estimator,
        const typename RefineCoarsenRtreeVisitor<Value, Options, Translator, Box, Allocators>::LeafEstimatorFunction &leaf_est,
        size_t max_cells_per_agglomerate_ = std::numeric_limits<size_t>::max());


      /**
       * Implements the visitor interface for InternalNode objects. If the node
       * belongs to the level next to @p target_level, then fill the bounding
       * box vector for that node.
       */
      inline void
      operator()(const InternalNode &node);

      /**
       * Implements the visitor interface for Leaf objects.
       */
      inline void
      operator()(const Leaf &);
    };



    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    RefineCoarsenRtreeVisitor<Value, Options, Translator, Box, Allocators>::RefineCoarsenRtreeVisitor(
      const Translator  &translator,
      std::vector<std::vector<typename Triangulation<
        boost::geometry::dimension<Box>::value>::active_cell_iterator>>
                                                      &agglomerates_,
      std::vector<types::global_cell_index>           &n_nodes_per_level_,
      std::map<std::pair<types::global_cell_index, types::global_cell_index>,
               std::vector<types::global_cell_index>> &parent_to_children,
      const EstimatorFunction                         &estimator,
      const LeafEstimatorFunction                     &leaf_est,
      size_t                                            max_cells_per_agglomerate_)
      : translator(translator)
      , level(0)
      , agglomerate_counter(0)
      , agglomerates(agglomerates_)
      , n_nodes_per_level(n_nodes_per_level_)
      , parent_node_to_children_nodes(parent_to_children)
      , posterior_estimator(estimator)
      , leaf_estimator(leaf_est)
      , mode(Mode::normal)
      , max_cells_per_agglomerate(max_cells_per_agglomerate_)
    {}



    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    void
    RefineCoarsenRtreeVisitor<Value, Options, Translator, Box, Allocators>::operator()(
      const RefineCoarsenRtreeVisitor::InternalNode &node)
    {
      using elements_type = 
        typename boost::geometry::index::detail::rtree::elements_type<
          InternalNode>::type;
      const elements_type &elements = 
        boost::geometry::index::detail::rtree::elements(node);

      // do nothing if in higher level it's already good enough 
      // and decide to agglomerate
      if (mode == Mode::collecting)
        {
          size_t level_backup = level;
          ++level;
          for (const auto &it : elements)
            boost::geometry::index::detail::rtree::apply_visitor(*this, *it.second);
          level = level_backup;
          return;
        }
      // get the status of current level
      Mode next_action = posterior_estimator(level, node);
      Mode mode_backup = mode;

      if (next_action == Mode::collecting)
        {
          agglomerates.push_back({});
          mode = Mode::collecting;
          size_t level_backup = level;
          ++level;
          for (const auto &it : elements)
            boost::geometry::index::detail::rtree::apply_visitor(*this, *it.second);
          level = level_backup;
          agglomerate_counter++;
          mode = mode_backup;
          return;
        }

      if (next_action == Mode::refining)
        {
          mode = Mode::refining;
        }  
      size_t level_backup = level;
      ++level;
      for (const auto &it : elements)
        boost::geometry::index::detail::rtree::apply_visitor(*this, *it.second);
      level = level_backup;
      mode = mode_backup;
    }



    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    void
    RefineCoarsenRtreeVisitor<Value, Options, Translator, Box, Allocators>::operator()(
      const RefineCoarsenRtreeVisitor::Leaf &leaf)
    {
      using elements_type =
        typename boost::geometry::index::detail::rtree::elements_type<
          Leaf>::type; //  pairs of bounding box and pointer to child node
      const elements_type &elements =
        boost::geometry::index::detail::rtree::elements(leaf);
        
      auto collect_and_force_refinement = [&](CellIterator base_cell, bool force_refine) 
        {
          std::function<void(CellIterator)> traverse = [&](CellIterator c) {
            if (c->is_active()) {
              ActiveCellIterator active_c = c;

              // Cheap safety net: while in Mode::collecting, a single
              // agglomerate can otherwise absorb an unbounded number of
              // fine cells. This is now a secondary backstop only -- the
              // PRIMARY size control lives one level up, in
              // extract_error_based_agglomerates()'s estimator, which
              // checks (before ever deciding to collect a node) whether
              // doing so would make a polytope too big relative to its
              // smallest fine cell, and backs off to the tree's next
              // level down instead of collecting if so. This cell-count
              // check only matters at/near the deepest tree level, where
              // there's no "next level down" left to back off to and
              // collecting happens regardless of size.
              if (mode == Mode::collecting &&
                  !agglomerates.empty() &&
                  agglomerates.back().size() >= max_cells_per_agglomerate)
                {
                  agglomerates.push_back({});
                  agglomerate_counter++;
                }

              agglomerates.back().push_back(active_c);
              
              // if needs refinement on leaf node，
              // set refine flag for deal.II to refine the mesh
              // if (evaluate_refinement && leaf_estimator) {
              //   if (leaf_estimator(level, active_c) == Mode::refining) {
              //     active_c->set_refine_flag();
              //   }
              // }
              if (force_refine)
                {
                  active_c->set_refine_flag();
                }
            } else {
              // if the grid has been refined by deal.II, recursively collect its child cells
              for (unsigned int i = 0; i < c->n_children(); ++i) {
                traverse(c->child(i));
              }
            }
          };
          traverse(base_cell);
        };
      if (mode == Mode::refining)
        {
          for (const auto &it : elements)
            {
              agglomerates.push_back({});
              collect_and_force_refinement(it.second, true);
              agglomerate_counter++;
            }
          return;
        }
      if (mode == Mode::collecting)
        {
          for (const auto &it : elements)
            collect_and_force_refinement(it.second, false);
          return;
        }
      

      agglomerates.push_back({});
      for (const auto &it : elements)
        collect_and_force_refinement(it.second, false);
      agglomerate_counter++;
    }
  } // namespace internal



  /**
   * Helper class which handles agglomeration based on the R-tree data
   * structure. Notice that the R-tree type is assumed to be an R-star-tree.
   */
  template <int dim, typename RtreeType>
  class CellsAgglomerator
  {
  public:
    template <int, int>
    friend class ::AgglomerationHandler;

    /**
     * Constructor. It takes a given rtree and an integer representing the
     * index of the level to be extracted.
     */
    CellsAgglomerator(const RtreeType   &rtree,
                      const unsigned int extraction_level);

    // /**
    //  * Extract agglomerates based on the current tree and the extraction level.
    //  * This function returns a reference to
    //  */
    // const std::vector<
    //   std::vector<typename Triangulation<dim>::active_cell_iterator>> &
    // extract_agglomerates();

    using RtreeView = boost::geometry::index::detail::rtree::utilities::view<RtreeType>;
    using InternalNode = typename boost::geometry::index::detail::rtree::internal_node<
        typename RtreeView::value_type,
        typename RtreeView::options_type::parameters_type,
        typename RtreeView::box_type,
        typename RtreeView::allocators_type,
        typename RtreeView::options_type::node_tag>::type;

    using EstimatorFunction = std::function<internal::Mode(size_t /*level*/, const InternalNode &)>;
    using LeafEstimatorFunction = std::function<internal::Mode(size_t /*level*/, const typename Triangulation<dim>::active_cell_iterator &)>;

    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &
    extract_adaptive_agglomerates(
      const EstimatorFunction &estimator,
      const LeafEstimatorFunction &leaf_est,
      size_t max_cells_per_agglomerate = std::numeric_limits<size_t>::max());


    template <typename VectorType>
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &
    extract_error_based_agglomerates(
      const VectorType &estimated_error,
      const double refine_threshold,
      const double coarsen_threshold,
      const size_t min_collecting_level = 1,
      size_t max_cells_per_agglomerate = std::numeric_limits<size_t>::max(),
      // Secondary, complementary size control (see max_cells_per_
      // agglomerate's own documentation for why IT is the one that
      // actually bounds cross-polytope size mismatch -- confirmed via
      // A/B testing in adaptive.cc). This one catches a narrower,
      // different failure mode: before collecting a node's whole subtree
      // into one polytope, check the RATIO between the largest and
      // smallest fine-cell diameter actually found under that node (i.e.
      // does this candidate polytope internally straddle a refinement-
      // level boundary). If that ratio exceeds max_diameter_ratio, don't
      // collect here -- return Mode::normal so the recursion backs off to
      // the tree's next level down instead, letting each child be
      // independently re-evaluated (which may itself collect, refine
      // further, or back off again). Note this is a ratio between the
      // node's OWN member cells' sizes, not the node's absolute physical
      // extent -- a large but internally-uniform region (same-size cells
      // throughout) always passes THIS check regardless of how many
      // cells/how much area it covers, which is exactly why
      // max_cells_per_agglomerate is still needed alongside it. Defaults
      // to "no cap" (infinity) so existing behavior is unaffected unless
      // explicitly opted in.
      double max_diameter_ratio = std::numeric_limits<double>::infinity());

    /**
     * Get total number of levels.
     */
    inline unsigned int
    get_n_levels() const;

    /**
     * Return the number of nodes present in level @p level.
     */
    inline types::global_cell_index
    get_n_nodes_per_level(const unsigned int level) const;

    /**
     * This function returns a map which associates to each node on level
     * @p extraction_level a list of children.
     */
    inline const std::map<
      std::pair<types::global_cell_index, types::global_cell_index>,
      std::vector<types::global_cell_index>> &
    get_hierarchy() const;

    static void assign_material_id(
      const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &agglomerates
    );

    static bool check_pollution(
      const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &agglomerates,
      bool print_warnings = true
    );

    template <typename MappingType>
    static std::map<types::material_id, std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>>>
    group_cells_by_material_id(
      const Triangulation<dim> &tria,
      const MappingType &mapping
    );

  private:
    /**
     * Raw pointer to the actual R-tree.
     */
    RtreeType *rtree;

    /**
     * Extraction level.
     */
    const unsigned int extraction_level;

    /**
     * Store agglomerates obtained after recursive extraction on nodes of
     * level @p extraction_level.
     */
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      agglomerates_on_level;

    /**
     * Vector storing the number of nodes (and, ultimately, agglomerates) for
     * each level.
     */
    std::vector<types::global_cell_index> n_nodes_per_level;

    /**
     * Map which maps a node parent @n on level @p l to a vector of integers
     * which stores the index of children.
     */
    std::map<std::pair<types::global_cell_index, types::global_cell_index>,
             std::vector<types::global_cell_index>>
      parent_node_to_children_nodes;
  };



  template <int dim, typename RtreeType>
  CellsAgglomerator<dim, RtreeType>::CellsAgglomerator(
    const RtreeType   &tree,
    const unsigned int extraction_level_)
    : extraction_level(extraction_level_)
  {
    rtree = const_cast<RtreeType *>(&tree);
    Assert(n_levels(*rtree), ExcMessage("At least two levels are needed."));
  }



  // template <int dim, typename RtreeType>
  // const std::vector<
  //   std::vector<typename Triangulation<dim>::active_cell_iterator>> &
  // CellsAgglomerator<dim, RtreeType>::extract_agglomerates()
  // {
  //   AssertThrow(extraction_level <= n_levels(*rtree),
  //               ExcInternalError("You are trying to extract level " +
  //                                std::to_string(extraction_level) +
  //                                " of the tree, but it only has a total of " +
  //                                std::to_string(n_levels(*rtree)) +
  //                                " levels."));
  //   using RtreeView =
  //     boost::geometry::index::detail::rtree::utilities::view<RtreeType>;
  //   RtreeView rtv(*rtree);

  //   n_nodes_per_level.resize(rtv.depth() +
  //                            1); // store how many nodes we have for each level.

  //   if (rtv.depth() == 0)
  //     {
  //       // The below algorithm does not work for `rtv.depth()==0`, which might
  //       // happen if the number entries in the tree is too small.
  //       agglomerates_on_level.resize(1);
  //       agglomerates_on_level[0].resize(1);
  //     }
  //   else
  //     {
  //       const unsigned int target_level =
  //         std::min<unsigned int>(extraction_level, rtv.depth());

  //       internal::RefineCoarsenRtreeVisitor<typename RtreeView::value_type,
  //                               typename RtreeView::options_type,
  //                               typename RtreeView::translator_type,
  //                               typename RtreeView::box_type,
  //                               typename RtreeView::allocators_type>
  //         extractor_visitor(rtv.translator(),
  //                           target_level,
  //                           agglomerates_on_level,
  //                           n_nodes_per_level,
  //                           parent_node_to_children_nodes);


  //       rtv.apply_visitor(extractor_visitor);
  //     }
  //   return agglomerates_on_level;
  // }

  template <int dim, typename RtreeType>
  const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &
  CellsAgglomerator<dim, RtreeType>::extract_adaptive_agglomerates(
    const EstimatorFunction &estimator,
    const LeafEstimatorFunction &leaf_est,
    size_t max_cells_per_agglomerate)
  {
      agglomerates_on_level.clear();

      RtreeView rtv(*rtree);

      n_nodes_per_level.resize(rtv.depth() + 1); 

      if (rtv.depth() == 0)
        {
          agglomerates_on_level.resize(1);
          agglomerates_on_level[0].resize(1);
        }
      else
        {
          internal::RefineCoarsenRtreeVisitor<
              typename RtreeView::value_type,
              typename RtreeView::options_type,
              typename RtreeView::translator_type,
              typename RtreeView::box_type,
              typename RtreeView::allocators_type>
            adaptive_visitor(
                rtv.translator(),
                agglomerates_on_level,
                n_nodes_per_level,
                parent_node_to_children_nodes,
                estimator,
                leaf_est,
                max_cells_per_agglomerate);

          rtv.apply_visitor(adaptive_visitor);
        }
      
      return agglomerates_on_level;
  }

  // Declared in CellsAgglomerator's public section.
  template <int dim, typename RtreeType>
  template <typename VectorType>
  const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &
  CellsAgglomerator<dim, RtreeType>::extract_error_based_agglomerates(
      const VectorType &estimated_error,
      const double refine_threshold,
      const double coarsen_threshold,
      const size_t min_collecting_level,
      size_t max_cells_per_agglomerate,
      double max_diameter_ratio)
  {
      const size_t max_tree_depth = this->get_n_levels();

      // Diagnostics: how many times does the estimator actually get
      // called (i.e. how many internal tree nodes get individually
      // re-evaluated), and how long does the whole extraction take wall-
      // clock-wise? This directly distinguishes "genuinely more work
      // because there's now real size disparity spread across a big
      // chunk of the domain" from "still a bug forcing needless descent"
      // -- rather than guessing at a third blind fix. Printed
      // unconditionally (every rank prints its own numbers, same
      // convention as the rest of this file's per-rank diagnostics).
      size_t estimator_call_count = 0;
      dealii::Timer extraction_timer;
      extraction_timer.start();

      auto estimator = [this, &estimated_error, refine_threshold, coarsen_threshold, max_tree_depth, min_collecting_level, max_diameter_ratio, &estimator_call_count]
                       (size_t level, const auto &node) -> internal::Mode 
      {
          ++estimator_call_count;

          if (level == 0) return internal::Mode::normal;

          using NodeType = std::decay_t<decltype(node)>;
          using elements_type = typename boost::geometry::index::detail::rtree::elements_type<NodeType>::type;
          const elements_type &elements = boost::geometry::index::detail::rtree::elements(node);
          if (elements.empty()) return internal::Mode::normal;
          // if level is too shallow, skip this level
          if (level < min_collecting_level) {
              return internal::Mode::normal;
          }
          // compute the bounding box of the whole node, 
          // and get all the leaves that is contained in the bounding box
          auto node_bbox = elements[0].first;
          for (size_t idx = 1; idx < elements.size(); ++idx) 
              boost::geometry::expand(node_bbox, elements[idx].first); 

          std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>> leaves_inside;

          this->rtree->query(boost::geometry::index::intersects(node_bbox), std::back_inserter(leaves_inside));
          // compute the maximum error among the leaves, and (for the size
          // check below) the min/max fine-cell diameter among them -- this
          // node might already span cells of different refinement levels
          // (the R-tree groups by spatial proximity, not by level).
          double max_error_in_node = 0.0;
          double min_leaf_diameter = std::numeric_limits<double>::max();
          double max_leaf_diameter = 0.0;
          for (const auto &leaf : leaves_inside) {
              max_error_in_node = std::max(max_error_in_node, (double)estimated_error[leaf.second->active_cell_index()]);
              min_leaf_diameter = std::min(min_leaf_diameter, leaf.second->diameter());
              max_leaf_diameter = std::max(max_leaf_diameter, leaf.second->diameter());
          }

          if (max_error_in_node > refine_threshold) 
            return internal::Mode::refining;

          // Would collecting this WHOLE node into one polytope merge fine
          // cells of very different SIZES together? This checks the ratio
          // between the LARGEST and SMALLEST fine-cell diameter actually
          // found under this node -- i.e. does this candidate polytope
          // straddle a refinement-level boundary internally -- NOT the
          // node's absolute physical extent relative to its smallest cell.
          //
          // That distinction matters a lot: a node spanning a large but
          // perfectly UNIFORM region (many same-sized coarse cells, e.g.
          // far from the Gaussian peaks) has max_leaf_diameter ==
          // min_leaf_diameter, so this check passes regardless of how many
          // cells/how large an area it covers -- collecting it in one shot
          // is completely safe (h_f between it and an equally-uniform
          // neighbor stays ~1). An earlier version of this check compared
          // the node's absolute bounding-box diagonal against its smallest
          // cell's diameter instead -- that's wrong: a uniform node
          // spanning k same-size cells has diagonal ~ sqrt(k) * cell_size
          // purely from packing more cells over a larger area, so it
          // flagged "too big" for almost any node with more than a
          // handful of cells even when there was zero actual size
          // disparity -- forcing near-total descent to the tree's deepest
          // level EVERYWHERE, not just near sharp features, which is what
          // made this pass dramatically slower without actually improving
          // conditioning any more than the correct (ratio-based) check
          // below does.
          const bool node_too_big =
              std::isfinite(max_diameter_ratio) &&
              min_leaf_diameter > 0.0 &&
              min_leaf_diameter < std::numeric_limits<double>::max() &&
              (max_leaf_diameter / min_leaf_diameter > max_diameter_ratio);

          const bool at_deepest_level = (level >= max_tree_depth - 1);

          if (max_error_in_node < coarsen_threshold) {
              if (node_too_big && !at_deepest_level)
                  return internal::Mode::normal; // back off one level down
              else
                  return internal::Mode::collecting; // safe size, or no deeper level to retreat to
          } else {
              return at_deepest_level ? internal::Mode::collecting : internal::Mode::normal;
          }
      };

      auto leaf_est = [](size_t, const auto &) { 
          return internal::Mode::collecting; 
      };

      const auto &result = this->extract_adaptive_agglomerates(estimator, leaf_est, max_cells_per_agglomerate);

      extraction_timer.stop();

      return result;
  }

  template <int dim, typename RtreeType>
  void CellsAgglomerator<dim, RtreeType>::assign_material_id(
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &agglomerates) 
  {
    unsigned int poly_id = 0;
    for (const auto &agglo : agglomerates) 
      {
        for (auto &cell : agglo) 
          {
            cell->set_material_id(poly_id);
          }
        poly_id++;
      }
  }

  template <int dim, typename RtreeType>
  bool CellsAgglomerator<dim, RtreeType>::check_pollution(
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &agglomerates,
    bool print_warnings
  )  
  {
    bool has_pollution = false;
    for (size_t agg_idx = 0; agg_idx < agglomerates.size(); ++agg_idx) 
      {
        const auto &agglo = agglomerates[agg_idx];
        if (agglo.empty()) continue;

        types::material_id reference_id = agglo.front()->material_id();

        for (const auto  &cell : agglo) 
          {
            if (cell->material_id() != reference_id) 
              {
                if (print_warnings) {
                    std::cerr << "Find pollution!"
                              << "Polytope id: " << agg_idx 
                              << " containing cells with different material_ids: " 
                              << reference_id << " and " << cell->material_id() 
                              << std::endl;
                }
                has_pollution = true;
              }
          }
      }
    return has_pollution;
  }

  template <int dim, typename RtreeType>
  template <typename MappingType>
  std::map<types::material_id, std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>>>
  CellsAgglomerator<dim, RtreeType>::group_cells_by_material_id(
      const Triangulation<dim> &tria,
      const MappingType &mapping)
  {
      std::map<types::material_id, std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>>> boxes_by_macro;
      for (const auto &cell : tria.active_cell_iterators()) {
          // LOCALLY-OWNED ONLY. Without this filter, ghost (and
          // artificial) active cells were bundled into the material-id
          // groups too -- and estimated_error is only ever WRITTEN for
          // locally-owned cells (every writer in adaptive_test.cc loops
          // with an is_locally_owned() guard), so a ghost cell's entry
          // is whatever reinit() zero-initialized it to: exactly 0.0.
          // Feeding those phantom zero-error members into group_ok
          // dragged whole groups' max-error verdicts toward "clean
          // enough to keep/merge" -- disproportionately at the peaks,
          // where p4est's partition lines cut through the refinement
          // hotspots and ghost density is highest. Confirmed directly
          // by the [recover-partial-ghost-check] diagnostics: in the
          // 8-process run, 568 merge events had err_max EXACTLY 0.0
          // (520 of them touching ghosts, median max/thr = 0 near the
          // peaks), while the single-process run -- no ghost cells at
          // all -- had zero such events and a near-peak median max/thr
          // of 1.0 (i.e. only legitimate at-the-line threshold cases).
          // This, not the (also-fixed) skipped cross-rank jump terms,
          // was the primary driver of the multi-process-only "holes"
          // in the peak cores. Agglomerates are per-rank-local by
          // design (a polytope never spans ranks), so non-owned cells
          // have no business in these groups in the first place.
          if (!cell->is_locally_owned())
              continue;
          boxes_by_macro[cell->material_id()].push_back(std::make_pair(mapping.get_bounding_box(cell), cell));
      }
      return boxes_by_macro;
  }

  // ------------------------------ inline functions -------------------------

  template <int dim, typename RtreeType>
  inline unsigned int
  CellsAgglomerator<dim, RtreeType>::get_n_levels() const
  {
    return n_levels(*rtree);
  }



  template <int dim, typename RtreeType>
  inline types::global_cell_index
  CellsAgglomerator<dim, RtreeType>::get_n_nodes_per_level(
    const unsigned int level) const
  {
    return n_nodes_per_level[level];
  }



  template <int dim, typename RtreeType>
  inline const std::map<
    std::pair<types::global_cell_index, types::global_cell_index>,
    std::vector<types::global_cell_index>> &
  CellsAgglomerator<dim, RtreeType>::get_hierarchy() const
  {
    Assert(parent_node_to_children_nodes.size(),
           ExcMessage(
             "The hierarchy has not been computed. Did you forget to call"
             " extract_agglomerates() first?"));
    return parent_node_to_children_nodes;
  }
} // namespace dealii


// ===========================================================================
// PolytopalAdaptivity: generic, problem-independent utilities for adaptive
// polytopal (agglomerated) discretizations. Extracted from the Gaussian-AMR
// driver once they had stabilized there; nothing in this namespace knows
// about a specific PDE, RHS, or error indicator -- everything is
// parameterized by a per-cell error field and scale-free thresholds.
//
// Contents:
//  * split_polytope_until_small_enough   -- diameter-targeted recursive
//                                           R-tree splitting of one group
//  * extract_agglomerates_by_level_retreat -- error/uniformity-driven
//                                           top-down regrouping of one group
//                                           with root-level acceptance
//  * split_into_connected_components     -- face-adjacency connected-
//                                           component decomposition (R-tree
//                                           packing is proximity-based and
//                                           does NOT guarantee connectivity)
//  * report_disconnected_polytopes       -- diagnostic-only connectivity scan
//  * compute_bulk_threshold              -- Doerfler (error-mass) marking
//                                           threshold, MPI-collective-safe
//  * compute_count_capped_threshold      -- fraction-of-cells marking
//                                           threshold, dual to the above
//                                           (guards against the mass-based
//                                           criterion collapsing to ~0 on
//                                           spatially-uniform error fields)
//  * mark_singleton_cells_for_refinement -- two-tier marking: only cells
//                                           that are their own polytope and
//                                           exceed hysteresis*L get refine
//                                           flags (mesh refinement inside a
//                                           multi-cell polytope adds no DoFs)
//  * apply_polytope_diameter_cap         -- absolute h_f_max control
// ===========================================================================

#include <deal.II/base/mpi.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping.h>

#include <map>
#include <queue>
#include <unordered_set>

namespace PolytopalAdaptivity
{
  using namespace dealii;

  // Self-contained replacement for constexpr_pow -- including
  // poly_utils.h here is not possible (it has include-order dependencies
  // on the full AgglomerationHandler declaration and deal.II FE headers,
  // which are not available at this point in the include graph).
  constexpr unsigned int
  constexpr_pow(const unsigned int base, const unsigned int exp)
  {
    return exp == 0 ? 1u : base * constexpr_pow(base, exp - 1);
  }


// NOTE (post-ablation cleanup): find_oversized_neighbor_polytope_targets
// and the probe/re-split neighbor-size correction that used it were
// deleted after ablation Test A showed them unnecessary once the
// interface-quadrature pairing fix landed in agglomeration_handler.cc
// (see the note at the top of STEP 3). split_polytope_until_small_enough
// below is KEPT: it is still used by STEP 2's polytope-diameter cap.


// Splits one polytope's fine-cell set until every resulting sub-polytope's
// diameter is <= target_max_diameter (see the rationale in the comment
// block above). Reuses the same underlying recursive tree-retreat
// machinery (extract_adaptive_agglomerates()) as normal agglomeration,
// just with a custom criterion tailored to this externally-known target
// instead of the internal max/min-cell-ratio check.
//
// If cells.size() is already 1, there's nothing left to split -- this
// polytope IS a single, unrefined background cell that simply hasn't
// itself been through enough mesh refinement, and no amount of
// re-agglomeration can shrink it further. The caller can detect this case
// (the returned single group's size will still be 1 and its diameter will
// still exceed target_max_diameter) to decide whether a genuinely
// different fix -- forcing actual mesh refinement on that specific cell,
// not re-agglomeration -- is needed instead.
template <int dim, typename RtreeTag>
std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
split_polytope_until_small_enough(
    const std::vector<typename Triangulation<dim>::active_cell_iterator> &cells,
    const Mapping<dim> &mapping,
    double target_max_diameter)
{
    // A group this small can't build a meaningful R-tree either --
    // CellsAgglomerator asserts "at least two levels are needed", which
    // fails for any element count up to max_elem_per_node (not just a
    // single cell -- this guard used to only check <= 1, which is why it
    // still crashed on groups of 2-4 cells). Handle it directly instead:
    // check whether keeping them as one group already satisfies
    // target_max_diameter; if not, the only thing left to try is
    // exploding to individual cells.
    constexpr unsigned int local_max_elem_per_node = constexpr_pow(2, dim);
    if (cells.size() <= local_max_elem_per_node)
    {
        if (cells.size() <= 1)
            return {cells};

        BoundingBox<dim> merged = mapping.get_bounding_box(cells[0]);
        for (size_t i = 1; i < cells.size(); ++i)
            merged.merge_with(mapping.get_bounding_box(cells[i]));
        const auto bp     = merged.get_boundary_points();
        const double diag = (bp.second - bp.first).norm();

        if (diag <= target_max_diameter)
            return {cells};

        std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> singletons;
        singletons.reserve(cells.size());
        for (const auto &cell : cells)
            singletons.push_back({cell});
        return singletons;
    }

    std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>> local_boxes;
    local_boxes.reserve(cells.size());
    for (const auto &cell : cells)
        local_boxes.push_back(std::make_pair(mapping.get_bounding_box(cell), cell));

    auto local_tree = pack_rtree<RtreeTag>(local_boxes);
    CellsAgglomerator<dim, decltype(local_tree)> local_agglomerator{local_tree, 0};
    const size_t max_tree_depth = local_agglomerator.get_n_levels();

    auto estimator = [target_max_diameter, max_tree_depth]
                     (size_t level, const auto &node) -> dealii::internal::Mode {
        if (level == 0)
            return dealii::internal::Mode::normal; // root: keep descending past it

        using NodeType = std::decay_t<decltype(node)>;
        using elements_type = typename boost::geometry::index::detail::rtree::elements_type<NodeType>::type;
        const elements_type &elements = boost::geometry::index::detail::rtree::elements(node);
        if (elements.empty())
            return dealii::internal::Mode::normal;

        auto node_bbox = elements[0].first;
        for (size_t idx = 1; idx < elements.size(); ++idx)
            boost::geometry::expand(node_bbox, elements[idx].first);

        const double node_diagonal =
            boost::geometry::distance(node_bbox.min_corner(), node_bbox.max_corner());

        const bool at_deepest_level = (level >= max_tree_depth - 1);

        // Collect here if this node is ALREADY small enough (the actual
        // criterion we care about, unlike extract_error_based_
        // agglomerates's internal-uniformity check), or if there's no
        // deeper level left to retreat to.
        if (node_diagonal <= target_max_diameter || at_deepest_level)
            return dealii::internal::Mode::collecting;
        else
            return dealii::internal::Mode::normal; // retreat one level down
    };

    auto leaf_est = [](size_t, const auto &) { return dealii::internal::Mode::collecting; };

    auto result = local_agglomerator.extract_adaptive_agglomerates(estimator, leaf_est);

    // Final cleanup: the tree's leaf-node granularity may still bundle a
    // handful of cells together even at the deepest level, which could
    // leave a group whose diameter still exceeds target_max_diameter.
    // Explode just that group down to individual cells -- the only thing
    // left to try. If even a single cell's own diameter exceeds
    // target_max_diameter, that cell is returned as its own (size-1)
    // group unchanged -- see the function-level comment on why that
    // specific case can't be fixed by re-agglomeration at all.
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> final_result;
    final_result.reserve(result.size());
    for (auto &group : result)
    {
        if (group.size() <= 1)
        {
            final_result.push_back(std::move(group));
            continue;
        }

        BoundingBox<dim> merged = mapping.get_bounding_box(group[0]);
        for (size_t i = 1; i < group.size(); ++i)
            merged.merge_with(mapping.get_bounding_box(group[i]));
        const auto bp   = merged.get_boundary_points();
        const double diag = (bp.second - bp.first).norm();

        if (diag > target_max_diameter)
        {
            for (const auto &cell : group)
                final_result.push_back({cell});
        }
        else
        {
            final_result.push_back(std::move(group));
        }
    }
    return final_result;
}



// Forward declaration: defined later in this file, but needed here
// (with explicit template-argument syntax, split_into_connected_components<dim>(...))
// by extract_agglomerates_by_level_retreat's partial-recovery cleanup
// below. Without this, the compiler doesn't yet know the name is a
// template and misparses the '<' as less-than.
template <int dim>
std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
split_into_connected_components(
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &groups);


// =====================================================================
// Level-by-level retreat version of the "large macro group" branch of
// group_cells_by_material_id re-agglomeration (STEP 2's else-branch,
// local_boxes.size() > max_elem_per_node case). Reuses exactly the same
// recursive tree-retreat machinery as split_polytope_until_small_enough
// above, but with an error/diameter-ratio-based stopping criterion
// instead of a fixed diameter target: a candidate node is accepted
// (collected as one polytope) as soon as its max cell error is <=
// refine_threshold AND its internal max/min cell-diameter ratio is <=
// max_diameter_ratio (and its cell count is <= max_cells_per_agglomerate);
// otherwise it retreats one level further down, splitting into
// ~max_elem_per_node (4 in 2D / 8 in 3D) smaller sub-polytopes, all the
// way to individual leaf cells if nothing else satisfies the criterion.
//
// Replaces a single one-shot call to extract_error_based_agglomerates
// with a genuine top-down walk of the local tree: a low-error,
// internally-uniform region is accepted EARLY at a shallow level (=> a
// large polytope), while only the sub-tree that is still too erroneous
// or too disparate keeps retreating. This gives the
// group_cells_by_material_id reuse path a real per-cycle "is this
// group still small/uniform enough to stay one polytope, or does it
// need to split into 4 smaller ones" decision, instead of the previous
// binary keep-whole/explode-to-singletons choice used for small groups.
// =====================================================================
template <int dim, typename RtreeTag>
std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
extract_agglomerates_by_level_retreat(
    const std::vector<typename Triangulation<dim>::active_cell_iterator> &cells,
    const Mapping<dim> &mapping,
    const Vector<float> &estimated_error,
    double refine_threshold,
    double max_diameter_ratio,
    size_t max_cells_per_agglomerate,
    double accept_margin = 0.6,
    double small_group_penalty = 0.6,
    // MERGE FORECAST GATE (option B from the dg_degree>=2 non-convergence
    // investigation -- replaces the earlier option-A curvature-relative-
    // variation heuristic, which fired far too broadly outside the
    // peaks and needed an untunable magic threshold). estimated_error's
    // volume term stops being layout-independent once the Delta u_h
    // correction is added (Delta u_h is evaluated from whichever
    // polytope's SHARED polynomial a cell currently belongs to -- see
    // compute_polygonal_error_indicator), so comparing a candidate
    // group's cells' pre-merge estimated_error against merge_threshold
    // is not a valid forecast of the post-merge error at degree >= 2.
    // Instead of reading stale per-cell values, DIRECTLY forecast the
    // best case: solve a small local least-squares problem for the
    // degree-p polynomial (in monomial basis, box-midpoint-centered,
    // half-extent-scaled local coordinates -- same convention as the
    // library's own MappingBox) whose Laplacian best cancels f over the
    // candidate box in the L2 sense, then measure the SAME h_K^2-
    // weighted ||f + Delta(fit)||^2 quantity the real indicator uses.
    // If even the best-possible degree-p fit leaves a large residual,
    // merging is doomed regardless of what any particular previous
    // solution happens to look like nearby. Reuses merge_threshold_for(...)
    // as the bar (same units, same scale as the real indicator), so no
    // new tunable constant is introduced for the volume part.
    //
    // NEIGHBOR COUPLING (added after empirically finding the volume-
    // only forecast wasn't enough): the volume-only version above
    // still let some groups through whose OWN best-case f-fit looked
    // fine in isolation, but whose true post-merge error (measured
    // later the same cycle, once the real SIPG-coupled solve ran) was
    // 13-16x their singleton neighbors' -- traced to exactly this: the
    // volume-only forecast has no way to know that the real DG solve
    // has to compromise between fitting f locally AND staying
    // consistent with its neighbors across the SIPG interface penalty,
    // so an isolated best-fit can be optimistic relative to what the
    // coupled solve actually produces. neighbor_lookup (if provided)
    // is a { fine-cell CellId -> {that cell's LAST cycle's polytope
    // bounding box, that polytope's LAST cycle's solved DOF
    // coefficients} } map, built once per cycle in adaptive_test.cc
    // (CellId, not active_cell_index, because the latter is renumbered
    // by refinement and this map must survive being read the cycle
    // after it's built). For every face of the candidate group that
    // borders a cell OUTSIDE the group, if that neighbor (or an
    // ancestor, walking up if it was refined since the map was built)
    // is found in the lookup, the neighbor's actual last-known value at
    // the face midpoint is added to the SAME least-squares system as an
    // extra row (value-jump only, not the full grad+value SIPG form --
    // scoped down for a first pass), and to the reported forecast_norm
    // using the same val_jump_sq/h_e0 convention as compute_polygonal_
    // error_indicator's real face term. A face with no available
    // neighbor info (boundary, or the neighbor isn't in the map)
    // silently contributes nothing -- degrades gracefully to the
    // volume-only forecast rather than failing.
    const Function<dim> *rhs_function        = nullptr,
    bool                 check_merge_forecast = false,
    unsigned int         forecast_fe_degree   = 2,
    const std::map<CellId, std::pair<BoundingBox<dim>, std::vector<double>>> *neighbor_lookup = nullptr)
{
    constexpr unsigned int local_max_elem_per_node = constexpr_pow(2, dim);

    // MERGE MARGIN: every keep-together/merge decision in this function
    // uses a merge_threshold below refine_threshold instead of
    // refine_threshold itself. Motivation (measured, not hypothesized):
    // the [root-accept-check] diagnostics showed the 8-process run
    // accepting 31 small groups whole INSIDE the peak cores at
    // max/thr ratios of 0.612..0.99 -- errors computed correctly, but
    // sitting just under the line -- while the 1-process run had ZERO
    // such events. The difference is purely group COMPOSITION: a
    // polytope can't span ranks, so p4est's partition line through a
    // peak splits the core's cells into small per-rank groups, and a
    // group that happens to receive only the "cooler" cells (its
    // hotter neighbors landing in the other rank's group) passes the
    // bare <= refine_threshold check, merges, and thereby exits the
    // two-tier h-refinement candidate pool -- the multi-process-only
    // "holes" in the peak cores. Requiring merges to clear a real
    // margin (0.6 sits just below the lowest observed offending ratio,
    // 0.612) eliminates exactly those at-the-line merges while leaving
    // genuinely-converged regions (whose errors sit far below the
    // line) untouched. Splitting/marking decisions elsewhere still use
    // the full refine_threshold / mark_threshold -- this margin only
    // makes MERGING more conservative, i.e. it's a hysteresis band:
    // merge only below accept_margin*L, split above L, no flip-flopping
    // between.
    //
    // SMALL-GROUP EXTRA PENALTY: even with the margin above, small
    // (2-8 cell) merged polytopes near the peaks kept showing up as
    // persistent "holes" -- but a full diagnostic chain (compute_
    // polygonal_error_indicator's [island-vs-singleton-check] and
    // [island-face-breakdown]) ruled out every candidate NUMERICAL
    // explanation in turn: the volume/RHS term matched their singleton
    // neighbors to within 1-3%; no single "hot" neighboring face
    // dominated (contributions were fairly even across the whole
    // boundary); and refinement level matched the deepest level near
    // the same peak exactly (island_levels_behind was 0 in every
    // observed case). What was NOT ruled out: face_error_sq's
    // measure-proportional per-cell sharing (see compute_polygonal_
    // error_indicator's "NON-UNIFORM per-cell distribution" comment)
    // is structurally biased upward for small polytopes -- their
    // island_facesq_mean ran 60-140% higher than same-level singleton
    // neighbors' facesq_mean, purely from how a small agglomerate's
    // external-boundary jump total gets divided by its (small) cell
    // count, independent of true local solution quality. Rather than
    // rework that sharing formula (which every other part of the
    // pipeline already depends on), small groups get an EXTRA
    // multiplicative penalty on top of accept_margin: they must clear
    // accept_margin * small_group_penalty * refine_threshold to merge,
    // roughly compensating for the observed 1.6x-2.4x over-reporting
    // instead of just accepting it at face value.
    constexpr size_t small_group_max_size = 8; // matches the diagnostics' filter range
    auto merge_threshold_for = [&](size_t grp_size) {
        double m = accept_margin;
        if (grp_size >= 2 && grp_size <= small_group_max_size)
            m *= small_group_penalty;
        return m * refine_threshold;
    };

    // [merge-forecast] see the MERGE FORECAST GATE note on the function
    // parameters above. Enumerates the tensor-product monomial basis
    // {prod_d xi_d^a_d : 0 <= a_d <= forecast_fe_degree} in box-
    // midpoint-centered, half-extent-scaled local coordinates xi (same
    // dimension, (p+1)^dim, as FE_DGQ(p)'s dof count -- spans the same
    // polynomial space regardless of which basis is used, so this is a
    // fair stand-in for "what could FE_DGQ(p) represent here").
    auto forecast_too_bad = [&](const std::vector<typename Triangulation<dim>::active_cell_iterator> &grp) {
        if (!check_merge_forecast || rhs_function == nullptr || grp.size() < 2)
            return false;

        Point<dim> bbox_min, bbox_max;
        for (unsigned int d = 0; d < dim; ++d) {
            bbox_min[d] = std::numeric_limits<double>::max();
            bbox_max[d] = std::numeric_limits<double>::lowest();
        }
        for (const auto &cell : grp)
            for (const auto v : cell->vertex_indices()) {
                const auto &vertex = cell->vertex(v);
                for (unsigned int d = 0; d < dim; ++d) {
                    bbox_min[d] = std::min(bbox_min[d], vertex[d]);
                    bbox_max[d] = std::max(bbox_max[d], vertex[d]);
                }
            }

        Point<dim> center, half_extent;
        for (unsigned int d = 0; d < dim; ++d) {
            center[d]      = 0.5 * (bbox_min[d] + bbox_max[d]);
            half_extent[d] = 0.5 * (bbox_max[d] - bbox_min[d]);
            if (half_extent[d] <= 0.0)
                return false; // degenerate box, nothing to forecast
        }

        // Multi-indices for the (forecast_fe_degree+1)^dim tensor basis.
        const unsigned int p1 = forecast_fe_degree + 1;
        unsigned int n_basis = 1;
        for (unsigned int d = 0; d < dim; ++d)
            n_basis *= p1;
        std::vector<std::array<unsigned int, dim>> multi_index(n_basis);
        for (unsigned int lin = 0; lin < n_basis; ++lin) {
            unsigned int rem = lin;
            for (unsigned int d = 0; d < dim; ++d) {
                multi_index[lin][d] = rem % p1;
                rem /= p1;
            }
        }

        // Gather quadrature points/weights across every cell in the
        // candidate group, using a throwaway FE_Q(1) purely to drive
        // FEValues' geometry machinery (update_quadrature_points |
        // update_JxW_values doesn't touch shape functions at all).
        const unsigned int quadrature_degree = 2 * forecast_fe_degree + 2;
        static const FE_Q<dim>  dummy_fe(1);
        const QGauss<dim>       quad(quadrature_degree);
        FEValues<dim>           fe_values(mapping, dummy_fe, quad,
                                          update_quadrature_points | update_JxW_values);

        std::vector<Point<dim>> q_points_all;
        std::vector<double>     jxw_all;
        std::vector<double>     h2_all;
        q_points_all.reserve(grp.size() * quad.size());
        jxw_all.reserve(grp.size() * quad.size());
        h2_all.reserve(grp.size() * quad.size());
        for (const auto &cell : grp) {
            fe_values.reinit(cell);
            const double h2 = cell->diameter() * cell->diameter();
            for (unsigned int q = 0; q < quad.size(); ++q) {
                q_points_all.push_back(fe_values.quadrature_point(q));
                jxw_all.push_back(fe_values.JxW(q));
                h2_all.push_back(h2);
            }
        }
        const unsigned int n_q = q_points_all.size();

        // Design matrix A(q,i) = physical Laplacian of monomial i at
        // q_points_all[q]. Chain rule for the diagonal affine map
        // xi_d = (x_d - center_d) / half_extent_d (same convention
        // established for MappingBox's own Hessian transform in
        // compute_polygonal_error_indicator): d^2/dx_d^2 (xi_d^a) =
        // a*(a-1)*xi_d^(a-2) / half_extent_d^2.
        FullMatrix<double> A(n_q, n_basis);
        std::vector<double> f_vals(n_q);
        for (unsigned int q = 0; q < n_q; ++q) {
            std::array<double, dim> xi;
            for (unsigned int d = 0; d < dim; ++d)
                xi[d] = (q_points_all[q][d] - center[d]) / half_extent[d];
            f_vals[q] = rhs_function->value(q_points_all[q]);

            for (unsigned int i = 0; i < n_basis; ++i) {
                double lap = 0.0;
                for (unsigned int d = 0; d < dim; ++d) {
                    const unsigned int a = multi_index[i][d];
                    if (a < 2)
                        continue; // second derivative of xi_d^0 or xi_d^1 is 0
                    double term = a * (a - 1) * std::pow(xi[d], static_cast<int>(a) - 2)
                                  / (half_extent[d] * half_extent[d]);
                    for (unsigned int d2 = 0; d2 < dim; ++d2)
                        if (d2 != d)
                            term *= std::pow(xi[d2], static_cast<int>(multi_index[i][d2]));
                    lap += term;
                }
                A(q, i) = lap;
            }
        }

        // NEIGHBOR COUPLING: gather one value-matching sample per
        // external face that borders a cell present in neighbor_lookup
        // (see the long comment on the neighbor_lookup parameter above).
        // B(k,i) = monomial i's VALUE (not Laplacian) at that face's
        // midpoint, in the CANDIDATE's local coordinates; g(k) = the
        // neighbor's actual last-known value there, evaluated from its
        // own stored bbox + DOF coefficients via genuine FE_DGQ shape
        // functions (so this reflects the real basis the solve uses,
        // not the fit's monomial basis).
        const double h_e0 = 2.0 * std::sqrt([&]{
            double s = 0.0;
            for (unsigned int d = 0; d < dim; ++d) s += half_extent[d] * half_extent[d];
            return s;
        }());
        std::vector<double>              face_weight;      // value-jump weight: measure/h_e0
        std::vector<double>              face_weight_grad; // grad-jump weight: measure*h_e0
        std::vector<std::array<double, dim>> face_xi;   // candidate-local coords of each sample
        std::vector<std::array<double, dim>> face_normal; // outward unit normal, physical
        std::vector<double>              face_g;         // neighbor's value there
        std::vector<double>              face_g_gradn;    // neighbor's physical grad, dotted with face_normal
        if (neighbor_lookup != nullptr)
        {
            std::unordered_set<unsigned int> grp_indices;
            grp_indices.reserve(grp.size());
            for (const auto &c : grp)
                grp_indices.insert(c->active_cell_index());

            static const FE_DGQ<dim> neighbor_fe(forecast_fe_degree);
            const unsigned int n_dofs_nb = neighbor_fe.n_dofs_per_cell();

            for (const auto &cell : grp)
                for (const auto f : cell->face_indices())
                {
                    if (cell->at_boundary(f))
                        continue;
                    auto neighbor = cell->neighbor(f);
                    if (!neighbor->is_active() || grp_indices.count(neighbor->active_cell_index()))
                        continue; // hanging-node neighbor (skipped for scope) or internal face

                    // Walk up to an ancestor if this exact cell was refined
                    // since neighbor_lookup was built (map lookup misses on
                    // the exact CellId then).
                    typename Triangulation<dim>::cell_iterator anc = neighbor;
                    typename std::map<CellId, std::pair<BoundingBox<dim>, std::vector<double>>>::const_iterator it
                        = neighbor_lookup->end();
                    for (unsigned int walk = 0; walk < 4; ++walk)
                    {
                        it = neighbor_lookup->find(anc->id());
                        if (it != neighbor_lookup->end() || anc->level() == 0)
                            break;
                        anc = anc->parent();
                    }
                    if (it == neighbor_lookup->end())
                        continue; // no info for this neighbor -- skip, graceful degradation

                    const BoundingBox<dim> &bbox_nb  = it->second.first;
                    const std::vector<double> &coeffs_nb = it->second.second;
                    if (coeffs_nb.size() != n_dofs_nb)
                        continue; // stored solve used a different degree -- skip rather than misread

                    const Point<dim> x_face = cell->face(f)->center();
                    const auto nb_bp = bbox_nb.get_boundary_points();
                    Point<dim> unit_nb;
                    bool ok = true;
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        const double extent = nb_bp.second[d] - nb_bp.first[d];
                        if (extent <= 0.0) { ok = false; break; }
                        unit_nb[d] = (x_face[d] - nb_bp.first[d]) / extent;
                    }
                    if (!ok)
                        continue;

                    double nb_value = 0.0;
                    for (unsigned int i = 0; i < n_dofs_nb; ++i)
                        nb_value += coeffs_nb[i] * neighbor_fe.shape_value(i, unit_nb);

                    // Neighbor's PHYSICAL gradient at x_face: reference
                    // gradient scaled by 1/extent_nb per axis (chain rule
                    // for unit_nb[d] = (x[d]-lower[d])/extent[d]), dotted
                    // with the outward unit normal. For an axis-aligned
                    // Cartesian quad, the direction from the cell's own
                    // center to the face's center IS that face's outward
                    // normal -- avoids needing FEFaceValues/GeometryInfo
                    // just for this.
                    std::array<double, dim> normal;
                    {
                        double norm_len_sq = 0.0;
                        for (unsigned int d = 0; d < dim; ++d)
                        {
                            normal[d] = x_face[d] - cell->center()[d];
                            norm_len_sq += normal[d] * normal[d];
                        }
                        const double norm_len = std::sqrt(norm_len_sq);
                        if (norm_len <= 0.0)
                            continue;
                        for (unsigned int d = 0; d < dim; ++d)
                            normal[d] /= norm_len;
                    }
                    double nb_gradn = 0.0;
                    for (unsigned int i = 0; i < n_dofs_nb; ++i)
                    {
                        const auto ref_grad = neighbor_fe.shape_grad(i, unit_nb);
                        double phys_grad_dot_n = 0.0;
                        for (unsigned int d = 0; d < dim; ++d)
                        {
                            const double extent = nb_bp.second[d] - nb_bp.first[d];
                            phys_grad_dot_n += (ref_grad[d] / extent) * normal[d];
                        }
                        nb_gradn += coeffs_nb[i] * phys_grad_dot_n;
                    }

                    std::array<double, dim> xi;
                    for (unsigned int d = 0; d < dim; ++d)
                        xi[d] = (x_face[d] - center[d]) / half_extent[d];

                    const double measure = cell->face(f)->measure();
                    face_weight.push_back(measure / h_e0);
                    face_weight_grad.push_back(measure * h_e0);
                    face_xi.push_back(xi);
                    face_normal.push_back(normal);
                    face_g.push_back(nb_value);
                    face_g_gradn.push_back(nb_gradn);
                }
        }

        // Monomial VALUE (not Laplacian) at local coords xi, for the
        // value-jump face rows below.
        auto monomial_value = [&](unsigned int i, const std::array<double, dim> &xi) {
            double v = 1.0;
            for (unsigned int d = 0; d < dim; ++d)
                v *= std::pow(xi[d], static_cast<int>(multi_index[i][d]));
            return v;
        };

        // Monomial GRADIENT (physical units, chain rule through
        // xi_d = (x_d-center_d)/half_extent_d), dotted with the given
        // outward normal -- for the grad-jump face rows below. Same
        // affine-map convention as the Laplacian design matrix A above.
        auto monomial_grad_normal = [&](unsigned int i, const std::array<double, dim> &xi,
                                         const std::array<double, dim> &normal) {
            double g = 0.0;
            for (unsigned int d = 0; d < dim; ++d) {
                const unsigned int a = multi_index[i][d];
                if (a == 0)
                    continue;
                double term = a * std::pow(xi[d], static_cast<int>(a) - 1) / half_extent[d];
                for (unsigned int d2 = 0; d2 < dim; ++d2)
                    if (d2 != d)
                        term *= std::pow(xi[d2], static_cast<int>(multi_index[i][d2]));
                g += term * normal[d];
            }
            return g;
        };

        // Normal equations for the JxW-weighted least-squares fit
        // minimizing || A c - (-f) ||^2_{JxW}
        //   + sum_faces w_val_k (B_k . c - g_k)^2       (value jump)
        //   + sum_faces w_grad_k (C_k . c - gn_k)^2      (grad jump, normal component)
        // Small Tikhonov term guards against a near-singular system for
        // degenerate/very small candidate groups rather than letting
        // gauss_jordan() blow up.
        FullMatrix<double> AtA(n_basis, n_basis);
        dealii::Vector<double> Atb(n_basis);
        for (unsigned int q = 0; q < n_q; ++q) {
            const double w = jxw_all[q];
            for (unsigned int i = 0; i < n_basis; ++i) {
                Atb(i) += w * A(q, i) * (-f_vals[q]);
                for (unsigned int j = 0; j < n_basis; ++j)
                    AtA(i, j) += w * A(q, i) * A(q, j);
            }
        }
        // FITTING weight vs REPORTING weight, deliberately different:
        // the real SIPG assembly's boundary/interface penalty is
        // penalty_constant/h_e (see solve_polygonal_system's own
        // `penalty_constant = 5.0 * p * (p+1)`, C0=5 -- reused here
        // verbatim, not a new constant), chosen large specifically so it
        // DOMINATES and enforces near-continuity for coercivity,
        // independent of how many volume quadrature points exist. Using
        // just the unscaled measure/h_e0 weight (matching the real a
        // posteriori INDICATOR's post-hoc val_jump_sq/h_e0 measurement
        // convention) to determine the fit itself under-weights the
        // neighbor constraint relative to what the real globally-
        // assembled solve would enforce -- confirmed empirically: with
        // the unscaled weight, several 4-cell sub-candidates near a
        // given spot were correctly REJECTED, yet a larger (14-16 cell)
        // composition covering the same physical area was accepted,
        // because more interior quadrature points (which scale with
        // cell count) simply out-voted the same few boundary samples
        // (which scale with external perimeter -- shrinking relative to
        // area as the box gets bigger/rounder) in ONE combined least-
        // squares objective. Scaling only the FITTING weight (not the
        // reporting one, which must stay in the real indicator's units
        // for merge_threshold_for(...) to remain meaningful) fixes the
        // fit itself without changing what "too bad" means.
        const double penalty_constant = 5.0 * forecast_fe_degree * (forecast_fe_degree + 1);
        std::vector<std::vector<double>> B_rows(face_xi.size(), std::vector<double>(n_basis));
        std::vector<std::vector<double>> C_rows(face_xi.size(), std::vector<double>(n_basis));
        for (unsigned int k = 0; k < face_xi.size(); ++k) {
            const double wv_fit = penalty_constant * face_weight[k];
            const double wg_fit = face_weight_grad[k];
            for (unsigned int i = 0; i < n_basis; ++i) {
                B_rows[k][i] = monomial_value(i, face_xi[k]);
                C_rows[k][i] = monomial_grad_normal(i, face_xi[k], face_normal[k]);
                Atb(i) += wv_fit * B_rows[k][i] * face_g[k] + wg_fit * C_rows[k][i] * face_g_gradn[k];
                for (unsigned int j = 0; j < n_basis; ++j)
                    AtA(i, j) += wv_fit * B_rows[k][i] * B_rows[k][j] + wg_fit * C_rows[k][i] * C_rows[k][j];
            }
        }
        double trace = 0.0;
        for (unsigned int i = 0; i < n_basis; ++i)
            trace += AtA(i, i);
        const double reg = 1e-12 * std::max(trace / n_basis, 1.0);
        for (unsigned int i = 0; i < n_basis; ++i)
            AtA(i, i) += reg;

        AtA.gauss_jordan(); // in-place inverse
        dealii::Vector<double> c(n_basis);
        AtA.vmult(c, Atb);

        // Forecasted volume error under the BEST POSSIBLE degree-p fit:
        // same h_K^2-weighted ||f + Delta(fit)||^2_{L2} quantity the
        // real indicator computes, just evaluated on a fitted polynomial
        // instead of the actual (not-yet-built) polytope solution --
        // PLUS the face-coupling mismatch, same val_jump_sq/h_e0
        // convention as compute_polygonal_error_indicator's real face
        // term, using the SAME fitted polynomial.
        double forecast_vol_sum = 0.0;
        for (unsigned int q = 0; q < n_q; ++q) {
            double lap_fit = 0.0;
            for (unsigned int i = 0; i < n_basis; ++i)
                lap_fit += c(i) * A(q, i);
            const double resid = f_vals[q] + lap_fit;
            forecast_vol_sum += h2_all[q] * resid * resid * jxw_all[q];
        }
        double forecast_face_sum = 0.0;
        for (unsigned int k = 0; k < face_xi.size(); ++k) {
            double fit_value = 0.0, fit_gradn = 0.0;
            for (unsigned int i = 0; i < n_basis; ++i) {
                fit_value += c(i) * B_rows[k][i];
                fit_gradn += c(i) * C_rows[k][i];
            }
            const double diff_val  = fit_value - face_g[k];
            const double diff_grad = fit_gradn - face_g_gradn[k];
            forecast_face_sum += face_weight[k] * diff_val * diff_val
                               + face_weight_grad[k] * diff_grad * diff_grad;
        }
        const double forecast_sum  = forecast_vol_sum + forecast_face_sum;
        const double forecast_norm = std::sqrt(std::max(forecast_sum, 0.0));
        const double vol_norm      = std::sqrt(std::max(forecast_vol_sum, 0.0));
        const double thr           = merge_threshold_for(grp.size());

        // Total external-face count (faces bordering a cell outside the group).
        unsigned int n_ext_faces = 0;
        {
            std::unordered_set<unsigned int> gi;
            gi.reserve(grp.size());
            for (const auto &c : grp) gi.insert(c->active_cell_index());
            for (const auto &c : grp)
                for (const auto f : c->face_indices())
                    if (!c->at_boundary(f) && c->neighbor(f)->is_active()
                        && !gi.count(c->neighbor(f)->active_cell_index()))
                        ++n_ext_faces;
        }
        const double coverage =
            (n_ext_faces > 0) ? double(face_xi.size()) / n_ext_faces : 1.0;

        // FAIL-CLOSED on inadequate face coverage. The forecast's face term
        // is ~200x the volume term for a peak-core merge, so it is the real
        // discriminator; when p4est repartitioning migrates a candidate's
        // region to a rank whose neighbor_lookup has no data for its
        // neighbors, face coverage collapses (empirically to 0/20 on the
        // cycle-8 peak islands) and the forecast silently degrades to the
        // volume-only value, which for those islands sat just UNDER threshold
        // and wrongly accepted the merge -- under-resolving the peak and
        // spiking true L2/H1 error. The gate's job is to CERTIFY a merge as
        // safe; when it is blind to the dominant term it must not certify.
        // Refusing the merge only ever keeps resolution (never hurts
        // accuracy), so fail closed. Gated on vol_norm to spare genuinely
        // smooth far-field merges: those cancel f almost perfectly and sit at
        // vol_norm/thr ~ 1e-4 (measured), ~500x below the 0.01 bar, while the
        // bad peak-ish merges sit at vol_norm/thr ~ 0.05-0.64 -- a wide, clean
        // separation. Applies only to genuinely large groups (small groups
        // already carry the separate small_group_penalty and under-resolve
        // far fewer cells if wrong).
        constexpr double coverage_min = 0.5;
        constexpr double vol_gate     = 0.01; // fraction of thr
        // Only fail closed when neighbor-coupling data is EXPECTED this cycle,
        // i.e. the lookup is populated from a previous cycle's solve. At
        // cycle 0 the lookup is empty by design (no prior solution), so 0
        // coverage there is the intended volume-only bootstrap, not a
        // partition-migration blind spot -- failing closed on it would veto
        // the entire initial smooth-region agglomeration (measured: 168
        // spurious rejections at cycle 0, tripling the initial polytope count
        // and diverging the whole trajectory from the baseline).
        const bool coupling_available = (neighbor_lookup != nullptr) && !neighbor_lookup->empty();
        const bool fail_closed = coupling_available
                                 && grp.size() > small_group_max_size
                                 && coverage < coverage_min
                                 && vol_norm > vol_gate * thr;

        return (forecast_norm > thr || fail_closed);
    };

    // Does this candidate group satisfy the "small/uniform enough to
    // stay one polytope" criterion?
    auto group_ok = [&](const std::vector<typename Triangulation<dim>::active_cell_iterator> &grp) {
        if (grp.size() > max_cells_per_agglomerate)
            return false;
        double max_err  = 0.0;
        double min_diam = std::numeric_limits<double>::max();
        double max_diam = 0.0;
        for (const auto &cell : grp) {
            max_err  = std::max(max_err, (double)estimated_error[cell->active_cell_index()]);
            const double d = cell->diameter();
            min_diam = std::min(min_diam, d);
            max_diam = std::max(max_diam, d);
        }
        const bool disparity_ok = !(std::isfinite(max_diameter_ratio) && min_diam > 0.0 &&
                                     max_diam / min_diam > max_diameter_ratio);
        return (max_err <= merge_threshold_for(grp.size())) && disparity_ok && !forecast_too_bad(grp);
    };

    // PARTIAL-RECOVERY helper: given a group that fails group_ok,
    // recover as much of it as possible instead of exploding every
    // cell to its own singleton. Defined once, right here (rather than
    // only next to the tree-based retreat further down), so it can
    // ALSO be used by the small-group fast path immediately below --
    // previously that path did a cruder unconditional explosion, which
    // meant a group of e.g. 4 cells with only ONE offending cell still
    // shattered its other 3 (individually fine) cells for no reason.
    //
    // group_ok can fail for THREE distinct reasons, treated three
    // different ways:
    //  (a) group.size() > max_cells_per_agglomerate -- a WHOLE-GROUP
    //      property (currently the only cap actually in effect in
    //      adaptive_test.cc, where max_diameter_ratio = infinity), but
    //      NOT an unfixable one: rebuilding a fresh, smaller-scoped
    //      local R-tree on just this group and recursing can still
    //      find a valid split, so this case recurses instead of
    //      exploding (see the !size_ok branch below).
    //  (b) internal max/min diameter ratio > max_diameter_ratio -- also
    //      a WHOLE-GROUP property, but unlike (a) a size-based resplit
    //      doesn't obviously fix it (which two cells straddle the
    //      ratio isn't tied to how many cells are in the group) -- kept
    //      as full explosion. Currently inert whenever max_diameter_
    //      ratio is infinity, but the check stays correct if a finite
    //      ratio is reintroduced later.
    //  (c) some cell's OWN error > refine_threshold -- a PER-CELL
    //      property; isolating just that cell can fix it.
    // Only (b) still gets unconditional full explosion (no single cell,
    // or resplit-by-size, is unambiguously at fault). (a) recurses via
    // a fresh local tree; (c) isolates offenders and re-merges the rest
    // -- and either path's "recovered" pieces are re-validated against
    // group_ok before being accepted, so a recovered piece can never
    // silently bypass max_cells_per_agglomerate or the other checks.
    std::function<std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>(
        std::vector<typename Triangulation<dim>::active_cell_iterator> &)>
        recover_partial =
        [&](std::vector<typename Triangulation<dim>::active_cell_iterator> &group)
        -> std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    {
        if (group.size() <= 1 || group_ok(group))
            return {std::move(group)};

        const bool size_ok = group.size() <= max_cells_per_agglomerate;

        double min_diam = std::numeric_limits<double>::max();
        double max_diam = 0.0;
        for (const auto &c : group)
        {
            const double d = c->diameter();
            min_diam = std::min(min_diam, d);
            max_diam = std::max(max_diam, d);
        }
        const bool disparity_ok = !(std::isfinite(max_diameter_ratio) && min_diam > 0.0 &&
                                     max_diam / min_diam > max_diameter_ratio);

        // forecast_ok is, like disparity_ok, a WHOLE-GROUP (bounding-box)
        // property -- no single cell's estimated_error is "at fault" the
        // way the per-cell error check below is, so the per-cell isolate-
        // and-reconnect logic further down has no way to fix a forecast
        // failure: it only ever isolates cells whose estimated_error
        // exceeds this_merge_threshold, which does nothing when EVERY
        // cell already clears that (i.e. group_ok failed on the merge
        // forecast alone). Left unhandled, `clean` would end up as the
        // WHOLE group again, split_into_connected_components would hand
        // back the identical component, and the recursive re-validation
        // would fail group_ok again with nothing changed -- infinite
        // recursion (this is exactly what happened with the earlier
        // option-A curvature gate before this guard existed). Route it
        // through the same full-explosion path as disparity_ok for the
        // same reason: it's the only branch here that's unconditionally
        // correct regardless of size, and is guaranteed to terminate
        // (singletons can never fail forecast_too_bad, which requires
        // grp.size() >= 2).
        const bool forecast_ok = !forecast_too_bad(group);

        if (!disparity_ok || !forecast_ok)
        {
            // Internal diameter ratio is a WHOLE-GROUP property that a
            // fresh re-split by size alone wouldn't necessarily fix --
            // no single cell (or size-based resplit) is unambiguously
            // "at fault" the way an error threshold is. Full explosion,
            // as before.
            std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> singles;
            singles.reserve(group.size());
            for (const auto &c : group)
                singles.push_back({c});
            return singles;
        }

        if (!size_ok)
        {
            // SIZE-ONLY failure: disparity is fine, and this group is
            // simply larger than max_cells_per_agglomerate -- typically
            // because an ancestor InternalNode was force-collected via
            // at_deepest_level using the OUTER local tree's depth
            // (computed for the whole material-id group), not a depth
            // scoped to this particular subset. Rather than shatter
            // every one of these (possibly individually fine) cells to
            // singletons, rebuild a FRESH local R-tree scoped to just
            // this group's cells and recurse into this same function:
            // its own max_tree_depth is computed for THIS group's
            // (smaller) size, so genuine further retreat becomes
            // reachable here that the outer descent's shared depth
            // budget cut off prematurely. Guaranteed to terminate: each
            // fresh rebuild either finds a subtree that now satisfies
            // group_ok, or keeps shrinking the candidate set (via real
            // R-tree fan-out) until it hits the cells.size() <=
            // local_max_elem_per_node fast path at the top of this
            // function, which never recurses further.
            return extract_agglomerates_by_level_retreat<dim, RtreeTag>(
                group, mapping, estimated_error, refine_threshold,
                max_diameter_ratio, max_cells_per_agglomerate, accept_margin, small_group_penalty,
                rhs_function, check_merge_forecast, forecast_fe_degree, neighbor_lookup);
        }

        // Only failure reason left: at least one cell's own error
        // exceeds merge_threshold_for(group.size()). Isolate those,
        // keep the rest ("clean") together.
        //
        // TERMINATION-CRITICAL: this isolation cut MUST use the exact
        // same merge_threshold_for(group.size()) group_ok just used for
        // THIS group, NOT the raw refine_threshold. With group_ok's
        // error check at that threshold, a group whose max error sits
        // in (merge_threshold, refine_threshold] fails group_ok -- but
        // cutting at refine_threshold instead would find ZERO
        // offenders, so `clean` would be the whole group, split_into_
        // connected_components would hand back the identical component,
        // and the recursive re-validation below would fail group_ok
        // again with nothing changed: infinite recursion. Cutting at
        // the SAME threshold guarantees every cell left in `clean` is
        // <= that threshold, so a re-merged component always passes
        // group_ok's error check and the recursion terminates. (The
        // recursive call further down re-evaluates this per-group each
        // time using THAT group's own -- usually smaller -- size, so a
        // small reconnected piece correctly gets re-checked against its
        // own, possibly more penalized, threshold rather than the
        // parent group's.)
        const double this_merge_threshold = merge_threshold_for(group.size());
        std::vector<typename Triangulation<dim>::active_cell_iterator> clean;
        std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> out;
        for (const auto &c : group)
        {
            if (estimated_error[c->active_cell_index()] > this_merge_threshold)
                out.push_back({c}); // offending cell: isolate as its own singleton
            else
                clean.push_back(c);
        }

        if (!clean.empty())
        {
            // The clean cells are NOT guaranteed to still be
            // face-connected once the offending ones are pulled out of
            // the middle of the group -- re-split by connectivity
            // before re-accepting them as polytopes (same guard used
            // elsewhere in this file against silently-disconnected
            // groups).
            auto components = split_into_connected_components<dim>({clean});
            for (auto &comp : components)
            {
                // Recursively re-validate: a connected clean component
                // could in principle still fail group_ok on its own
                // (e.g. still too big relative to max_cells_per_
                // agglomerate) -- recursing lets that case hit the
                // whole-group branch above and explode (or, for a
                // size-only failure, further resplit) just THAT piece,
                // instead of silently exceeding the cap.
                auto pieces = recover_partial(comp);
                for (auto &p : pieces)
                    out.push_back(std::move(p));
            }
        }
        return out;
    };

    if (cells.size() <= 1)
        return {cells};

    // ROOT-LEVEL ACCEPTANCE (fix for monotone far-field shrinkage):
    // check the ENTIRE group against group_ok BEFORE any tree descent.
    // Without this, the estimator below unconditionally descends past
    // level 0 (the root = the whole group), so a group that fully
    // satisfies the criterion -- low error, internally uniform, within
    // the cell cap -- was STILL force-split into its ~max_elem_per_node
    // level-1 children on every pass through the reuse path:
    // 64 cells -> ~4x16 -> ~4x4, converging to ~max_elem_per_node-cell
    // fragments over 2-3 cycles even where nothing was refined. That
    // one-way shrinkage is exactly the failure mode this function was
    // written to prevent; keeping a satisfied group whole is the entire
    // point of the macro-reuse path (far-field groups stay constant
    // across cycles).
    if (group_ok(cells))
        return {cells};

    // Same guard as split_polytope_until_small_enough: a group this
    // small can't build a meaningful R-tree (CellsAgglomerator needs at
    // least two levels), so decide directly -- but now via the SAME
    // partial-recovery logic as everywhere else in this function
    // (group_ok already returned above, so a small group reaching here
    // failed the criterion). Previously this exploded unconditionally;
    // now a 4-cell group with only one offending cell keeps its other
    // 3 (individually fine) cells merged instead of shattering all 4.
    if (cells.size() <= local_max_elem_per_node)
    {
        std::vector<typename Triangulation<dim>::active_cell_iterator> cells_copy(cells.begin(), cells.end());
        return recover_partial(cells_copy);
    }

    std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>> local_boxes;
    local_boxes.reserve(cells.size());
    for (const auto &cell : cells)
        local_boxes.push_back(std::make_pair(mapping.get_bounding_box(cell), cell));

    auto local_tree = pack_rtree<RtreeTag>(local_boxes);
    CellsAgglomerator<dim, decltype(local_tree)> local_agglomerator{local_tree, 0};
    const size_t max_tree_depth = local_agglomerator.get_n_levels();

    auto estimator = [&](size_t level, const auto &node) -> dealii::internal::Mode {
        if (level == 0)
            return dealii::internal::Mode::normal; // root: always keep descending past it

        using NodeType = std::decay_t<decltype(node)>;
        using elements_type = typename boost::geometry::index::detail::rtree::elements_type<NodeType>::type;
        const elements_type &elements = boost::geometry::index::detail::rtree::elements(node);
        if (elements.empty())
            return dealii::internal::Mode::normal;

        // IMPORTANT: at an InternalNode, elements() gives (bounding-box,
        // child-node-pointer) pairs -- NOT (bounding-box, cell_iterator)
        // pairs. Only Leaf nodes hold actual cell iterators. So we can
        // only read .first (the box) off `elements` directly here; to
        // get the real cell iterators under this node we query the
        // local tree itself for every leaf entry geometrically inside
        // this node's merged bounding box -- exactly what
        // extract_error_based_agglomerates's own estimator does with
        // this->rtree->query(...) in adaptive_agglomerator.h, just using
        // `local_tree` (which this free function owns) instead of a
        // CellsAgglomerator member we don't have access to here.
        auto node_bbox = elements[0].first;
        for (size_t idx = 1; idx < elements.size(); ++idx)
            boost::geometry::expand(node_bbox, elements[idx].first);

        std::vector<std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>> leaves_inside;
        local_tree.query(boost::geometry::index::intersects(node_bbox), std::back_inserter(leaves_inside));

        std::vector<typename Triangulation<dim>::active_cell_iterator> candidate;
        candidate.reserve(leaves_inside.size());
        for (const auto &leaf : leaves_inside)
            candidate.push_back(leaf.second);

        const bool at_deepest_level = (level >= max_tree_depth - 1);

        // NOTE: at_deepest_level force-collects regardless of group_ok, so
        // a candidate that fails group_ok (error, disparity, or the merge
        // forecast) can still be merged into one polytope here -- the
        // "final cleanup" recover_partial pass below re-validates every
        // resulting group and explodes/re-splits it if it still fails.
        if (group_ok(candidate) || at_deepest_level)
            return dealii::internal::Mode::collecting;
        else
            return dealii::internal::Mode::normal; // retreat one level further down
    };

    auto leaf_est = [](size_t, const auto &) { return dealii::internal::Mode::collecting; };

    // Pass max_cells_per_agglomerate through as the hard leaf-level
    // backstop too (see RefineCoarsenRtreeVisitor::operator()(Leaf) in
    // adaptive_agglomerator.h): a node's "elements" count at the level
    // our estimator inspects isn't necessarily the final active-cell
    // count once Mode::collecting recurses down to leaves (a tree entry
    // can expand into several active children if the mesh was refined
    // after the tree was built), so this is a second, independent cap.
    auto result = local_agglomerator.extract_adaptive_agglomerates(estimator, leaf_est, max_cells_per_agglomerate);

    // Final cleanup: apply the SAME partial-recovery logic used by the
    // small-group fast path above (recover_partial, defined right after
    // group_ok) to every group this tree-based retreat produced -- the
    // deepest tree level (and the leaf estimator above, which
    // unconditionally returns Mode::collecting without consulting
    // group_ok) may still have bundled together cells that don't
    // jointly satisfy group_ok.
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> final_result;
    final_result.reserve(result.size());
    for (auto &group : result)
    {
        auto pieces = recover_partial(group);
        for (auto &p : pieces)
            final_result.push_back(std::move(p));
    }
    return final_result;
}



// Checks whether every polytope's constituent fine cells form a SINGLE
// connected component (via face adjacency), as opposed to being split
// into two-or-more physically disconnected islands that just happen to
// share one material_id/agglomerate slot. This is a DIFFERENT failure
// mode from a merely small polytope (which report_condition_number_
// estimate's smallest-polytopes companion diagnostic already checks):
// a disconnected polytope can have a perfectly normal bounding-box
// diameter, yet its LOCAL SIPG stiffness contribution can still be
// rank-deficient, since there is no direct penalty coupling enforced
// between the disconnected pieces within what the discretization
// treats as a single DoF's element -- a plausible candidate for the
// lambda_min~inf events seen with jump-term-weighted marking, since
// those change which cells get merged into which agglomerate group
// without necessarily changing any polytope's overall size.
//
// This is a purely local (no MPI) connectivity scan over
// LOCALLY-OWNED groups (the ones passed to define_agglomerate() on
// this rank) -- ghost portions of a cross-rank group are out of scope,
// matching the same limitation already accepted elsewhere in this file
// (find_oversized_neighbor_polytope_targets, report_h_f_extremes).
template <int dim>
void report_disconnected_polytopes(
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &groups,
    MPI_Comm mpi_communicator,
    const std::string &tag = "")
{
    unsigned int n_disconnected = 0;
    for (size_t gi = 0; gi < groups.size(); ++gi)
    {
        const auto &grp = groups[gi];
        if (grp.size() <= 1)
            continue;

        std::unordered_set<unsigned int> members;
        std::map<unsigned int, typename Triangulation<dim>::active_cell_iterator> idx_to_cell;
        for (const auto &c : grp)
        {
            members.insert(c->active_cell_index());
            idx_to_cell[c->active_cell_index()] = c;
        }

        std::unordered_set<unsigned int> visited;
        std::queue<unsigned int> q;
        const unsigned int start_idx = grp.front()->active_cell_index();
        visited.insert(start_idx);
        q.push(start_idx);

        while (!q.empty())
        {
            const unsigned int idx = q.front();
            q.pop();
            const auto &cell = idx_to_cell[idx];

            for (const auto f : cell->face_indices())
            {
                if (cell->at_boundary(f))
                    continue;

                if (cell->neighbor(f)->has_children())
                {
                    // Neighbor is more refined: check each child touching
                    // this face individually.
                    for (unsigned int sub = 0; sub < cell->face(f)->n_children(); ++sub)
                    {
                        const auto neighbor_child = cell->neighbor_child_on_subface(f, sub);
                        if (!neighbor_child->is_active())
                            continue;
                        const unsigned int nidx = neighbor_child->active_cell_index();
                        if (members.count(nidx) && !visited.count(nidx))
                        {
                            visited.insert(nidx);
                            q.push(nidx);
                        }
                    }
                }
                else
                {
                    const auto &neigh = cell->neighbor(f);
                    if (!neigh->is_active())
                        continue; // coarser neighbor with children handled from its own side
                    const unsigned int nidx = neigh->active_cell_index();
                    if (members.count(nidx) && !visited.count(nidx))
                    {
                        visited.insert(nidx);
                        q.push(nidx);
                    }
                }
            }
        }

        if (visited.size() != grp.size())
        {
            ++n_disconnected;
            std::cout << "  [connectivity-check" << (tag.empty() ? "" : (" " + tag)) << ", rank "
                      << Utilities::MPI::this_mpi_process(mpi_communicator) << "] polytope " << gi
                      << " is DISCONNECTED: only " << visited.size() << "/" << grp.size()
                      << " cells reachable from the first cell (diameter alone would NOT catch this)."
                      << std::endl;
        }
    }
    if (n_disconnected > 0)
        std::cout << "  [connectivity-check" << (tag.empty() ? "" : (" " + tag)) << ", rank "
                  << Utilities::MPI::this_mpi_process(mpi_communicator) << "] " << n_disconnected
                  << " disconnected polytope(s) found this cycle." << std::endl;
}



// Splits every group into its face-adjacency connected components,
// leaving already-connected groups untouched (returned as a single
// group, same as input). This is the ACTIVE FIX corresponding to the
// diagnostic in report_disconnected_polytopes above -- same BFS
// adjacency logic, but instead of just reporting a disconnected group,
// this finds ALL of its connected components (not just "how much of it
// is reachable from the first cell") and returns them as separate
// groups. Already-connected groups incur one BFS pass and are returned
// unchanged (as a single-element result), so this is safe to call
// unconditionally on every group, every cycle.
//
// Works on whatever cells a group happens to contain (locally-owned,
// ghost, or a mix -- e.g. group_cells_by_material_id's output before
// the locally-owned filter is applied) since it only relies on
// ordinary Triangulation cell/face/neighbor queries, not on any
// AgglomerationHandler/polytope-level API.
template <int dim>
std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
split_into_connected_components(
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &groups)
{
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> result;
    result.reserve(groups.size());

    for (const auto &grp : groups)
    {
        if (grp.size() <= 1)
        {
            result.push_back(grp);
            continue;
        }

        std::unordered_set<unsigned int> members;
        std::map<unsigned int, typename Triangulation<dim>::active_cell_iterator> idx_to_cell;
        for (const auto &c : grp)
        {
            members.insert(c->active_cell_index());
            idx_to_cell[c->active_cell_index()] = c;
        }

        std::unordered_set<unsigned int> assigned;
        for (const auto &c0 : grp)
        {
            const unsigned int start_idx = c0->active_cell_index();
            if (assigned.count(start_idx))
                continue; // already claimed by an earlier component's BFS

            std::vector<typename Triangulation<dim>::active_cell_iterator> component;
            std::queue<unsigned int> q;
            q.push(start_idx);
            assigned.insert(start_idx);

            while (!q.empty())
            {
                const unsigned int idx = q.front();
                q.pop();
                const auto &cell = idx_to_cell[idx];
                component.push_back(cell);

                for (const auto f : cell->face_indices())
                {
                    if (cell->at_boundary(f))
                        continue;

                    if (cell->neighbor(f)->has_children())
                    {
                        for (unsigned int sub = 0; sub < cell->face(f)->n_children(); ++sub)
                        {
                            const auto neighbor_child = cell->neighbor_child_on_subface(f, sub);
                            if (!neighbor_child->is_active())
                                continue;
                            const unsigned int nidx = neighbor_child->active_cell_index();
                            if (members.count(nidx) && !assigned.count(nidx))
                            {
                                assigned.insert(nidx);
                                q.push(nidx);
                            }
                        }
                    }
                    else
                    {
                        const auto &neigh = cell->neighbor(f);
                        if (!neigh->is_active())
                            continue;
                        const unsigned int nidx = neigh->active_cell_index();
                        if (members.count(nidx) && !assigned.count(nidx))
                        {
                            assigned.insert(nidx);
                            q.push(nidx);
                        }
                    }
                }
            }

            result.push_back(std::move(component));
        }
    }

    return result;
}


  // Doerfler (bulk) threshold: the error value t such that cells with
  // error > t carry ~theta of the TOTAL squared error mass. Robust where
  // count-based quantiles fail (e.g. localized sources leaving most
  // cells at numerically-zero error, which drags a count quantile into
  // the noise floor). Fixed bisection iteration count and identical
  // reduced values on every rank -- safe to call collectively. If
  // total_error_mass_out is non-null, receives sqrt(sum err^2)
  // (useful for relative-reduction stopping criteria).
  // restrict_mask (optional): when given, ONLY cells whose
  // active_cell_index() is true in the mask enter the statistic. This is what
  // separates the two roles L plays. The estimator is an element quantity
  // (one eta_K per polytope, Thm 21); marking has to pick BACKGROUND CELLS to
  // refine, and only a singleton cell can usefully be refined at all --
  // refining inside a multi-cell polytope adds no DoFs. Ranking singletons
  // against cells that sit inside agglomerates therefore compares two things
  // that are not comparable: under the theory's h_K weight the agglomerated
  // cells are inflated by ~(#cells in the polytope), they pull the Doerfler
  // line up, and no singleton can ever clear it. Restricting the statistic to
  // the refinable set removes that coupling. The bisection count is fixed and
  // every reduced value is identical on all ranks, so this stays
  // collective-safe.
  template <int dim>
  double compute_bulk_threshold(const Triangulation<dim> &tria,
                                 const Vector<float>      &err,
                                 MPI_Comm                  mpi_communicator,
                                 const double              theta = 0.5,
                                 double *total_error_mass_out    = nullptr,
                                 const std::vector<bool> *restrict_mask = nullptr)
  {
    const auto included = [&](const typename Triangulation<dim>::active_cell_iterator &c) {
      return restrict_mask == nullptr || (*restrict_mask)[c->active_cell_index()];
    };
    double local_max = 0.0;
    double local_sum_sq = 0.0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && included(cell))
        {
          const double e = err[cell->active_cell_index()];
          local_max = std::max(local_max, e);
          local_sum_sq += e * e;
        }
    const double global_max = Utilities::MPI::max(local_max, mpi_communicator);
    const double total_sq   = Utilities::MPI::sum(local_sum_sq, mpi_communicator);
    if (total_error_mass_out != nullptr)
      *total_error_mass_out = std::sqrt(total_sq);
    if (total_sq <= 0.0 || global_max <= 0.0)
      return 0.0;

    double lo = 0.0, hi = global_max;
    for (unsigned int it = 0; it < 40; ++it)
      {
        const double mid = 0.5 * (lo + hi);
        double local_above_sq = 0.0;
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() && included(cell))
            {
              const double e = err[cell->active_cell_index()];
              if (e > mid)
                local_above_sq += e * e;
            }
        const double above_sq = Utilities::MPI::sum(local_above_sq, mpi_communicator);
        if (above_sq > theta * total_sq)
          lo = mid;
        else
          hi = mid;
      }
    return hi;
  }

  // Count-capped marking threshold: the error value t such that AT MOST
  // target_fraction of all active cells have error > t. Complements
  // compute_bulk_threshold (a MASS-based Doerfler criterion) -- the two
  // are dual failure modes of each other:
  //  - compute_bulk_threshold degenerates when error is spread almost
  //    UNIFORMLY across the domain (no localized peaks, e.g. a smooth
  //    manufactured solution's RHS): no cell concentrates enough error
  //    mass on its own, so the mass-based bisection is forced to
  //    include almost EVERY cell just to reach theta of the total, and
  //    mark_threshold collapses close to zero -- observed directly on
  //    the SinSinRHS test as global (~4x cells/cycle) refinement every
  //    single cycle.
  //  - a naive count-based quantile (reading off the raw VALUE at some
  //    percentile rank) degenerates the OPPOSITE way when most cells
  //    sit at numerically-zero error (e.g. a highly localized source):
  //    a huge share of cells cluster at the noise floor, so the
  //    percentile VALUE lands in that floor and "error > that value"
  //    ends up selecting far more than the intended fraction.
  // This function avoids the second failure mode by doing the SAME
  // bisection-on-the-actual-error-VALUES approach as compute_bulk_
  // threshold, just with the bisection criterion swapped from "error
  // mass above mid" to "cell COUNT above mid" -- it always finds a
  // value that closes on exactly target_fraction of cells, regardless
  // of how many cells happen to share the same (possibly noise-floor)
  // value.
  //
  // Intended use: call both this and compute_bulk_threshold and take
  // the max of the two results as the actual marking threshold L. That
  // makes L "whichever criterion says mark FEWER cells wins": a
  // problem with a localized RHS keeps behaving exactly as before
  // (mass-based L already sits high, count-based L rarely binds
  // tighter), while a problem with a spread-out RHS (like a smooth
  // manufactured solution) gets its runaway mass-based L overridden by
  // a hard cap on how large a fraction of the mesh can be marked in
  // one cycle -- without needing to hand-tune theta per RHS shape.
  // Fixed bisection iteration count and identical reduced values on
  // every rank -- safe to call collectively, same as compute_bulk_
  // threshold.
  template <int dim>
  double compute_count_capped_threshold(const Triangulation<dim> &tria,
                                         const Vector<float>      &err,
                                         MPI_Comm                  mpi_communicator,
                                         const double              target_fraction = 0.2,
                                         const std::vector<bool> *restrict_mask = nullptr)
  {
    // Same restriction as compute_bulk_threshold; see the note there.
    const auto included = [&](const typename Triangulation<dim>::active_cell_iterator &c) {
      return restrict_mask == nullptr || (*restrict_mask)[c->active_cell_index()];
    };
    double local_max = 0.0;
    unsigned long local_count = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && included(cell))
        {
          local_max = std::max(local_max, (double)err[cell->active_cell_index()]);
          ++local_count;
        }
    const double global_max = Utilities::MPI::max(local_max, mpi_communicator);
    const unsigned long total_count = Utilities::MPI::sum(local_count, mpi_communicator);
    if (global_max <= 0.0 || total_count == 0)
      return 0.0;

    const double target_count = target_fraction * static_cast<double>(total_count);

    double lo = 0.0, hi = global_max;
    for (unsigned int it = 0; it < 40; ++it)
      {
        const double mid = 0.5 * (lo + hi);
        unsigned long local_above = 0;
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() && included(cell))
            {
              const double e = err[cell->active_cell_index()];
              if (e > mid)
                ++local_above;
            }
        const unsigned long above = Utilities::MPI::sum(local_above, mpi_communicator);
        if (static_cast<double>(above) > target_count)
          lo = mid;
        else
          hi = mid;
      }
    return hi;
  }


  // single-cell polytope (the agglomeration has bottomed out there --
  // in a polytopal method the DoF count belongs to polytopes, so
  // refining cells inside a multi-cell polytope adds no resolution) and
  // (b) carry error > refine_hysteresis * mark_threshold (only cells
  // COMFORTABLY above the agglomeration split line, so their children,
  // inheriting that error, cannot be re-fused by marginal threshold
  // drift). Clears all refine flags first, so any flags left behind by
  // upstream extraction code cannot union in. Returns the GLOBAL number
  // of marked cells (one MPI sum -- call collectively).
  template <int dim>
  unsigned long mark_singleton_cells_for_refinement(
    Triangulation<dim> &tria,
    const Vector<float> &err,
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &groups,
    const double mark_threshold,
    MPI_Comm     mpi_communicator,
    const double refine_hysteresis = 1.5)
  {
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->clear_refine_flag();

    std::vector<bool> is_singleton(tria.n_active_cells(), false);
    for (const auto &grp : groups)
      if (grp.size() == 1)
        is_singleton[grp.front()->active_cell_index()] = true;

    unsigned long local_marked = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() &&
          is_singleton[cell->active_cell_index()] &&
          err[cell->active_cell_index()] > refine_hysteresis * mark_threshold)
        {
          cell->set_refine_flag();
          ++local_marked;
        }
    return Utilities::MPI::sum(local_marked, mpi_communicator);
  }

  // Absolute polytope-size (h_f_max) control: any group whose merged
  // bounding-box diagonal exceeds max_polytope_diameter is recursively
  // split (via split_polytope_until_small_enough) until every piece
  // fits. Returns the number of groups that were split. Purely local.
  template <int dim, typename RtreeTag>
  unsigned int apply_polytope_diameter_cap(
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &groups,
    const Mapping<dim> &mapping,
    const double        max_polytope_diameter)
  {
    unsigned int n_split = 0;
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> result;
    result.reserve(groups.size());
    for (auto &group : groups)
      {
        if (group.size() <= 1)
          {
            result.push_back(std::move(group));
            continue;
          }
        BoundingBox<dim> merged = mapping.get_bounding_box(group[0]);
        for (size_t i = 1; i < group.size(); ++i)
          merged.merge_with(mapping.get_bounding_box(group[i]));
        const auto   bp   = merged.get_boundary_points();
        const double diag = (bp.second - bp.first).norm();
        if (diag <= max_polytope_diameter)
          {
            result.push_back(std::move(group));
            continue;
          }
        auto pieces = split_polytope_until_small_enough<dim, RtreeTag>(
          group, mapping, max_polytope_diameter);
        for (auto &p : pieces)
          result.push_back(std::move(p));
        ++n_split;
      }
    groups.swap(result);
    return n_split;
  }

} // namespace PolytopalAdaptivity

#endif // ADAPTIVE_AGGLOMERATOR_H