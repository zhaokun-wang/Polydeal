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


#ifndef polytope_repartition_h
#define polytope_repartition_h


#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <parmetis.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <vector>


namespace dealii::PolytopeRepartition
{
  /**
   * A copy of a p4est triangulation as a
   * parallel::fullydistributed::Triangulation, partitioned by polytopes.
   *
   * The p4est triangulation of an adaptive polytopal mesh is never
   * repartitioned, so that every polytope stays on one rank. The degrees of
   * freedom then pile up on the ranks that own the refined regions. p4est
   * could not balance them anyway: it partitions cells along a space-filling
   * curve, while the degrees of freedom live on polytopes of very different
   * numbers of cells.
   *
   * This class builds, in every cycle, a copy of the triangulation on which
   * the system is assembled, solved and the error is estimated. The copy
   * is partitioned by ParMETIS on the graph of the polytopes, with one vertex
   * per polytope and an edge between polytopes that share a face, under two
   * constraints, degrees of freedom and cells. The global polytope index of a
   * polytope is its position in the list of the polytopes of all ranks, taken
   * rank by rank. It is also the index of its vertex in the graph. Every
   * polytope lies on one rank of the copy, and the polytopes are the same as
   * on the p4est triangulation.
   *
   * Per-cell data move between the two triangulations polytope by polytope,
   * in an order both sides know, so that no CellId is sent. The error
   * indicators and, for time-dependent problems, the solution go back to the
   * p4est triangulation, because everything that lasts from one cycle to the
   * next lives there: the R-tree, which every rank builds once over the
   * initial cells it owns, the refinement and coarsening, and the transfer of
   * data to the next mesh. A parallel::fullydistributed::Triangulation can
   * not be refined, so the copy is built anew in every cycle.
   */
  template <int dim>
  class PartitionedCopy
  {
  public:
    using CellIterator = typename Triangulation<dim>::active_cell_iterator;

    /**
     * Constructor.
     */
    PartitionedCopy(const parallel::distributed::Triangulation<dim> &tria);

    /**
     * Build the copy for the locally owned polytopes @p polytopes of the p4est
     * triangulation, each with @p dofs_per_polytope degrees of freedom.
     *
     * An AgglomerationHandler built on the previous copy must be destroyed
     * before. It reinitializes itself whenever its triangulation changes, so
     * rebuilding the copy would make it reinitialize from the polytopes of the
     * previous copy, whose cells no longer exist.
     */
    void
    reinit(const std::vector<std::vector<CellIterator>> &polytopes,
           const unsigned int                            dofs_per_polytope);

    /**
     * The copy.
     */
    const parallel::fullydistributed::Triangulation<dim> &
    get_triangulation() const;

    /**
     * The polytopes this rank owns on the copy, each a list of locally owned
     * cells of the copy in CellId order.
     */
    const std::vector<std::vector<CellIterator>> &
    get_polytopes() const;

    /**
     * Largest over mean number of degrees of freedom per rank on the copy.
     */
    double
    dof_imbalance() const;

    /**
     * One value per cell of the copy, sent to the cells of the p4est
     * triangulation.
     */
    void
    copy_to_p4est(const Vector<float> &copy_values,
                  Vector<float>       &p4est_values) const;

    /**
     * Data of any length per cell of the copy, sent to the cells of the
     * p4est triangulation.
     */
    void
    copy_to_p4est(const std::vector<std::vector<double>> &copy_data,
                  std::vector<std::vector<double>>       &p4est_data) const;

    /**
     * Data of any length per cell of the p4est triangulation, sent to the
     * cells of the copy.
     */
    void
    p4est_to_copy(const std::vector<std::vector<double>> &p4est_data,
                  std::vector<std::vector<double>>       &copy_data) const;

  private:
    /**
     * Send per-cell data from the polytopes of one triangulation to the same
     * polytopes on the other. Polytope @p k of @p from_polytopes goes to the
     * rank @p destination[k], and polytope @p k of @p to_polytopes comes from
     * the rank @p source[k]. Between two ranks, both sides list the polytopes
     * they share in the order of their global polytope index and the cells
     * of each in CellId order, so a message holds only, cell by cell, the
     * length of its data and the data. @p to_data is indexed by the active
     * cell index of the triangulation of @p to_polytopes.
     */
    void
    send_cell_data(const std::vector<std::vector<CellIterator>> &from_polytopes,
                   const std::vector<unsigned int>              &destination,
                   const std::vector<std::vector<double>>       &from_data,
                   const std::vector<std::vector<CellIterator>> &to_polytopes,
                   const std::vector<unsigned int>              &source,
                   std::vector<std::vector<double>>             &to_data) const;

    /**
     * The triangulation that is copied.
     */
    const parallel::distributed::Triangulation<dim> &p4est_tria;

    /**
     * The communicator of both triangulations.
     */
    const MPI_Comm comm;

    /**
     * The copy.
     */
    parallel::fullydistributed::Triangulation<dim> copy_tria;

    /**
     * The polytopes this rank owns on the copy.
     */
    std::vector<std::vector<CellIterator>> copy_polytopes;

    /**
     * For every polytope of copy_polytopes, the rank that owns it on the
     * p4est triangulation.
     */
    std::vector<unsigned int> p4est_rank_of_copy_polytope;

    /**
     * The polytopes this rank owns on the p4est triangulation, as given to
     * reinit(). They stay valid until the p4est triangulation is refined, and
     * reinit() is called again before the next transfer anyway.
     */
    std::vector<std::vector<CellIterator>> p4est_polytopes;

    /**
     * For every polytope of p4est_polytopes, the rank that holds it on the
     * copy.
     */
    std::vector<unsigned int> copy_rank_of_p4est_polytope;

    /**
     * Largest over mean number of degrees of freedom per rank on the copy.
     */
    double imbalance;
  };



  template <int dim>
  PartitionedCopy<dim>::PartitionedCopy(
    const parallel::distributed::Triangulation<dim> &tria)
    : p4est_tria(tria)
    , comm(tria.get_mpi_communicator())
    , copy_tria(tria.get_mpi_communicator())
    , imbalance(1.)
  {}



  template <int dim>
  void
  PartitionedCopy<dim>::reinit(
    const std::vector<std::vector<CellIterator>> &polytopes,
    const unsigned int                            dofs_per_polytope)
  {
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

    // ParMETIS holds the graph distributed in contiguous blocks of vertices:
    // rank r holds the vertices from vertex_distribution[r] to
    // vertex_distribution[r + 1] - 1. Every rank holds the vertices of its own
    // polytopes, so the global polytope index of the first polytope of rank r
    // is vertex_distribution[r]. ParMETIS needs at least one vertex on every
    // rank.
    const std::uint64_t n_local_polytopes = polytopes.size();
    AssertThrow(Utilities::MPI::min(n_local_polytopes, comm) > 0,
                ExcMessage("A rank owns no polytope."));
    std::vector<idx_t> vertex_distribution{0};
    for (const std::uint64_t count :
         Utilities::MPI::all_gather(comm, n_local_polytopes))
      vertex_distribution.push_back(vertex_distribution.back() + idx_t(count));
    const std::uint64_t first_polytope = vertex_distribution[my_rank];
    const std::uint64_t n_vertices     = vertex_distribution[n_ranks];

    // The vertex of the polytope of every cell, in a vector over the global
    // active cells of the p4est triangulation, so that the ghost cells know
    // the vertex of their polytope too.
    const auto partitioner =
      p4est_tria.global_active_cell_index_partitioner().lock();
    LinearAlgebra::distributed::Vector<double> polytope_of_cell(partitioner);
    for (std::uint64_t polytope = 0; polytope < n_local_polytopes; ++polytope)
      for (const auto &cell : polytopes[polytope])
        polytope_of_cell[cell->global_active_cell_index()] =
          double(first_polytope + polytope);
    polytope_of_cell.update_ghost_values();

    // The neighbours of every local polytope: the polytopes of the cells
    // across its faces. Across a hanging face the neighbours are the children
    // of the neighbouring cell on that face.
    std::vector<std::set<std::uint64_t>> neighbors(n_local_polytopes);
    for (std::uint64_t polytope = 0; polytope < n_local_polytopes; ++polytope)
      for (const auto &cell : polytopes[polytope])
        {
          const std::uint64_t this_polytope = first_polytope + polytope;
          const auto          add_neighbor  = [&](const auto &neighbor_cell) {
            const std::uint64_t neighbor_polytope = std::uint64_t(
              polytope_of_cell[neighbor_cell->global_active_cell_index()]);
            if (neighbor_polytope != this_polytope)
              neighbors[polytope].insert(neighbor_polytope);
          };
          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                continue;
              if (cell->neighbor(f)->is_active())
                add_neighbor(cell->neighbor(f));
              else
                for (unsigned int subface = 0;
                     subface < cell->face(f)->n_children();
                     ++subface)
                  add_neighbor(cell->neighbor_child_on_subface(f, subface));
            }
        }

    // The graph in the compressed format of ParMETIS: the neighbours of
    // local vertex v are adjacency[adjacency_offsets[v]], ...,
    // adjacency[adjacency_offsets[v + 1] - 1]. Every edge is seen from both
    // of its polytopes, so the adjacency is symmetric, as ParMETIS requires.
    std::vector<idx_t> adjacency_offsets{0}, adjacency;
    for (std::uint64_t polytope = 0; polytope < n_local_polytopes; ++polytope)
      {
        for (const std::uint64_t neighbor : neighbors[polytope])
          adjacency.push_back(idx_t(neighbor));
        adjacency_offsets.push_back(idx_t(adjacency.size()));
      }

    // Two weights per vertex, degrees of freedom and cells.
    std::vector<idx_t> vertex_weights(2 * n_local_polytopes);
    for (std::uint64_t polytope = 0; polytope < n_local_polytopes; ++polytope)
      {
        vertex_weights[2 * polytope] = idx_t(std::max(1u, dofs_per_polytope));
        vertex_weights[2 * polytope + 1] = idx_t(polytopes[polytope].size());
      }

    // Partition into one part per rank, with weights on the vertices only
    // (weight_flag 2), numbering from zero, equal target weights for every
    // part and a tolerance of five percent on each constraint.
    idx_t weight_flag = 2, numbering = 0, n_constraints = 2,
          n_parts = idx_t(n_ranks), edges_cut = 0;
    std::vector<real_t> target_part_weights(n_constraints * n_parts,
                                            real_t(1) / real_t(n_parts));
    std::vector<real_t> imbalance_tolerance(n_constraints, real_t(1.05));
    idx_t               options[3] = {0, 0, 0};
    std::vector<idx_t>  part_of_vertex(n_local_polytopes, 0);
    MPI_Comm            parmetis_comm = comm;
    const int           parmetis_status =
      ParMETIS_V3_PartKway(vertex_distribution.data(),
                           adjacency_offsets.data(),
                           adjacency.empty() ? nullptr : adjacency.data(),
                           vertex_weights.data(),
                           nullptr,
                           &weight_flag,
                           &numbering,
                           &n_constraints,
                           &n_parts,
                           target_part_weights.data(),
                           imbalance_tolerance.data(),
                           options,
                           &edges_cut,
                           part_of_vertex.data(),
                           &parmetis_comm);

    std::vector<idx_t> rank_of_vertex;
    for (const auto &parts_from_rank :
         Utilities::MPI::all_gather(comm, part_of_vertex))
      rank_of_vertex.insert(rank_of_vertex.end(),
                            parts_from_rank.begin(),
                            parts_from_rank.end());

    // A rank without a polytope is not allowed by AgglomerationHandler. If
    // ParMETIS fails or leaves one, split the vertices into contiguous blocks
    // of equal size. The vertices are in the order of the cells of the p4est
    // triangulation, so every block is a connected part of the domain as far
    // as the space-filling curve of p4est is.
    const bool parmetis_succeeded =
      Utilities::MPI::min(parmetis_status == METIS_OK ? 1 : 0, comm) == 1;
    std::vector<std::uint64_t> polytopes_per_rank(n_ranks, 0);
    if (parmetis_succeeded)
      for (const idx_t rank : rank_of_vertex)
        ++polytopes_per_rank[rank];
    if (!parmetis_succeeded || *std::min_element(polytopes_per_rank.begin(),
                                                 polytopes_per_rank.end()) == 0)
      {
        polytopes_per_rank.assign(n_ranks, 0);
        for (std::uint64_t vertex = 0; vertex < n_vertices; ++vertex)
          {
            rank_of_vertex[vertex] = idx_t((vertex * n_ranks) / n_vertices);
            ++polytopes_per_rank[rank_of_vertex[vertex]];
          }
      }

    // Every polytope has the same number of degrees of freedom, so the
    // largest over mean number of degrees of freedom per rank of the copy is
    // that of polytopes.
    imbalance = double(*std::max_element(polytopes_per_rank.begin(),
                                         polytopes_per_rank.end())) /
                (double(n_vertices) / n_ranks);

    // The rank on the copy of every local polytope, and of every cell in
    // copy_partition, a vector over the global active cells of the p4est
    // triangulation, the form create_description_from_triangulation() takes.
    p4est_polytopes = polytopes;
    copy_rank_of_p4est_polytope.resize(n_local_polytopes);
    LinearAlgebra::distributed::Vector<double> copy_partition(partitioner);
    for (std::uint64_t polytope = 0; polytope < n_local_polytopes; ++polytope)
      {
        const auto rank = rank_of_vertex[first_polytope + polytope];
        copy_rank_of_p4est_polytope[polytope] = rank;
        for (const auto &cell : polytopes[polytope])
          copy_partition[cell->global_active_cell_index()] = rank;
      }

    // Build the copy: every cell goes to the rank of its polytope.
    copy_tria.clear();
    copy_tria.create_triangulation(
      TriangulationDescription::Utilities::
        create_description_from_triangulation(p4est_tria, copy_partition));

    // Every polytope, sent to the rank that holds it on the copy: its global
    // polytope index, its number of cells and the CellIds of its cells in
    // binary form, four words each, in the order of the cells in the polytope.
    std::map<unsigned int, std::vector<std::uint64_t>> polytopes_to_send;
    for (std::uint64_t polytope = 0; polytope < n_local_polytopes; ++polytope)
      {
        const std::uint64_t global_polytope = first_polytope + polytope;
        auto &message = polytopes_to_send[static_cast<unsigned int>(
          rank_of_vertex[global_polytope])];
        message.push_back(global_polytope);
        message.push_back(polytopes[polytope].size());
        for (const auto &cell : polytopes[polytope])
          for (const unsigned int word : cell->id().template to_binary<dim>())
            message.push_back(word);
      }
    const auto polytopes_received =
      Utilities::MPI::some_to_some(comm, polytopes_to_send);

    // The polytopes of the copy. The messages arrive in the order of the
    // sending ranks, and every message lists its polytopes in the order of
    // their global polytope index, so the polytopes of the copy are in that
    // order too, and their cells in CellId order, as on the p4est
    // triangulation.
    copy_polytopes.clear();
    p4est_rank_of_copy_polytope.clear();
    for (const auto &[p4est_rank, message] : polytopes_received)
      for (std::size_t position = 0; position < message.size();)
        {
          const std::size_t n_cells = message[position + 1];
          position += 2;
          std::vector<CellIterator> cells;
          for (std::size_t cell = 0; cell < n_cells; ++cell, position += 4)
            {
              std::array<unsigned int, 4> cell_id;
              for (unsigned int word = 0; word < 4; ++word)
                cell_id[word] =
                  static_cast<unsigned int>(message[position + word]);
              cells.emplace_back(
                copy_tria.create_cell_iterator(CellId(cell_id)));
              Assert(cells.back()->is_locally_owned(), ExcInternalError());
            }
          copy_polytopes.push_back(std::move(cells));
          p4est_rank_of_copy_polytope.push_back(p4est_rank);
        }
  }



  template <int dim>
  const parallel::fullydistributed::Triangulation<dim> &
  PartitionedCopy<dim>::get_triangulation() const
  {
    return copy_tria;
  }



  template <int dim>
  const std::vector<std::vector<typename PartitionedCopy<dim>::CellIterator>> &
  PartitionedCopy<dim>::get_polytopes() const
  {
    return copy_polytopes;
  }



  template <int dim>
  double
  PartitionedCopy<dim>::dof_imbalance() const
  {
    return imbalance;
  }



  template <int dim>
  void
  PartitionedCopy<dim>::copy_to_p4est(const Vector<float> &copy_values,
                                      Vector<float>       &p4est_values) const
  {
    // Sent as data of length one per cell.
    std::vector<std::vector<double>> copy_data(copy_tria.n_active_cells()),
      p4est_data;
    for (const auto &cell : copy_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        copy_data[cell->active_cell_index()] = {
          copy_values[cell->active_cell_index()]};
    copy_to_p4est(copy_data, p4est_data);
    p4est_values.reinit(p4est_tria.n_active_cells());
    for (const auto &cell : p4est_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        p4est_values[cell->active_cell_index()] =
          static_cast<float>(p4est_data[cell->active_cell_index()][0]);
  }



  template <int dim>
  void
  PartitionedCopy<dim>::send_cell_data(
    const std::vector<std::vector<CellIterator>> &from_polytopes,
    const std::vector<unsigned int>              &destination,
    const std::vector<std::vector<double>>       &from_data,
    const std::vector<std::vector<CellIterator>> &to_polytopes,
    const std::vector<unsigned int>              &source,
    std::vector<std::vector<double>>             &to_data) const
  {
    // One message per destination rank: for every cell of every polytope it
    // receives, the length of the data of the cell and the data.
    std::map<unsigned int, std::vector<double>> data_to_send;
    for (unsigned int polytope = 0; polytope < from_polytopes.size();
         ++polytope)
      {
        auto &message = data_to_send[destination[polytope]];
        for (const auto &cell : from_polytopes[polytope])
          {
            const auto &cell_data = from_data[cell->active_cell_index()];
            message.push_back(cell_data.size());
            message.insert(message.end(), cell_data.begin(), cell_data.end());
          }
      }
    const auto data_received = Utilities::MPI::some_to_some(comm, data_to_send);

    // The polytopes from each rank, read in the order they were written.
    to_data.clear();
    if (!to_polytopes.empty())
      to_data.resize(to_polytopes[0][0]->get_triangulation().n_active_cells());
    std::map<unsigned int, std::size_t> position;
    for (unsigned int polytope = 0; polytope < to_polytopes.size(); ++polytope)
      {
        const auto &message = data_received.at(source[polytope]);
        auto       &read    = position[source[polytope]];
        for (const auto &cell : to_polytopes[polytope])
          {
            const std::size_t length = static_cast<std::size_t>(message[read]);
            to_data[cell->active_cell_index()].assign(
              message.begin() + read + 1, message.begin() + read + 1 + length);
            read += 1 + length;
          }
      }
  }



  template <int dim>
  void
  PartitionedCopy<dim>::copy_to_p4est(
    const std::vector<std::vector<double>> &copy_data,
    std::vector<std::vector<double>>       &p4est_data) const
  {
    send_cell_data(copy_polytopes,
                   p4est_rank_of_copy_polytope,
                   copy_data,
                   p4est_polytopes,
                   copy_rank_of_p4est_polytope,
                   p4est_data);
  }



  template <int dim>
  void
  PartitionedCopy<dim>::p4est_to_copy(
    const std::vector<std::vector<double>> &p4est_data,
    std::vector<std::vector<double>>       &copy_data) const
  {
    send_cell_data(p4est_polytopes,
                   copy_rank_of_p4est_polytope,
                   p4est_data,
                   copy_polytopes,
                   p4est_rank_of_copy_polytope,
                   copy_data);
  }
} // namespace dealii::PolytopeRepartition

#endif
