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
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <boost/serialization/utility.hpp>

#include <parmetis.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>


namespace dealii::PolytopeRepartition
{
  namespace internal
  {
    /**
     * What one rank of the p4est triangulation sends to one rank of the copy
     * for the description of the copy: the coarse cells, their ids and their
     * vertices, numbered as on the p4est triangulation, whose coarse mesh is
     * the same on every rank, and the cells of every level.
     */
    template <int dim>
    struct DescriptionPart
    {
      std::vector<dealii::CellData<dim>>  coarse_cells;
      std::vector<types::coarse_cell_id>  coarse_cell_ids;
      std::vector<unsigned int>           vertex_indices;
      std::vector<Point<dim>>             vertices;
      std::vector<std::vector<TriangulationDescription::CellData<dim>>>
        cell_infos;

      template <class Archive>
      void
      serialize(Archive &ar, const unsigned int /*version*/)
      {
        ar &coarse_cells;
        ar &coarse_cell_ids;
        ar &vertex_indices;
        ar &vertices;
        ar &cell_infos;
      }
    };
  } // namespace internal



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
   * polytope is the index of its vertex in the graph. By default it is the
   * position of the polytope in the list of the polytopes of all ranks, taken
   * rank by rank. Every polytope lies on one rank of the copy, and the
   * polytopes are the same as on the p4est triangulation. A polytope may also
   * hold cells of several ranks of the p4est triangulation, as with
   * PolytopeAdaptivity::InitialPartition::metis: its pieces on these ranks
   * are joined on the copy.
   *
   * Per-cell data move between the two triangulations piece by piece, in an
   * order both sides know, so that no CellId is sent. The error indicators
   * and, for time-dependent problems, the solution go back to the p4est
   * triangulation, because the refinement and coarsening and the transfer of
   * data to the next mesh happen there. A
   * parallel::fullydistributed::Triangulation can not be refined, so the copy
   * is built anew in every cycle.
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
     * Build the copy for polytopes that may hold cells of several ranks of
     * the p4est triangulation, as PolytopeAdaptivity::AdaptivePolytopalMesh
     * gives them with PolytopeAdaptivity::InitialPartition::metis. @p pieces
     * are the parts of the polytopes on this rank, each a list of locally
     * owned cells, and @p polytope_indices their global polytope indices;
     * several pieces, on one rank or on several, may have the same index.
     * Rank r holds the vertices of the polytopes with the indices from
     * @p polytope_index_offsets[r] to @p polytope_index_offsets[r+1] - 1 in
     * the graph that ParMETIS partitions. On the copy, the pieces of a
     * polytope are joined into one polytope, in the order of the ranks they
     * come from.
     *
     * If @p copy_ranks is not empty, it gives the rank on the copy of every
     * piece, the same for all pieces of a polytope, and ParMETIS is not
     * called. With the tree ranks of
     * PolytopeAdaptivity::AdaptivePolytopalMesh::get_tree_ranks(), the copy
     * keeps the partition of the initial cells, which tells what the
     * partition by ParMETIS gains.
     */
    void
    reinit(const std::vector<std::vector<CellIterator>> &pieces,
           const std::vector<types::global_cell_index>  &polytope_indices,
           const std::vector<types::global_cell_index>  &polytope_index_offsets,
           const unsigned int                            dofs_per_polytope,
           const std::vector<unsigned int>              &copy_ranks = {});

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
     * The rank on the copy of every polytope, by global polytope index, from
     * ParMETIS on the graph of the polytopes, for the arguments of reinit().
     */
    std::vector<unsigned int>
    partition_by_parmetis(
      const std::vector<std::vector<CellIterator>> &pieces,
      const std::vector<types::global_cell_index>  &polytope_indices,
      const std::vector<types::global_cell_index>  &polytope_index_offsets,
      const unsigned int                            dofs_per_polytope) const;

    /**
     * The description of the copy on this rank, for the rank on the copy of
     * every locally owned cell in @p copy_partition. It is the description
     * that
     * TriangulationDescription::Utilities::create_description_from_triangulation()
     * gives, with the coarse cells sorted by their id. A rank of the copy
     * gets its cells, every active cell that shares a vertex with one of
     * them, and the parents of all these. Each rank of the p4est
     * triangulation sends these cells for its locally owned cells, which it
     * finds among its locally owned and ghost cells. The deal.II function
     * instead loops, for every rank it sends to, over all cells and vertices
     * of every level, the coarse cells of the whole mesh included, so it
     * becomes slower as the number of ranks grows.
     */
    TriangulationDescription::Description<dim>
    create_description(
      const LinearAlgebra::distributed::Vector<double> &copy_partition) const;

    /**
     * Send per-cell data from the polytopes of one triangulation to the same
     * polytopes on the other. Polytope @p k of @p from_polytopes goes to the
     * rank @p destination[k], and polytope @p k of @p to_polytopes comes from
     * the rank @p source[k]. Between two ranks, both sides list the polytopes
     * they share in the order of their global polytope index and the cells
     * of each in CellId order, so no CellId is sent. The cells of a polytope
     * often hold the same data, the polynomial of the polytope, so a message
     * holds, polytope by polytope, every distinct data of its cells once,
     * with its length, followed by the position of the data of every cell
     * among them. The positions are left out when all cells hold the same
     * data, or when every cell holds different data. If the positions would
     * make the message longer than the data of every cell, the data of every
     * cell is sent instead, as if all were different. @p to_data is indexed by
     * the active cell index of the triangulation of @p to_polytopes.
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
     * The pieces of the polytopes of copy_polytopes, as they came from the
     * ranks of the p4est triangulation, and for each that rank. A polytope
     * that lies on one rank of the p4est triangulation is one piece.
     */
    std::vector<std::vector<CellIterator>> copy_pieces;
    std::vector<unsigned int>              p4est_rank_of_copy_piece;

    /**
     * The pieces of polytopes this rank owns on the p4est triangulation, as
     * given to reinit(). They stay valid until the p4est triangulation is
     * refined, and reinit() is called again before the next transfer anyway.
     */
    std::vector<std::vector<CellIterator>> p4est_pieces;

    /**
     * For every piece of p4est_pieces, the rank that holds it on the copy.
     */
    std::vector<unsigned int> copy_rank_of_p4est_piece;

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
    // Every polytope is one piece, and the polytopes of rank r have the
    // global polytope indices from the number of polytopes on the ranks
    // before r on, in the order given.
    std::vector<types::global_cell_index> polytope_index_offsets{0};
    for (const auto count :
         Utilities::MPI::all_gather(comm,
                                    types::global_cell_index(polytopes.size())))
      polytope_index_offsets.push_back(polytope_index_offsets.back() + count);
    const types::global_cell_index first_polytope =
      polytope_index_offsets[Utilities::MPI::this_mpi_process(comm)];
    std::vector<types::global_cell_index> polytope_indices(polytopes.size());
    for (std::size_t polytope = 0; polytope < polytopes.size(); ++polytope)
      polytope_indices[polytope] = first_polytope + polytope;
    reinit(polytopes,
           polytope_indices,
           polytope_index_offsets,
           dofs_per_polytope);
  }



  template <int dim>
  void
  PartitionedCopy<dim>::reinit(
    const std::vector<std::vector<CellIterator>> &pieces,
    const std::vector<types::global_cell_index>  &polytope_indices,
    const std::vector<types::global_cell_index>  &polytope_index_offsets,
    const unsigned int                            dofs_per_polytope,
    const std::vector<unsigned int>              &copy_ranks)
  {
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
    AssertDimension(pieces.size(), polytope_indices.size());
    AssertDimension(polytope_index_offsets.size(), n_ranks + 1);
    Assert(copy_ranks.empty() || copy_ranks.size() == pieces.size(),
           ExcDimensionMismatch(copy_ranks.size(), pieces.size()));

    // The rank on the copy of every polytope, by global polytope index.
    const types::global_cell_index n_polytopes =
      polytope_index_offsets[n_ranks];
    std::vector<unsigned int> rank_of_polytope;
    if (copy_ranks.empty())
      rank_of_polytope = partition_by_parmetis(pieces,
                                               polytope_indices,
                                               polytope_index_offsets,
                                               dofs_per_polytope);
    else
      {
        // Every polytope has at least one piece, and all its pieces have the
        // same rank on the copy.
        rank_of_polytope.assign(n_polytopes, 0);
        for (std::size_t piece = 0; piece < pieces.size(); ++piece)
          rank_of_polytope[polytope_indices[piece]] = copy_ranks[piece];
        Utilities::MPI::max(ArrayView<const unsigned int>(rank_of_polytope),
                            comm,
                            make_array_view(rank_of_polytope));
      }
    std::vector<std::uint64_t> polytopes_per_rank(n_ranks, 0);
    for (const unsigned int rank : rank_of_polytope)
      ++polytopes_per_rank[rank];
    AssertThrow(*std::min_element(polytopes_per_rank.begin(),
                                  polytopes_per_rank.end()) > 0,
                ExcMessage("A rank owns no polytope on the copy."));

    // Every polytope has the same number of degrees of freedom, so the
    // largest over mean number of degrees of freedom per rank of the copy is
    // that of polytopes.
    imbalance = double(*std::max_element(polytopes_per_rank.begin(),
                                         polytopes_per_rank.end())) /
                (double(n_polytopes) / n_ranks);

    // The rank on the copy of every local piece, and of every cell in
    // copy_partition, a vector over the global active cells of the p4est
    // triangulation, the form create_description_from_triangulation() takes.
    p4est_pieces = pieces;
    copy_rank_of_p4est_piece.resize(pieces.size());
    const auto partitioner =
      p4est_tria.global_active_cell_index_partitioner().lock();
    LinearAlgebra::distributed::Vector<double> copy_partition(partitioner);
    for (std::size_t piece = 0; piece < pieces.size(); ++piece)
      {
        const auto rank = rank_of_polytope[polytope_indices[piece]];
        copy_rank_of_p4est_piece[piece] = rank;
        for (const auto &cell : pieces[piece])
          copy_partition[cell->global_active_cell_index()] = rank;
      }

    // Build the copy: every cell goes to the rank of its polytope.
    copy_tria.clear();
    copy_tria.create_triangulation(create_description(copy_partition));

    // Every piece, sent to the rank that holds its polytope on the copy: its
    // global polytope index, its number of cells and the CellIds of its
    // cells in binary form, four words each, in the order of the cells in
    // the piece.
    std::map<unsigned int, std::vector<std::uint64_t>> cells_to_send;
    for (std::size_t piece = 0; piece < pieces.size(); ++piece)
      {
        auto &message = cells_to_send[copy_rank_of_p4est_piece[piece]];
        message.push_back(polytope_indices[piece]);
        message.push_back(pieces[piece].size());
        for (const auto &cell : pieces[piece])
          for (const unsigned int word : cell->id().template to_binary<dim>())
            message.push_back(word);
      }

    // The pieces on the copy, in the order of the ranks they come from and,
    // from each rank, in the order it sent them, which both sides use to
    // transfer data. The polytopes of the copy join the pieces with the same
    // global polytope index and are in the order of that index. A polytope
    // of one piece keeps the order of its cells on the p4est triangulation.
    copy_pieces.clear();
    p4est_rank_of_copy_piece.clear();
    std::map<std::uint64_t, std::vector<CellIterator>> cells_of_polytope;
    for (const auto &[p4est_rank, message] :
         Utilities::MPI::some_to_some(comm, cells_to_send))
      for (std::size_t position = 0; position < message.size();)
        {
          const std::uint64_t polytope = message[position];
          const std::size_t   n_cells  = message[position + 1];
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
          auto &joined = cells_of_polytope[polytope];
          joined.insert(joined.end(), cells.begin(), cells.end());
          copy_pieces.push_back(std::move(cells));
          p4est_rank_of_copy_piece.push_back(p4est_rank);
        }
    copy_polytopes.clear();
    for (auto &[polytope, cells] : cells_of_polytope)
      copy_polytopes.push_back(std::move(cells));
  }



  template <int dim>
  std::vector<unsigned int>
  PartitionedCopy<dim>::partition_by_parmetis(
    const std::vector<std::vector<CellIterator>> &pieces,
    const std::vector<types::global_cell_index>  &polytope_indices,
    const std::vector<types::global_cell_index>  &polytope_index_offsets,
    const unsigned int                            dofs_per_polytope) const
  {
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

    // ParMETIS holds the graph distributed in contiguous blocks of vertices:
    // rank r holds the vertices from vertex_distribution[r] to
    // vertex_distribution[r + 1] - 1, the vertex of a polytope being its
    // global polytope index. ParMETIS needs at least one vertex on every
    // rank.
    std::vector<idx_t>  vertex_distribution(polytope_index_offsets.begin(),
                                           polytope_index_offsets.end());
    const std::uint64_t first_polytope = vertex_distribution[my_rank];
    const std::uint64_t n_local_polytopes =
      vertex_distribution[my_rank + 1] - first_polytope;
    const std::uint64_t n_vertices = vertex_distribution[n_ranks];
    AssertThrow(Utilities::MPI::min(n_local_polytopes, comm) > 0,
                ExcMessage("A rank owns no polytope."));
    const auto rank_of_index = [&](const std::uint64_t index) {
      return static_cast<unsigned int>(
        std::upper_bound(polytope_index_offsets.begin(),
                         polytope_index_offsets.end(),
                         index) -
        polytope_index_offsets.begin() - 1);
    };

    // The vertex of the polytope of every cell, in a vector over the global
    // active cells of the p4est triangulation, so that the ghost cells know
    // the vertex of their polytope too.
    const auto partitioner =
      p4est_tria.global_active_cell_index_partitioner().lock();
    LinearAlgebra::distributed::Vector<double> polytope_of_cell(partitioner);
    for (std::size_t piece = 0; piece < pieces.size(); ++piece)
      for (const auto &cell : pieces[piece])
        polytope_of_cell[cell->global_active_cell_index()] =
          double(polytope_indices[piece]);
    polytope_of_cell.update_ghost_values();

    // The neighbours of every piece: the polytopes of the cells across its
    // faces. Across a hanging face the neighbours are the children of the
    // neighbouring cell on that face. Every piece sends its number of cells
    // and its neighbours to the rank that holds the vertex of its polytope:
    // the global polytope index, the number of cells, the number of
    // neighbours and the neighbours.
    std::map<unsigned int, std::vector<std::uint64_t>> pieces_to_send;
    for (std::size_t piece = 0; piece < pieces.size(); ++piece)
      {
        const std::uint64_t     this_polytope = polytope_indices[piece];
        std::set<std::uint64_t> neighbors;
        for (const auto &cell : pieces[piece])
          {
            const auto add_neighbor = [&](const auto &neighbor_cell) {
              const std::uint64_t neighbor_polytope = std::uint64_t(
                polytope_of_cell[neighbor_cell->global_active_cell_index()]);
              if (neighbor_polytope != this_polytope)
                neighbors.insert(neighbor_polytope);
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
        auto &message = pieces_to_send[rank_of_index(this_polytope)];
        message.push_back(this_polytope);
        message.push_back(pieces[piece].size());
        message.push_back(neighbors.size());
        message.insert(message.end(), neighbors.begin(), neighbors.end());
      }

    // The number of cells and the neighbours of every local vertex, from
    // all its pieces.
    std::vector<std::uint64_t>           n_cells(n_local_polytopes, 0);
    std::vector<std::set<std::uint64_t>> vertex_neighbors(n_local_polytopes);
    for (const auto &[rank, message] :
         Utilities::MPI::some_to_some(comm, pieces_to_send))
      for (std::size_t position = 0; position < message.size();)
        {
          const std::uint64_t vertex = message[position] - first_polytope;
          n_cells[vertex] += message[position + 1];
          const std::size_t n_neighbors = message[position + 2];
          vertex_neighbors[vertex].insert(message.begin() + position + 3,
                                          message.begin() + position + 3 +
                                            n_neighbors);
          position += 3 + n_neighbors;
        }

    // The graph in the compressed format of ParMETIS: the neighbours of
    // local vertex v are adjacency[adjacency_offsets[v]], ...,
    // adjacency[adjacency_offsets[v + 1] - 1]. Every edge is seen from both
    // of its polytopes, so the adjacency is symmetric, as ParMETIS requires.
    std::vector<idx_t> adjacency_offsets{0}, adjacency;
    for (std::uint64_t vertex = 0; vertex < n_local_polytopes; ++vertex)
      {
        for (const std::uint64_t neighbor : vertex_neighbors[vertex])
          adjacency.push_back(idx_t(neighbor));
        adjacency_offsets.push_back(idx_t(adjacency.size()));
      }

    // Two weights per vertex, degrees of freedom and cells.
    std::vector<idx_t> vertex_weights(2 * n_local_polytopes);
    for (std::uint64_t vertex = 0; vertex < n_local_polytopes; ++vertex)
      {
        vertex_weights[2 * vertex]     = idx_t(std::max(1u, dofs_per_polytope));
        vertex_weights[2 * vertex + 1] = idx_t(n_cells[vertex]);
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
    // of equal size. The vertices are numbered rank by rank, so every block
    // covers the polytopes of one or two neighbouring ranks, a connected
    // part of the domain as far as the partition of these ranks is.
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

    return std::vector<unsigned int>(rank_of_vertex.begin(),
                                     rank_of_vertex.end());
  }



  template <int dim>
  TriangulationDescription::Description<dim>
  PartitionedCopy<dim>::create_description(
    const LinearAlgebra::distributed::Vector<double> &copy_partition) const
  {
    Assert(p4est_tria.get_periodic_face_map().empty(), ExcNotImplemented());

    // The rank on the copy of a locally owned or ghost cell.
    copy_partition.update_ghost_values();
    const auto rank_on_copy = [&](const auto &cell) {
      return static_cast<unsigned int>(
        copy_partition[cell->global_active_cell_index()]);
    };

    // A cell of any level as one number, its level and its index on the
    // level, which sorts faster than an iterator.
    using AnyCellIterator = typename Triangulation<dim>::cell_iterator;
    const auto key_of = [](const AnyCellIterator &cell) {
      return (std::uint64_t(cell->level()) << 32) |
             std::uint64_t(cell->index());
    };
    const auto cell_of = [&](const std::uint64_t key) {
      return AnyCellIterator(&p4est_tria,
                             int(key >> 32),
                             int(key & 0xffffffff));
    };

    // The locally owned and ghost cells at each of their vertices, sorted by
    // the index of the vertex, with the rank on the copy of the locally owned
    // ones. This is the only loop over all active cells, which include the
    // coarse cells of the whole mesh away from this rank.
    struct CellAtVertex
    {
      unsigned int  vertex;
      std::uint64_t cell;
      unsigned int  rank;
    };
    std::vector<CellAtVertex> cells_at_vertex;
    for (const auto &cell : p4est_tria.active_cell_iterators())
      if (!cell->is_artificial())
        {
          const unsigned int rank = cell->is_locally_owned() ?
                                      rank_on_copy(cell) :
                                      numbers::invalid_unsigned_int;
          for (const unsigned int v : cell->vertex_indices())
            cells_at_vertex.push_back(
              {cell->vertex_index(v), key_of(cell), rank});
        }
    std::sort(cells_at_vertex.begin(),
              cells_at_vertex.end(),
              [](const auto &a, const auto &b) { return a.vertex < b.vertex; });

    // The cells every rank of the copy gets from this rank: the locally owned
    // cells that go to it, the active cells that share a vertex with them,
    // and the parents of all these, each once. All cells at a vertex go to
    // the ranks of the locally owned cells at it.
    std::map<unsigned int, std::vector<std::uint64_t>> cells_for_rank;
    std::vector<unsigned int>                          ranks_at_vertex;
    for (auto first = cells_at_vertex.begin(); first != cells_at_vertex.end();)
      {
        auto last = first;
        ranks_at_vertex.clear();
        for (; last != cells_at_vertex.end() && last->vertex == first->vertex;
             ++last)
          if (last->rank != numbers::invalid_unsigned_int &&
              std::find(ranks_at_vertex.begin(),
                        ranks_at_vertex.end(),
                        last->rank) == ranks_at_vertex.end())
            ranks_at_vertex.push_back(last->rank);
        for (const unsigned int rank : ranks_at_vertex)
          {
            auto &cells = cells_for_rank[rank];
            for (auto entry = first; entry != last; ++entry)
              cells.push_back(entry->cell);
          }
        first = last;
      }
    for (auto &[rank, cells] : cells_for_rank)
      {
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        std::unordered_set<std::uint64_t> parents;
        for (const std::uint64_t key : cells)
          for (auto cell = cell_of(key); cell->level() > 0;)
            {
              cell = cell->parent();
              if (!parents.insert(key_of(cell)).second)
                break;
            }
        cells.insert(cells.end(), parents.begin(), parents.end());
      }

    // The part of the description for every rank, filled as the deal.II
    // function does. The active cells get their rank on the copy, and the
    // parents are artificial. No level of the copy is owned, since the copy
    // has no multigrid hierarchy.
    const unsigned int n_levels = p4est_tria.n_global_levels();
    std::map<unsigned int, internal::DescriptionPart<dim>> parts;
    for (const auto &[rank, cells] : cells_for_rank)
      {
        auto &part = parts[rank];
        part.cell_infos.resize(n_levels);
        for (const std::uint64_t key : cells)
          {
            const auto cell = cell_of(key);
            TriangulationDescription::CellData<dim> cell_info;
            cell_info.id = cell->id().template to_binary<dim>();
            for (const unsigned int f : cell->face_indices())
              {
                const types::boundary_id boundary_id =
                  cell->face(f)->boundary_id();
                if (boundary_id != numbers::internal_face_boundary_id)
                  cell_info.boundary_ids.emplace_back(f, boundary_id);
              }
            cell_info.manifold_id = cell->manifold_id();
            if constexpr (dim >= 2)
              for (const unsigned int line : cell->line_indices())
                cell_info.manifold_line_ids[line] =
                  cell->line(line)->manifold_id();
            if constexpr (dim == 3)
              for (const unsigned int f : cell->face_indices())
                cell_info.manifold_quad_ids[f] = cell->quad(f)->manifold_id();
            cell_info.subdomain_id =
              cell->is_active() ? rank_on_copy(cell) :
                                  numbers::artificial_subdomain_id;
            cell_info.level_subdomain_id = numbers::artificial_subdomain_id;
            part.cell_infos[cell->level()].push_back(cell_info);

            if (cell->level() == 0)
              {
                dealii::CellData<dim> coarse_cell(cell->n_vertices());
                coarse_cell.material_id = cell->material_id();
                coarse_cell.manifold_id = cell->manifold_id();
                for (const unsigned int v : cell->vertex_indices())
                  {
                    coarse_cell.vertices[v] = cell->vertex_index(v);
                    part.vertex_indices.push_back(cell->vertex_index(v));
                    part.vertices.push_back(cell->vertex(v));
                  }
                part.coarse_cells.push_back(coarse_cell);
                part.coarse_cell_ids.push_back(cell->id().get_coarse_cell_id());
              }
          }
      }

    // Join the parts from all ranks. A cell may come from several ranks, the
    // same from each, and is kept once. The coarse cells, the vertices and
    // the cells of every level are sorted by their ids, which fixes the
    // numbering of the degrees of freedom on the copy, and with it the linear
    // solver, in every run.
    std::vector<std::pair<types::coarse_cell_id, dealii::CellData<dim>>>
                                                    coarse_cells;
    std::vector<std::pair<unsigned int, Point<dim>>> vertices;
    std::vector<std::vector<TriangulationDescription::CellData<dim>>>
      cell_infos(n_levels);
    for (const auto &[rank, part] : Utilities::MPI::some_to_some(comm, parts))
      {
        for (std::size_t i = 0; i < part.coarse_cells.size(); ++i)
          coarse_cells.emplace_back(part.coarse_cell_ids[i],
                                    part.coarse_cells[i]);
        for (std::size_t i = 0; i < part.vertices.size(); ++i)
          vertices.emplace_back(part.vertex_indices[i], part.vertices[i]);
        for (unsigned int level = 0; level < n_levels; ++level)
          cell_infos[level].insert(cell_infos[level].end(),
                                   part.cell_infos[level].begin(),
                                   part.cell_infos[level].end());
      }
    const auto by_first = [](const auto &a, const auto &b) {
      return a.first < b.first;
    };
    const auto same_first = [](const auto &a, const auto &b) {
      return a.first == b.first;
    };
    std::sort(coarse_cells.begin(), coarse_cells.end(), by_first);
    coarse_cells.erase(
      std::unique(coarse_cells.begin(), coarse_cells.end(), same_first),
      coarse_cells.end());
    std::sort(vertices.begin(), vertices.end(), by_first);
    vertices.erase(std::unique(vertices.begin(), vertices.end(), same_first),
                   vertices.end());

    TriangulationDescription::Description<dim> description;
    description.comm      = comm;
    description.smoothing = p4est_tria.get_mesh_smoothing();
    for (const auto &[index, vertex] : vertices)
      description.coarse_cell_vertices.push_back(vertex);
    for (auto &[id, coarse_cell] : coarse_cells)
      {
        for (auto &vertex : coarse_cell.vertices)
          vertex = std::lower_bound(vertices.begin(),
                                    vertices.end(),
                                    vertex,
                                    [](const auto &a, const unsigned int b) {
                                      return a.first < b;
                                    }) -
                   vertices.begin();
        description.coarse_cells.push_back(coarse_cell);
        description.coarse_cell_index_to_coarse_cell_id.push_back(id);
      }
    description.cell_infos.resize(n_levels);
    for (unsigned int level = 0; level < n_levels; ++level)
      {
        auto &infos = cell_infos[level];
        std::sort(infos.begin(), infos.end(), [](const auto &a, const auto &b) {
          return a.id < b.id;
        });
        for (const auto &cell_info : infos)
          if (description.cell_infos[level].empty() ||
              description.cell_infos[level].back().id != cell_info.id)
            description.cell_infos[level].push_back(cell_info);
      }
    return description;
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
    // One message per destination rank. For every polytope it receives: the
    // number of distinct data of its cells, each of them with its length,
    // and the position of the data of every cell among them. The positions
    // are left out if there is one distinct data, which every cell then
    // holds, or as many as cells, in which case cell i holds the i-th. A
    // polytope of one cell has one data, and its number is left out too.
    std::map<unsigned int, std::vector<double>> data_to_send;
    for (unsigned int polytope = 0; polytope < from_polytopes.size();
         ++polytope)
      {
        const auto &cells = from_polytopes[polytope];
        std::map<std::vector<double>, unsigned int> position_of_data;
        std::vector<const std::vector<double> *> distinct_data;
        std::vector<unsigned int>                position_of_cell(cells.size());
        for (unsigned int i = 0; i < cells.size(); ++i)
          {
            const auto &cell_data = from_data[cells[i]->active_cell_index()];
            const auto [entry, is_new] =
              position_of_data.emplace(cell_data, distinct_data.size());
            if (is_new)
              distinct_data.push_back(&cell_data);
            position_of_cell[i] = entry->second;
          }

        // If listing the distinct data and the positions takes more values
        // than listing the data of every cell, every cell is listed instead,
        // as if all were different.
        std::size_t n_values_cells = 0, n_values_distinct = 0;
        for (const auto &cell : cells)
          n_values_cells += 1 + from_data[cell->active_cell_index()].size();
        for (const auto *data : distinct_data)
          n_values_distinct += 1 + data->size();
        if (distinct_data.size() != 1 && distinct_data.size() != cells.size())
          n_values_distinct += cells.size();
        if (n_values_distinct > n_values_cells)
          {
            distinct_data.clear();
            for (const auto &cell : cells)
              distinct_data.push_back(&from_data[cell->active_cell_index()]);
          }

        auto &message = data_to_send[destination[polytope]];
        if (cells.size() > 1)
          message.push_back(distinct_data.size());
        for (const auto *data : distinct_data)
          {
            message.push_back(data->size());
            message.insert(message.end(), data->begin(), data->end());
          }
        if (distinct_data.size() != 1 && distinct_data.size() != cells.size())
          message.insert(message.end(),
                         position_of_cell.begin(),
                         position_of_cell.end());
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
        const auto &cells   = to_polytopes[polytope];

        // The start and the length of every distinct data in the message.
        const std::size_t n_distinct =
          cells.size() > 1 ? static_cast<std::size_t>(message[read++]) : 1;
        std::vector<std::pair<std::size_t, std::size_t>> distinct_data(
          n_distinct);
        for (auto &[start, length] : distinct_data)
          {
            length = static_cast<std::size_t>(message[read]);
            start  = read + 1;
            read += 1 + length;
          }

        const bool positions_sent =
          n_distinct != 1 && n_distinct != cells.size();
        for (std::size_t i = 0; i < cells.size(); ++i)
          {
            const std::size_t data_index =
              positions_sent ? static_cast<std::size_t>(message[read + i]) :
                               (n_distinct == 1 ? 0 : i);
            const auto [start, length] = distinct_data[data_index];
            to_data[cells[i]->active_cell_index()].assign(
              message.begin() + start, message.begin() + start + length);
          }
        if (positions_sent)
          read += cells.size();
      }
  }



  template <int dim>
  void
  PartitionedCopy<dim>::copy_to_p4est(
    const std::vector<std::vector<double>> &copy_data,
    std::vector<std::vector<double>>       &p4est_data) const
  {
    send_cell_data(copy_pieces,
                   p4est_rank_of_copy_piece,
                   copy_data,
                   p4est_pieces,
                   copy_rank_of_p4est_piece,
                   p4est_data);
  }



  template <int dim>
  void
  PartitionedCopy<dim>::p4est_to_copy(
    const std::vector<std::vector<double>> &p4est_data,
    std::vector<std::vector<double>>       &copy_data) const
  {
    send_cell_data(p4est_pieces,
                   copy_rank_of_p4est_piece,
                   p4est_data,
                   copy_pieces,
                   p4est_rank_of_copy_piece,
                   copy_data);
  }
} // namespace dealii::PolytopeRepartition

#endif
