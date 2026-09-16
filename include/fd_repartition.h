#ifndef FD_REPARTITION_H
#define FD_REPARTITION_H

// =====================================================================
// Polytope-balanced repartitioning: p4est adapts, fullydistributed solves.
//
// WHY. p4est partitions by CELL COUNT along its Morton curve, but the
// unknowns live on POLYTOPES (dofs_per_polytope is constant regardless of how
// many cells a polytope holds). Since the solve is 80%+ of the run and scales
// with dofs, p4est balances the cheap currency: measured dofs max/mean is
// 1.27-2.68 where cells max/mean is 1.01. p4est cannot fix this -- its
// partition is always a contiguous range of a space-filling curve, and its
// weight signal is a single scalar, so "balance cells" and "balance dofs"
// have to be traded off against each other rather than both satisfied.
//
// WHAT. Partition the POLYTOPE graph with METIS (two constraints: dofs and
// cells) and rebuild the mesh as a parallel::fullydistributed::Triangulation
// with that partition. Measured payoff: dofs max/mean drops to 1.02-1.03 at
// every cycle while cells max/mean only degrades 1.006 -> 1.027. Cost: ~4ms
// to rebuild, ~22ms for serial METIS. See
// comparison_charts/loadbal/FD_PROBE_FINDINGS.md.
//
// WHY THIS SPLIT. fullydistributed cannot refine or coarsen itself (both
// overrides are DEAL_II_NOT_IMPLEMENTED) -- but it does not have to.
// copy_triangulation()/create_description_from_triangulation() explicitly
// accept an ALREADY REFINED parallel::distributed::Triangulation, so p4est
// keeps doing mark/refine/coarsen/CellDataTransfer exactly as before and only
// the agglomerate+assemble+solve stage moves onto the rebuilt copy.
// parallel::shared is not an option: AgglomerationHandler asserts against it
// (source/agglomeration_handler.cc:33).
//
// KEY ACROSS THE TWO TRIANGULATIONS. The two meshes hold the same cells but
// number them differently (each has its own global_active_cell_index). CellId
// is the partition-independent key -- it encodes coarse cell id plus the child
// path, and create_description_from_triangulation preserves it -- so every
// exchange here is keyed on CellId.
// =====================================================================

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <metis.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace FDRepartition
{
  using namespace dealii;

  // What one cell has to carry from the p4est mesh to its new owner: which
  // polytope it belongs to, and where it lived in the source numbering (so
  // per-cell results can be sent back without a second key exchange).
  struct CellPayload
  {
    std::array<unsigned int, 4> id;       // CellId::to_binary<dim>()
    std::uint64_t               poly_gid; // global polytope index
    std::uint64_t               src_gaci; // source global_active_cell_index

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &id &poly_gid &src_gaci;
    }
  };


  // Timing/quality report of one repartition, for the [fd-repart] line.
  struct Report
  {
    std::uint64_t n_poly_global = 0;
    double        dof_imbalance_before = 0.;
    double        dof_imbalance_after  = 0.;
    double        cell_imbalance_after = 0.;
    double        t_graph = 0., t_metis = 0., t_build = 0., t_migrate = 0.;
    idx_t         edgecut = 0;

    // Set when the cost gate decided this cycle was not worth repartitioning.
    bool   skipped          = false;
    double predicted_saving = 0.;
    double gate_overhead    = 0.; // rank-uniform overhead the gate compared against

    // How many of the n_ranks parts came out in more than one connected
    // piece. This -- not edgecut -- is what METIS_OPTION_CONTIG controls.
    unsigned int n_parts_disconnected = 0;

    double
    overhead() const
    {
      return t_graph + t_metis + t_build + t_migrate;
    }
  };


  // Running state the cost gate needs from the previous cycle. Kept by the
  // caller so this header stays free of globals.
  struct GateState
  {
    double last_overhead   = 0.; // what one repartition cost last time
    double last_solve_time = 0.; // what one solve cost last time

    // Global polytope count at the cycle last_solve_time was measured on.
    // The gate has to predict THIS cycle's solve from LAST cycle's, and under
    // AMR the mesh in between only ever grows, so the raw previous value is a
    // systematic underestimate. This lets the gate scale it by how much the
    // problem grew. Must be updated by the caller EVERY cycle, including
    // cycles the gate skipped -- last_solve_time is measured every cycle, so
    // the two have to refer to the same cycle or the growth ratio spans the
    // wrong interval. See examples/fd_gate_test.cc.
    std::uint64_t last_n_poly = 0;
  };


  inline double
  max_over_mean(const std::vector<std::uint64_t> &per_rank)
  {
    if (per_rank.empty())
      return 0.;
    const std::uint64_t mx =
      *std::max_element(per_rank.begin(), per_rank.end());
    std::uint64_t sum = 0;
    for (const auto v : per_rank)
      sum += v;
    if (sum == 0)
      return 0.;
    return double(mx) / (double(sum) / per_rank.size());
  }


  // -------------------------------------------------------------------
  // Rebuild `dst_tria` as a polytope-balanced copy of `src_tria`, and
  // reconstruct the polytope grouping on it.
  //
  // Whole polytopes are moved, never split -- so the discretization is
  // bit-identical to what the source partition would have produced; only the
  // rank that owns each polytope changes.
  //
  // Outputs:
  //   dst_tria       -- created (must be empty on entry)
  //   dst_polytopes  -- the same polytopes, as cell iterators into dst_tria
  //   dst_to_src     -- per dst active_cell_index, the SOURCE
  //                     global_active_cell_index (used to send results back)
  // Returns false and leaves everything untouched if the partition could not
  // be computed (single rank, degenerate graph, METIS failure) -- callers then
  // just keep using the p4est mesh, which is always correct, only slower.
  // -------------------------------------------------------------------
  template <int dim>
  bool
  build_balanced_copy(
    const parallel::distributed::Triangulation<dim> &src_tria,
    const std::vector<
      std::vector<typename Triangulation<dim>::active_cell_iterator>>
                      &src_polytopes,
    const unsigned int dofs_per_polytope,
    const MPI_Comm     comm,
    parallel::fullydistributed::Triangulation<dim> &dst_tria,
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
                                           &dst_polytopes,
    std::vector<types::global_cell_index>  &dst_to_src,
    Report                                 &report,
    const GateState                        &gate = GateState{})
  {
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

    if (n_ranks < 2)
      return false;

    // --- 1. global polytope numbering (contiguous blocks in rank order) ---
    std::vector<unsigned int> local_poly_sizes;
    local_poly_sizes.reserve(src_polytopes.size());
    for (const auto &grp : src_polytopes)
      if (!grp.empty() && grp.front()->is_locally_owned())
        local_poly_sizes.push_back(grp.size());

    const std::uint64_t n_loc_poly  = local_poly_sizes.size();
    std::uint64_t       poly_offset = 0;
    int                 ierr        = MPI_Exscan(
      &n_loc_poly, &poly_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
    AssertThrowMPI(ierr);
    const std::uint64_t n_glob_poly = Utilities::MPI::sum(n_loc_poly, comm);
    report.n_poly_global            = n_glob_poly;

    if (n_glob_poly < n_ranks)
      return false;

    // --- 1b. COST GATE: is this cycle even worth repartitioning? ---
    //
    // dofs are dofs_per_polytope per polytope, so the current imbalance is
    // just max/mean of the polytope COUNT -- one reduction, no graph needed.
    // Repartitioning turns a solve that waits for the busiest rank into one
    // that waits for ~the mean, so the saving is (1 - 1/imbalance) times a
    // solve; spend the overhead only when that is bigger.
    //
    // Self-calibrating on purpose. A fixed imbalance threshold is wrong
    // because the break-even point depends on how expensive the solve is:
    // at ~1s/solve and ~12ms overhead it sits around imbalance 1.008, so on
    // THIS workload the gate should essentially never fire (measured
    // end-of-run imbalance is still 1.098, worth ~160ms against ~12ms). On a
    // problem with a cheap solve the same rule skips automatically.
    // FD_GATE=0 disables the gate entirely.
    const std::uint64_t max_loc_poly = Utilities::MPI::max(n_loc_poly, comm);
    const double        imbalance_now =
      double(max_loc_poly) / (double(n_glob_poly) / n_ranks);
    report.dof_imbalance_before = imbalance_now;

    static const bool gate_enabled = []{
      const char *e = std::getenv("FD_GATE");
      return !(e && std::string(e) == "0");
    }();

    // THE GATE'S INPUTS MUST BE RANK-UNIFORM, and they are not as measured:
    // t_metis is nonzero only on rank 0 (the partition is serial there) and
    // t_solve is each rank's own timer. Comparing per-rank values would let
    // some ranks take the early return below while others walk into the
    // gather/broadcast further down -- a call-count mismatch, i.e. a hang.
    // (Observed: a 2.3s run wedged past 10 minutes.) Reduce both with
    // unconditional collectives BEFORE any branch, so every rank compares the
    // same two numbers and reaches the same verdict. max is also the right
    // semantics: the critical path is set by the slowest rank.
    const double last_overhead =
      Utilities::MPI::max(gate.last_overhead, comm);
    const double last_solve =
      Utilities::MPI::max(gate.last_solve_time, comm);
    const std::uint64_t last_n_poly =
      Utilities::MPI::max(gate.last_n_poly, comm);

    if (gate_enabled && last_overhead > 0. && last_solve > 0.)
      {
        // GROWTH CORRECTION. last_solve is the previous cycle's solve, but the
        // saving being weighed is on THIS cycle's solve, and AMR grows the
        // problem between the two -- measured on this workload, solve time
        // rose in 10 of 12 cycle transitions, by up to 1.83x in one step. Left
        // uncorrected the predicted saving is biased low, and the error is
        // one-directional: the gate skips repartitions that would have paid,
        // never the reverse.
        //
        // Scaling by the polytope-count ratio is the cheap correction: dofs
        // are dofs_per_polytope per polytope, so the ratio IS the dof growth,
        // and n_glob_poly was already reduced two lines up -- no extra
        // collective, no model of how the solver scales beyond "cost follows
        // problem size". Clamped so one anomalous cycle (or a first cycle with
        // no history) cannot extrapolate wildly.
        //
        // examples/fd_gate_test.cc replays a measured trace and quantifies
        // this: the total band of overheads that provoke a wrong skip shrinks
        // from 149.1 ms to 43.4 ms, and at the 23.3 ms overhead actually
        // measured with FD_METIS_CONTIG=1 the wrong skip disappears. It also
        // checks the correction does not simply become a rubber stamp.
        // FD_GATE_GROWTH=0 restores the uncorrected predictor, so the
        // end-to-end A/B can be run on one binary.
        static const bool growth_enabled = []{
          const char *e = std::getenv("FD_GATE_GROWTH");
          return !(e && std::string(e) == "0");
        }();

        double growth = 1.;
        if (growth_enabled && last_n_poly > 0)
          growth = std::min(4.0,
                            std::max(0.25,
                                     double(n_glob_poly) / double(last_n_poly)));

        report.predicted_saving =
          (1. - 1. / std::max(imbalance_now, 1.)) * last_solve * growth;
        report.gate_overhead = last_overhead;
        if (report.predicted_saving < last_overhead)
          {
            report.skipped = true;
            return false;
          }
      }

    // --- 2. publish "which polytope owns me" into the ghost layer ---
    const auto partitioner =
      src_tria.global_active_cell_index_partitioner().lock();
    Assert(partitioner, ExcInternalError());

    LinearAlgebra::distributed::Vector<double> poly_gid(partitioner);
    poly_gid = -1.;
    {
      std::uint64_t g = poly_offset;
      for (const auto &grp : src_polytopes)
        {
          if (grp.empty() || !grp.front()->is_locally_owned())
            continue;
          for (const auto &cell : grp)
            poly_gid[cell->global_active_cell_index()] = double(g);
          ++g;
        }
    }
    poly_gid.update_ghost_values();

    // --- 3. polytope adjacency graph ---
    //
    // Hanging-node pitfall (documented in adaptive_agglomerator.h's
    // group_touches_ghost): cell->neighbor(f) is only guaranteed ACTIVE when
    // the neighbor is at the same level or finer. At a hanging-node face it
    // is the COARSER, non-active cell and its global_active_cell_index is
    // meaningless -- walk neighbor_child_on_subface() instead.
    Timer t_graph;
    std::set<std::pair<std::uint64_t, std::uint64_t>> local_edges;
    auto add_edge = [&local_edges](const double a, const double b) {
      if (a < 0. || b < 0.)
        return;
      const auto ga = std::uint64_t(a), gb = std::uint64_t(b);
      if (ga != gb)
        local_edges.emplace(std::min(ga, gb), std::max(ga, gb));
    };

    for (const auto &cell : src_tria.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        const double my_g = poly_gid[cell->global_active_cell_index()];
        for (const auto f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;
            const auto nb = cell->neighbor(f);
            if (nb->is_active())
              {
                if (!nb->is_artificial())
                  add_edge(my_g, poly_gid[nb->global_active_cell_index()]);
              }
            else
              for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                {
                  const auto ch = cell->neighbor_child_on_subface(f, sf);
                  if (ch.state() == IteratorState::valid && ch->is_active() &&
                      !ch->is_artificial())
                    add_edge(my_g, poly_gid[ch->global_active_cell_index()]);
                }
          }
      }
    t_graph.stop();
    report.t_graph = t_graph.wall_time();

    // --- 4. serial METIS on rank 0 ---
    //
    // ParMETIS is not needed: the graph has one vertex per POLYTOPE, not per
    // cell (a few thousand where the mesh has hundreds of thousands of
    // cells), so gathering it costs nothing. ncon=2 is the whole point of
    // leaving p4est behind -- constraint 0 is dofs (balances the SOLVE),
    // constraint 1 is cell count (balances ASSEMBLE); p4est's scalar weight
    // can only trade one against the other.
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> local_edge_vec(
      local_edges.begin(), local_edges.end());
    const auto all_edges = Utilities::MPI::gather(comm, local_edge_vec, 0);
    const auto all_sizes = Utilities::MPI::gather(comm, local_poly_sizes, 0);

    std::vector<unsigned int> poly_to_rank;
    Timer                     t_metis;
    bool                      metis_ok = false;

    if (my_rank == 0)
      {
        std::vector<unsigned int> poly_cells;
        poly_cells.reserve(n_glob_poly);
        for (const auto &blk : all_sizes)
          poly_cells.insert(poly_cells.end(), blk.begin(), blk.end());

        std::vector<std::set<std::uint64_t>> adj(n_glob_poly);
        for (const auto &blk : all_edges)
          for (const auto &e : blk)
            {
              adj[e.first].insert(e.second);
              adj[e.second].insert(e.first);
            }

        std::vector<idx_t> xadj, adjncy;
        xadj.reserve(n_glob_poly + 1);
        xadj.push_back(0);
        for (std::uint64_t v = 0; v < n_glob_poly; ++v)
          {
            for (const auto w : adj[v])
              adjncy.push_back(idx_t(w));
            xadj.push_back(idx_t(adjncy.size()));
          }

        std::vector<idx_t> vwgt(2 * n_glob_poly);
        for (std::uint64_t v = 0; v < n_glob_poly; ++v)
          {
            vwgt[2 * v + 0] = idx_t(dofs_per_polytope);
            vwgt[2 * v + 1] = idx_t(poly_cells[v]);
          }

        // RECURSIVE BISECTION IS THE DEFAULT, measured not assumed. k-way cost
        // ~22ms and was almost independent of graph size (22.3ms at 458
        // vertices, 25.4ms at 5014) -- not the profile of a graph algorithm.
        // Recursive bisection, which METIS itself recommends when nparts is
        // small, is 3x faster AND balances better on BOTH constraints
        // (8 cycles, np=8): 22.5 -> 7.5ms, dofs max/mean 1.027 -> 1.010,
        // cells 1.028 -> 1.011.
        //
        // Knobs, all off by default:
        //   FD_METIS_ALG=kway   -- back to k-way.
        //   FD_METIS_NCON=1     -- drop the cell-count constraint. REJECTED as
        //                          a default: 2.0ms but cells max/mean blows
        //                          out to 1.31, i.e. it buys speed by giving
        //                          up the second currency, which is the whole
        //                          reason for leaving p4est.
        //   FD_METIS_NITER=<n>  -- refinement passes; measured no effect.
        //   FD_METIS_CONTIG=1   -- ask for connected parts. METIS only honours
        //                          CONTIG in the k-way routine, so this forces
        //                          k-way back on and is a genuine trade, not a
        //                          free extra: see the measurement in
        //                          comparison_charts/loadbal/FD_PROBE_FINDINGS.md.
        static const bool want_contig = []{
          const char *e = std::getenv("FD_METIS_CONTIG");
          return e && std::string(e) == "1";
        }();
        static const bool use_recursive = []{
          const char *e = std::getenv("FD_METIS_ALG");
          return !(e && std::string(e) == "kway");
        }() && !want_contig;
        static const idx_t ncon_env = []{
          const char *e = std::getenv("FD_METIS_NCON");
          return (e && std::string(e) == "1") ? idx_t(1) : idx_t(2);
        }();
        static const int niter_env = []{
          const char *e = std::getenv("FD_METIS_NITER");
          return e ? std::atoi(e) : -1;
        }();

        idx_t              n      = idx_t(n_glob_poly);
        idx_t              ncon   = ncon_env;
        idx_t              nparts = idx_t(n_ranks);
        std::vector<idx_t> part(n_glob_poly, 0);
        idx_t              options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        if (niter_env > 0)
          options[METIS_OPTION_NITER] = idx_t(niter_env);
        if (want_contig)
          options[METIS_OPTION_CONTIG] = 1;

        // With ncon=1 only the first constraint (dofs) is passed through.
        std::vector<idx_t> vwgt1;
        if (ncon == 1)
          {
            vwgt1.resize(n_glob_poly);
            for (std::uint64_t v = 0; v < n_glob_poly; ++v)
              vwgt1[v] = idx_t(dofs_per_polytope);
          }
        idx_t *vwgt_ptr = (ncon == 1) ? vwgt1.data() : vwgt.data();

        t_metis.restart();
        const int mierr =
          adjncy.empty() ?
            METIS_ERROR :
            (use_recursive ? METIS_PartGraphRecursive(&n,
                                                      &ncon,
                                                      xadj.data(),
                                                      adjncy.data(),
                                                      vwgt_ptr,
                                                      nullptr,
                                                      nullptr,
                                                      &nparts,
                                                      nullptr,
                                                      nullptr,
                                                      options,
                                                      &report.edgecut,
                                                      part.data()) :
                             METIS_PartGraphKway(&n,
                                                 &ncon,
                                                 xadj.data(),
                                                 adjncy.data(),
                                                 vwgt_ptr,
                                                 nullptr,
                                                 nullptr,
                                                 &nparts,
                                                 nullptr,
                                                 nullptr,
                                                 options,
                                                 &report.edgecut,
                                                 part.data()));
        t_metis.stop();
        report.t_metis = t_metis.wall_time();
        metis_ok       = (mierr == METIS_OK);

        if (metis_ok)
          {
            poly_to_rank.resize(n_glob_poly);
            for (std::uint64_t v = 0; v < n_glob_poly; ++v)
              poly_to_rank[v] = static_cast<unsigned int>(part[v]);

            std::vector<std::uint64_t> before(n_ranks, 0), after_d(n_ranks, 0),
              after_c(n_ranks, 0);
            for (unsigned int r = 0; r < all_sizes.size(); ++r)
              before[r] =
                std::uint64_t(all_sizes[r].size()) * dofs_per_polytope;
            for (std::uint64_t v = 0; v < n_glob_poly; ++v)
              {
                after_d[poly_to_rank[v]] += dofs_per_polytope;
                after_c[poly_to_rank[v]] += poly_cells[v];
              }
            report.dof_imbalance_before = max_over_mean(before);
            report.dof_imbalance_after  = max_over_mean(after_d);
            report.cell_imbalance_after = max_over_mean(after_c);

            // How many connected pieces did each part come out in? This is
            // what METIS_OPTION_CONTIG is actually about -- edgecut is a
            // communication-volume proxy and does NOT answer it. One BFS per
            // part over the graph we already built, on rank 0 only.
            {
              std::vector<char> seen(n_glob_poly, 0);
              report.n_parts_disconnected = 0;
              std::vector<unsigned int> pieces(n_ranks, 0);
              for (std::uint64_t v = 0; v < n_glob_poly; ++v)
                {
                  if (seen[v])
                    continue;
                  const unsigned int    part_of_v = poly_to_rank[v];
                  std::vector<std::uint64_t> stack{v};
                  seen[v] = 1;
                  while (!stack.empty())
                    {
                      const auto u = stack.back();
                      stack.pop_back();
                      for (const auto w : adj[u])
                        if (!seen[w] && poly_to_rank[w] == part_of_v)
                          {
                            seen[w] = 1;
                            stack.push_back(w);
                          }
                    }
                  ++pieces[part_of_v];
                }
              for (const auto p : pieces)
                if (p > 1)
                  ++report.n_parts_disconnected;
            }
          }
      }

    metis_ok = (Utilities::MPI::sum(metis_ok ? 1 : 0, comm) > 0);
    if (!metis_ok)
      return false;
    poly_to_rank = Utilities::MPI::broadcast(comm, poly_to_rank, 0);

    // --- 5. build the fullydistributed copy with that partition ---
    Timer t_build;
    LinearAlgebra::distributed::Vector<double> partition(partitioner);
    for (const auto &cell : src_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const double g = poly_gid[cell->global_active_cell_index()];
          partition[cell->global_active_cell_index()] =
            (g < 0.) ? double(my_rank) : double(poly_to_rank[std::uint64_t(g)]);
        }
    partition.update_ghost_values();

    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(src_tria, partition);
    dst_tria.create_triangulation(description);
    t_build.stop();
    report.t_build = t_build.wall_time();

    // --- 6. migrate the polytope definitions, keyed on CellId ---
    Timer t_migrate;
    std::map<unsigned int, std::vector<CellPayload>> outgoing;
    for (const auto &cell : src_tria.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        const double g = poly_gid[cell->global_active_cell_index()];
        if (g < 0.)
          continue;
        const std::uint64_t gid    = std::uint64_t(g);
        const unsigned int  target = poly_to_rank[gid];
        outgoing[target].push_back(
          CellPayload{cell->id().template to_binary<dim>(),
                      gid,
                      std::uint64_t(cell->global_active_cell_index())});
      }

    const auto incoming = Utilities::MPI::some_to_some(comm, outgoing);

    std::map<CellId, std::pair<std::uint64_t, std::uint64_t>> by_id;
    for (const auto &[from, payloads] : incoming)
      {
        (void)from;
        for (const auto &p : payloads)
          by_id[CellId(p.id)] = {p.poly_gid, p.src_gaci};
      }

    // Group the received cells back into polytopes. std::map keeps the
    // polytope order deterministic across ranks and runs, which matters
    // because the polytope index is what material_id and every downstream
    // diagnostic key on.
    dst_to_src.assign(dst_tria.n_active_cells(),
                      numbers::invalid_dof_index);
    std::map<std::uint64_t,
             std::vector<typename Triangulation<dim>::active_cell_iterator>>
      grouped;

    unsigned int n_unmatched = 0;
    for (const auto &cell : dst_tria.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        const auto it = by_id.find(cell->id());
        if (it == by_id.end())
          {
            ++n_unmatched;
            continue;
          }
        grouped[it->second.first].push_back(cell);
        dst_to_src[cell->active_cell_index()] = it->second.second;
      }

    // A cell the new owner holds but nobody sent it means the partition and
    // the migration disagree -- bail out rather than solve on a mesh whose
    // polytopes have holes.
    const unsigned int n_unmatched_global =
      Utilities::MPI::sum(n_unmatched, comm);
    if (n_unmatched_global > 0)
      {
        if (my_rank == 0)
          std::cout << "  [fd-repart] ABORT: " << n_unmatched_global
                    << " cell(s) arrived without a polytope assignment"
                    << std::endl;
        return false;
      }

    dst_polytopes.clear();
    dst_polytopes.reserve(grouped.size());
    for (auto &[gid, cells] : grouped)
      {
        (void)gid;
        dst_polytopes.push_back(std::move(cells));
      }
    t_migrate.stop();
    report.t_migrate = t_migrate.wall_time();

    return true;
  }


  // -------------------------------------------------------------------
  // Send a per-cell result computed on the rebuilt mesh back to the p4est
  // mesh, so marking / refinement / CellDataTransfer can go on working on the
  // triangulation they have always used.
  //
  // Uses a Partitioner whose OWNED range is the source mesh's own range and
  // whose GHOST set is every source index this rank happens to hold on the
  // destination mesh; compress() then routes each value to its source owner.
  //
  // The reduction is ADD, not INSERT, and that is load-bearing. Every source
  // cell is held by exactly one destination rank, so exactly one rank ever
  // writes a nonzero and summing into a zeroed vector reproduces it exactly.
  // INSERT looks like the natural choice but silently loses data here: the
  // source owner's own entry was set to 0 by the zero-fill, and when the value
  // arrives from a different rank as a ghost contribution the two disagree --
  // the local 0 wins and the indicator comes back empty. That produced a mesh
  // where gamma-coarsening flagged 3404 of 4096 cells on cycle 0 and the run
  // stopped after three cycles.
  // -------------------------------------------------------------------
  template <int dim>
  void
  transfer_to_source(
    const parallel::distributed::Triangulation<dim>      &src_tria,
    const parallel::fullydistributed::Triangulation<dim> &dst_tria,
    const std::vector<types::global_cell_index>          &dst_to_src,
    const Vector<float>                                  &dst_values,
    const MPI_Comm                                        comm,
    Vector<float>                                        &src_values)
  {
    const auto partitioner =
      src_tria.global_active_cell_index_partitioner().lock();
    Assert(partitioner, ExcInternalError());

    const IndexSet &owned = partitioner->locally_owned_range();

    IndexSet ghost(src_tria.n_global_active_cells());
    {
      std::vector<types::global_dof_index> extra;
      for (const auto &cell : dst_tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto s = dst_to_src[cell->active_cell_index()];
            if (s != numbers::invalid_dof_index && !owned.is_element(s))
              extra.push_back(s);
          }
      std::sort(extra.begin(), extra.end());
      extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
      ghost.add_indices(extra.begin(), extra.end());
    }

    LinearAlgebra::distributed::Vector<double> v(
      std::make_shared<const Utilities::MPI::Partitioner>(owned, ghost, comm));
    v = 0.;

    for (const auto &cell : dst_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto s = dst_to_src[cell->active_cell_index()];
          if (s != numbers::invalid_dof_index)
            v[s] = dst_values[cell->active_cell_index()];
        }
    v.compress(VectorOperation::add);

    src_values.reinit(src_tria.n_active_cells());
    for (const auto &cell : src_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        src_values[cell->active_cell_index()] =
          static_cast<float>(v[cell->global_active_cell_index()]);
  }

} // namespace FDRepartition

#endif
