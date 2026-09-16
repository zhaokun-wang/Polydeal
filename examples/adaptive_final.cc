#include <cstdlib>
#include <algorithm>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <boost/geometry.hpp>
#include <deal.II/lac/sparsity_pattern.h>
// METIS is called directly rather than through SparsityTools::partition:
// deal.II only calls METIS_SetDefaultOptions(), leaving METIS_OPTION_CONTIG
// at 0 (parts NOT required to be connected), and it routes nparts <= 8 to
// METIS_PartGraphRecursive, which does not support CONTIG at all.
#include <metis.h>
#include <unordered_map>
#include <adaptive_agglomerator.h>

namespace bgi = boost::geometry::index; // needed at file scope: the class
                                        // declaration (DummyTreeType) and the
                                        // method bodies both use it.

// Generic adaptive-polytopal utilities now live in adaptive_agglomerator.h
// (namespace PolytopalAdaptivity); imported unqualified so the code below
// keeps its original call syntax.
using PolytopalAdaptivity::split_polytope_until_small_enough;
using PolytopalAdaptivity::extract_agglomerates_by_level_retreat;
using PolytopalAdaptivity::split_into_connected_components;
using PolytopalAdaptivity::compute_bulk_threshold;
using PolytopalAdaptivity::compute_count_capped_threshold;
using PolytopalAdaptivity::mark_singleton_cells_for_refinement;

#include <agglomeration_handler.h>
#include <poly_utils.h>
#include <fe_agglodgp.h>
#include <fd_repartition.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/cell_data_transfer.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
// Raw ML parameter list, needed only for the amg_block preconditioner below
// (deal.II's AdditionalData cannot express ML's block-smoother block size).
#include <Teuchos_ParameterList.hpp>
#include <ml_MultiLevelPreconditioner.h>
#include <deal.II/base/index_set.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>
#include <queue>
#include <unordered_set>
#include <set>
#include <optional>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/grid_refinement.h>
#include <numeric>
#include <iomanip>

using namespace dealii;

// Element degree for the SIPG polytopal solve. Kept in one place (mirrors
// adaptive_test.cc / adaptive1.cc): FE_DGQ(dg_degree) for the polygonal
// system, FE_Q of the same degree for the cycle-0 bootstrap. The windowed-
// Gaussian manufactured solution is C-infinity, so raising this raises the
// true convergence rate.
// =====================================================================
// METIS agglomeration -- the one thing that differs from adaptive.cc.
//
// With METIS_OPTION_CONTIG enabled METIS produces polytopes that are always
// face-connected (where R-tree bounding-box grouping needs a repair pass),
// and it balances polytope sizes better. Its cost is per-DoF
// accuracy: FE_AggloDGP's Legendre basis is orthogonal on the BOUNDING BOX,
// so bounding-box-aligned grouping (the R-tree's natural output) fills its
// box better and conditions better.
//
// Error drives the partition through METIS vertex WEIGHTS: METIS balances
// weight per part, so at a fixed k a high-error region ends up with fewer
// cells per polytope than a quiet one.
// =====================================================================

// Partition ONE CONNECTED graph into n_parts connected parts.
// PRECONDITION: the graph is connected -- CONTIG is silently ignored on a
// disconnected graph, which is why callers split first.
inline bool
metis_partition_contiguous(const DynamicSparsityPattern    &dsp,
                           const std::vector<unsigned int> &weights,
                           const unsigned int               n_parts,
                           std::vector<unsigned int>       &partition_indices)
{
    const idx_t n = static_cast<idx_t>(dsp.n_rows());
    if (n_parts < 2 || n < 2)
        return false;

    std::vector<idx_t> xadj, adjncy;
    xadj.reserve(n + 1);
    adjncy.reserve(4 * n);
    xadj.push_back(0);
    for (idx_t row = 0; row < n; ++row) {
        for (auto col = dsp.begin(row); col != dsp.end(row); ++col)
            if (static_cast<idx_t>(col->column()) != row)
                adjncy.push_back(static_cast<idx_t>(col->column()));
        xadj.push_back(static_cast<idx_t>(adjncy.size()));
    }
    if (adjncy.empty())
        return false;

    std::vector<idx_t> vwgt(weights.begin(), weights.end());
    std::vector<idx_t> part(n, 0);
    idx_t ncon = 1, nparts = static_cast<idx_t>(n_parts), objval = 0;

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = 1;

    const int ierr = METIS_PartGraphKway(const_cast<idx_t *>(&n), &ncon,
                                         xadj.data(), adjncy.data(),
                                         vwgt.data(), nullptr, nullptr,
                                         &nparts, nullptr, nullptr, options,
                                         &objval, part.data());
    if (ierr != METIS_OK)
        return false;
    partition_indices.assign(part.begin(), part.end());
    return true;
}



// Global error quantile: the level above which about `hot_fraction` of ALL
// locally-owned cells across ALL ranks sit.
//
// Must be ONE number shared by every rank. Computing it rank-locally makes
// each rank apply a different absolute threshold, which at np=8 collapsed an
// entire rank to singletons and blew the run up. Bisection runs a FIXED
// number of iterations so every rank issues the same collectives in the same
// order; call it exactly once per cycle from a path all ranks reach, never
// from inside a per-group loop.
template <int dim>
double
compute_global_hot_threshold(const Triangulation<dim> &tria,
                             const Vector<float>      &estimated_error,
                             const double              refine_threshold,
                             const double              hot_fraction,
                             const MPI_Comm            mpi_communicator)
{
    std::vector<double> loc_errs;
    loc_errs.reserve(tria.n_active_cells());
    double loc_max = 0.;
    for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned()) {
            const double e = estimated_error[cell->active_cell_index()];
            loc_errs.push_back(e);
            loc_max = std::max(loc_max, e);
        }
    std::sort(loc_errs.begin(), loc_errs.end());

    size_t loc_above = 0;
    if (refine_threshold > 0.)
        loc_above = loc_errs.end() - std::lower_bound(loc_errs.begin(),
                                                      loc_errs.end(),
                                                      refine_threshold);

    const double glob_max   = Utilities::MPI::max(loc_max, mpi_communicator);
    const size_t glob_n     = Utilities::MPI::sum(loc_errs.size(), mpi_communicator);
    const size_t glob_above = Utilities::MPI::sum(loc_above, mpi_communicator);
    if (glob_n == 0)
        return std::numeric_limits<double>::max();

    // Never fewer than the cells tier-2 could mark: each of those has to be a
    // singleton for tier-2 to see it at all.
    size_t target = static_cast<size_t>(std::ceil(hot_fraction * glob_n));
    target = std::max(target, glob_above);
    target = std::min(target, glob_n);
    if (target == 0)      return std::numeric_limits<double>::max();
    if (target >= glob_n) return 0.;

    double lo = 0., hi = glob_max;
    for (unsigned int iter = 0; iter < 50; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const size_t loc_count = loc_errs.end() -
            std::lower_bound(loc_errs.begin(), loc_errs.end(), mid);
        const size_t glob_count = Utilities::MPI::sum(loc_count, mpi_communicator);
        if (glob_count > target) lo = mid;
        else                     hi = mid;
    }
    return hi;
}



// Agglomerate `cells` with METIS.
//
// hot_threshold must come from compute_global_hot_threshold (same value on
// every rank). input_is_connected lets a caller that already knows its input
// forms one face-connected patch (e.g. a surviving polytope) skip the
// connected-component pass.
template <int dim>
std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
extract_agglomerates_by_metis(
    const std::vector<typename Triangulation<dim>::active_cell_iterator> &cells,
    const Vector<float> &estimated_error,
    const double         hot_threshold,
    const size_t         max_cells_per_agglomerate,
    const size_t         target_cells_per_polytope = 16,
    const bool           input_is_connected        = false,
    const unsigned int   recursion_depth           = 0)
{
    using CellIt = typename Triangulation<dim>::active_cell_iterator;
    std::vector<std::vector<CellIt>> result;

    const size_t n = cells.size();
    if (n == 0) return result;
    if (n == 1) { result.push_back(cells); return result; }

    // --- cells that must stand alone -------------------------------------
    // Tier-2 h-refinement can only mark SINGLE-CELL polytopes, so anything it
    // might want to refine has to be a singleton. Two cases:
    //   * error at or above the global hot quantile;
    //   * just created by refinement (user_index == 1) -- its error is
    //     INHERITED, not yet measured on the new mesh, so the hot test cannot
    //     see it, and merging it now would undo the refinement just paid for.
    std::vector<CellIt> cool_cells;
    cool_cells.reserve(n);
    for (const auto &cell : cells) {
        if (estimated_error[cell->active_cell_index()] >= hot_threshold ||
            cell->user_index() == 1)
            result.push_back({cell});
        else
            cool_cells.push_back(cell);
    }
    const size_t n_cool = cool_cells.size();
    if (n_cool == 0) return result;
    if (n_cool == 1) { result.push_back(cool_cells); return result; }

    // --- connected components --------------------------------------------
    // Needed because CONTIG is meaningless on a disconnected graph, and
    // removing the hot cells above generally disconnects things. Skipped when
    // the caller guarantees connectivity AND nothing was removed.
    std::vector<std::vector<CellIt>> components;
    if (input_is_connected && n_cool == n)
        components.push_back(std::move(cool_cells));
    else
        components = split_into_connected_components<dim>({cool_cells});

    for (auto &component : components) {
        const size_t n_c = component.size();
        if (n_c == 0) continue;

        // Fast path: too small to split. No graph, no METIS call. This is the
        // common case once the mesh is refined, and building a graph just to
        // discover k == 1 was a large part of the original cost.
        if (n_c <= target_cells_per_polytope &&
            (max_cells_per_agglomerate == 0 || n_c <= max_cells_per_agglomerate)) {
            result.push_back(std::move(component));
            continue;
        }

        const size_t k_target =
            (target_cells_per_polytope > 0
                 ? static_cast<size_t>(std::ceil(static_cast<double>(n_c) /
                                                 target_cells_per_polytope)) : 1);
        const size_t k_cap =
            (max_cells_per_agglomerate > 0
                 ? static_cast<size_t>(std::ceil(static_cast<double>(n_c) /
                                                 max_cells_per_agglomerate)) : 1);
        size_t k = std::min(std::max<size_t>({k_target, k_cap, 1}), n_c);

        if (k >= n_c) {
            for (const auto &cell : component) result.push_back({cell});
            continue;
        }

        // unordered_map: this lookup runs once per face of every cell.
        std::unordered_map<unsigned int, unsigned int> global_to_local;
        global_to_local.reserve(2 * n_c);
        for (size_t i = 0; i < n_c; ++i)
            global_to_local[component[i]->active_cell_index()] =
                static_cast<unsigned int>(i);

        DynamicSparsityPattern dsp(n_c, n_c);
        for (size_t i = 0; i < n_c; ++i) {
            const auto &cell = component[i];
            for (const auto f : cell->face_indices()) {
                if (cell->at_boundary(f)) continue;
                const auto add_edge = [&](const CellIt &nb) {
                    const auto it = global_to_local.find(nb->active_cell_index());
                    if (it == global_to_local.end()) return;
                    dsp.add(i, it->second);
                    dsp.add(it->second, i); // METIS wants a symmetric graph
                };
                if (cell->face(f)->has_children()) {
                    for (unsigned int sub = 0; sub < cell->face(f)->n_children(); ++sub)
                        add_edge(cell->neighbor_child_on_subface(f, sub));
                } else {
                    const auto nb = cell->neighbor(f);
                    if (nb->is_active()) add_edge(nb);
                }
            }
        }

        constexpr unsigned int weight_scale = 1000;
        std::vector<unsigned int> weights(n_c);
        for (size_t i = 0; i < n_c; ++i) {
            const double err = (hot_threshold > 0.
                ? estimated_error[component[i]->active_cell_index()] / hot_threshold
                : 0.);
            weights[i] = static_cast<unsigned int>(std::min<double>(
                std::max<double>(std::round(err * weight_scale), 1.),
                10. * weight_scale));
        }

        std::vector<std::vector<CellIt>> groups;
        std::vector<unsigned int>        part_idx;
        if (metis_partition_contiguous(dsp, weights,
                                       static_cast<unsigned int>(k), part_idx)) {
            // CONTIG guarantees each part is connected, so no repair pass.
            std::vector<std::vector<CellIt>> by_part(k);
            for (size_t i = 0; i < n_c; ++i)
                if (part_idx[i] < k) by_part[part_idx[i]].push_back(component[i]);
            for (auto &pt : by_part)
                if (!pt.empty()) groups.push_back(std::move(pt));
        } else {
            // METIS declined. Keeping the component whole is safe; the size
            // cap below still applies. Connectivity holds because `component`
            // came from a connected-component split.
            groups.push_back(std::move(component));
        }

        for (auto &group : groups) {
            if (max_cells_per_agglomerate > 0 &&
                group.size() > max_cells_per_agglomerate) {
                if (recursion_depth >= 4 || group.size() == n_c) {
                    for (const auto &cell : group) result.push_back({cell});
                } else {
                    auto pieces = extract_agglomerates_by_metis<dim>(
                        group, estimated_error, hot_threshold,
                        max_cells_per_agglomerate, target_cells_per_polytope,
                        true /*a METIS part is connected*/, recursion_depth + 1);
                    result.insert(result.end(), pieces.begin(), pieces.end());
                }
            } else if (!group.empty()) {
                result.push_back(std::move(group));
            }
        }
    }
    return result;
}


// Agglomerator selection: env AGGLOMERATOR = rtree (default) | metis.
// The two paths are configuration, not instrumentation -- they build the
// polytopes differently and diverge structurally in prepare_cycle's cycle>0
// branch (METIS carries a stability gate and batches debris; the R-tree path
// groups debris by inherited material_id), so both branches are kept in full
// rather than reduced to a swapped function call.
// TEMPORARY A/B switch for the estimator/marking role separation (see
// mark_threshold_refine). Default off = shipped behaviour, bit-for-bit.
inline bool separate_marking_pool()
{
    static const bool v = []{
        const char *e = std::getenv("MARK_POOL");
        return e != nullptr && std::string(e) == "singleton";
    }();
    return v;
}

inline bool use_metis_agglomerator()
{
  static const bool v = []{
      const char *e = std::getenv("AGGLOMERATOR");
      return e != nullptr && std::string(e) == "metis";
  }();
  return v;
}

// AGGLOMERATOR=none turns agglomeration OFF: every locally owned background
// cell becomes its own polytope, on every cycle. This is the pure DG-AMR
// setting the a posteriori literature works in (Karakashian-Pascal,
// Houston-Schoetzau-Wihler), and it exists as the control for the effectivity
// study -- it isolates whether the I_eff plateau has anything to do with
// agglomeration at all. It OVERRIDES whatever STEP 2 built rather than
// short-circuiting it, so the carryover/marking machinery stays on its normal
// code path and the only variable is the polytope layout.
inline bool no_agglomeration()
{
  static const bool v = []{
      const char *e = std::getenv("AGGLOMERATOR");
      return e != nullptr && std::string(e) == "none";
  }();
  return v;
}

constexpr unsigned int dg_degree = 4;

// =====================================================================
// Problem definitions.
//
// Two test problems are provided:
//  - GaussianRHS: a right-hand side with two localized peaks. Has no known
//    closed-form solution, so it's good for exercising AMR on a genuinely
//    non-trivial error distribution, but no exact-error norms can be
//    computed for it (only the residual-based indicator is available).
//  - SinSinRHS (+ SinSinSolution): the classic manufactured solution for
//    -Delta u = f on [0,1]^2 with homogeneous Dirichlet BCs, f = sin(pi x)
//    sin(pi y). Since the exact solution is known, this is the problem to
//    use when you want real L2/H1 error norms via PolyUtils::compute_global_error
//    (e.g. to check convergence rates or sanity-check the parallel assembly).
//
// Switch between them from the single toggle in main() (search for
// "PROBLEM SELECTION").
// =====================================================================

template <int dim>
class GaussianRHS : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    // peak1: (0.25, 0.25)
    double r1_sq = p.distance_square(Point<dim>(0.25, 0.25));
    double peak1 = 1000.0 * std::exp(-100.0 * r1_sq);

    // peak2: (0.75, 0.75)
    double r2_sq = p.distance_square(Point<dim>(0.75, 0.75));
    double peak2 = 1000.0 * std::exp(-100.0 * r2_sq);

    return peak1 + peak2;
  }
};

template <int dim>
class SinSinRHS : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};

// Exact solution of -Delta u = sin(pi x) sin(pi y) on [0,1]^2 with u=0 on
// the boundary. Provided together with its gradient so that
// PolyUtils::compute_global_error can compute both the L2 error and the
// H1 seminorm error.
template <int dim>
class SinSinSolution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) /
           (2.0 * numbers::PI * numbers::PI);
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override
  {
    Tensor<1, dim> grad;
    grad[0] = std::cos(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) / (2.0 * numbers::PI);
    grad[1] = std::sin(numbers::PI * p[0]) * std::cos(numbers::PI * p[1]) / (2.0 * numbers::PI);
    return grad;
  }
};

// Windowed-Gaussian manufactured solution (shared design across all four
// adaptive drivers). See the eval() comment below for the formula and why
// it replaced the earlier C^2 smoothstep-cutoff bump.
namespace GaussianPeakedDetail
{
  // Gaussian decay rate. Peak width ~ 1/sqrt(k): k=100 -> 0.10, k=400 -> 0.05,
  // k=1600 -> 0.025, which is only ~1.6 cells wide on the 64x64 background mesh
  // -- the peak is then essentially unresolved at cycle 0, the error mass
  // collapses onto a handful of cells, Doerfler's L rises to meet the maximum
  // cell indicator, and the tier-2 hysteresis bar can no longer be cleared, so
  // the run stops after two quiet cycles.
  //
  // env K overrides it, so a parameter sweep cannot leave a stale value baked
  // into the source.
  inline double k_rate()
  {
    static const double v =
      std::getenv("K") ? std::atof(std::getenv("K")) : 400.0;
    return v;
  }
  constexpr double A  = 5.0;   // amplitude; u_peak = A*sin(pi/4)^2 ~= 2.5,
                                // matching the old bump's peak value so the
                                // indicator's error mass / marking scale stay
                                // comparable to prior runs

  inline const std::vector<Point<2>> &peaks()
  {
    static const std::vector<Point<2>> p = {Point<2>(0.25, 0.25), Point<2>(0.75, 0.75)};
    return p;
  }

  // Evaluate u, grad u, and Delta u at p for the smooth WINDOWED Gaussian
  //   u = A * s(x,y) * G(x,y),   s = sin(pi x) sin(pi y),   G = sum_i exp(-k r_i^2).
  // The sin window is C^inf and vanishes EXACTLY on the whole boundary of the
  // unit square, enforcing the homogeneous Dirichlet BC with NO piecewise
  // cutoff/junction -- unlike the previous quintic-smoothstep window, whose
  // r=R1/R2 junctions were only C^2 (u''' jumps), capping degree>=2 L2
  // convergence at O(h^3) and seeding non-monotone true-error jumps. The
  // Gaussian factor still localizes each feature to its peak (adaptive stays
  // meaningful). Closed form via the product rule, C^inf everywhere:
  //   Delta u = A ( Delta s * G + 2 grad s . grad G + s * Delta G ),
  //   Delta s = -2 pi^2 s,  grad g_i = -2k (p-c_i) g_i,
  //   Delta g_i = (4 k^2 r_i^2 - 4 k) g_i.
  template <int dim>
  inline void eval(const Point<dim> &p, double &u, Tensor<1, dim> &grad, double &lap)
  {
    const double pi = numbers::PI;
    const double k = k_rate();
    const double x = p[0], y = p[1];
    const double sinx = std::sin(pi * x), cosx = std::cos(pi * x);
    const double siny = std::sin(pi * y), cosy = std::cos(pi * y);
    const double s     = sinx * siny;
    const double s_x   = pi * cosx * siny;
    const double s_y   = pi * sinx * cosy;
    const double lap_s = -2.0 * pi * pi * s;

    double G = 0.0, G_x = 0.0, G_y = 0.0, lap_G = 0.0;
    for (const auto &c : peaks())
    {
      const double dx = x - c[0], dy = y - c[1];
      const double rho2 = dx * dx + dy * dy;
      const double g = std::exp(-k * rho2);
      G     += g;
      G_x   += -2.0 * k * dx * g;
      G_y   += -2.0 * k * dy * g;
      lap_G += (4.0 * k * k * rho2 - 4.0 * k) * g;
    }

    u = A * s * G;
    grad[0] = A * (s_x * G + s * G_x);
    grad[1] = A * (s_y * G + s * G_y);
    lap = A * (lap_s * G + 2.0 * (s_x * G_x + s_y * G_y) + s * lap_G);
  }
} // namespace GaussianPeakedDetail

template <int dim>
class GaussianPeakedSolution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    double u, lap; Tensor<1, dim> grad;
    GaussianPeakedDetail::eval(p, u, grad, lap);
    return u;
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override
  {
    double u, lap; Tensor<1, dim> grad;
    GaussianPeakedDetail::eval(p, u, grad, lap);
    return grad;
  }

private:
};

// Right-hand side f = -Delta u for GaussianPeakedSolution above. See
// GaussianPeakedDetail's comment for the derivation.
template <int dim>
class GaussianPeakedRHS : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    double u, lap; Tensor<1, dim> grad;
    GaussianPeakedDetail::eval(p, u, grad, lap);
    return -lap; // f = -Delta u
  }
};

// ---------------------------------------------------------------------------
// LOW-REGULARITY manufactured solution -- an algebraic point singularity, the
// h-refinement-favoring counterpart to the C^inf windowed Gaussian above.
//
//   u = sin(a (x-x0)) sin(b (y-y0)) * r^alpha,  r = |(x,y) - c|, alpha = 2/3,
//
// where the sine window is stretched to span exactly one half-period across
// the background domain (see window() below) and c is a vertex of the
// background mesh (see center()).
//
// The window is C^inf and vanishes EXACTLY on the whole boundary of that
// domain, so the homogeneous Dirichlet BC holds with no junction -- the same
// trick as GaussianPeakedDetail, so make_grid() and the SIPG assembly are
// reused UNCHANGED. That exactness matters here: the boundary branch of
// solve_polygonal_system imposes u=0 weakly and has no non-homogeneous
// Nitsche right-hand-side term, so a window that failed to vanish on the
// boundary would make the reported L2/H1 "error" mostly BC mismatch.
// alpha = 2/3 is the
// exponent of the classic L-shaped re-entrant-corner singularity, realized as
// an INTERIOR point singularity on the existing square domain: u in
// H^{1+2/3-eps} = H^{5/3-eps}, in H^1 but NOT H^2, so degree>=2 convergence is
// capped by the singularity (p-refinement STALLS) and only local h-refinement
// clustered at the singular point drives the error down. f = -Delta u is
// singular ~ r^{-4/3} at the centre (L^1, not L^2) but never sampled at r=0
// (singularity sits on a coarse-mesh vertex, Gauss points are cell-interior;
// eval() guards r=0 anyway). Closed form via the product rule
// (Delta(r^a) = a^2 r^{a-2} in 2D, grad(r^a) = a r^{a-2} (p-c)).
namespace CornerSingularityDetail
{
  constexpr double alpha = 2.0 / 3.0; // r^alpha singularity exponent

  // Half-period sine window fitted to the background domain: the unit square
  // by default, and the rectangle [-1.25,1.25]x[-0.5,1.25] spanned by
  // meshes/t2.msh under USE_T2. On the unit square this is literally
  // sin(pi x) sin(pi y), so the structured-mesh runs are unaffected.
  struct Window
  {
    double x0, y0, ax, ay;
  };

  inline const Window &window()
  {
    static const Window w =
      (std::getenv("USE_T2") != nullptr)
        ? Window{-1.25, -0.5, numbers::PI / 2.5, numbers::PI / 1.75}
        : Window{0.0, 0.0, numbers::PI, numbers::PI};
    return w;
  }

  // The singular point must coincide with a background-mesh VERTEX. f ~
  // r^{-4/3} is then never sampled near r=0 on any refinement level, and the
  // cells around it see the singularity symmetrically. (0.5,0.5) is a vertex
  // of the structured unit square at every refinement level; under USE_T2 the
  // value below is the interior vertex of t2.msh closest to that mesh's
  // centre, copied verbatim from the .msh file so the match is exact.
  inline const Point<2> &center()
  {
    static const Point<2> c =
      (std::getenv("USE_T2") != nullptr)
        ? Point<2>(0.1054517676749012, 0.3907521701589055)
        : Point<2>(0.5, 0.5);
    return c;
  }

  template <int dim>
  inline void eval(const Point<dim> &p, double &u, Tensor<1, dim> &grad, double &lap)
  {
    const auto  &w = window();
    const double x = p[0], y = p[1];
    const double X = w.ax * (x - w.x0), Y = w.ay * (y - w.y0);
    const double sinx = std::sin(X), cosx = std::cos(X);
    const double siny = std::sin(Y), cosy = std::cos(Y);
    const double s     = sinx * siny;
    const double s_x   = w.ax * cosx * siny;
    const double s_y   = w.ay * sinx * cosy;
    const double lap_s = -(w.ax * w.ax + w.ay * w.ay) * s;

    const double dx = x - center()[0], dy = y - center()[1];
    const double r2 = dx * dx + dy * dy;

    // Defensive guard: r=0 is a coarse-mesh vertex, never a quadrature point.
    if (r2 < 1e-300)
      {
        u = 0.0;
        grad[0] = 0.0;
        grad[1] = 0.0;
        lap = 0.0;
        return;
      }

    const double r     = std::sqrt(r2);
    const double R     = std::pow(r, alpha);       // r^alpha
    const double R_am2 = std::pow(r, alpha - 2.0); // r^{alpha-2}

    u = s * R;
    grad[0] = s_x * R + s * alpha * R_am2 * dx;
    grad[1] = s_y * R + s * alpha * R_am2 * dy;
    lap = lap_s * R
          + 2.0 * alpha * R_am2 * (s_x * dx + s_y * dy)
          + s * alpha * alpha * R_am2;
  }
} // namespace CornerSingularityDetail

template <int dim>
class CornerSingularitySolution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    double u, lap; Tensor<1, dim> grad;
    CornerSingularityDetail::eval(p, u, grad, lap);
    return u;
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override
  {
    double u, lap; Tensor<1, dim> grad;
    CornerSingularityDetail::eval(p, u, grad, lap);
    return grad;
  }
};

template <int dim>
class CornerSingularityRHS : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    double u, lap; Tensor<1, dim> grad;
    CornerSingularityDetail::eval(p, u, grad, lap);
    return -lap; // f = -Delta u
  }
};

// Reads back the polytope id a fine background cell belongs to (encoded in
// its material_id by CellsAgglomerator::assign_material_id).
template <int dim>
inline int get_polytope_id(const typename Triangulation<dim>::active_cell_iterator &cell)
{
    return static_cast<int>(cell->material_id());
}

// Draws the fine background mesh (light gray, thin) and the polytope
// boundaries (dark blue, thicker) in one picture -- much more direct than
// trying to pick a discrete colormap for material_id: you see both the
// underlying fine mesh AND the agglomeration boundaries at a glance,
// without having to visually disambiguate which color belongs to which
// polytope.
// MPI-aware version of export_merged_svg: each rank only ever sees its own
// locally-owned (+ ghost) slice of a distributed triangulation, so calling
// export_merged_svg() directly on each rank only ever draws that rank's
// local piece -- to see the WHOLE domain in one picture, every rank's
// lines need to be gathered onto one rank (here, rank 0) and written into
// a single file. Restricted to is_locally_owned() cells/polytopes so the
// shared geometry at partition boundaries doesn't get drawn twice by two
// neighboring ranks. Same visual style as export_merged_svg (light gray
// fine mesh, dark blue bolder polytope boundaries) -- just assembled from
// every rank instead of just one.
template <int dim>
void export_merged_svg_global(
    const Triangulation<dim>        &tria,
    const AgglomerationHandler<dim> &agglomeration_handler,
    const std::string               &filename,
    MPI_Comm                         mpi_communicator)
{
    static_assert(dim == 2, "Merged SVG export is only supported for 2D.");

    // Global bounding box (reduced across all ranks) so the viewBox covers
    // the WHOLE domain, not just whichever rank happens to compute it.
    double local_min_x = 1e10, local_max_x = -1e10;
    double local_min_y = 1e10, local_max_y = -1e10;
    for (const auto &cell : tria.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;
        for (const auto v : cell->vertex_indices()) {
            auto p = cell->vertex(v);
            local_min_x = std::min(local_min_x, p[0]);
            local_max_x = std::max(local_max_x, p[0]);
            local_min_y = std::min(local_min_y, p[1]);
            local_max_y = std::max(local_max_y, p[1]);
        }
    }
    const double min_x = Utilities::MPI::min(local_min_x, mpi_communicator);
    const double max_x = Utilities::MPI::max(local_max_x, mpi_communicator);
    const double min_y = Utilities::MPI::min(local_min_y, mpi_communicator);
    const double max_y = Utilities::MPI::max(local_max_y, mpi_communicator);

    // Every rank builds its own fine-mesh and polytope-boundary line
    // strings, restricted to what it locally owns.
    std::ostringstream fine_lines;
    for (const auto &cell : tria.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        for (const auto f : cell->face_indices())
        {
            auto p1 = cell->face(f)->vertex(0);
            auto p2 = cell->face(f)->vertex(1);
            fine_lines << "    <line x1=\"" << p1[0] << "\" y1=\"" << p1[1]
                       << "\" x2=\"" << p2[0] << "\" y2=\"" << p2[1] << "\" />\n";
        }
    }

    // Polytope-boundary lines: reconstructed directly from the fine
    // Triangulation's own cell/face/neighbor structure and material_id
    // (which assign_material_id(), called at the end of STEP 3's
    // correction loop, keeps in sync with whatever polytope grouping was
    // actually used to solve this cycle) -- NOT via
    // polytope->polytope_boundary(). That AgglomerationHandler-internal
    // method is suspected to have gaps specifically at faces bordering a
    // GHOST polytope (the same general class of "ghost polytope handling
    // is incomplete" issue already known elsewhere in this codebase, e.g.
    // setup_ghost_polytopes() on hanging-node neighbors). material_id, by
    // contrast, is an ordinary per-cell Triangulation property that deal.II
    // already keeps automatically synchronized to ghost cells across MPI
    // boundaries as part of standard ghost-cell-data exchange -- so
    // comparing it directly, even across a partition boundary, doesn't
    // depend on any agglomeration-specific ghost bookkeeping at all.
    //
    // Rule: a fine cell's face is a polytope-boundary edge if it's on the
    // actual domain boundary, OR its neighbor (across that face) has a
    // DIFFERENT material_id -- i.e. belongs to a different polytope.
    // Faces where the neighbor is more refined (hanging node) are handled
    // by checking each of the neighbor's children touching this face
    // individually, since different children can belong to different
    // polytopes.
    std::ostringstream coarse_lines;
    for (const auto &cell : tria.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (const auto f : cell->face_indices())
        {
            if (cell->at_boundary(f))
            {
                auto p1 = cell->face(f)->vertex(0);
                auto p2 = cell->face(f)->vertex(1);
                coarse_lines << "    <line x1=\"" << p1[0] << "\" y1=\"" << p1[1]
                             << "\" x2=\"" << p2[0] << "\" y2=\"" << p2[1] << "\" />\n";
                continue;
            }

            if (cell->neighbor(f)->has_children())
            {
                // Hanging node: the neighbor is more refined than this
                // cell -- check each child touching this face separately,
                // since they might belong to different polytopes from one
                // another (and from this cell).
                for (unsigned int sub = 0; sub < cell->face(f)->n_children(); ++sub)
                {
                    const auto neighbor_child = cell->neighbor_child_on_subface(f, sub);
                    if (neighbor_child->material_id() != cell->material_id())
                    {
                        auto p1 = cell->face(f)->child(sub)->vertex(0);
                        auto p2 = cell->face(f)->child(sub)->vertex(1);
                        coarse_lines << "    <line x1=\"" << p1[0] << "\" y1=\"" << p1[1]
                                     << "\" x2=\"" << p2[0] << "\" y2=\"" << p2[1] << "\" />\n";
                    }
                }
            }
            else
            {
                if (cell->neighbor(f)->material_id() != cell->material_id())
                {
                    auto p1 = cell->face(f)->vertex(0);
                    auto p2 = cell->face(f)->vertex(1);
                    coarse_lines << "    <line x1=\"" << p1[0] << "\" y1=\"" << p1[1]
                                 << "\" x2=\"" << p2[0] << "\" y2=\"" << p2[1] << "\" />\n";
                }
            }
        }
    }

    // Gather every rank's line strings onto rank 0 -- this is a collective,
    // so every rank must call it (only rank 0's return value is actually
    // used below).
    const std::vector<std::string> all_fine_lines =
        Utilities::MPI::gather(mpi_communicator, fine_lines.str(), 0);
    const std::vector<std::string> all_coarse_lines =
        Utilities::MPI::gather(mpi_communicator, coarse_lines.str(), 0);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
        return; // only rank 0 writes the merged file

    const double w = max_x - min_x;
    const double h = max_y - min_y;
    const double pad_x = w * 0.05;
    const double pad_y = h * 0.05;
    const double svg_min_x = min_x - pad_x;
    const double svg_min_y = -max_y - pad_y;
    const double svg_w = w + 2 * pad_x;
    const double svg_h = h + 2 * pad_y;
    const double fine_stroke = std::max(w, h) * 0.0005;
    const double coarse_stroke = std::max(w, h) * 0.001;

    std::ofstream myfile(filename + ".svg");
    myfile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    myfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\""
           << svg_min_x << " " << svg_min_y << " " << svg_w << " " << svg_h << "\" "
           << "width=\"800\" height=\"800\" style=\"background-color: white;\">\n";
    myfile << "<g transform=\"scale(1, -1)\">\n";
    myfile << "  <g stroke=\"lightgray\" stroke-width=\"" << fine_stroke << "\" fill=\"none\">\n";
    for (const auto &s : all_fine_lines)
        myfile << s;
    myfile << "  </g>\n";
    myfile << "  <g stroke=\"#003366\" stroke-width=\"" << coarse_stroke << "\" stroke-linecap=\"round\" fill=\"none\">\n";
    for (const auto &s : all_coarse_lines)
        myfile << s;
    myfile << "  </g>\n";
    myfile << "</g>\n</svg>\n";
    myfile.close();
}

// Correct ||u_h||_{L2(Omega)}, i.e. sqrt(sum_K integral_K u_h^2 dx), computed
// via the same quadrature already set up on `ah`. This is NOT the same
// thing as poly_solution.l2_norm() (the Euclidean norm of the raw DoF
// coefficient vector): with N growing DoFs and O(1) coefficient values,
// the raw vector norm grows like sqrt(N) purely from summing more terms,
// completely independent of whether the underlying function u_h is
// actually converging. The properly quadrature-weighted integral below is
// the physically meaningful quantity -- it should stay bounded (converge)
// under refinement if the true solution is well-behaved, unlike the raw
// vector norm.
template <int dim>
double compute_function_l2_norm(AgglomerationHandler<dim> &ah,
                                 const TrilinosWrappers::MPI::Vector &poly_solution,
                                 MPI_Comm mpi_communicator)
{
    double local_norm_sq = 0.0;
    const unsigned int dofs_per_cell = ah.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    std::vector<double> local_coeffs(dofs_per_cell);

    for (const auto &polytope : ah.polytope_iterators())
    {
        if (!polytope->is_locally_owned())
            continue;

        const auto &agglo_values = ah.reinit(polytope);
        polytope->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            local_coeffs[i] = poly_solution(dof_indices[i]);

        for (unsigned int q_index : agglo_values.quadrature_point_indices())
        {
            double u_at_q = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                u_at_q += local_coeffs[i] * agglo_values.shape_value(i, q_index);
            local_norm_sq += u_at_q * u_at_q * agglo_values.JxW(q_index);
        }
    }

    const double global_norm_sq = Utilities::MPI::sum(local_norm_sq, mpi_communicator);
    return std::sqrt(global_norm_sq);
}

// =====================================================================
// STEP 4 helper: residual-based error indicator on the polytopes.
//
// Standard IP-DG a posteriori residual estimator, three terms per
// polytope K:
//
//   eta_K^2 = h_K^2 * ||f + Delta(u_h)||^2_{L2(K)}
//           + sum_{e subset dK, interior} h_e * ||[grad(u_h).n]||^2_{L2(e)}
//           + sum_{e subset dK}           (1/h_e) * ||[u_h]||^2_{L2(e)}
//
// (the last sum runs over ALL faces of K, including boundary faces,
// where the Dirichlet BC u=0 makes the "jump" just u_h itself).
//
// Since FE_DGQ(1) is piecewise LINEAR, Delta(u_h) is identically 0 inside
// every polytope, so the first term's "f + Delta(u_h)" reduces to just f.
//
// A shared interior face is only visited ONCE (from the lower-id() side,
// same convention as the assembly in solve_polygonal_system), but its
// jump energy is added to BOTH neighboring polytopes' totals -- a shared
// face's jump belongs to both adjacent elements, not just the one whose
// turn it was to compute it.
// =====================================================================
// =====================================================================
// Cangiani-Dong-Georgoulis (SINUM 61(5) 2023) estimator, env ESTIMATOR=cdg.
//
// The classic estimator weights a polytope's jump terms by the POLYTOPE
// DIAMETER. On an agglomerated mesh one polytopal face is a POLYLINE of many
// background-cell faces, so that diameter is a poor stand-in for the edge
// measure actually multiplying each jump. The CDG estimator instead treats
// every background face as its own interface with its own |e_i|.
//
// Measured on t2.msh at cycle 0 (the fully agglomerated state), effectivity
// index I_eff = eta_total/||e||_H1, background refine_global level 2..5:
//
//   level   dofs    classic   cdg
//     2     1000     7.836   5.700
//     3     3250     7.918   5.910
//     4    10500     6.402   4.346
//     5    39825     5.461   2.762   <- the paper reports 1.2-2.6
//
// The advantage GROWS with resolution (-27% -> -49%) because finer background
// meshes put more background faces on each polytopal face, which is exactly
// what the polytope-diameter weight cannot see. Term by term at level 5:
// the per-face measure carries the whole effect (5.461 -> 2.725), the
// face-shared min(h_P,h_Q) sigma is numerically a no-op here (the value-jump
// term is only ~0.7% of eta on this problem) and the tangential term costs
// 1.4% of I_eff. Both are kept anyway: they are what make the estimator a
// provable upper bound, and neither is free to drop from the theory.
//
// NOTE this only bites while the polytopes still HOLD error. In an
// error-driven adaptive run the agglomerator correctly sheds polytopes
// wherever the error concentrates, so from cycle ~2 on the two estimators
// agree to within 3% and are equivalent at equal accuracy.
// =====================================================================
inline bool use_cdg_estimator()
{
  static const bool v = []{
      const char *e = std::getenv("ESTIMATOR");
      return e != nullptr && std::string(e) == "cdg";
  }();
  return v;
}

// The background faces making up one polytopal interface, as vertex pairs.
// No library change is needed: ah.get_interface() is public, and the library
// canonically sorts each side's quadrature points by physical coordinate, so
// both sides of an interface land on the same partition and compute the same
// per-sub-edge |e_i| by construction rather than by convention.
// Hanging-node faces are expanded to their children, the granularity the
// finer side sees.
template <int dim>
std::vector<std::pair<Point<dim>, Point<dim>>>
collect_interface_segments(const AgglomerationHandler<dim> &ah,
                           const CellId &id_in,
                           const CellId &id_out)
{
    static_assert(dim == 2, "The CDG estimator assumes 2D faces are segments.");
    std::vector<std::pair<Point<dim>, Point<dim>>> segments;

    const auto &iface = ah.get_interface();
    const auto  it    = iface.find({id_in, id_out});
    if (it == iface.end())
        return segments;

    for (const auto &[deal_cell, local_face_idx] : it->second)
    {
        const auto face = deal_cell->face(local_face_idx);
        if (face->has_children())
            for (unsigned int c = 0; c < face->n_children(); ++c)
                segments.emplace_back(face->child(c)->vertex(0),
                                      face->child(c)->vertex(1));
        else
            segments.emplace_back(face->vertex(0), face->vertex(1));
    }
    return segments;
}

// Index of the background segment a quadrature point sits on. The point lies
// ON one of them up to roundoff (verified: max deviation 2.2e-16 over three
// problems x 20 cycles), so nearest-segment is exact in practice.
template <int dim>
unsigned int
segment_of_point(const Point<dim> &p,
                 const std::vector<std::pair<Point<dim>, Point<dim>>> &segments)
{
    unsigned int best      = 0;
    double       best_dist = std::numeric_limits<double>::max();
    for (unsigned int s = 0; s < segments.size(); ++s)
    {
        const Point<dim> &a  = segments[s].first;
        const Point<dim> &b  = segments[s].second;
        const Tensor<1, dim> ab = b - a;
        const Tensor<1, dim> ap = p - a;
        const double len_sq = ab * ab;
        double t = (len_sq > 0.0) ? ((ap * ab) / len_sq) : 0.0;
        t = std::max(0.0, std::min(1.0, t));
        const Point<dim> proj = a + t * ab;
        const double d = p.distance(proj);
        if (d < best_dist) { best_dist = d; best = s; }
    }
    return best;
}

// =====================================================================
// Jump part of the DG ENERGY NORM of the true error.
//
//   |||e|||^2 = sum_K |grad(u - u_h)|^2_K  +  sum_F sigma_F ||[u - u_h]||^2_F
//
// PolyUtils::compute_global_error returns only the first sum (the BROKEN H1
// seminorm: it integrates over polytope interiors and adds no face term). For
// a DG discretisation the jump IS part of the error -- two solutions with the
// same interior gradients but different jumps are not equally good -- and the
// reliability bound the estimator is built on is stated in |||.|||, not in the
// broken seminorm. Reporting the ratio eta/|e|_{H1} instead of eta/|||e|||
// divides by something strictly smaller and so overstates the estimator's
// error by a factor no theorem describes.
//
// This function supplies the missing second sum, and it is nearly free:
//
//   * u is the EXACT solution, hence in H^1, hence [u] = 0 on every interior
//     face -- so [u - u_h] = -[u_h], which is the same jump the assembly
//     already integrates;
//   * the manufactured solutions here all carry a window vanishing on the
//     whole boundary, so u = 0 there and (u - u_h) = -u_h on boundary faces.
//
// sigma_F is taken from the ASSEMBLY (penalty_constant / min(h_P,h_Q), and
// the polytope's own diameter on the boundary), not from the estimator: the
// energy norm has to be the one the bilinear form is coercive in.
//
// Each face is counted ONCE globally -- from the lower id() side, ghost faces
// included, since the other rank visits the same face with the roles swapped
// and its own id then compares as the larger one.
// =====================================================================
template <int dim>
double compute_dg_jump_seminorm_sq(const AgglomerationHandler<dim> &ah,
                                   const TrilinosWrappers::MPI::Vector &poly_solution,
                                   const double penalty_constant,
                                   // Same role as in the estimator: this is the
                                   // jump seminorm of the ERROR u - u_h. The
                                   // exact solution is continuous, so interior
                                   // jumps reduce to [u_h]; on the boundary
                                   // u = g_D, so the jump is (u_h - g_D), not
                                   // u_h. nullptr means g_D == 0.
                                   const Function<dim> *dirichlet_data = nullptr)
{
    // A ghost neighbour's coefficients are NOT readable from the completely
    // distributed solution vector the solver produces -- doing so silently
    // returns garbage, which showed up as a DG energy norm of 74 against an
    // H1 seminorm of 2.7e-4 at np=4 (np=1 and np=2 happened to be clean).
    // Same remedy as the error indicator's caller: make every DOF ghost
    // visible. Cheap here because this is a POLYTOPE-level vector -- a handful
    // of DOFs per polytope, not per fine cell -- so a full copy on every rank
    // costs nothing regardless of mesh resolution.
    const IndexSet locally_owned = ah.agglo_dh.locally_owned_dofs();
    IndexSet all_dofs(ah.agglo_dh.n_dofs());
    all_dofs.add_range(0, ah.agglo_dh.n_dofs());
    TrilinosWrappers::MPI::Vector solution_ghosted(locally_owned, all_dofs, MPI_COMM_WORLD);
    solution_ghosted = poly_solution;

    const unsigned int dofs_per_cell = ah.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices0(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices1(dofs_per_cell);
    std::vector<double> coeffs0(dofs_per_cell), coeffs1(dofs_per_cell);

    double local_sum = 0.0;

    for (const auto &polytope : ah.polytope_iterators())
    {
        if (!polytope->is_locally_owned())
            continue;

        polytope->get_dof_indices(dof_indices0);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            coeffs0[i] = solution_ghosted(dof_indices0[i]);

        const unsigned int n_faces = polytope->n_faces();
        for (unsigned int f = 0; f < n_faces; ++f)
        {
            if (polytope->at_boundary(f))
            {
                const auto &fe_face = ah.reinit(polytope, f);
                const double sigma = penalty_constant / polytope->diameter();
                const auto &bdry_qpts = fe_face.get_quadrature_points();
                for (unsigned int q : fe_face.quadrature_point_indices())
                {
                    double v = 0.0;
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        v += coeffs0[i] * fe_face.shape_value(i, q);
                    if (dirichlet_data != nullptr)
                        v -= dirichlet_data->value(bdry_qpts[q]);
                    local_sum += sigma * v * v * fe_face.JxW(q);
                }
                continue;
            }

            const auto &neigh_polytope = polytope->neighbor(f);
            // Count each interface exactly once, from the lower-id side.
            if (!(polytope->id() < neigh_polytope->id()))
                continue;

            const unsigned int nofn = polytope->neighbor_of_agglomerated_neighbor(f);
            const auto &fe_faces  = ah.reinit_interface(polytope, neigh_polytope, f, nofn);
            const auto &fe_faces0 = fe_faces.first;

            neigh_polytope->get_dof_indices(dof_indices1);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                coeffs1[i] = solution_ghosted(dof_indices1[i]);

            if (!neigh_polytope->is_locally_owned())
            {
                // diameter() is unsafe on a ghost polytope; the bounding-box
                // diagonal is what the assembly uses there too.
                const auto ghost_bp = neigh_polytope->get_bounding_box().get_boundary_points();
                const double neigh_diameter = (ghost_bp.second - ghost_bp.first).norm();
                const double sigma =
                    penalty_constant / std::min(polytope->diameter(), neigh_diameter);

                const types::subdomain_id neigh_rank = neigh_polytope->subdomain_id();
                const auto &recv_values = ah.recv_values.at(neigh_rank).at({neigh_polytope->id(), nofn});
                const auto &recv_jxws   = ah.recv_jxws.at(neigh_rank).at({neigh_polytope->id(), nofn});

                for (unsigned int q : fe_faces0.quadrature_point_indices())
                {
                    double v0 = 0.0, v1 = 0.0;
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        v0 += coeffs0[i] * fe_faces0.shape_value(i, q);
                        v1 += coeffs1[i] * recv_values[i][q];
                    }
                    const double jump = v0 - v1;
                    local_sum += sigma * jump * jump * recv_jxws[q];
                }
                continue;
            }

            const auto &fe_faces1 = fe_faces.second;
            const double sigma =
                penalty_constant / std::min(polytope->diameter(), neigh_polytope->diameter());
            for (unsigned int q : fe_faces0.quadrature_point_indices())
            {
                double v0 = 0.0, v1 = 0.0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    v0 += coeffs0[i] * fe_faces0.shape_value(i, q);
                    v1 += coeffs1[i] * fe_faces1.shape_value(i, q);
                }
                const double jump = v0 - v1;
                local_sum += sigma * jump * jump * fe_faces0.JxW(q);
            }
        }
    }

    return Utilities::MPI::sum(local_sum, MPI_COMM_WORLD);
}

template <int dim>
void compute_polygonal_error_indicator(
    const Triangulation<dim> &tria,
    AgglomerationHandler<dim> &ah,
    FE_AggloDGP<dim> &dg_fe,
    const Function<dim> &rhs_function,
    const TrilinosWrappers::MPI::Vector &poly_solution,
    const std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> &final_agglomerates,
    Vector<float> &estimated_error,
    // Dirichlet datum g_D for the boundary terms. Thm 21 weighs
    // ||sigma^{1/2}(u_h - g_D)|| and ||sqrt(h_K) grad_T(u_h - g_D)||, NOT
    // ||u_h||: on a boundary where g_D is O(1) the value-jump term measures
    // ||g_D||^2/h_e, which does NOT decay under refinement and makes the
    // effectivity index diverge (measured elsewhere: 3.25 / 72.4 / 831 /
    // 31776 across four levels, R_J holding 99.94% of eta).
    // nullptr means g_D == 0, which is exactly what THIS driver imposes:
    // solve_polygonal_system has no non-homogeneous Nitsche RHS term, so u = 0
    // is what is weakly enforced. Passing the manufactured solution here
    // instead would make the estimator inconsistent with the solve.
    const Function<dim> *dirichlet_data = nullptr)
{
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "--> Computing error indicator on Polygons "
                 "(h_K-weighted ||f + Delta u_h|| + jump terms)..." << std::endl;

    const bool use_cdg = use_cdg_estimator();

    // TEMPORARY PROBE (env H_PROBE=1): how far apart are the two candidate
    // volume weights, polytope diameter vs background cell diameter, and in
    // particular do they COINCIDE on singletons? Tier-2 only ever marks
    // singletons, so if the ratio is 1 there, marking is already being driven
    // by the theory's weight and the two fields differ only where the
    // agglomeration/coarsening criteria read them. Reported separately for
    // singleton and multi-cell polytopes. Remove once the question is settled.
    const bool h_probe = (std::getenv("H_PROBE") != nullptr);
    double probe_s_min = 1e300, probe_s_max = 0.0, probe_s_sum = 0.0;
    double probe_a_min = 1e300, probe_a_max = 0.0, probe_a_sum = 0.0;
    long   probe_s_n = 0, probe_a_n = 0;

    estimated_error.reinit(tria.n_active_cells());
    estimated_error = 0.0;

    const unsigned int quadrature_degree = 2 * dg_fe.get_degree() + 2;
    const unsigned int dofs_per_cell = ah.n_dofs_per_cell();
    // Fixed per-cell block size in the polytope's agglomerated quadrature
    // (agglomerated_quadrature() concatenates each cell's own
    // QGauss(quadrature_degree) points in get_agglomerate() order).
    const unsigned int n_q_per_cell = QGauss<dim>(quadrature_degree).size();

    // update_hessians ADDED so the volume term below can request the physical
    // Delta u_h via get_function_hessians() through the library's own box
    // mapping. At degree >= 2 the strong-form residual is f + Delta u_h (not
    // just f) -- the missing piece that leaves the ||f||-only version frozen
    // at higher order; at degree 1 the bilinear Laplacian is 0 so it reduces
    // to the old ||f|| term exactly. Face jump terms need values/gradients.
    ah.initialize_fe_values(QGauss<dim>(quadrature_degree),
                            update_quadrature_points | update_JxW_values |
                            update_values | update_gradients | update_hessians,
                            QGauss<dim - 1>(quadrature_degree));

    // Per-polytope accumulators, keyed by polytope->index() (a plain
    // unsigned int, already used elsewhere in this function) -- safe here
    // because we only use it as an opaque map key within this single
    // function call (same `ah`, not modified in between the two passes
    // below), never to index into final_agglomerates directly (which is
    // where index() is documented to be unsafe, since that vector can
    // have empty gaps for polytopes with no locally-owned cells).
    // map::operator[] default-constructs missing keys to 0.0, so no
    // separate zero-init pass is needed.
    std::map<unsigned int, double> volume_error_sq;
    std::map<unsigned int, double> face_error_sq;

    // Per-CELL volume contributions (see the cell-local volume term
    // note in the loop below) + a cell-level FEValues for integrating
    // f^2 on individual background cells.
    std::vector<double> cell_volume_error_sq(tria.n_active_cells(), 0.0);
    MappingQ1<dim> cell_mapping;
    FEValues<dim> cell_fe_values(cell_mapping, dg_fe,
                                 QGauss<dim>(quadrature_degree),
                                 update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> dof_indices0(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices1(dofs_per_cell);
    std::vector<double> coeffs0(dofs_per_cell), coeffs1(dofs_per_cell);

    for (const auto &polytope : ah.polytope_iterators())
    {
        if (!polytope->is_locally_owned())
            continue;

        // --- volume term, CELL-LOCAL version: h_cell^2 * ||f||^2_{L2(cell)}
        // per background cell, NOT a single polytope-level h_K^2 *
        // ||f||^2_{L2(K)} shared uniformly by every cell in the
        // polytope. A uniform per-cell share would (a) turn level-
        // retreat's group_ok into an all-or-nothing decision (constant
        // field within a group => every subtree fails together =>
        // whole regions explode to singletons at once), and (b) couple
        // the per-cell error FIELD to the agglomeration LAYOUT
        // (splitting a polytope discontinuously jumps its cells'
        // values), destabilizing the marking threshold and marking
        // counts cycle to cycle. Cell-local h and cell-local
        // integration keep the volume part of the field layout-
        // INDEPENDENT, smooth, and non-uniform within groups -- a
        // stable quantile and partial (subtree-level) splitting.
        // (the h_cell^2 weight is applied per
        // cell.)
        double poly_vol_sq = 0.0;
        {
            // Strong-form residual f + Delta u_h on the polytope's own
            // agglomerated quadrature: ah.reinit(polytope) gives the library's
            // FEValues (real box mapping), get_function_hessians() gives the
            // physical Delta u_h, and the combined arrays slice back per cell
            // in get_agglomerate() order (see n_q_per_cell above). Replaces the
            // old per-cell ||f||-only integration, which had no way to see the
            // solution's Laplacian and so was wrong at degree >= 2.
            const bool need_laplacian = (dg_fe.get_degree() >= 2);
            const auto &fe_agglo = ah.reinit(polytope);
            const auto &q_points_v = fe_agglo.get_quadrature_points();
            std::vector<double> rhs_values(q_points_v.size());
            rhs_function.value_list(q_points_v, rhs_values);
            std::vector<Tensor<2, dim>> hessians;
            if (need_laplacian)
            {
                hessians.resize(q_points_v.size());
                fe_agglo.get_function_hessians(poly_solution, hessians);
            }
            const auto &agglo_cells = polytope->get_agglomerate();
            unsigned int q_offset = 0;
            for (const auto &cell : agglo_cells)
            {
                double cell_f_sq = 0.0;
                for (unsigned int qi = 0; qi < n_q_per_cell; ++qi)
                {
                    const unsigned int q = q_offset + qi;
                    double resid = rhs_values[q];
                    if (need_laplacian)
                    {
                        double lap_uh = 0.0;
                        for (unsigned int d = 0; d < dim; ++d)
                            lap_uh += hessians[q][d][d];
                        resid += lap_uh; // f + Delta u_h
                    }
                    cell_f_sq += resid * resid * fe_agglo.JxW(q);
                }
                // VOL_H=poly weights the volume residual by the POLYTOPE's
                // diameter (the theory's mesh function h|_K = h_K) instead of
                // the background cell's. They coincide on singletons and
                // differ by ~sqrt(#cells) on an agglomerate. Temporary A/B
                // switch for the effectivity study; default = shipped behavior.
                static const bool vol_h_poly = []{
                    const char *e = std::getenv("VOL_H");
                    return e != nullptr && std::string(e) == "poly";
                }();
                if (h_probe)
                {
                    const double r = polytope->diameter() / cell->diameter();
                    if (agglo_cells.size() == 1) {
                        probe_s_min = std::min(probe_s_min, r);
                        probe_s_max = std::max(probe_s_max, r);
                        probe_s_sum += r; ++probe_s_n;
                    } else {
                        probe_a_min = std::min(probe_a_min, r);
                        probe_a_max = std::max(probe_a_max, r);
                        probe_a_sum += r; ++probe_a_n;
                    }
                }
                const double h_c = vol_h_poly ? polytope->diameter() : cell->diameter();
                const double w   = h_c * h_c;
                cell_volume_error_sq[cell->active_cell_index()] = w * cell_f_sq;
                poly_vol_sq += w * cell_f_sq;
                q_offset += n_q_per_cell;
            }
        }
        volume_error_sq[polytope->index()] += poly_vol_sq;

        // --- face terms: gradient jump + value jump ---
        polytope->get_dof_indices(dof_indices0);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            coeffs0[i] = poly_solution(dof_indices0[i]);

        const unsigned int n_faces = polytope->n_faces();
        for (unsigned int f = 0; f < n_faces; ++f)
        {
            if (polytope->at_boundary(f))
            {
                // Dirichlet BC (u=0): the "jump" is just u_h itself,
                // weighted by 1/h_e -- same scaling convention as the
                // SIPG penalty term and the standard boundary
                // contribution in IP-DG error estimators.
                const auto &fe_face = ah.reinit(polytope, f);
                const auto &bdry_normals = fe_face.get_normal_vectors();

                // The library lumps a polytope's ENTIRE domain boundary into
                // ONE face (keyed on the pseudo-neighbour {own id, own id}),
                // so this is where per-background-face splitting changes most.
                std::vector<std::pair<Point<dim>, Point<dim>>> segments;
                std::vector<unsigned int> q_group;
                unsigned int n_groups = 1;
                if (use_cdg)
                {
                    segments = collect_interface_segments<dim>(ah, polytope->id(), polytope->id());
                    if (!segments.empty())
                    {
                        n_groups = segments.size();
                        const auto &qpts = fe_face.get_quadrature_points();
                        q_group.resize(qpts.size());
                        for (unsigned int q = 0; q < qpts.size(); ++q)
                            q_group[q] = segment_of_point<dim>(qpts[q], segments);
                    }
                }
                std::vector<double> g_val(n_groups, 0.0), g_tang(n_groups, 0.0),
                                    g_meas(n_groups, 0.0);

                // Thm 21's boundary terms are built on (u_h - g_D), not on
                // u_h. With dirichlet_data == nullptr (this driver's case,
                // g_D == 0) the two coincide bit-for-bit; the subtraction is
                // written out so that a driver which DOES impose
                // non-homogeneous data gets a correct estimator instead of a
                // silently divergent one.
                const auto &bdry_qpts = fe_face.get_quadrature_points();
                for (unsigned int q : fe_face.quadrature_point_indices())
                {
                    double val0 = 0.0;
                    Tensor<1, dim> grad0;
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        val0 += coeffs0[i] * fe_face.shape_value(i, q);
                        if (use_cdg)
                            grad0 += coeffs0[i] * fe_face.shape_grad(i, q);
                    }
                    if (dirichlet_data != nullptr)
                    {
                        val0 -= dirichlet_data->value(bdry_qpts[q]);
                        if (use_cdg)
                            grad0 -= dirichlet_data->gradient(bdry_qpts[q]);
                    }
                    const unsigned int g = q_group.empty() ? 0u : q_group[q];
                    g_val[g]  += val0 * val0 * fe_face.JxW(q);
                    g_meas[g] += fe_face.JxW(q);
                    if (use_cdg)
                    {
                        const Tensor<1, dim> grad_t =
                            grad0 - (grad0 * bdry_normals[q]) * bdry_normals[q];
                        g_tang[g] += (grad_t * grad_t) * fe_face.JxW(q);
                    }
                }

                // A boundary face has only ONE adjacent element, so the
                // face-shared sigma degenerates to this polytope's own h.
                const bool split_active = use_cdg && !segments.empty();
                for (unsigned int g = 0; g < n_groups; ++g)
                {
                    // A segment can legitimately receive NO quadrature point:
                    // the interface list expands a hanging-node face into ALL
                    // its children, while the library integrates only those
                    // whose neighbour really lies on the other side. Such a
                    // group carries zero energy, so skipping it is exact and
                    // keeps a zero |e_i| out of the denominator.
                    if (!(g_meas[g] > 0.0))
                        continue;
                    const double h_e = split_active ? g_meas[g] : polytope->diameter();
                    face_error_sq[polytope->index()] += g_val[g] / h_e;
                    if (use_cdg)
                        face_error_sq[polytope->index()] += h_e * g_tang[g];
                }
                continue;
            }

            const auto &neigh_polytope = polytope->neighbor(f);
            const unsigned int nofn = polytope->neighbor_of_agglomerated_neighbor(f);
            const auto &fe_faces = ah.reinit_interface(polytope, neigh_polytope, f, nofn);
            const auto &fe_faces0 = fe_faces.first;
            const auto &normals = fe_faces0.get_normal_vectors();

            if (!neigh_polytope->is_locally_owned())
            {
                // NO id-ordering skip here, unlike the locally-owned
                // branch below. That convention ("assemble each
                // interface once, from the lower-id() side") is only
                // valid when a SINGLE visit updates BOTH sides' shares
                // at once (as the locally-owned branch does, writing to
                // both polytope->index() and neigh_polytope->index() in
                // the same iteration) -- true for matrix assembly too,
                // since Trilinos' distribute_local_to_global + compress
                // correctly routes the "other side"'s matrix rows across
                // ranks regardless of who computed them.
                //
                // face_error_sq has no such cross-process reduction: it
                // is a plain per-rank-local std::map. If this rank's
                // polytope happens to have the HIGHER id of the two,
                // skipping this face here would mean this rank's own
                // face_error_sq[polytope->index()] never gets ANY
                // contribution from this shared interface -- the other
                // rank computes its OWN share for its OWN polytope
                // independently (from its own polytope_iterators() loop,
                // symmetric to this one), and neither side's computation
                // can update the other's local map, so both sides must
                // run this branch unconditionally whenever the neighbor
                // is a ghost, regardless of id() order.
                //
                // Requires a GHOSTED poly_solution passed in from the
                // caller. Mirrors solve_polygonal_system's own
                // ghost-neighbor branch -- same recv_values/
                // recv_gradients/recv_jxws (basis-function data already
                // communicated once per cycle by distribute_
                // agglomerated_dofs()'s exchange_interface_values(), so
                // no new communication is needed here) and the same
                // FIXED ghost diameter (get_bounding_box() diagonal,
                // safe on a ghost polytope; NOT ->diameter(), which
                // reads a non-ghosted is_slave_cell() vector). Combined
                // with the ghosted poly_solution, this gives an actual
                // coeffs1 (the neighbor's real DOF values, not just its
                // basis functions), letting grad1/val1 -- hence the
                // jump -- be computed for real instead of implicitly
                // treated as zero: skipping this face would make any
                // polytope bordering a ghost neighbor look artificially
                // well-resolved (missing jump contribution =>
                // underestimated face_error_sq => can fall below
                // refine_threshold and get merged even when genuinely
                // under-resolved).
                neigh_polytope->get_dof_indices(dof_indices1);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    coeffs1[i] = poly_solution(dof_indices1[i]); // valid: poly_solution is now ghosted

                const types::subdomain_id neigh_rank = neigh_polytope->subdomain_id();
                const auto &recv_values    = ah.recv_values.at(neigh_rank).at({neigh_polytope->id(), nofn});
                const auto &recv_gradients = ah.recv_gradients.at(neigh_rank).at({neigh_polytope->id(), nofn});
                const auto &recv_jxws      = ah.recv_jxws.at(neigh_rank).at({neigh_polytope->id(), nofn});

                // get_bounding_box() IS safe on a ghost polytope (diameter()
                // is not: it reads a non-ghosted is_slave_cell() vector).
                const auto ghost_bp = neigh_polytope->get_bounding_box().get_boundary_points();
                const double neigh_diameter = (ghost_bp.second - ghost_bp.first).norm();

                std::vector<std::pair<Point<dim>, Point<dim>>> segments;
                std::vector<unsigned int> q_group;
                unsigned int n_groups = 1;
                if (use_cdg)
                {
                    segments = collect_interface_segments<dim>(ah, polytope->id(),
                                                               neigh_polytope->id());
                    if (!segments.empty())
                    {
                        n_groups = segments.size();
                        const auto &qpts = fe_faces0.get_quadrature_points();
                        q_group.resize(qpts.size());
                        for (unsigned int q = 0; q < qpts.size(); ++q)
                            q_group[q] = segment_of_point<dim>(qpts[q], segments);
                    }
                }
                std::vector<double> g_grad(n_groups, 0.0), g_val(n_groups, 0.0),
                                    g_tang(n_groups, 0.0), g_meas(n_groups, 0.0);

                for (unsigned int q : fe_faces0.quadrature_point_indices())
                {
                    Tensor<1, dim> grad0, grad1;
                    double val0 = 0.0, val1 = 0.0;
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        grad0 += coeffs0[i] * fe_faces0.shape_grad(i, q);
                        val0  += coeffs0[i] * fe_faces0.shape_value(i, q);
                        grad1 += coeffs1[i] * recv_gradients[i][q];
                        val1  += coeffs1[i] * recv_values[i][q];
                    }
                    const Tensor<1, dim> grad_jump = grad0 - grad1;
                    const double grad_jump_n = grad_jump * normals[q];
                    const double val_jump    = val0 - val1;
                    const unsigned int g = q_group.empty() ? 0u : q_group[q];
                    g_grad[g] += grad_jump_n * grad_jump_n * recv_jxws[q];
                    g_val[g]  += val_jump * val_jump * recv_jxws[q];
                    g_meas[g] += recv_jxws[q];
                    if (use_cdg)
                    {
                        const Tensor<1, dim> grad_jump_t =
                            grad_jump - grad_jump_n * normals[q];
                        g_tang[g] += (grad_jump_t * grad_jump_t) * recv_jxws[q];
                    }
                }

                // Same "each side gets its own contribution, weighted by
                // its own diameter" convention as the local-neighbor
                // branch below -- see that branch's comment for why. The
                // two flux terms are ELEMENT quantities and keep their own
                // h; only the value-jump sigma is a FACE quantity, shared.
                const bool split_active = use_cdg && !segments.empty();
                double my_face_contrib_ghost = 0.0;
                for (unsigned int g = 0; g < n_groups; ++g)
                {
                    if (!(g_meas[g] > 0.0))
                        continue;
                    const double h_e0 = split_active ? g_meas[g] : polytope->diameter();
                    const double h_e1 = split_active ? g_meas[g] : neigh_diameter;
                    const double h_jump = use_cdg ? std::min(h_e0, h_e1) : h_e0;
                    my_face_contrib_ghost += h_e0 * g_grad[g] + g_val[g] / h_jump;
                    if (use_cdg)
                        my_face_contrib_ghost += h_e0 * g_tang[g];
                }
                face_error_sq[polytope->index()] += my_face_contrib_ghost;

                // Cannot accumulate into the ghost neighbor's OWN
                // face_error_sq entry here -- it's owned by another
                // rank and this map is purely local. That side's
                // process computes ITS own h_e1-weighted contribution
                // for this same face independently, the next time it
                // visits this interface from its own polytope_iterators()
                // loop (with polytope/neigh_polytope's roles swapped) --
                // same "each side computes its own share" principle
                // that already applies to the locally-owned branch,
                // just split across two processes instead of computed
                // together in one loop iteration.
                continue;
            }

            // Locally-owned neighbor: id-ordering "assemble once" guard
            // belongs HERE, not before the ghost branch above -- this is
            // the case where a single visit updates BOTH sides' shares
            // together (face_error_sq[polytope] AND face_error_sq[neigh_
            // polytope] both written below), so visiting from just the
            // lower-id() side is correct and avoids double-counting.
            if (!(polytope->id() < neigh_polytope->id()))
                continue;

            const auto &fe_faces1 = fe_faces.second;

            neigh_polytope->get_dof_indices(dof_indices1);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                coeffs1[i] = poly_solution(dof_indices1[i]);

            // NOT the same convention as the SIPG penalty on purpose.
            // The assembly's h_e = min(diam0, diam1) is a coercivity
            // requirement: the penalty must be large enough to cover
            // BOTH sides' trace inequality, and the smaller side's is
            // the tighter constraint -- that's a correctness floor and
            // must not change here (solve_polygonal_system is untouched).
            //
            // The a posteriori indicator's jump term has no such
            // constraint: standard residual-based DG estimator theory
            // weights each element's own jump contribution by its OWN
            // h, not a face-shared min. Sharing the smaller neighbor's h
            // would inflate a LARGE polytope's face/jump error purely
            // from bordering a small one, contaminating the split/merge
            // signal with a geometric artifact (who you happen to
            // border) unrelated to whether YOUR solution is actually
            // under-resolved.
            std::vector<std::pair<Point<dim>, Point<dim>>> segments;
            std::vector<unsigned int> q_group;
            unsigned int n_groups = 1;
            if (use_cdg)
            {
                segments = collect_interface_segments<dim>(ah, polytope->id(),
                                                           neigh_polytope->id());
                if (!segments.empty())
                {
                    n_groups = segments.size();
                    const auto &qpts = fe_faces0.get_quadrature_points();
                    q_group.resize(qpts.size());
                    for (unsigned int q = 0; q < qpts.size(); ++q)
                        q_group[q] = segment_of_point<dim>(qpts[q], segments);
                }
            }
            std::vector<double> g_grad(n_groups, 0.0), g_val(n_groups, 0.0),
                                g_tang(n_groups, 0.0), g_meas(n_groups, 0.0);

            for (unsigned int q : fe_faces0.quadrature_point_indices())
            {
                Tensor<1, dim> grad0, grad1;
                double val0 = 0.0, val1 = 0.0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    grad0 += coeffs0[i] * fe_faces0.shape_grad(i, q);
                    val0  += coeffs0[i] * fe_faces0.shape_value(i, q);
                    grad1 += coeffs1[i] * fe_faces1.shape_grad(i, q);
                    val1  += coeffs1[i] * fe_faces1.shape_value(i, q);
                }
                const Tensor<1, dim> grad_jump = grad0 - grad1;
                const double grad_jump_n = grad_jump * normals[q];
                const double val_jump    = val0 - val1;
                const unsigned int g = q_group.empty() ? 0u : q_group[q];
                g_grad[g] += grad_jump_n * grad_jump_n * fe_faces0.JxW(q);
                g_val[g]  += val_jump * val_jump * fe_faces0.JxW(q);
                g_meas[g] += fe_faces0.JxW(q);
                if (use_cdg)
                {
                    const Tensor<1, dim> grad_jump_t =
                        grad_jump - grad_jump_n * normals[q];
                    g_tang[g] += (grad_jump_t * grad_jump_t) * fe_faces0.JxW(q);
                }
            }

            // Each side gets its OWN contribution, weighted by each polytope's
            // OWN diameter (classic) or by the shared per-background-face
            // measure |e_i| (CDG) -- the former avoids a large polytope's jump
            // error being inflated purely by bordering a small one; the latter
            // is the edge measure the theory actually asks for. sigma is a FACE
            // quantity (same on both sides); the two flux terms are ELEMENT
            // quantities and each keeps its own h.
            const bool split_active = use_cdg && !segments.empty();
            double my_face_contrib = 0.0, neigh_face_contrib = 0.0;
            for (unsigned int g = 0; g < n_groups; ++g)
            {
                if (!(g_meas[g] > 0.0))
                    continue;
                const double h_e0 = split_active ? g_meas[g] : polytope->diameter();
                const double h_e1 = split_active ? g_meas[g] : neigh_polytope->diameter();
                const double h_jump0 = use_cdg ? std::min(h_e0, h_e1) : h_e0;
                const double h_jump1 = use_cdg ? std::min(h_e0, h_e1) : h_e1;
                my_face_contrib    += h_e0 * g_grad[g] + g_val[g] / h_jump0;
                neigh_face_contrib += h_e1 * g_grad[g] + g_val[g] / h_jump1;
                if (use_cdg)
                {
                    my_face_contrib    += h_e0 * g_tang[g];
                    neigh_face_contrib += h_e1 * g_tang[g];
                }
            }
            face_error_sq[polytope->index()]       += my_face_contrib;
            face_error_sq[neigh_polytope->index()] += neigh_face_contrib;
        }
    }

    // --- combine volume + face contributions, map back onto fine cells ---
    for (const auto &polytope : ah.polytope_iterators())
    {
        if (!polytope->is_locally_owned())
            continue;

        const unsigned int p_idx = polytope->index();
        if (p_idx < final_agglomerates.size())
        {
            // NON-UNIFORM per-cell distribution: each cell gets its OWN
            // cell-local volume term plus a measure-proportional slice
            // of the polytope's face/jump total. A uniform share would
            // make the field constant within each polytope, which (a)
            // makes group_ok all-or-nothing (whole regions explode to
            // singletons in one cycle) and (b) ties the field to the
            // agglomeration layout, destabilizing the global quantile L
            // and the marking counts cycle to cycle. The volume part is
            // layout-independent; only the (subordinate) jump share
            // still depends on the current polytope.
            double poly_measure = 0.0;
            for (const auto &cell : final_agglomerates[p_idx])
                poly_measure += cell->measure();
            const double face_sq = face_error_sq[polytope->index()];

            for (const auto &cell : final_agglomerates[p_idx])
            {
                const double face_share =
                    (poly_measure > 0.0) ? face_sq * (cell->measure() / poly_measure) : 0.0;
                const double vol_part = cell_volume_error_sq[cell->active_cell_index()];
                estimated_error[cell->active_cell_index()] = static_cast<float>(
                    std::sqrt(vol_part + face_share));
            }
        }
    }

    if (h_probe)
    {
        // Collectives outside the polytope loop and called the same number of
        // times on every rank (h_probe comes from the environment, which
        // mpirun hands to all ranks identically).
        const double smin = Utilities::MPI::min(probe_s_min, MPI_COMM_WORLD);
        const double smax = Utilities::MPI::max(probe_s_max, MPI_COMM_WORLD);
        const double ssum = Utilities::MPI::sum(probe_s_sum, MPI_COMM_WORLD);
        const long   sn   = Utilities::MPI::sum(probe_s_n,   MPI_COMM_WORLD);
        const double amin = Utilities::MPI::min(probe_a_min, MPI_COMM_WORLD);
        const double amax = Utilities::MPI::max(probe_a_max, MPI_COMM_WORLD);
        const double asum = Utilities::MPI::sum(probe_a_sum, MPI_COMM_WORLD);
        const long   an   = Utilities::MPI::sum(probe_a_n,   MPI_COMM_WORLD);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "  [h-probe] singleton n=" << sn
                      << " ratio min/mean/max = "
                      << (sn ? smin : 0.0) << " / " << (sn ? ssum / sn : 0.0)
                      << " / " << (sn ? smax : 0.0)
                      << " | agglomerated n=" << an
                      << " ratio min/mean/max = "
                      << (an ? amin : 0.0) << " / " << (an ? asum / an : 0.0)
                      << " / " << (an ? amax : 0.0) << std::endl;
    }

    // Global estimated-error norm eta_total = sqrt(sum_cells estimated_error^2)
    // over locally-owned cells. Pair with the printed true H1 seminorm error
    // to read off the effectivity index I_eff = eta_total / ||e||_H1, the
    // measure of whether the estimator stays well-calibrated.
    double local_eta_sq = 0.0;
    for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            const double e = estimated_error[cell->active_cell_index()];
            local_eta_sq += e * e;
        }
    const double eta_total = std::sqrt(Utilities::MPI::sum(local_eta_sq, MPI_COMM_WORLD));
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "    -> [estimator] eta_total = " << eta_total << std::endl;
}

// =====================================================================
// STEP 3: assemble and solve the SIPG system on the current set of
// polytopes (parallel, MPI-aware: handles both locally-owned and ghosted
// neighbor polytopes across process boundaries).
//
// If exact_solution is non-null, the true L2/H1-seminorm error is computed
// via PolyUtils::compute_global_error (mirrors the pattern used in
// poisson.cc). Otherwise, only the (physically meaningless without a known
// exact solution) solution norm is printed, just to confirm the solve
// produced something sane.
// =====================================================================
template <int dim>
void solve_polygonal_system(
    AgglomerationHandler<dim> &ah,
    FE_AggloDGP<dim> &dg_fe,
    const Function<dim> &rhs_function,
    const Function<dim> *exact_solution,
    TrilinosWrappers::MPI::Vector &poly_solution,
    MPI_Comm mpi_communicator,
    // Optional per-phase wall-time accumulators (nullptr = don't measure).
    double *t_assemble_out = nullptr,
    double *t_solve_out = nullptr)
{
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        std::cout << "--> Assembling and solving SIPG on Polygons..." << std::endl;

    dealii::Timer _t_asm; _t_asm.start(); // measures assembly (sparsity + matrix/rhs)

    const IndexSet locally_owned_dofs = ah.agglo_dh.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(ah.agglo_dh);

    TrilinosWrappers::SparsityPattern dsp;
    ah.create_agglomeration_sparsity_pattern(dsp);
    TrilinosWrappers::SparseMatrix system_matrix;
    system_matrix.reinit(dsp);

    poly_solution.reinit(locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector system_rhs(locally_owned_dofs, mpi_communicator);

    const unsigned int dofs_per_cell = ah.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    FullMatrix<double> M11(dofs_per_cell, dofs_per_cell), M12(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M21(dofs_per_cell, dofs_per_cell), M22(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    AffineConstraints<double> constraints;
    constraints.close();

    // SIPG penalty constant. Needed >= some problem-dependent threshold for
    // coercivity; how large depends on how extreme the polytope-size ratios
    // in the mesh get (very non-uniform agglomerations -- e.g. near
    // aggressively refined regions -- need a larger constant).
    // Over-penalizing doesn't just cost accuracy -- it inflates lambda_max
    // (the penalty term dominates the bilinear form on small faces)
    // without helping lambda_min, which makes cond(A) *worse*, not better.
    // SIPG penalty must scale with the element degree p to stay coercive:
    // C0*p*(p+1). C0 = 2 is calibrated for the COMPLETE P_p space
    // (FE_AggloDGP); the tensor Q_p this replaced needed C0 = 5, consistent
    // with Q_p's larger trace/inverse constants (max total degree 2p vs p).
    // Measured at p = 4: C0 = 2 is the L2 optimum, CG near-minimal, and 2x
    // above the C0 ~ 1 point where conditioning collapses. Cross-checked at
    // p = 2: CG-optimal there too, L2 inside the flat band.
    const unsigned int p_penalty = dg_fe.get_degree();
    const double penalty_constant = 2.0 * p_penalty * (p_penalty + 1);

    for (const auto &polytope : ah.polytope_iterators())
    {
        if (!polytope->is_locally_owned())
            continue;
        cell_matrix = 0;
        cell_rhs = 0;
        const auto &agglo_values = ah.reinit(polytope);
        const auto &q_points  = agglo_values.get_quadrature_points();
        std::vector<double> rhs(q_points.size());
        rhs_function.value_list(q_points, rhs);

        // cell (volume) integrals
        for (unsigned int q_index : agglo_values.quadrature_point_indices()) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += agglo_values.shape_grad(i, q_index) * agglo_values.shape_grad(j, q_index) * agglo_values.JxW(q_index);
                }
                cell_rhs(i) += agglo_values.shape_value(i, q_index) * rhs[q_index] * agglo_values.JxW(q_index);
            }
        }
        polytope->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

        // face (interface/boundary) integrals
        const unsigned int n_faces = polytope->n_faces();
        for (unsigned int f = 0; f < n_faces; ++f) {
            const double current_element_diameter = std::fabs(polytope->diameter());

            if (polytope->at_boundary(f)) {
                // Weak (Nitsche-type) imposition of the homogeneous Dirichlet BC.
                const auto &fe_face = ah.reinit(polytope, f);
                const auto &normals = fe_face.get_normal_vectors();
                const double penalty = penalty_constant * (1. / current_element_diameter);
                cell_matrix = 0.;
                for (unsigned int q_index : fe_face.quadrature_point_indices()) {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            cell_matrix(i, j) += (-fe_face.shape_grad(i, q_index) * normals[q_index] * fe_face.shape_value(j, q_index)
                                                  -fe_face.shape_grad(j, q_index) * normals[q_index] * fe_face.shape_value(i, q_index)
                                                  + penalty * fe_face.shape_value(i, q_index) * fe_face.shape_value(j, q_index))
                                                 * fe_face.JxW(q_index);
                        }
                    }
                }
                constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
            } else {
                const auto &neigh_polytope = polytope->neighbor(f);

                // Only assemble each interior interface once, from the side
                // with the lower CellId -- this must hold across process
                // boundaries too, which is why we compare id() (not index(),
                // which is only a local/per-process numbering).
                if (polytope->id() < neigh_polytope->id()) {
                    unsigned int nofn = polytope->neighbor_of_agglomerated_neighbor(f);
                    const auto &fe_faces = ah.reinit_interface(polytope, neigh_polytope, f, nofn);
                    const auto &fe_faces0 = fe_faces.first;

                    std::vector<types::global_dof_index> local_dof_indices_neighbor(dofs_per_cell);
                    neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

                    M11 = 0.; M12 = 0.; M21 = 0.; M22 = 0.;
                    const auto &normals = fe_faces0.get_normal_vectors();

                    if (neigh_polytope->is_locally_owned()) {
                        // Neighbor lives on this process: both FEFaceValues
                        // objects are available directly, and it's safe to
                        // call neigh_polytope->diameter() here. Use the
                        // SMALLER of the two sides' diameters, not just the
                        // owning side's -- see the comment on the ghosted
                        // branch below for why this matters.
                        const double h_f = std::min(polytope->diameter(), neigh_polytope->diameter());
                        const double penalty = penalty_constant * (1.0 / h_f);
                        const auto &fe_faces1 = fe_faces.second;

                        // M12/M21/M22 are integrated against fe_faces1.JxW,
                        // matching upstream polyDEAL: agglomeration_
                        // handler.cc's reinit_master canonically sorts both
                        // sides' quadrature points by physical location, so
                        // fe_faces0.JxW(q) and fe_faces1.JxW(q) are
                        // identical for every q at that point -- which
                        // side's JxW is used here is immaterial.
                        for (unsigned int q_index : fe_faces0.quadrature_point_indices()) {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                    M11(i, j) += (-0.5 * fe_faces0.shape_grad(i, q_index) * normals[q_index] * fe_faces0.shape_value(j, q_index)
                                                  -0.5 * fe_faces0.shape_grad(j, q_index) * normals[q_index] * fe_faces0.shape_value(i, q_index)
                                                  + penalty * fe_faces0.shape_value(i, q_index) * fe_faces0.shape_value(j, q_index)) * fe_faces0.JxW(q_index);

                                    M12(i, j) += ( 0.5 * fe_faces0.shape_grad(i, q_index) * normals[q_index] * fe_faces1.shape_value(j, q_index)
                                                  -0.5 * fe_faces1.shape_grad(j, q_index) * normals[q_index] * fe_faces0.shape_value(i, q_index)
                                                  - penalty * fe_faces0.shape_value(i, q_index) * fe_faces1.shape_value(j, q_index)) * fe_faces1.JxW(q_index);

                                    M21(i, j) += (-0.5 * fe_faces1.shape_grad(i, q_index) * normals[q_index] * fe_faces0.shape_value(j, q_index)
                                                  +0.5 * fe_faces0.shape_grad(j, q_index) * normals[q_index] * fe_faces1.shape_value(i, q_index)
                                                  - penalty * fe_faces1.shape_value(i, q_index) * fe_faces0.shape_value(j, q_index)) * fe_faces1.JxW(q_index);

                                    M22(i, j) += ( 0.5 * fe_faces1.shape_grad(i, q_index) * normals[q_index] * fe_faces1.shape_value(j, q_index)
                                                  +0.5 * fe_faces1.shape_grad(j, q_index) * normals[q_index] * fe_faces1.shape_value(i, q_index)
                                                  + penalty * fe_faces1.shape_value(i, q_index) * fe_faces1.shape_value(j, q_index)) * fe_faces1.JxW(q_index);
                                }
                            }
                        }
                    } else {
                        // Neighbor is ghosted (owned by another rank): there
                        // is no local FEFaceValues for it, so we use the
                        // shape values/gradients/JxW already communicated by
                        // AgglomerationHandler::exchange_interface_values().
                        //
                        // neigh_polytope->diameter() is NOT safe to call
                        // here: AgglomerationAccessor::diameter()
                        // internally calls is_slave_cell(), which reads a
                        // distributed vector with no ghost entries for
                        // polytopes owned by another process.
                        // get_bounding_box() IS safe on a ghost polytope
                        // (it doesn't go through is_slave_cell()), so use
                        // its diagonal as the ghost neighbor's diameter
                        // and take the min with our own side, matching the
                        // locally-owned branch above.
                        const auto ghost_bp = neigh_polytope->get_bounding_box().get_boundary_points();
                        const double neigh_diameter = (ghost_bp.second - ghost_bp.first).norm();
                        const double h_f = std::min(polytope->diameter(), neigh_diameter);
                        const double penalty = penalty_constant * (1.0 / h_f);

                        const types::subdomain_id neigh_rank = neigh_polytope->subdomain_id();
                        const auto &recv_values    = ah.recv_values.at(neigh_rank).at({neigh_polytope->id(), nofn});
                        const auto &recv_gradients = ah.recv_gradients.at(neigh_rank).at({neigh_polytope->id(), nofn});

                        // M12/M21/M22 are integrated against recv_jxws,
                        // matching upstream polyDEAL: agglomeration_
                        // handler.cc's exchange_interface_values() +
                        // reinit_master's canonical point sort make
                        // recv_jxws and fe_faces0.JxW identical pointwise.
                        const auto &recv_jxws = ah.recv_jxws.at(neigh_rank).at({neigh_polytope->id(), nofn});
                        for (unsigned int q_index : fe_faces0.quadrature_point_indices()) {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                    M11(i, j) += (-0.5 * fe_faces0.shape_grad(i, q_index) * normals[q_index] * fe_faces0.shape_value(j, q_index)
                                                  -0.5 * fe_faces0.shape_grad(j, q_index) * normals[q_index] * fe_faces0.shape_value(i, q_index)
                                                  + penalty * fe_faces0.shape_value(i, q_index) * fe_faces0.shape_value(j, q_index)) * fe_faces0.JxW(q_index);

                                    M12(i, j) += ( 0.5 * fe_faces0.shape_grad(i, q_index) * normals[q_index] * recv_values[j][q_index]
                                                  -0.5 * recv_gradients[j][q_index] * normals[q_index] * fe_faces0.shape_value(i, q_index)
                                                  - penalty * fe_faces0.shape_value(i, q_index) * recv_values[j][q_index]) * recv_jxws[q_index];

                                    M21(i, j) += (-0.5 * recv_gradients[i][q_index] * normals[q_index] * fe_faces0.shape_value(j, q_index)
                                                  +0.5 * fe_faces0.shape_grad(j, q_index) * normals[q_index] * recv_values[i][q_index]
                                                  - penalty * recv_values[i][q_index] * fe_faces0.shape_value(j, q_index)) * recv_jxws[q_index];

                                    M22(i, j) += ( 0.5 * recv_gradients[i][q_index] * normals[q_index] * recv_values[j][q_index]
                                                  +0.5 * recv_gradients[j][q_index] * normals[q_index] * recv_values[i][q_index]
                                                  + penalty * recv_values[i][q_index] * recv_values[j][q_index]) * recv_jxws[q_index];
                                }
                            }
                        }
                    }

                    // All four blocks get written unconditionally: M21/M22
                    // land in rows owned by the neighbor's process, but
                    // Trilinos handles this fine via the sparsity pattern +
                    // compress(VectorOperation::add) below.
                    constraints.distribute_local_to_global(M11, local_dof_indices, system_matrix);
                    constraints.distribute_local_to_global(M12, local_dof_indices, local_dof_indices_neighbor, system_matrix);
                    constraints.distribute_local_to_global(M21, local_dof_indices_neighbor, local_dof_indices, system_matrix);
                    constraints.distribute_local_to_global(M22, local_dof_indices_neighbor, system_matrix);
                }
            }
        }
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    _t_asm.stop(); if (t_assemble_out) *t_assemble_out += _t_asm.wall_time();
    dealii::Timer _t_slv; _t_slv.start(); // measures the linear solve (AMG setup + CG)

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        std::cout << "    -> Solving the linear system..." << std::endl;

    // Relative residual reduction demanded of the linear solve. The
    // discretisation error dominates the algebraic one at 1e-7, so the extra
    // digits a tighter tolerance buys never reach the reported L2/H1 norms.
    ReductionControl solver_control(10000, 1e-13, 1e-7);
    solver_control.enable_history_data();

    {
        // CG preconditioned by algebraic multigrid whose FINEST-LEVEL smoother
        // is Ifpack block relaxation over the per-polytope blocks. The matrix
        // is SPD, so CG applies; cond(A) grows with AMR depth, so the [CG]
        // iteration count below is the health signal to watch.
        //
        // The preconditioner SETUP and the CG iterations are timed separately:
        // the phase breakdown reports only their sum, and the split is what
        // tells apart an AMG setup that got expensive from a CG that needed
        // more iterations.
        dealii::Timer _t_pre, _t_cg;
        _t_pre.start();
        std::unique_ptr<TrilinosWrappers::PreconditionBase> preconditioner;
        // One polytope's self-coupling block = (p+1)^dim dofs, numbered
        // contiguously (same layout as FE_DGQ(p)) -- the block size the
        // finest-level smoother below inverts exactly.
        const unsigned int poly_block = dg_fe.n_dofs_per_cell();

        auto amg = std::make_unique<TrilinosWrappers::PreconditionAMG>();
        Teuchos::ParameterList ml;
        ML_Epetra::SetDefaults("SA", ml);
        ml.set("coarse: type", "Amesos-KLU");
        ml.set("initialize random seed", true);
        ml.set("smoother: sweeps", 2);
        ml.set("cycle applications", 1);
        ml.set("prec type", "MGV");
        ml.set("aggregation: threshold", 0.02);
        ml.set("coarse: max size", 2000);
        ml.set("ML output", 0);

        // The basis is MODAL (FE_AggloDGP, Legendre): the discrete constant is
        // (1,0,...,0) per polytope, not the all-ones vector ML defaults to. So
        // ML must be told the per-polytope block structure -- with
        // "PDE equations" = n its default near-null-space becomes n vectors,
        // the first of which IS the discrete constant. Without this the CG
        // count is 4.2x higher (measured); with it, the residual 1.4x matches
        // what point-Jacobi says the matrix itself costs.
        // The coarse levels
        // keep ML's Chebyshev smoother. The per-polytope block structure only
        // exists on the FINEST level -- ML's aggregation destroys it further
        // down -- so a globally-set block smoother would be handed coarse
        // matrices with fewer rows than the requested block count, which
        // Ifpack rejects (IFPACK ERROR -1 in Ifpack_OverlappingPartitioner)
        // and ML then segfaults on. Confining it to level 0 (ML_INCREASING is
        // the default, so level 0 is the finest) is what makes it usable.
        ml.set("PDE equations", static_cast<int>(poly_block));
        ml.set("smoother: type", "Chebyshev");
        ml.set("smoother: Chebyshev alpha", 10.);

        const auto range      = system_matrix.local_range();
        const int  local_rows = static_cast<int>(range.second - range.first);

        // Finest-level smoother: Ifpack block relaxation over the per-polytope
        // blocks. That block is where the FE_AggloDGP conditioning penalty
        // lives -- its basis is orthonormal on the bounding box but integrated
        // over the polytope, so the intra-polytope coupling degenerates as the
        // polytope stops filling its box, which a point smoother cannot touch.
        // One Jacobi sweep is the benchmarked setting: each application costs
        // one dense poly_block x poly_block solve per polytope, and further
        // sweeps cost more than the CG iterations they save.
        //
        // THREE TEMPORARY PROBES (HYST=0.9 rerun, 2026-09-15), all bit-neutral
        // at their defaults:
        //   BLOCK_SMOOTHER=0  -- skip the level-0 block smoother entirely, so
        //                        every level keeps ML's Chebyshev (section 6.3's
        //                        on/off A/B). The DoF trajectory is unaffected
        //                        by this switch, so CG counts stay comparable.
        //   IFPACK_SWEEPS=n   -- inner relaxation sweeps (default 1).
        //   IFPACK_RELAX=name -- inner relaxation type (default Jacobi).
        static const bool block_smoother_on = []{
            const char *e = std::getenv("BLOCK_SMOOTHER");
            return e ? (std::atoi(e) != 0) : true;
        }();
        static const int ifpack_sweeps = []{
            const char *e = std::getenv("IFPACK_SWEEPS");
            return e ? std::atoi(e) : 1;
        }();
        static const std::string ifpack_relax = []{
            const char *e = std::getenv("IFPACK_RELAX");
            return std::string(e ? e : "Jacobi");
        }();
        if (block_smoother_on) {
            Teuchos::ParameterList &lvl0 = ml.sublist("smoother: list (level 0)");
            lvl0.set("smoother: type", "IFPACK");
            lvl0.set("smoother: ifpack type", "block relaxation stand-alone");
            lvl0.set("smoother: sweeps", 1);
            Teuchos::ParameterList &ifp = lvl0.sublist("smoother: ifpack list");
            ifp.set("relaxation: type", ifpack_relax);
            ifp.set("relaxation: sweeps", ifpack_sweeps);
            ifp.set("partitioner: type", "linear");
            ifp.set("partitioner: local parts",
                    local_rows / static_cast<int>(poly_block));
        }

        amg->initialize(system_matrix, ml);
        preconditioner = std::move(amg);

        _t_pre.stop();
        dealii::SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

        try {
            _t_cg.start();
            solver.solve(system_matrix, poly_solution, system_rhs, *preconditioner);
            _t_cg.stop();
            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                std::cout << "  [CG] converged in " << solver_control.last_step()
                          << " steps, final residual = " << solver_control.last_value()
                          << " | precond_setup=" << _t_pre.wall_time()
                          << " cg=" << _t_cg.wall_time()
                          << " per_iter=" << _t_cg.wall_time() /
                                             std::max(1u, solver_control.last_step())
                          << std::endl;
        } catch (const std::exception &e) {
            std::cout << "  [CG] FAILED at step " << solver_control.last_step()
                      << ", last residual = " << solver_control.last_value()
                      << " -- " << e.what() << std::endl;
            throw;
        }
    }

    _t_slv.stop(); if (t_solve_out) *t_solve_out += _t_slv.wall_time();

    if (exact_solution != nullptr) {
        // Real error norms against the known exact solution, following the
        // same pattern as poisson.cc.
        std::vector<double> errors;
        PolyUtils::compute_global_error(ah,
                                        poly_solution,
                                        *exact_solution,
                                        {VectorTools::L2_norm, VectorTools::H1_seminorm},
                                        errors);

        // Full DG energy norm of the true error: the broken H1 seminorm above
        // plus the jump term the theory's reliability bound is stated in.
        // Reported alongside (not instead of) the H1 seminorm, so effectivity
        // indices can be formed in either convention.
        const double jump_sq =
            compute_dg_jump_seminorm_sq<dim>(ah, poly_solution, penalty_constant);
        const double dg_energy_error = std::sqrt(errors[1] * errors[1] + jump_sq);

        const types::global_dof_index n_dofs = poly_solution.size();

        // Rank-gated: compute_global_error() reduces onto rank 0, so every
        // other rank would print a line of zeros. Also report this cycle's own
        // solve time, which is what a time-to-accuracy comparison needs (the
        // phase-timing line at the end only gives the sum over all cycles).
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            std::cout << "    -> Solved. Error (L2): " << errors[0]
                      << ", Error (H1 seminorm): " << errors[1]
                      << ", Error (DG energy): " << dg_energy_error
                      << ", n_dofs: " << n_dofs
                      << ", solve_time: " << _t_slv.wall_time() << std::endl;

        // TEMPORARY PROBE: per-rank DoF spread, the "imbalance" column of the
        // section 6.4 coarsening table. The three reductions run on EVERY rank
        // (collectives must be unconditional); only the print is rank-gated.
        {
            const unsigned int n_local = ah.agglo_dh.n_locally_owned_dofs();
            const unsigned int lo  = Utilities::MPI::min(n_local, mpi_communicator);
            const unsigned int hi  = Utilities::MPI::max(n_local, mpi_communicator);
            const unsigned int tot = Utilities::MPI::sum(n_local, mpi_communicator);
            const unsigned int nproc = Utilities::MPI::n_mpi_processes(mpi_communicator);
            const double mean = static_cast<double>(tot) / nproc;
            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                std::cout << "  [loadbal] dofs: min=" << lo << " max=" << hi
                          << " mean=" << mean
                          << " max/mean=" << (mean > 0.0 ? hi / mean : 0.0)
                          << std::endl;
        }

        // Track the previous cycle's error/DoF count (this function runs
        // once per AMR cycle) so we can report the *observed* convergence
        // rate rather than just the raw error value -- useful evidence for
        // "is the error small enough / do I need more cycles" when running
        // the manufactured-solution problem. NOTE: this h ~ n_dofs^{-1/2}
        // proxy assumes roughly uniform 2D refinement; it is only a rough
        // indicator under adaptive (non-uniform) refinement.
        static double previous_L2_error = -1.0;
        static double previous_H1_error = -1.0;
        static types::global_dof_index previous_n_dofs = 0;

        if (previous_n_dofs > 0 && previous_L2_error > 0.0 && previous_H1_error > 0.0)
        {
            const double dof_ratio   = static_cast<double>(n_dofs) / static_cast<double>(previous_n_dofs);
            const double log_h_ratio = 0.5 * std::log(dof_ratio);

            if (log_h_ratio > 1e-12) {
                const double L2_rate = std::log(previous_L2_error / errors[0]) / log_h_ratio;
                const double H1_rate = std::log(previous_H1_error / errors[1]) / log_h_ratio;

                std::cout << "    -> Observed rate vs. previous cycle: "
                          << "L2 ~ h^" << L2_rate
                          << ", H1 semi ~ h^" << H1_rate
                          << "  (theory for FE_DGQ(1) SIPG: ~2 and ~1)" << std::endl;
            }
        }

        previous_L2_error = errors[0];
        previous_H1_error = errors[1];
        previous_n_dofs   = n_dofs;
    } else {
        // No exact solution available for this RHS (e.g. GaussianRHS).
        // Report BOTH quantities, since they answer different questions:
        //  - poly_solution.l2_norm(): Euclidean norm of the raw DoF
        //    coefficient vector. Grows like sqrt(n_dofs) purely from
        //    summing more terms, REGARDLESS of whether u_h itself is
        //    converging -- not physically meaningful on its own across
        //    different mesh resolutions.
        //  - compute_function_l2_norm(...): the actual ||u_h||_{L2(Omega)},
        //    quadrature-weighted. This is the quantity that should
        //    converge/stay bounded under refinement if the true solution
        //    is well-behaved.
        const double vector_norm = poly_solution.l2_norm();
        const double true_l2_norm = compute_function_l2_norm(ah, poly_solution, mpi_communicator);
        std::cout << "    -> Solved. (No exact solution available; "
                  << "reporting solution norms instead of an error norm.) "
                  << "||coeff vector||_2 = " << vector_norm
                  << " ,  ||u_h||_L2(Omega) = " << true_l2_norm << std::endl;
    }
}


// ===========================================================================
// AdaptivePolytopalPoisson: class-based driver for the adaptive polytopal
// (agglomerated SIPG) Poisson solver, structured after polyDEAL's
// diffusion_reaction.cc example. The per-cycle pipeline is
//   prepare_cycle()          -- STEP 1 (error assessment / bootstrap) +
//                               STEP 2 (adaptive agglomeration: level-retreat
//                               regrouping, connectivity split, diameter cap)
//   solve_estimate_output()  -- STEP 3 (SIPG solve on the polytopes) +
//                               STEP 4 (error indicator + Doerfler line L +
//                               relative-mass stop check) + STEP 5 (output)
//   mark_and_refine()        -- STEP 6 (two-tier marking + inheritance +
//                               mesh refinement)
// Generic agglomeration utilities live in adaptive_agglomerator.h
// (namespace PolytopalAdaptivity); the SIPG assembly/solve, the error
// indicator and the SVG export are the free functions defined above.
// ===========================================================================
// Problem selection:
//   GaussianTwoPeak          -- two-peak GaussianRHS, no known exact
//                                solution; indicator-driven two-tier AMR;
//                                ||u_h|| is reported instead of an error norm.
//   SinSinUniform            -- globally smooth SinSinRHS/SinSinSolution;
//                                the residual indicator has essentially no
//                                discriminating power for this RHS (see
//                                mark_and_refine()'s comment), so this mode
//                                forces uniform refinement every cycle to
//                                isolate the solver's own convergence rate
//                                from AMR-marking noise.
//   GaussianPeakedManufactured -- GaussianPeakedRHS/GaussianPeakedSolution;
//                                has both a known exact solution (real
//                                L2/H1 error norms) AND genuine local
//                                features, so it is driven through the
//                                same indicator-based two-tier adaptive
//                                marking as GaussianTwoPeak instead of
//                                forced uniform refinement.
enum class ProblemType
{
  GaussianTwoPeak,
  SinSinUniform,
  GaussianPeakedManufactured,
  CornerSingularityManufactured // low-regularity r^{2/3} point singularity
};

template <int dim>
class AdaptivePolytopalPoisson
{
public:
  AdaptivePolytopalPoisson(const ProblemType problem_type,
                           const MPI_Comm comm);
  void run();

private:
  void make_grid();
  void prepare_cycle();
  void solve_estimate_output();
  void mark_and_refine();

  // --- communication ---
  const MPI_Comm     mpi_communicator;
  const unsigned int this_mpi_process;

  // --- configuration ---
  const ProblemType problem_type;
  // env CYCLES overrides the cycle budget (singularity runs use 15).
  const unsigned int max_amr_cycles   =
    std::getenv("CYCLES") ? std::atoi(std::getenv("CYCLES")) : 20;
  // Relative-mass stop. It has to be this tight for a convergence benchmark:
  // the windowed-Gaussian error mass collapses fast enough that a 1% bar fires
  // after ~2 cycles at dg_degree=4, while 1e-6 effectively lets the run use
  // its whole cycle budget.
  const double stop_rel_tolerance     = 1e-6;
  const double dorfler_theta          = 0.5;  // bulk fraction for the marking line L
  // Fraction-of-cells cap on the marking line L, dual to dorfler_theta:
  // L is taken as max(mass-based L, count-based L), so at most roughly
  // this fraction of cells ever gets marked in one cycle even when the
  // error field is spread nearly uniformly (dorfler_theta's mass-based
  // bisection collapses toward 0 in that case -- see
  // compute_count_capped_threshold's doc comment in
  // adaptive_agglomerator.h). 0.2 is a reasonable, RHS-shape-agnostic
  // starting point.
  const double mark_count_fraction    = 0.2;
  // Master switch for h-refinement of the background mesh. true (default,
  // matches all previous behavior): STEP 6 below marks and refines
  // single-cell polytopes with error > 1.5*mark_threshold, same as
  // always. false: STEP 6 never sets a refine flag on anything -- the
  // background mesh stays exactly as built by make_grid() for the
  // entire run, and each cycle's adaptivity comes ONLY from STEP 2
  // re-agglomerating (splitting/merging polytopes) over that fixed
  // mesh. Useful for isolating the agglomeration logic's own behavior
  // (e.g. does it converge to a stable partition, does recover_partial
  // behave as expected) from mesh-refinement effects, or for cheaply
  // exploring agglomeration parameters without paying for repeated
  // mesh growth. NOTE: with this false, the "no refinement candidates
  // for two consecutive cycles" stop criterion is skipped (it would
  // trivially fire after 2 cycles since nothing is ever marked) -- the
  // loop instead runs until the STEP 4 relative-error-mass criterion
  // (stop_rel_tolerance) fires or max_amr_cycles is reached. Since the
  // mesh never gets finer, current_total_error is bounded by whatever
  // this fixed mesh can resolve -- if that floor is above
  // stop_rel_tolerance's target, the loop will run to max_amr_cycles
  // without ever satisfying the error-mass stop, so consider lowering
  // max_amr_cycles when using this for a quick agglomeration-only check.
  // TEMPORARY PROBE (rerun at HYST=0.9, 2026-09-15). Env REFINE=0 freezes the
  // background mesh -- configuration B/U of section 6.5. Same probe the
  // 2026-09-07 P_p rerun used and then removed; re-added verbatim so the
  // tables stay reproducible. Default true => bit-neutral.
  const bool   enable_mesh_refinement = []{
      const char *e = std::getenv("REFINE");
      return e ? (std::atoi(e) != 0) : true;
  }();
  // TEMPORARY PROBE. Env COARSEN=0 turns gamma-coarsening off, for the
  // section 6.4 A/B. Default true => bit-neutral.
  const bool   coarsening_enabled = []{
      const char *e = std::getenv("COARSEN");
      return e ? (std::atoi(e) != 0) : true;
  }();
  // Master switch for ADAPTIVE agglomeration. true (default): STEP 2's
  // extract_agglomerates_by_level_retreat is error-driven -- it retreats
  // levels (splits) wherever the local estimated_error exceeds
  // refine_threshold, producing a partition that is fine near the peaks and
  // coarse elsewhere. false: the extractor is made error-BLIND (refine_
  // threshold forced to +infinity, max_cells_per_agglomerate forced to
  // uniform_agglomerate_size below), so every polytope is a uniform
  // fixed-size block regardless of the solution -- a plain, non-adaptive
  // agglomeration of the background mesh. Combined with enable_mesh_
  // refinement=false this gives a pure, zero-adaptivity polytopal-DG
  // baseline: the partition is uniform AND static, so it is fully formed at
  // cycle 0 and never changes -- 1-2 cycles suffice. Isolates the raw
  // agglomerated-DG discretization (and its p-convergence) from BOTH tiers
  // of adaptivity.
  // TEMPORARY PROBE. Env UNIFORM_AGGLOM=1 makes the extractor error-blind --
  // configuration U of section 6.5. Default true => bit-neutral.
  const bool   enable_adaptive_agglomeration = []{
      const char *e = std::getenv("UNIFORM_AGGLOM");
      return e ? (std::atoi(e) == 0) : true;
  }();
  // Uniform block size (cells per polytope) used ONLY when
  // enable_adaptive_agglomeration=false. 4 = 2x2 blocks over the fixed mesh
  // (~1024 polytopes at refine_global(6)), chosen so the uniform run lands
  // near the adaptive agglomeration-only run's stabilized dof count, making
  // the "adaptive vs uniform partition" comparison roughly equal-dof.
  const size_t uniform_agglomerate_size = 4;
  static constexpr unsigned int max_elem_per_node = PolyUtils::constexpr_pow(2, dim);
  using DummyTreeType = bgi::rtree<
      std::pair<BoundingBox<dim>, typename Triangulation<dim>::active_cell_iterator>,
      bgi::rstar<max_elem_per_node>>;

  // --- problem definition ---
  std::unique_ptr<Function<dim>> rhs_function;
  std::unique_ptr<Function<dim>> exact_solution; // nullptr for the Gaussian problem
  parallel::distributed::Triangulation<dim> tria;
  MappingQ1<dim> mapping;
  FE_Q<dim>      fe{dg_degree};

  // --- AMR state (names match the original locals so the transplanted
  //     step bodies compile unchanged) ---
  bool         needs_more_refinement    = true;
  unsigned int amr_cycle                = 0;
  // Uniform background refinement applied in make_grid(); also the level
  // at which the far field sits, so coarsening (STEP 6) reclaims dofs on
  // whichever level the background lives on.
  unsigned int       base_refine_level  = 6;   // USE_T2 lowers it to 3
  double       mark_threshold           = 0.0;
  // Tier-2's own Doerfler line. ROLE SEPARATION (env MARK_POOL=singleton):
  // mark_threshold is the agglomeration/coarsening ruler and is computed over
  // ALL cells; this one is computed over the REFINABLE set only (singletons),
  // because that is the set tier-2 can actually act on. Equal to
  // mark_threshold when the separation is off.
  double       mark_threshold_refine    = 0.0;
  double       current_total_error      = 0.0;
  double       reference_total_error    = -1.0;
  unsigned int consecutive_zero_marks   = 0;
  bool         macro_boundaries_defined = false;
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> final_agglomerates;
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> locally_defined_agglomerates;
  Vector<float>                  estimated_error;
  IndexSet                       locally_owned_dofs;
  TrilinosWrappers::MPI::Vector  solution;

  // Neighbor-coupling lookup for the merge-forecast gate in
  // extract_agglomerates_by_level_retreat (adaptive_agglomerator.h):
  // { fine-cell CellId -> {that cell's last-cycle polytope bounding box,
  // that polytope's last-cycle solved DOF coefficients} }, built at the end
  // of solve_estimate_output() from the converged ah/solution and read by the
  // NEXT cycle's prepare_cycle(). Parallel: after p4est repartitioning a
  // candidate group can migrate to a rank whose lookup lacks its neighbors,
  // so this is also what the fail-closed branch keys off (missing coverage =>
  // don't certify the merge). Ghost polytopes are registered under their
  // master CellId (get_agglomerate() throws on ghosts).
  std::map<CellId, std::pair<BoundingBox<dim>, std::vector<double>>> prev_polytope_lookup;
  TrilinosWrappers::SparseMatrix system_matrix;
  dealii::Timer                  timer_total;
  // Per-phase wall-time accumulators (summed over all AMR cycles). Reduced
  // (max over ranks) and printed at the end of run() for the phase-breakdown
  // chart. assemble/solve are filled from inside solve_polygonal_system.
  // t_handler covers rebuilding AgglomerationHandler each cycle
  // (define_agglomerate per polytope, initialize_fe_values,
  // distribute_agglomerated_dofs). It sits between the other phases --
  // t_agglo stops when prepare_cycle returns and assemble/solve only start
  // inside solve_polygonal_system -- so without its own accumulator it is
  // counted nowhere, and the reported total (a sum of the phases) hides
  // that rather than showing a discrepancy.
  double t_agglo = 0.0, t_assemble = 0.0, t_solve = 0.0, t_estimator = 0.0, t_refine = 0.0,
         t_handler = 0.0;
};


template <int dim>
AdaptivePolytopalPoisson<dim>::AdaptivePolytopalPoisson(
    const ProblemType problem_type_,
    const MPI_Comm comm)
  : mpi_communicator(comm)
  , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
  , problem_type(problem_type_)
  , tria(comm, Triangulation<dim>::MeshSmoothing::limit_level_difference_at_vertices)
{
  switch (problem_type)
    {
      case ProblemType::SinSinUniform:
        rhs_function   = std::make_unique<SinSinRHS<dim>>();
        exact_solution = std::make_unique<SinSinSolution<dim>>();
        break;
      case ProblemType::GaussianPeakedManufactured:
        rhs_function   = std::make_unique<GaussianPeakedRHS<dim>>();
        exact_solution = std::make_unique<GaussianPeakedSolution<dim>>();
        break;
      case ProblemType::CornerSingularityManufactured:
        rhs_function   = std::make_unique<CornerSingularityRHS<dim>>();
        exact_solution = std::make_unique<CornerSingularitySolution<dim>>();
        break;
      case ProblemType::GaussianTwoPeak:
      default:
        rhs_function   = std::make_unique<GaussianRHS<dim>>();
        exact_solution = nullptr;
        break;
    }
}


template <int dim>
void AdaptivePolytopalPoisson<dim>::make_grid()
{
      // 1. generate initial fine mesh
      //
      // IMPORTANT: SinSinSolution (u = sin(pi x) sin(pi y) / (2 pi^2)) is
      // only the exact solution of -Delta u = f with u=0 on the boundary
      // when the domain is the unit square [0,1]^2. GaussianRHS's two
      // peaks at (0.25,0.25) and (0.75,0.75) are also unit-square
      // coordinates -- they're centrally symmetric about (0.5,0.5)
      // (0.5-0.25 = 0.25, 0.5+0.25 = 0.75) precisely because they were
      // written assuming this domain.
      // USE_T2=1 swaps the structured unit square for the unstructured
      // quad mesh meshes/t2.msh (85 irregular quads on
      // [-1.25,1.25]x[-0.5,1.25]). Which background mesh is used decides
      // whether agglomeration does anything at all: uniform structured
      // cells make a polytope of them worthless and the pipeline sheds
      // them, whereas on t2.msh the polytopes stay large for the whole
      // run and the cell cap binds. Conclusions about agglomeration have
      // to be drawn here, not on the structured mesh.
      //
      // NOTE: the boundary branch of assemble contributes to the MATRIX
      // only -- there is NO non-homogeneous Nitsche right-hand side term
      // anywhere in this driver, so u = 0 is imposed weakly whether or not
      // the manufactured solution agrees with it. Every manufactured
      // solution here therefore carries a window that vanishes on the
      // boundary of whatever domain is in use. 85 quads are already many,
      // so the base refinement is lighter.
      if (std::getenv("USE_T2") != nullptr) {
          GridIn<dim> grid_in;
          grid_in.attach_triangulation(tria);
          std::ifstream gmsh_file("../../meshes/t2.msh");
          grid_in.read_msh(gmsh_file);
          base_refine_level = 3;
          if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
              std::cout << "[USE_T2] read unstructured t2.msh (85 quads), refine_global(3)" << std::endl;
      } else {
          GridGenerator::hyper_cube(tria, 0.0, 1.0);
      }

      // Env REF_LEVEL overrides the background mesh resolution. Used by the
      // estimator-quality study, which compares effectivity indices across a
      // sequence of fully agglomerated meshes.
      if (const char *e = std::getenv("REF_LEVEL"))
          base_refine_level = static_cast<unsigned int>(std::atoi(e));

      tria.refine_global(base_refine_level);
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          std::cout << "Initial background cells: " << tria.n_global_active_cells() << std::endl;


}


template <int dim>
void AdaptivePolytopalPoisson<dim>::prepare_cycle()
{
          if (this_mpi_process == 0)
          {
              std::cout << "\n=============================================" << std::endl;
              std::cout << "=== AMR Cycle " << amr_cycle << " ===" << std::endl;
          }
          const unsigned int global_active_cells = tria.n_global_active_cells();
          if (this_mpi_process == 0)
              std::cout << "Background Active cells: " << global_active_cells << std::endl;

          // -------------------------------------------------------------
          // STEP 1: Error assessment
          // -------------------------------------------------------------
          if (amr_cycle == 0) {
              std::cout << "--> Bootstrapping with fine mesh solver for initial error..." << std::endl;
              // solve the PDE on fine mesh to get initial error in cycle 0
              DoFHandler<dim> dof_handler(tria);
              dof_handler.distribute_dofs(fe);

              locally_owned_dofs = dof_handler.locally_owned_dofs();
              IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

              AffineConstraints<double> constraints;
              constraints.clear();
              constraints.reinit(locally_relevant_dofs, locally_relevant_dofs);
              DoFTools::make_hanging_node_constraints(dof_handler, constraints);
              VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
              constraints.close();

              TrilinosWrappers::MPI::Vector solution(locally_owned_dofs, mpi_communicator);
              TrilinosWrappers::MPI::Vector system_rhs(locally_owned_dofs, mpi_communicator);

              TrilinosWrappers::SparsityPattern dsp(locally_owned_dofs,
                                                    locally_owned_dofs,
                                                    locally_relevant_dofs,
                                                    mpi_communicator);
              DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
              dsp.compress();

              system_matrix.reinit(dsp);
              MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), system_matrix, (const Function<dim>*)nullptr, constraints);
              VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), *rhs_function, system_rhs);

              system_matrix.compress(VectorOperation::add);
              system_rhs.compress(VectorOperation::add);

              std::map<types::global_dof_index, double> boundary_values;
              VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);
              MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);

              ReductionControl solver_control(10000, 1e-13, 1e-9);
              solver_control.enable_history_data();

              // See solve_polygonal_system() for the same set of
              // solver/preconditioner options, kept in sync for convenience.
              try {
                  TrilinosWrappers::SolverDirect::AdditionalData direct_data;
                  TrilinosWrappers::SolverDirect direct_solver(solver_control, direct_data);
                  direct_solver.solve(system_matrix, solution, system_rhs);
              } catch (const std::exception &e) {
                  std::cout << "Solver stopped at step " << solver_control.last_step()
                            << ", last residual = " << solver_control.last_value() << std::endl;
                  throw;
              }
              std::cout << "--> Polygonal PDE solved successfully." << std::endl;

              // Consistent-scale residual indicator: same ||f||_{L2(cell)}
              // formula as compute_polygonal_error_indicator() (used for
              // amr_cycle >= 1 below), so the raw estimated_error values --
              // and therefore max_abs_error compared against stop_threshold
              // -- are on the same physical scale in every cycle. A jump-
              // based estimator here (e.g. KellyErrorEstimator) would have
              // a different scale and make cross-cycle comparison of
              // max_abs_error meaningless.
              {
                  const unsigned int quad_degree = 2 * fe.degree + 1;
                  QGauss<dim> quadrature(quad_degree);
                  FEValues<dim> fe_values(fe, quadrature,
                                          update_quadrature_points | update_JxW_values);

                  for (const auto &cell : dof_handler.active_cell_iterators())
                  {
                      if (!cell->is_locally_owned())
                          continue;

                      fe_values.reinit(cell);
                      const auto &q_points = fe_values.get_quadrature_points();
                      std::vector<double> rhs_values(q_points.size());
                      rhs_function->value_list(q_points, rhs_values);

                      double cell_error_sq = 0.0;
                      for (unsigned int q = 0; q < q_points.size(); ++q)
                          cell_error_sq += rhs_values[q] * rhs_values[q] * fe_values.JxW(q);

                      estimated_error[cell->active_cell_index()] =
                          static_cast<float>(std::sqrt(cell_error_sq));
                  }
              }
          } else {
              // cycle > 0: we already have estimated_error inherited from the
              // previous cycle's polygonal solution (see STEP 6), so we can
              // skip directly to agglomeration.
          }

          double local_max_abs_error = estimated_error.linfty_norm();
          double max_abs_error = Utilities::MPI::max(local_max_abs_error, mpi_communicator);

          // =============================================================
          // MARK-THRESHOLD COUPLING, GENERATION-COHERENT VERSION.
          //
          // L (mark_threshold) is the single scale-free ruler shared by
          // the agglomeration (split > L, stay in [0.2L, L], merge
          // < 0.2L) and the mesh refinement (refine single-cell
          // polytopes with error > 1.5L). Two invariants:
          //
          //  1. L must be a Dörfler (error-mass) threshold, not a
          //     cell-count quantile: with Gaussian sources most cells
          //     carry numerically-zero error, so a naive count-based
          //     quantile collapses toward the noise floor and shatters
          //     the Gaussian mid-tail into single-cell polytopes.
          //
          //  1b. DUAL FAILURE MODE: the mass-based L has the OPPOSITE
          //     degenerate case -- a spatially uniform error field (no
          //     localized peaks) forces the mass-bisection to include
          //     almost every cell just to reach dorfler_theta of the
          //     total, collapsing L toward 0 and marking (effectively)
          //     everything. L is therefore max(mass-based, count-based),
          //     where the count-based side (compute_count_capped_
          //     threshold) caps how large a FRACTION of cells can ever
          //     be marked in one cycle, computed by bisecting on the
          //     actual error VALUES (not a raw percentile), so it
          //     doesn't reintroduce the noise-floor failure from 1.
          //     above. Taking the max means whichever criterion says
          //     "mark fewer cells" wins.
          //
          //  2. GENERATION COHERENCE: L must be recomputed only where
          //     the error field itself is recomputed -- at the cycle-0
          //     bootstrap fill (here) and after every STEP 4 -- and
          //     NEVER from the inherited field. Recomputing from the
          //     inherited field mixes generations: STEP 6 would compare
          //     the freshly-recomputed STEP 4 field against a line
          //     derived from the PREVIOUS field, and inheritance's
          //     duplication of refined cells' values can inflate the
          //     count quantile enough that children of legitimately-
          //     refined cells land BELOW the new line and get re-fused.
          //     One field, one line, same generation.
          // =============================================================
          if (amr_cycle == 0)
          {
              const double L_mass  = compute_bulk_threshold<dim>(tria, estimated_error, mpi_communicator,
                                                                  dorfler_theta, &current_total_error);
              const double L_count = compute_count_capped_threshold<dim>(tria, estimated_error, mpi_communicator,
                                                                          mark_count_fraction);
              mark_threshold = std::max(L_mass, L_count);
          }

          // -------------------------------------------------------------
          // STEP 2: Adaptive Agglomeration
          // -------------------------------------------------------------

          // Split/merge HYSTERESIS: descend (split) when max error in a
          // node exceeds the marking line; but only COLLECT (merge into
          // a bigger polytope) when comfortably below it. The gap
          // matters because the h_K^2-weighted indicator jumps by ~4x
          // when two polytopes merge (h_K doubles): with a single
          // threshold, a group whose error sits just under the line
          // merges, quadruples its indicator, splits again next cycle,
          // and flaps forever. 0.2 < 1/4 leaves margin: whatever merges
          // stays merged unless the error genuinely grows.
          // When adaptive agglomeration is OFF, force the error criterion
          // to never fire (+infinity) so the extractor groups purely by
          // size -> a uniform, non-adaptive partition.
          const double refine_threshold  = enable_adaptive_agglomeration
                                               ? mark_threshold
                                               : std::numeric_limits<double>::infinity();
          // max_cells_per_agglomerate bounds a polytope's ABSOLUTE cell
          // count uniformly everywhere, regardless of whether its
          // members are all the same size -- this is what actually
          // keeps h_f = min(diam(K), diam(K')) bounded between two
          // separately-formed NEIGHBORING polytopes. max_diameter_ratio
          // below only bounds size disparity WITHIN a single candidate
          // polytope (does it internally straddle a refinement-level
          // boundary) -- a narrower, complementary check. The cap here
          // mainly guards against absurdly large single polytopes
          // hurting LOCAL approximation quality; the [quadrature-probe]
          // canary and ||u_h|| monotonicity are the safety net when
          // sweeping this value.
          const size_t max_cells_per_agglomerate = enable_adaptive_agglomeration
                                                       ? 256
                                                       : uniform_agglomerate_size;
          // max_diameter_ratio disabled (infinity): size-disparity
          // control relies purely on max_cells_per_agglomerate plus the
          // error-driven mark_threshold coupling above (the error
          // indicator self-corrects for genuinely bad local
          // approximation quality, so a static ratio would just
          // duplicate that with a mesh-scale-dependent magic number).
          const double max_diameter_ratio        = std::numeric_limits<double>::infinity();

          // accept_margin/small_group_penalty tighter than
          // extract_agglomerates_by_level_retreat's own defaults
          // (0.6/0.6): small (2-8 cell) merged polytopes otherwise
          // persist as visible "holes" inside the peak cores at the
          // deeper refinement levels this problem reaches.
          const double accept_margin       = 0.4;
          const double small_group_penalty = 0.35;

          const bool use_metis = use_metis_agglomerator();

          // METIS granularity: k = n_cells / this. Error does NOT set k --
          // it enters through the vertex weights, so quiet regions get more
          // cells per polytope than busy ones at the same k. Coupling k to
          // the Doerfler threshold instead made the polytope count oscillate
          // 370 -> 20 -> 522 -> 19 on consecutive cycles, because L is
          // recomputed every cycle and the layout feeds back into it.
          // TEMPORARY PROBE (HYST=0.9 rerun, 2026-09-15): env METIS_TARGET
          // overrides the granularity, for the section 6.3 tuning sweep.
          // Default 16 => bit-neutral.
          static const size_t metis_target_cells_per_polytope = []{
              const char *e = std::getenv("METIS_TARGET");
              return e ? static_cast<size_t>(std::atoi(e)) : size_t(16);
          }();

          // Fraction of cells treated as "hot" -- the ones kept out of any
          // agglomerate so tier-2 h-refinement can still see them. A dose
          // response confirmed this is the knob that decides how much error
          // mass agglomerates are allowed to hold; 0.05 is the validated
          // setting.
          const double metis_hot_fraction = 0.05;

          // PLACEMENT IS LOAD-BEARING: before the macro_boundaries_defined
          // branch, so every rank evaluates it exactly once per cycle no
          // matter which path it takes. It contains collectives and must
          // never move into a per-group loop (rank-dependent trip counts ->
          // deadlock). Evaluated unconditionally rather than under
          // `use_metis` for the same reason -- the collectives stay outside
          // every branch.
          const double metis_hot_threshold =
              compute_global_hot_threshold<dim>(tria, estimated_error,
                                                refine_threshold,
                                                metis_hot_fraction,
                                                mpi_communicator);

          // NOTE: final_agglomerates is deliberately NOT cleared here.
          // The else branch below CONSUMES the value mark_and_refine()
          // left in it (the intact-polytope carryover) via
          // carried.swap(final_agglomerates) -- clearing it first
          // silently discards every intact polytope, leaving all their
          // cells outside ANY agglomerate and crashing polyDEAL's
          // distribute_agglomerated_dofs with map::at on the very first
          // reuse-path cycle. The fresh-build branch clears it itself
          // immediately below.

          if (!macro_boundaries_defined) {
              final_agglomerates.clear();
              // Cycle 0 uses the same extract_agglomerates_by_level_
              // retreat used by every later cycle's reuse path: real
              // root-level acceptance, a size check that's actually
              // load-bearing (group_ok's size_ok, with the recursive
              // fresh-tree resplit for size-only failures), and
              // recover_partial's per-cell isolation instead of an
              // all-or-nothing/order-based fallback. One grouping
              // standard for the whole run.
              std::vector<typename Triangulation<dim>::active_cell_iterator> local_cells;
              local_cells.reserve(tria.n_locally_owned_active_cells());
              for (const auto &cell : tria.active_cell_iterators())
                  if (cell->is_locally_owned())
                      local_cells.push_back(cell);

              if (use_metis)
                  final_agglomerates = extract_agglomerates_by_metis<dim>(
                      local_cells, estimated_error, metis_hot_threshold,
                      max_cells_per_agglomerate, metis_target_cells_per_polytope,
                      /*input_is_connected=*/false);
              else
                  final_agglomerates = extract_agglomerates_by_level_retreat<dim, bgi::rstar<max_elem_per_node>>(
                      local_cells, mapping, estimated_error, refine_threshold, max_diameter_ratio, max_cells_per_agglomerate,
                      accept_margin, small_group_penalty,
                      rhs_function.get(), /*check_merge_forecast=*/true, dg_degree, &prev_polytope_lookup);

              // NOTE: material_id is NOT assigned here -- STEP 3's
              // post-hoc correction pass can still further re-split some
              // of these polytopes, so assigning material_id now would
              // just get silently overwritten/stale by the time that pass
              // runs. See the single assign_material_id() call at the end
              // of STEP 3 instead, which reflects the FINAL grouping
              // actually used for this cycle's solve (both this branch's
              // decisions AND any post-hoc corrections).
              bool polluted = CellsAgglomerator<dim, DummyTreeType>::check_pollution(final_agglomerates);
              // Report across all ranks (not just whichever rank happened to
              // print) so pollution can be correlated with the AMR cycle at
              // which ||u_h|| starts blowing up.
              {
                  const int polluted_global = Utilities::MPI::sum(polluted ? 1 : 0, mpi_communicator);
                  if (this_mpi_process == 0 && polluted_global > 0)
                      std::cout << "!!! [cycle " << amr_cycle << "] " << polluted_global
                                << " process(es) reported pollution -- the polytopes listed above may be problematic !!!" << std::endl;
              }

              // Macro boundaries are reused from cycle 1 onward: the
              // else-branch below regroups existing material_id groups
              // via extract_agglomerates_by_level_retreat instead of
              // rebuilding the global R-tree from scratch every cycle.
              // This keeps polytopes in regions untouched by local
              // refinement STABLE across cycles (no gratuitous
              // re-partitioning of the far field) and avoids the
              // per-cycle cost of a global R-tree rebuild.
              macro_boundaries_defined = true;
          } else {
              // mark_and_refine() already computed exactly the two
              // inputs this branch needs and left them for us:
              //
              //  1. final_agglomerates -- polytopes verified INTACT across
              //     the refine + p4est-repartition step (cell count matched
              //     the pre-refinement polytope_sizes record). The
              //     iterators were collected AFTER
              //     execute_coarsening_and_refinement(), and no mesh
              //     mutation happens between there and here, so they are
              //     still valid. Each such group goes to
              //     extract_agglomerates_by_level_retreat unchanged: its
              //     root-level group_ok fast path re-validates the group
              //     against THIS generation's estimated_error at ~zero
              //     cost (one O(n) max scan, no R-tree) and keeps it whole
              //     if still clean -- and if the fresh solve made this
              //     region's error grow past the line, the normal
              //     level-retreat machinery kicks in as it must. "Intact
              //     last cycle" is a starting point, never an exemption
              //     from re-validation.
              //
              //  2. user_flag -- set on every "debris" cell: genuinely
              //     newborn children (level grew during refinement) AND
              //     surviving members of polytopes that refinement broke
              //     up. Grouped by their (inherited) material_id so that
              //     each broken polytope's remains are re-agglomerated
              //     together.
              //
              // Using mark_and_refine's work this way also avoids a latent
              // cross-rank hazard: assign_material_id numbers polytopes
              // 0-based PER RANK, so after p4est repartitioning a cell
              // migrated from rank A carrying id 42 could be lumped into
              // rank B's own, unrelated polytope 42 by an id-based scan.
              // Intact polytopes handed over directly never go through an
              // id lookup at all; the hazard only touches the (much
              // smaller) debris set.
              //
              // MPI note: no collectives inside these loops (per-rank
              // group counts differ across ranks, so a per-group
              // collective is a classic call-count-mismatch deadlock) --
              // one pollution reduction after everything, exactly once
              // per rank.
              bool any_polluted = false;

              // --- 1. carried-over intact polytopes ---
              std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> carried;
              carried.swap(final_agglomerates); // final_agglomerates is now
                                                 // empty, ready to refill
              for (auto &grp : carried) {
                  if (use_metis) {
                      // STABILITY GATE. A carried polytope is re-partitioned
                      // only if it holds a cell that must stand alone;
                      // otherwise it is passed through untouched, so a quiet
                      // region keeps the very same polytope for as many cycles
                      // as it stays quiet. Without this, every group went back
                      // through METIS every cycle, and because METIS can only
                      // ever SUBDIVIDE a group it is handed (neighbours are
                      // never in the same graph), the mesh eroded monotonically
                      // -- mean cells per polytope 80.3 -> 7.8 -> 5.7 -> 3.6 ->
                      // 2.8 -> 2.4 over 8 cycles.
                      bool needs_repartition =
                          (grp.size() > max_cells_per_agglomerate);
                      if (!needs_repartition)
                          for (const auto &cell : grp)
                              if (estimated_error[cell->active_cell_index()] >=
                                      metis_hot_threshold ||
                                  cell->user_index() == 1) {
                                  needs_repartition = true;
                                  break;
                              }

                      if (!needs_repartition) {
                          final_agglomerates.push_back(grp);
                          continue;
                      }

                      // A surviving polytope is face-connected by construction.
                      auto local_aggs = extract_agglomerates_by_metis<dim>(
                          grp, estimated_error, metis_hot_threshold,
                          max_cells_per_agglomerate, metis_target_cells_per_polytope,
                          /*input_is_connected=*/true);
                      final_agglomerates.insert(final_agglomerates.end(), local_aggs.begin(), local_aggs.end());
                  } else {
                      auto local_aggs = extract_agglomerates_by_level_retreat<dim, bgi::rstar<max_elem_per_node>>(
                          grp, mapping, estimated_error, refine_threshold, max_diameter_ratio, max_cells_per_agglomerate,
                          accept_margin, small_group_penalty,
                          rhs_function.get(), /*check_merge_forecast=*/true, dg_degree, &prev_polytope_lookup);
                      final_agglomerates.insert(final_agglomerates.end(), local_aggs.begin(), local_aggs.end());
                      if (CellsAgglomerator<dim, DummyTreeType>::check_pollution(final_agglomerates))
                          any_polluted = true;
                  }
              }

              // --- 2. debris cells (user_flag) ---
              if (use_metis) {
                  // ONE batch, not one call per material_id. The R-tree path
                  // below groups debris by inherited material_id because
                  // level-retreat needs a tree per group anyway. For METIS
                  // that split is pure cost: it turns one graph problem into
                  // thousands of tiny ones, each paying a connected-component
                  // pass, a graph build and a METIS call. It also FORBIDS
                  // merging adjacent debris carrying different material_ids,
                  // an artificial restriction -- the ids are stale by this
                  // point. The connected-component pass inside the extractor
                  // splits the batch into genuinely separate regions anyway.
                  //
                  // check_pollution is deliberately NOT scored on this path:
                  // it asserts "one material_id per group", which the batched
                  // debris breaks ON PURPOSE, producing dozens of
                  // guaranteed-false warnings per run.
                  std::vector<typename Triangulation<dim>::active_cell_iterator> debris_cells;
                  for (const auto &cell : tria.active_cell_iterators())
                      if (cell->is_locally_owned() && cell->user_flag_set())
                          debris_cells.push_back(cell);

                  if (!debris_cells.empty()) {
                      auto local_aggs = extract_agglomerates_by_metis<dim>(
                          debris_cells, estimated_error, metis_hot_threshold,
                          max_cells_per_agglomerate, metis_target_cells_per_polytope,
                          /*input_is_connected=*/false);
                      final_agglomerates.insert(final_agglomerates.end(), local_aggs.begin(), local_aggs.end());
                  }
              } else {
                  std::map<types::material_id, std::vector<typename Triangulation<dim>::active_cell_iterator>> debris_by_macro;
                  for (const auto &cell : tria.active_cell_iterators())
                      if (cell->is_locally_owned() && cell->user_flag_set())
                          debris_by_macro[cell->material_id()].push_back(cell);

                  for (auto &[mat_id, local_cells] : debris_by_macro) {
                      (void)mat_id;
                      auto local_aggs = extract_agglomerates_by_level_retreat<dim, bgi::rstar<max_elem_per_node>>(
                          local_cells, mapping, estimated_error, refine_threshold, max_diameter_ratio, max_cells_per_agglomerate,
                          accept_margin, small_group_penalty,
                          rhs_function.get(), /*check_merge_forecast=*/true, dg_degree, &prev_polytope_lookup);
                      final_agglomerates.insert(final_agglomerates.end(), local_aggs.begin(), local_aggs.end());
                      if (CellsAgglomerator<dim, DummyTreeType>::check_pollution(final_agglomerates))
                          any_polluted = true;
                  }
              }

              // Single collective for the whole cycle, called exactly once
              // by every rank regardless of how many groups it happened to
              // have -- see the MPI note above for why this must NOT be
              // inside either loop.
              {
                  const int polluted_global = Utilities::MPI::sum(any_polluted ? 1 : 0, mpi_communicator);
                  if (this_mpi_process == 0 && polluted_global > 0)
                      std::cout << "!!! [cycle " << amr_cycle << ", macro-boundary path] " << polluted_global
                                << " process(es) reported pollution -- the polytopes listed above may be problematic !!!" << std::endl;
              }
          }

          // -------------------------------------------------------------
          // Connectivity post-processing: R-tree-based packing groups
          // cells by spatial/bounding-box proximity, which does NOT
          // guarantee face-adjacency: on a background mesh with highly
          // non-uniform cell sizes (heavy local refinement near the
          // Gaussian peaks, coarse far field), R-tree packing can
          // produce dozens to HUNDREDS of physically disconnected
          // "polytopes" per rank per cycle. A disconnected polytope can
          // have a perfectly normal bounding-box diameter (so none of
          // the size-based checks above catch it) yet its local SIPG
          // stiffness block can be rank-deficient, since the
          // discretization enforces no direct penalty coupling between
          // the disconnected pieces within what it treats as a single
          // polytope.
          //
          // Unconditionally split every group in final_agglomerates
          // (regardless of which STEP 2 branch produced it) into its
          // face-adjacency connected components before STEP 3 ever
          // calls define_agglomerate() -- this catches disconnection
          // from any source.
          final_agglomerates = split_into_connected_components<dim>(final_agglomerates);

          // Control experiment (see no_agglomeration() above): replace the
          // layout with one polytope per locally owned cell. Done AFTER the
          // split so it is the last word on the layout, and built from locally
          // owned cells only so that final_agglomerates lines up index-for-index
          // with locally_defined_agglomerates (STEP 3 indexes the former by
          // polytope->index()).
          if (no_agglomeration()) {
              final_agglomerates.clear();
              for (const auto &cell : tria.active_cell_iterators())
                  if (cell->is_locally_owned())
                      final_agglomerates.push_back(
                          std::vector<typename Triangulation<dim>::active_cell_iterator>{cell});
          }


}


template <int dim>
void AdaptivePolytopalPoisson<dim>::solve_estimate_output()
{
          // -------------------------------------------------------------
          // STEP 3: solve PDE on the adaptive polytopes
          // -------------------------------------------------------------
          GridTools::Cache<dim> cached_tria_current(tria, mapping);

          locally_defined_agglomerates.clear();
          for (const auto &agglo : final_agglomerates) {
              std::vector<typename Triangulation<dim>::active_cell_iterator> local_agglo;
              for (const auto &cell : agglo)
                  if (cell->is_locally_owned())
                      local_agglo.push_back(cell);
              // Only register with polyDEAL if this polytope has any
              // physical presence on the current process.
              if (!local_agglo.empty())
                  locally_defined_agglomerates.push_back(local_agglo);
          }

          // Second connectivity pass: split_polytope_until_small_enough
          // is itself R-tree-node-based -- it retreats to a smaller
          // node and collects it whole, with no connectivity check of
          // its own -- so it can reintroduce a disconnected fragment
          // even after the first split_into_connected_components pass
          // in STEP 2. Disconnected polytopes do NOT create rank
          // deficiency (the polytope's polynomial space lives on its
          // bounding box, so its DoFs stay coupled through the volume
          // term); they are kept split anyway because a single
          // polynomial spanning physically separated regions degrades
          // local approximation quality and muddies the error
          // indicator's meaning. Cheap to run unconditionally:
          // already-connected groups cost one BFS pass and come back
          // unchanged.
          locally_defined_agglomerates = split_into_connected_components<dim>(locally_defined_agglomerates);

          // =============================================================
          // POLYTOPE-BALANCED REPARTITION (env FD_REPART=1, default off).
          //
          // p4est balances CELLS along its Morton curve, but the unknowns sit
          // on POLYTOPES: a polytope carries n_dofs_per_cell() DoFs whether it
          // holds one background cell or two hundred. Since the solve is
          // 70-80% of the run and scales with DoFs, p4est is balancing the
          // cheap currency. Partition the POLYTOPE graph with METIS instead
          // and rebuild the mesh as a fullydistributed copy carrying that
          // partition; p4est keeps doing mark/refine/coarsen/CellDataTransfer
          // untouched, and only agglomerate+assemble+solve move onto the copy.
          //
          // Everything downstream of solve_estimate_output (marking,
          // CellDataTransfer, prepare_cycle's intact/debris split) still reads
          // the P4EST cells, so the error field is transferred back below and
          // locally_defined_agglomerates is restored at the end of this
          // function -- the migrated groups point into fd_tria, which dies
          // with this scope.
          // =============================================================
          std::unique_ptr<parallel::fullydistributed::Triangulation<dim>> fd_tria;
          std::unique_ptr<GridTools::Cache<dim>>                          fd_cache;
          std::vector<types::global_cell_index>                           fd_to_src;
          bool using_fd = false;
          std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> p4est_agglomerates;

          if (std::getenv("FD_REPART") != nullptr)
          {
              fd_tria = std::make_unique<
                  parallel::fullydistributed::Triangulation<dim>>(mpi_communicator);
              std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> migrated;
              FDRepartition::Report rep;

              // Cost-gate state carried across cycles. t_solve is cumulative,
              // so difference it for the last cycle's own solve. Both are zero
              // at cycle 0, which disables the gate there -- right, since the
              // first cycle has nothing to reason from and its imbalance is
              // the worst of the run.
              static FDRepartition::GateState gate;
              static double                   prev_t_solve = 0.0;
              gate.last_solve_time = t_solve - prev_t_solve;
              prev_t_solve         = t_solve;

              if (FDRepartition::build_balanced_copy<dim>(
                      tria, locally_defined_agglomerates,
                      // DoFs per polytope: the quantity p4est cannot see and
                      // this partition balances. dg_fe itself is declared
                      // further down, so ask the element type directly.
                      FE_AggloDGP<dim>(dg_degree).n_dofs_per_cell(), mpi_communicator,
                      *fd_tria, migrated, fd_to_src, rep, gate))
              {
                  gate.last_overhead = rep.overhead();
                  p4est_agglomerates.swap(locally_defined_agglomerates);
                  locally_defined_agglomerates.swap(migrated);
                  fd_cache = std::make_unique<GridTools::Cache<dim>>(*fd_tria, mapping);
                  using_fd = true;
                  if (this_mpi_process == 0)
                      std::cout << "  [fd-repart] cycle=" << amr_cycle
                                << " poly=" << rep.n_poly_global
                                << " edgecut=" << rep.edgecut
                                << " split_parts=" << rep.n_parts_disconnected
                                << " | dofs max/mean " << rep.dof_imbalance_before
                                << " -> " << rep.dof_imbalance_after
                                << " (cells " << rep.cell_imbalance_after << ")"
                                << " | graph=" << rep.t_graph
                                << "s metis=" << rep.t_metis
                                << "s build=" << rep.t_build
                                << "s migrate=" << rep.t_migrate << "s" << std::endl;
              }
              else
              {
                  fd_tria.reset();
                  if (this_mpi_process == 0 && rep.skipped)
                      std::cout << "  [fd-repart] cycle=" << amr_cycle
                                << " SKIPPED by cost gate: imbalance "
                                << rep.dof_imbalance_before
                                << " would save " << rep.predicted_saving
                                << "s, repartition costs " << rep.gate_overhead
                                << "s" << std::endl;
              }
              // The gate's growth correction needs the polytope count from the
              // cycle last_solve_time belongs to, so update it EVERY cycle,
              // skipped ones included.
              gate.last_n_poly = rep.n_poly_global;
          }

          // The mesh the solve and the estimator run on: the rebalanced copy
          // when it exists, the p4est mesh otherwise.
          const Triangulation<dim> &solve_tria =
              using_fd ? static_cast<const Triangulation<dim> &>(*fd_tria)
                       : static_cast<const Triangulation<dim> &>(tria);

          dealii::Timer _t_handler; _t_handler.start();
          AgglomerationHandler<dim> ah(using_fd ? *fd_cache : cached_tria_current);
          for (const auto &agglo : locally_defined_agglomerates)
              ah.define_agglomerate(agglo);

          FE_AggloDGP<dim> dg_fe(dg_degree);
          const unsigned int quadrature_degree      = 2 * dg_fe.get_degree() + 1;
          const unsigned int face_quadrature_degree = 2 * dg_fe.get_degree() + 1;

          // initialize_fe_values() must be called before
          // distribute_agglomerated_dofs(), since the latter's ghost-polytope
          // setup (exchange_interface_values) relies on the FEFaceValues
          // objects it allocates.
          ah.initialize_fe_values(QGauss<dim>(quadrature_degree),
                                  update_gradients | update_JxW_values |
                                  update_quadrature_points | update_values,
                                  QGauss<dim - 1>(face_quadrature_degree));
          ah.distribute_agglomerated_dofs(dg_fe);
          _t_handler.stop(); t_handler += _t_handler.wall_time();

          // Tag every cell with which polytope it belongs to -- reflects
          // the FINAL grouping actually used to solve this cycle, and
          // is what export_merged_svg_global() compares to draw
          // polytope boundaries.
          CellsAgglomerator<dim, DummyTreeType>::assign_material_id(locally_defined_agglomerates);

          solve_polygonal_system(ah, dg_fe, *rhs_function, exact_solution.get(), solution, mpi_communicator,
                                 &t_assemble, &t_solve);

          // -------------------------------------------------------------
          // STEP 4: calculate the new error indicator
          // -------------------------------------------------------------
          // Ghosted copy of the solution, same idiom used elsewhere in
          // this file (e.g. interpolated_solution_ghosted before VTU
          // export) -- needed so compute_polygonal_error_indicator can
          // look up a GHOST neighbor's DOF coefficients (poly_solution
          // is otherwise built with just locally_owned_dofs and throws
          // on any non-owned index). This is what lets the jump terms
          // below compute cross-rank interfaces instead of skipping
          // them.
          // DoFTools::extract_locally_relevant_dofs(ah.agglo_dh) is NOT
          // sufficient here -- it's based on ONE layer of standard
          // deal.II ghost cells, but setup_ghost_polytopes()'s own doc
          // comment (agglomeration_handler.h) explains that a
          // neighboring POLYTOPE is made of some ghost cells AND some
          // ARTIFICIAL ones, needing its own dedicated DOF exchange
          // (local_ghost_dofs/recv_ghost_dofs) that standard relevant-
          // dof extraction doesn't know about. recv_ghost_dofs holds
          // the correct list but is a PRIVATE AgglomerationHandler
          // member, not reachable from here -- so instead of
          // reconstructing it, just make every DOF ghost-visible
          // everywhere. This is a polytope-level vector (a handful of
          // DOFs per polytope, not per fine cell), so a full copy on
          // every rank is cheap regardless of mesh resolution.
          const IndexSet error_locally_owned = ah.agglo_dh.locally_owned_dofs();
          IndexSet error_all_dofs(ah.agglo_dh.n_dofs());
          error_all_dofs.add_range(0, ah.agglo_dh.n_dofs());
          TrilinosWrappers::MPI::Vector solution_ghosted(error_locally_owned, error_all_dofs, mpi_communicator);
          solution_ghosted = solution;

          { dealii::Timer _t_est; _t_est.start();
          {
              // The indicator is computed on whichever mesh the solve used and
              // then mapped back onto the p4est numbering, because everything
              // downstream (marking, CellDataTransfer) indexes estimated_error
              // by the p4est active_cell_index.
              Vector<float> solve_error;
              compute_polygonal_error_indicator(solve_tria, ah, dg_fe, *rhs_function, solution_ghosted, locally_defined_agglomerates, solve_error);
              if (using_fd)
                  FDRepartition::transfer_to_source<dim>(tria, *fd_tria, fd_to_src,
                                                         solve_error, mpi_communicator,
                                                         estimated_error);
              else
                  estimated_error.swap(solve_error);
          }
          _t_est.stop(); t_estimator += _t_est.wall_time(); }
          // Rebuild the neighbor-coupling lookup for the next cycle's
          // merge-forecast gate, from THIS cycle's converged ah/solution (the
          // only place both are alive). ah.polytope_iterators() yields only
          // locally-owned polytopes, so also walk each owned polytope's
          // neighbors and register any GHOST one (under its master CellId --
          // get_agglomerate() throws on ghosts), using solution_ghosted (built
          // just above) for its coefficients and get_bounding_box() for its
          // box. Without this a candidate group that migrated across a p4est
          // partition boundary would find zero neighbor coverage and the
          // forecast would silently degrade to volume-only.
          {
              prev_polytope_lookup.clear();
              const unsigned int dofs_per_cell_poly = ah.n_dofs_per_cell();
              std::vector<types::global_dof_index> poly_dof_indices(dofs_per_cell_poly);
              auto add_owned_polytope = [&](const auto &polytope) {
                  polytope->get_dof_indices(poly_dof_indices);
                  std::vector<double> coeffs(dofs_per_cell_poly);
                  for (unsigned int i = 0; i < dofs_per_cell_poly; ++i)
                      coeffs[i] = solution_ghosted(poly_dof_indices[i]);
                  const BoundingBox<dim> bbox = polytope->get_bounding_box();
                  for (const auto &cell : polytope->get_agglomerate())
                      prev_polytope_lookup[cell->id()] = {bbox, coeffs};
              };
              auto add_ghost_polytope = [&](const auto &polytope) {
                  polytope->get_dof_indices(poly_dof_indices);
                  std::vector<double> coeffs(dofs_per_cell_poly);
                  for (unsigned int i = 0; i < dofs_per_cell_poly; ++i)
                      coeffs[i] = solution_ghosted(poly_dof_indices[i]);
                  const BoundingBox<dim> bbox = polytope->get_bounding_box();
                  prev_polytope_lookup[polytope->id()] = {bbox, coeffs};
              };
              for (const auto &polytope : ah.polytope_iterators())
              {
                  if (!polytope->is_locally_owned())
                      continue;
                  add_owned_polytope(polytope);
                  for (unsigned int f = 0; f < polytope->n_faces(); ++f)
                  {
                      if (polytope->at_boundary(f))
                          continue;
                      const auto neigh_polytope = polytope->neighbor(f);
                      if (!neigh_polytope->is_locally_owned())
                          add_ghost_polytope(neigh_polytope);
                  }
              }
          }

          // Recompute L from the field that was JUST written -- this is
          // the value STEP 6 below marks against, and the value the
          // NEXT cycle's STEP 2 agglomerates the inherited copy of this
          // field against (see the generation-coherence note at the
          // cycle-start block). Computed before inheritance ever
          // duplicates values, so the duplication can't inflate it.
          // Same dual mass/count criterion as the cycle-0 bootstrap
          // site above -- see the "DUAL FAILURE MODE" note there.
          {
              const double L_mass  = compute_bulk_threshold<dim>(tria, estimated_error, mpi_communicator,
                                                                  dorfler_theta, &current_total_error);
              const double L_count = compute_count_capped_threshold<dim>(tria, estimated_error, mpi_communicator,
                                                                          mark_count_fraction);
              mark_threshold = std::max(L_mass, L_count);

              // ROLE SEPARATION. The line tier-2 marks against is recomputed
              // over the refinable set alone -- the cells that are their own
              // polytope. Under the theory's h_K volume weight a cell sitting
              // inside an agglomerate carries ~(#cells) times the indicator of
              // an otherwise identical singleton, so a shared line is set by
              // cells that tier-2 is structurally unable to refine, and
              // starves it. The agglomeration/coarsening ruler above is
              // deliberately NOT restricted: split/merge acts on polytopes,
              // for which every cell is a legitimate input.
              if (separate_marking_pool())
              {
                  std::vector<bool> refinable(tria.n_active_cells(), false);
                  for (const auto &grp : locally_defined_agglomerates)
                      if (grp.size() == 1)
                          refinable[grp.front()->active_cell_index()] = true;
                  const double Lr_mass  = compute_bulk_threshold<dim>(tria, estimated_error, mpi_communicator,
                                                                       dorfler_theta, nullptr, &refinable);
                  const double Lr_count = compute_count_capped_threshold<dim>(tria, estimated_error, mpi_communicator,
                                                                               mark_count_fraction, &refinable);
                  mark_threshold_refine = std::max(Lr_mass, Lr_count);
              }
              else
                  mark_threshold_refine = mark_threshold;
          }
          if (reference_total_error < 0.0 && current_total_error > 0.0)
              reference_total_error = current_total_error; // frozen at cycle 0's STEP 4

          // Relative-reduction stop: total estimated error mass has
          // dropped below stop_rel_tolerance of its initial value --
          // the discretization has done its job; skip STEP 6 and end.
          if (reference_total_error > 0.0 &&
              current_total_error < stop_rel_tolerance * reference_total_error)
          {
              if (this_mpi_process == 0)
                  std::cout << "==> STOP: total error mass reduced to "
                            << (current_total_error / reference_total_error)
                            << " of initial (< " << stop_rel_tolerance << "). Stopping AMR." << std::endl;
              needs_more_refinement = false;
          }

          // -------------------------------------------------------------
          // STEP 5: OUTPUT
          // -------------------------------------------------------------
          timer_total.stop();
          if (this_mpi_process == 0)
              std::cout << "\nTotal AMR time up to now: " << timer_total.wall_time() << " seconds" << std::endl;

// Per-cycle SVG/VTU visualization output. Off by default: the fine-grid
      // interpolation and the parallel VTU gather/write cost real time per
      // cycle and are not needed for convergence/timing runs. WRITE_MESH=1
      // turns it on to inspect meshes. The block is collective, and an
      // environment variable is uniform across ranks, so gating on it cannot
      // deadlock the way a rank-dependent condition would.
      if (std::getenv("WRITE_MESH") != nullptr)
      {
          // 1. output SVG: fine mesh (light gray) + polytope boundaries
          // (dark blue, bolder) merged into one picture, gathered from
          // EVERY rank into a single file (see export_merged_svg_global()
          // above) -- each rank only ever sees its own locally-owned (+
          // ghost) slice of the domain, so without gathering you'd only
          // ever get one rank's piece per file. This is a collective (the
          // MPI gather inside), so every rank must call it.
          export_merged_svg_global<dim>(
              solve_tria, ah,
              "real_amr_test" + std::to_string(amr_cycle),
              mpi_communicator);
          if (this_mpi_process == 0)
              std::cout << "SVG successfully generated: real_amr_test"
                        + std::to_string(amr_cycle) + ".svg" << std::endl << std::flush;


          // 2. interpolate and output VTU
          DataOut<dim> data_out;
          data_out.attach_triangulation(solve_tria);
          TrilinosWrappers::MPI::Vector interpolated_solution;
          PolyUtils::interpolate_to_fine_grid(ah,
                                              interpolated_solution,
                                              solution,
                                              true /*on_the_fly*/);

          // ---------------------------------------------------------------
          // IMPORTANT: interpolated_solution as returned by
          // interpolate_to_fine_grid() only has locally-owned entries (no
          // ghost values). Both VectorTools::point_value() and DataOut
          // internally need to read values for cells this process doesn't
          // own (or that sit near a partition boundary), so they need a
          // *ghosted* vector -- reading a non-ghosted Trilinos vector in a
          // way that isn't perfectly collective/symmetric across ranks (as
          // point_value's "only the owning rank succeeds" pattern is) can
          // leave its underlying Epetra map/communication state corrupted
          // for later calls. Build one ghosted copy here and reuse it below
          // for both the point-value diagnostic and add_data_vector, rather
          // than touching the raw non-ghosted vector directly.
          // ---------------------------------------------------------------
          const IndexSet output_locally_owned    = ah.output_dh.locally_owned_dofs();
          const IndexSet output_locally_relevant = DoFTools::extract_locally_relevant_dofs(ah.output_dh);
          TrilinosWrappers::MPI::Vector interpolated_solution_ghosted(
              output_locally_owned, output_locally_relevant, mpi_communicator);
          interpolated_solution_ghosted = interpolated_solution;

          // Diagnostics: since GaussianRHS has no known exact solution, we
          // can't compute a "true" L2/H1 error. These two global norms let
          // us judge whether interpolated_solution contains anything
          // obviously broken (NaN/huge outlier that would skew the ParaView
          // color scale).
          //
          // point_value() must NOT be used here to sample specific points
          // (e.g. the known Gaussian peak locations): its internal
          // GridTools::find_active_cell_around_point() search over the
          // (much larger) fine mesh owned by the rank with more
          // agglomerates can take far longer than on a rank with fewer,
          // and point_value's collective communication assumes every
          // rank finishes together -- a real deadlock risk. Sample
          // specific points in ParaView instead (Filters -> Data
          // Analysis -> Probe Location) on the exported VTU -- that
          // reads the fully-assembled mesh and has none of these
          // parallel point-location pitfalls.
          //
          // IMPORTANT: l2_norm()/linfty_norm() are themselves collective
          // (they do an internal MPI reduction across all ranks), so *every*
          // rank must call them -- only the std::cout print itself should be
          // rank-guarded. Guarding the calls themselves would make rank 0
          // wait forever in the reduction for a call the other ranks never
          // make -- the same class of deadlock as the point_value hazard
          // above.
          const double sol_l2_norm     = interpolated_solution.l2_norm();
          const double sol_linfty_norm = interpolated_solution.linfty_norm();
          if (this_mpi_process == 0)
              std::cout << "    -> interpolated_solution: ||u||_2 = " << sol_l2_norm
                        << ", max|u| = " << sol_linfty_norm << std::endl;


          data_out.attach_dof_handler(ah.output_dh);
          Vector<double> macro_ids(tria.n_active_cells());
          for (const auto &cell : tria.active_cell_iterators()) {
              // material_id() tracks which POLYTOPE this cell belongs to
              // (see assign_material_id(), called at the end of STEP 3's
              // correction loop so this reflects the actual final
              // grouping used for this cycle's solve) -- subdomain_id()
              // was used here before, which is just the OWNING MPI RANK,
              // not the polytope grouping; with only a handful of ranks
              // that field could only ever show a handful of colors, not
              // the individual polytopes.
              macro_ids[cell->active_cell_index()] = cell->material_id();
          }
          data_out.add_data_vector(macro_ids, "PolytopeID");
          // Rank ownership kept as its own, separate field (this is what
          // subdomain_id() actually is) -- still useful to see, just not
          // the same thing as which polytope a cell belongs to.
          Vector<double> rank_ids(tria.n_active_cells());
          for (const auto &cell : tria.active_cell_iterators())
              rank_ids[cell->active_cell_index()] = cell->subdomain_id();
          data_out.add_data_vector(rank_ids, "SubdomainID");
          data_out.add_data_vector(interpolated_solution_ghosted, "interpolated_solution");
          data_out.build_patches();

          std::string out_vtu("real_amr_test" + std::to_string(amr_cycle));
          data_out.write_vtu_with_pvtu_record(
            "./",
            out_vtu,
            amr_cycle,
            mpi_communicator,
            3);
          if (this_mpi_process == 0)
              std::cout << "Success! Open 'real_amr_test" + std::to_string(amr_cycle) + ".vtu' in ParaView" << std::endl;
      }

      // Hand the polytopes back in terms of P4EST cells before leaving: the
      // migrated groups above point into fd_tria, which dies with this scope,
      // and everything downstream -- tier-2 marking, CellDataTransfer packing,
      // prepare_cycle's intact/debris split -- reads material_id off the p4est
      // cells. This function has no early return; if one is ever added it must
      // run this block first, or mark_and_refine() will dereference iterators
      // into a destroyed triangulation.
      if (using_fd)
      {
          locally_defined_agglomerates.swap(p4est_agglomerates);
          CellsAgglomerator<dim, DummyTreeType>::assign_material_id(locally_defined_agglomerates);
      }
}


template <int dim>
void AdaptivePolytopalPoisson<dim>::mark_and_refine()
{
          // -------------------------------------------------------------
          // STEP 6: refine the background mesh if error is still large
          // -------------------------------------------------------------
          if (needs_more_refinement) {
              if (amr_cycle + 1 >= max_amr_cycles) {
                  needs_more_refinement = false;
              } else {
                  // Timer starts HERE, before any of this cycle's
                  // refinement work -- marking, the CellDataTransfer
                  // pack/ride/unpack, execute_coarsening_and_refinement()
                  // and the p4est repartition it triggers are all
                  // genuine AMR-pipeline compute and must be counted.
                  // The only thing the timer excludes is STEP 5's pure
                  // output (SVG / VTU / visualization interpolation),
                  // which stops it just before that block.
                  timer_total.start();
                  std::cout << "--> Inheriting error to new children..." << std::endl;

                  // Sizes keyed by material_id -- the quantity that actually
                  // rides through CellDataTransfer. Indexing final_agglomerates
                  // by position instead is a DIFFERENT numbering, because
                  // material_id is assigned at the end of STEP 3, after the
                  // second connected-component split has added polytopes.
                  std::map<int, int> polytope_sizes;
                  for (const auto &cell : tria.active_cell_iterators())
                      if (cell->is_locally_owned())
                          ++polytope_sizes[get_polytope_id<dim>(cell)];

                  Vector<float> id_vec(tria.n_active_cells());
                  Vector<float> level_vec(tria.n_active_cells());
                  Vector<float> error_vec(tria.n_active_cells());
                  // (owner rank, polytope size) packed into one payload.
                  //
                  // Needed because material_id is numbered 0-based PER RANK.
                  // After a p4est repartition a cell arriving from rank A with
                  // id 42 would be grouped with rank B's unrelated polytope 42;
                  // the mixed group's size then almost never matches, so a
                  // perfectly intact polytope is declared BROKEN and its cells
                  // -- including peak-region singletons -- are dumped into
                  // debris and re-merged. That was the np=8 period-2 error
                  // oscillation (true L2 jumping 35-100x while the residual
                  // indicator went DOWN).
                  //
                  // Packing both into one float keeps this to a single extra
                  // transfer. owner < 64 and size < 131072 (the cap is 256),
                  // so the product stays under 2^24 and is exact in float.
                  //
                  // This used to live only in the METIS driver. It applies to
                  // BOTH agglomerators here, because the collision is a
                  // property of material_id numbering, not of how polytopes
                  // are built -- the R-tree path merely masked it behind the
                  // merge-forecast gate. Measured effect on the R-tree path
                  // (np=4, 12 cycles, windowed Gaussian): cycle 0 identical,
                  // then the same accuracy at 2-6% FEWER dofs, which is what
                  // "stop declaring intact polytopes broken" should buy. It is
                  // a no-op at np=1-2, where no cell ever migrates ranks.
                  // NOTE this makes the R-tree trajectory at np>=4 differ
                  // slightly from the pre-consolidation adaptive.cc baseline.
                  constexpr int owner_shift = 131072;
                  Vector<float> aux_vec(tria.n_active_cells());
                  const unsigned int my_rank =
                      Utilities::MPI::this_mpi_process(mpi_communicator);

                  for (const auto &cell : tria.active_cell_iterators()) {
                      if (cell->is_locally_owned()) {
                          const int pid = get_polytope_id<dim>(cell);
                          id_vec[cell->active_cell_index()]    = static_cast<float>(pid);
                          level_vec[cell->active_cell_index()] = static_cast<float>(cell->level());
                          error_vec[cell->active_cell_index()] = estimated_error[cell->active_cell_index()];
                          aux_vec[cell->active_cell_index()] =
                              static_cast<float>(static_cast<int>(my_rank) * owner_shift +
                                                 polytope_sizes[pid]);
                      }
                  }

                  // Coarsening-aware strategies: the default CellDataTransfer
                  // constructor's check_equality coarsening strategy throws
                  // when 4 children carry differing data (which they do once
                  // STEP 6 coarsens). id/level use mean; error uses MAX so the
                  // merged (coarser, less resolved) parent inherits its
                  // LARGEST child error -- mean would underestimate it and the
                  // estimator would never refine the cell back, stalling.
                  // id (and, where present, the packed owner/size aux) are
                  // CATEGORICAL LABELS, not quantities, so they must NOT be
                  // averaged on coarsening. The mean of four differing polytope
                  // ids is an id that belongs to no polytope; the coarsened
                  // cell then lands in a fabricated bucket, is grouped with
                  // unrelated cells, and its bogus size overwrites the recorded
                  // polytope size -- corrupting the intact/broken decision.
                  // Take the value of ONE child instead: the one with the
                  // largest error, matching error_vec's max strategy. Every
                  // label payload uses the SAME selector, so labels that belong
                  // together always come from the same child.
                  const auto pick_worst_child =
                      [&error_vec](const typename Triangulation<dim>::cell_iterator &parent,
                                   const std::vector<float> &ch) -> float {
                          if (ch.empty()) return 0.0f;
                          unsigned int best     = 0;
                          float        best_err = -1.0f;
                          for (unsigned int c = 0;
                               c < ch.size() && c < parent->n_children(); ++c) {
                              const float e = error_vec[parent->child(c)->active_cell_index()];
                              if (e > best_err) { best_err = e; best = c; }
                          }
                          return ch[best];
                      };

                  dealii::parallel::distributed::CellDataTransfer<dim, dim, dealii::Vector<float>> id_transfer(
                      tria, false, &dealii::AdaptationStrategies::Refinement::preserve<dim, dim, float>,
                      pick_worst_child);
                  dealii::parallel::distributed::CellDataTransfer<dim, dim, dealii::Vector<float>> level_transfer(
                      tria, false, &dealii::AdaptationStrategies::Refinement::preserve<dim, dim, float>,
                      &dealii::AdaptationStrategies::Coarsening::mean<dim, dim, float>);
                  dealii::parallel::distributed::CellDataTransfer<dim, dim, dealii::Vector<float>> aux_transfer(
                      tria, false, &dealii::AdaptationStrategies::Refinement::preserve<dim, dim, float>,
                      pick_worst_child);
                  dealii::parallel::distributed::CellDataTransfer<dim, dim, dealii::Vector<float>> error_transfer(
                      tria, false, &dealii::AdaptationStrategies::Refinement::preserve<dim, dim, float>,
                      [](const typename Triangulation<dim>::cell_iterator &,
                         const std::vector<float> &ch) -> float {
                          return ch.empty() ? 0.0f : *std::max_element(ch.begin(), ch.end());
                      });


                  id_transfer.prepare_for_coarsening_and_refinement(id_vec);
                  level_transfer.prepare_for_coarsening_and_refinement(level_vec);
                  error_transfer.prepare_for_coarsening_and_refinement(error_vec);
                  aux_transfer.prepare_for_coarsening_and_refinement(aux_vec);

                  if (!enable_mesh_refinement) {
                      // Agglomerate-only mode: never set a refine flag,
                      // regardless of problem_type or the
                      // two-tier marking logic below. Explicitly clear
                      // any flags first -- STEP 2's extractor (or a
                      // stray Mode::refining path inside
                      // CellsAgglomerator, if ever exercised) could in
                      // principle have left flags set as a side effect,
                      // and this guarantees execute_coarsening_and_
                      // refinement() further down is a true no-op:
                      // n_active_cells() stays fixed, so every cycle's
                      // adaptivity comes only from STEP 2 re-agglomerating
                      // over the same fixed background mesh.
                      for (const auto &cell : tria.active_cell_iterators())
                          if (cell->is_locally_owned())
                              cell->clear_refine_flag();
                  } else if (problem_type == ProblemType::SinSinUniform) {
                      // For the globally-smooth manufactured-solution
                      // convergence test, use plain uniform refinement
                      // instead of the error-indicator-driven top-30%
                      // marking. The residual indicator (||f||_{L2(cell)})
                      // has essentially no discriminating power for a
                      // globally smooth RHS like sin(pi x) sin(pi y) --
                      // unlike GaussianRHS/GaussianPeakedRHS, there's no
                      // localized peak for it to home in on -- so which 30%
                      // of cells get marked each cycle is close to arbitrary
                      // and can easily pick regions that don't correspond to
                      // where the true solution error is largest. That
                      // mismatch alone can make the observed L2/H1 error go
                      // up between cycles even if the SIPG assembly/solver
                      // themselves are completely correct, contaminating the
                      // convergence-rate check we actually want here.
                      // Marking every cell keeps the sequence of meshes
                      // uniformly/consistently refined, isolating the
                      // solver's own convergence behavior from AMR-marking
                      // noise.
                      //
                      // GaussianPeakedManufactured does NOT take this path:
                      // its two Gaussian bumps give the residual indicator
                      // real discriminating power (like GaussianTwoPeak), so
                      // it falls through to the indicator-driven two-tier
                      // marking below instead, while still providing a known
                      // exact solution for real L2/H1 error norms.
                      tria.set_all_refine_flags();
                  } else {
                      // ===================================================
                      // TWO-TIER ADAPTIVITY, CONSISTENT-RULER VERSION.
                      //
                      // Both the agglomeration's split/merge decision and
                      // the mesh refinement decision must use the SAME
                      // scale-free ruler -- the marking line L (=
                      // mark_threshold) -- with hysteresis on both ends:
                      //   error < 0.2 L                          -> collectible (merge)
                      //   0.2 L .. L                             -> stay as-is
                      //   > L                                    -> polytope splits
                      //   > refine_hysteresis*L AND single-cell  -> MESH refines
                      // The refine-side hysteresis mirrors the 0.2
                      // merge-side one: only cells COMFORTABLY above the
                      // split line get refined, so their children
                      // (inheriting error > refine_hysteresis*L) sit far
                      // above the next cycle's merge line and cannot be
                      // re-fused by marginal threshold drift. Marking is
                      // bounded by construction (cells above
                      // refine_hysteresis*L are fewer than those above
                      // L). Purely local flag-setting -- no collectives,
                      // no rank mismatch. Refine flags are cleared first
                      // so any flags left behind by STEP 2's extractor
                      // can't union in cells this criterion didn't
                      // choose.
                      // ===================================================
                      {
                          // Two-tier marking, via the generic utility in
                          // adaptive_agglomerator.h (clears refine flags,
                          // marks only single-cell polytopes above
                          // refine_hysteresis*L, returns the global
                          // count). The bar still has to clear the 0.2*L
                          // merge line by enough that the refine-then-re-fuse
                          // oscillation the hysteresis exists to prevent stays
                          // impossible; 0.9 does (0.9 >> 0.2).
                          //
                          // 0.9 rather than the older 1.2, measured at EQUAL
                          // ACCURACY rather than at a fixed cycle budget: the
                          // bar sets how fast DoFs grow per cycle, not the
                          // error/DoF efficiency (equal-accuracy DoF ratio
                          // stays 0.99-1.04 across 1.2 -> 0.6). A high bar
                          // therefore buys nothing and costs a full round of
                          // per-cycle fixed work -- re-agglomerate, rebuild the
                          // handler, assemble, solve, estimate -- for each
                          // extra cycle it takes to cover the same ground. At
                          // 1.2 the run crawled from cycle 3 to 11 while DoFs
                          // grew only 70%. Measured time-to-accuracy at 0.9 vs
                          // 1.2: 0.60-0.63 on the structured mesh and on
                          // t2.msh, 0.56-0.64 for METIS at the deep targets.
                          // Below ~0.8 mark_count_fraction's 20%-of-cells cap
                          // takes over and the knob stops doing anything.
                          //
                          // One value now covers all three regimes: the point
                          // singularity used to need an explicit HYST=0.8
                          // because 1.2 starved tier-2 there, and 0.9 is
                          // measured to be within 1% of 0.8 on it.
                          // Env HYST still overrides, for the sweep that
                          // produced these numbers (roles/FINDINGS.md).
                          static const double tier2_hyst = []{
                              const char *e = std::getenv("HYST");
                              return e ? std::atof(e) : 0.9;
                          }();
                          auto global_marked = mark_singleton_cells_for_refinement<dim>(
                              tria, estimated_error, locally_defined_agglomerates,
                              mark_threshold_refine, mpi_communicator, /*refine_hysteresis=*/tier2_hyst);

                          // Nothing-left-to-do stop: no cell qualified
                          // for refinement for two consecutive cycles.
                          // (One quiet cycle can just mean tier-1
                          // splitting is still preparing candidates.)
                          // All ranks see the same reduced count, so
                          // the flag flips identically everywhere.

                          // PRESSURE VALVE. tier2_hyst is a hysteresis margin
                          // above the Doerfler line L, sized for BROAD features.
                          // When the error concentrates hard enough that L rises
                          // to meet the maximum cell indicator (a point
                          // singularity, or a Gaussian sharp enough that its peak
                          // spans only a cell or two), NOTHING clears hyst*L and
                          // the run stops with error mass still on the table --
                          // a parameter that is merely mismatched turns into a
                          // dead stop. So when the bar starves, re-mark once at
                          // the BARE Doerfler line 1.0*L, but only while the
                          // relative-mass target is still unmet.
                          //
                          // Safe against the refine/re-fuse oscillation the
                          // hysteresis exists to prevent: children inherit their
                          // parent's error, which is at or above L, so they sit
                          // above the 0.2*L merge line and cannot be re-fused.
                          //
                          // MPI: every input to this condition is already a
                          // globally reduced value (global_marked, the two error
                          // totals) or an env constant, so the branch is
                          // rank-uniform and the second collective inside
                          // mark_singleton_cells_for_refinement is entered by all
                          // ranks or none.
                          const bool mass_target_met =
                              (reference_total_error > 0.0 &&
                               current_total_error < stop_rel_tolerance * reference_total_error);
                          if (global_marked == 0 &&
                              tier2_hyst > 1.0 && !mass_target_met)
                          {
                              global_marked = mark_singleton_cells_for_refinement<dim>(
                                  tria, estimated_error, locally_defined_agglomerates,
                                  mark_threshold_refine, mpi_communicator, /*refine_hysteresis=*/1.0);
                              if (global_marked > 0 && this_mpi_process == 0)
                                  std::cout << "  [pressure valve] nothing cleared "
                                            << tier2_hyst << "*L; re-marked "
                                            << global_marked << " cell(s) at 1.0*L"
                                            << std::endl;
                          }

                          if (global_marked == 0)
                          {
                              ++consecutive_zero_marks;
                              if (consecutive_zero_marks >= 2)
                              {
                                  if (this_mpi_process == 0)
                                      std::cout << "==> STOP: no refinement candidates for two consecutive cycles. Stopping AMR." << std::endl;
                                  needs_more_refinement = false;
                              }
                          }
                          else
                              consecutive_zero_marks = 0;
                      }
                  }

                  // -----------------------------------------------------
                  // Far-field coarsening (right-currency / gamma mode).
                  //
                  // Coarsen a background cell only if its error is below an
                  // ABSOLUTE bar tied to the refinement line L (= mark_threshold):
                  //   err <= gamma_coarse * L,   gamma_coarse = 0.005.
                  // gamma << 1/100 guarantees that even after a ~100x post-
                  // coarsen error jump the merged cell stays below L and so
                  // cannot lift the Doerfler bulk line -- the mechanism that
                  // would otherwise starve tier-2 refinement. It buys ~13%
                  // fewer dofs at matched L2, and it doubles as a load
                  // balancer: dofs hang off polytopes, and dropping the
                  // far-field cells is what keeps the per-rank dof counts
                  // within ~1.2x of the mean instead of ~2-3x.
                  //
                  // Level guard: only touch background cells sitting between
                  // (base_refine_level - 2) and base_refine_level -- never a
                  // peak-refined cell (no refine/coarsen oscillation) and never
                  // a refine-flagged one. Purely local flag-setting; the
                  // reduction below is a single collective, same on every rank.
                  if (enable_mesh_refinement && coarsening_enabled) {
                      constexpr double   gamma_coarse = 0.005;
                      const double       L_coarse     = gamma_coarse * mark_threshold;
                      const unsigned int min_level    =
                          (base_refine_level >= 2) ? base_refine_level - 2 : 0u;
                      unsigned int n_coarsen = 0;
                      for (const auto &cell : tria.active_cell_iterators())
                          if (cell->is_locally_owned() && !cell->refine_flag_set()
                              && cell->level() > static_cast<int>(min_level)
                              && cell->level() <= static_cast<int>(base_refine_level)) {
                              if (static_cast<double>(estimated_error[cell->active_cell_index()]) <= L_coarse) {
                                  cell->set_coarsen_flag();
                                  ++n_coarsen;
                              }
                          }
                      const unsigned int gc = Utilities::MPI::sum(n_coarsen, mpi_communicator);
                      if (this_mpi_process == 0)
                          std::cout << "  [coarsen] cells flagged for coarsening: " << gc << std::endl;
                  }

                  tria.execute_coarsening_and_refinement();

                  // Vectors must be resized to the *new* n_active_cells()
                  // before unpack() -- it does not resize them itself.
                  id_vec.reinit(tria.n_active_cells());
                  level_vec.reinit(tria.n_active_cells());
                  error_vec.reinit(tria.n_active_cells());
                  aux_vec.reinit(tria.n_active_cells());
                  id_transfer.unpack(id_vec);
                  level_transfer.unpack(level_vec);
                  error_transfer.unpack(error_vec);
                  aux_transfer.unpack(aux_vec);

                  estimated_error.reinit(tria.n_active_cells());

                  // Keyed by (source rank, polytope id); size comes from the
                  // payload, so nothing here depends on a rank-local table or
                  // on the two numberings agreeing.
                  std::map<std::pair<int, int>,
                           std::vector<typename Triangulation<dim>::active_cell_iterator>> cells_per_polytope;
                  std::map<std::pair<int, int>, int> size_of_polytope;
                  std::vector<typename Triangulation<dim>::active_cell_iterator> newborn_cells;

                  for (const auto &cell : tria.active_cell_iterators()) {
                      // user_index marks cells that are GENUINELY NEW this
                      // cycle, unlike user_flag which also covers survivors of
                      // polytopes that refinement broke up. A just-refined cell
                      // still carries an INHERITED error, so the hot test
                      // cannot recognize it and the agglomerator would pack it
                      // into a full-size polytope, undoing the refinement just
                      // paid for -- a single 16-cell block of newborns can end
                      // up holding a third of a cycle's whole error.
                      cell->set_user_index(0);
                      if (cell->is_locally_owned()) {
                          const int   poly_id   = static_cast<int>(std::lround(id_vec[cell->active_cell_index()]));
                          const int   pre_level = static_cast<int>(std::lround(level_vec[cell->active_cell_index()]));
                          const float err_val   = error_vec[cell->active_cell_index()];
                          const int   aux       = static_cast<int>(std::lround(aux_vec[cell->active_cell_index()]));
                          const int   prev_owner = aux / owner_shift;
                          const int   prev_size  = aux % owner_shift;

                          // Re-sync material_id to the id that actually rode
                          // through the transfer. deal.II does NOT recompute a
                          // coarsened parent's material_id from its children --
                          // the parent keeps the stale value it had before it
                          // was ever refined -- so after STEP 6 coarsens,
                          // cell->material_id() and the transferred poly_id
                          // disagree. Nothing in the intact path reads
                          // material_id (it keys on the transferred pair), but
                          // check_pollution does, and so does the per-material_id
                          // debris grouping. Writing it back here makes the
                          // field truthful again for both. Safe w.r.t. the
                          // unsigned-char range: poly_id was itself a
                          // material_id when it was packed.
                          cell->set_material_id(static_cast<types::material_id>(poly_id));

                          const bool is_newborn = cell->level() > pre_level;
                          estimated_error[cell->active_cell_index()] = err_val;

                          if (!is_newborn) {
                              const auto key = std::make_pair(prev_owner, poly_id);
                              cells_per_polytope[key].push_back(cell);
                              size_of_polytope[key] = prev_size;
                          } else {
                              cell->set_user_index(1);
                              newborn_cells.push_back(cell);
                          }
                      }
                  }

                  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>> next_agglomerates;
                  for (const auto &[poly_id, cells] : cells_per_polytope)
                  {
                      if (static_cast<int>(cells.size()) == size_of_polytope[poly_id])
                          next_agglomerates.push_back(cells);
                      else
                          for (const auto &cell : cells)
                              newborn_cells.push_back(cell);
                  }
                  tria.clear_user_flags();
                  for (const auto &cell : newborn_cells)
                      cell->set_user_flag();

                  final_agglomerates = std::move(next_agglomerates);

                  amr_cycle++;
              }
          }

}


template <int dim>
void AdaptivePolytopalPoisson<dim>::run()
{
  make_grid();
  estimated_error.reinit(tria.n_active_cells());
  timer_total.start();

  while (needs_more_refinement && amr_cycle < max_amr_cycles)
    {
      { dealii::Timer _t; _t.start(); prepare_cycle();   _t.stop(); t_agglo  += _t.wall_time(); }
      solve_estimate_output();  // assemble/solve/estimator timed internally
      { dealii::Timer _t; _t.start(); mark_and_refine(); _t.stop(); t_refine += _t.wall_time(); }

      // Same breakdown as the final [phase-timing] line, but cumulative up to
      // and including this cycle. A time-to-accuracy comparison needs this:
      // two configurations reach a given error at different cycles, so the
      // end-of-run totals compare them at different accuracies. The four
      // reductions are unconditional and identical on every rank (a
      // rank-dependent collective count here would deadlock).
      {
        const double ca = Utilities::MPI::max(t_agglo,     mpi_communicator);
        const double cs = Utilities::MPI::max(t_assemble,  mpi_communicator);
        const double cv = Utilities::MPI::max(t_solve,     mpi_communicator);
        const double ce = Utilities::MPI::max(t_estimator, mpi_communicator);
        const double cr = Utilities::MPI::max(t_refine,    mpi_communicator);
        const double ch = Utilities::MPI::max(t_handler,   mpi_communicator);
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          std::cout << "[phase-cycle] cycle=" << (amr_cycle - 1)
                    << " agglomerate=" << ca << " assemble=" << cs
                    << " solve=" << cv << " estimator=" << ce
                    << " refine=" << cr << " handler=" << ch
                    << " total=" << (ca + cs + cv + ce + cr + ch)
                    << "  (cumulative, max over ranks)" << std::endl;
      }
    }

  // Per-phase breakdown (max wall time over ranks = effective critical path).
  const double a = Utilities::MPI::max(t_agglo,     mpi_communicator);
  const double as = Utilities::MPI::max(t_assemble, mpi_communicator);
  const double s  = Utilities::MPI::max(t_solve,    mpi_communicator);
  const double e  = Utilities::MPI::max(t_estimator,mpi_communicator);
  const double r  = Utilities::MPI::max(t_refine,   mpi_communicator);
  const double h  = Utilities::MPI::max(t_handler,  mpi_communicator);
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "\n[phase-timing] nprocs=" << Utilities::MPI::n_mpi_processes(mpi_communicator)
              << " agglomerate=" << a
              << " assemble=" << as
              << " solve=" << s
              << " estimator=" << e
              << " refine_repartition=" << r
              << " handler_rebuild=" << h
              << "  (seconds, summed over cycles, max over ranks)" << std::endl;
}


int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm mpi_communicator = MPI_COMM_WORLD;

  try
    {
      constexpr int dim = 2;

      // Problem selection:
      //   ProblemType::SinSinUniform            -- globally smooth manufactured
      //       solution, forced uniform refinement (real L2/H1 error norms,
      //       isolates the solver's own convergence rate from AMR-marking
      //       noise; see mark_and_refine()).
      //   ProblemType::GaussianTwoPeak           -- two-peak GaussianRHS, no
      //       known exact solution, indicator-driven two-tier AMR, ||u_h||
      //       reported instead of an error norm.
      //   ProblemType::GaussianPeakedManufactured -- two-peak manufactured
      //       solution (known exact solution + real L2/H1 error norms) that
      //       ALSO drives indicator-based two-tier adaptive marking instead
      //       of uniform refinement, since its localized peaks give the
      //       residual indicator real discriminating power.
      //   ProblemType::CornerSingularityManufactured -- r^(2/3) point
      //       singularity at the domain center (u in H^{5/3}); the
      //       low-regularity counterpart used for the h-vs-p benchmark.
      //       Ran with an explicit HYST=0.8 while the default was 1.2, which
      //       starved tier-2 here; the default is now 0.9 and needs no override.
      // env PROBLEM selects it without recompiling: PROBLEM=singular.
      ProblemType problem_type = ProblemType::GaussianPeakedManufactured;
      if (const char *p = std::getenv("PROBLEM"))
        {
          const std::string s(p);
          if (s == "singular")
            problem_type = ProblemType::CornerSingularityManufactured;
          else if (s == "sinsin")
            problem_type = ProblemType::SinSinUniform;
          else if (s == "twopeak")
            problem_type = ProblemType::GaussianTwoPeak;
        }

      AdaptivePolytopalPoisson<dim> problem(problem_type, mpi_communicator);
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}