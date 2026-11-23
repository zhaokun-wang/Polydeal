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


// The complete adaptive pipeline on polytopes, configured by a parameter
// file (default: adaptive_polytopal.prm):
// - adaptive refinement and coarsening of polytopes along fixed R-trees,
//   against a tolerance, one level or several per adaptation, the trees
//   built over the initial cells of the ranks of p4est or of METIS
//   (PolytopeAdaptivity);
// - assembly with or without quadrature (QuadratureFree);
// - solving on the p4est mesh or on a copy on which every polytope lies on
//   one rank, partitioned by ParMETIS on the graph of the polytopes or by
//   the ranks of the R-trees (PolytopeRepartition);
// - for the heat equation, adaptive time stepping by Algorithm 1 of
//   Cangiani, Georgoulis and Metcalfe (TimeAdaptivity).
//
// Four problems on the unit square or, with Dimension = 3, the unit cube,
// all with u = 0 on the boundary: the Poisson problem with the point
// singularity u = sin(pi x) sin(pi y) r^(2/3) (times sin(pi z) on the cube),
// the Poisson problem with the constant source f = 1, the Poisson problem with
// the polynomial solution u = x(1 - x) y(1 - y)(1 + x + 2 y^2) (times z(1 - z)
// on the cube), and the heat equation with a moving Gaussian peak. The load vector of the
// two polynomial right-hand sides needs no quadrature either. The two Poisson
// problems without an exact polynomial solution also run on an unstructured
// mesh read from a gmsh file, of a rectangle to which the window of the point
// singularity is fitted.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>
#include <polytope_adaptivity.h>
#include <polytope_repartition.h>
#include <quadrature_free_integration.h>
#include <time_adaptivity.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace dealii;



/**
 * The parameters of the program.
 */
class ProblemParameters : public ParameterAcceptor
{
public:
  ProblemParameters();

  /**
   * The dimension, the problem, the initial mesh and the degree. See the
   * descriptions in the constructor, which also appear in the parameter file.
   */
  unsigned int dimension      = 2;
  std::string  problem        = "singular";
  unsigned int n_subdivisions = 256;
  unsigned int degree         = 4;

  /**
   * An initial mesh read from a gmsh file instead, refined globally.
   */
  std::string  mesh_file;
  unsigned int n_mesh_file_refinements = 0;

  /**
   * The adaptivity of the polytopes.
   */
  unsigned int max_cells_per_polytope = 256;
  std::string  level_rule             = "multiple";
  double       coarsening_fraction    = 0.02;
  double       tolerance              = 5e-5;
  unsigned int n_cycles               = 20;
  std::string  initial_partition      = "p4est";

  /**
   * The assembly without quadrature and the solution on a partitioned copy.
   */
  bool quadrature_free = false;
  bool repartition     = false;
  bool use_parmetis    = true;

  /**
   * The time stepping of the heat equation.
   */
  unsigned int n_peaks_per_direction      = 1;
  double       final_time                 = 0.6;
  double       initial_dt                 = 0.02;
  double       max_dt                     = 0.02;
  double       space_tolerance            = 2.5e-3;
  double       time_tolerance             = 0.65;
  double       time_doubling_fraction     = 0.;
  unsigned int n_initial_adaptations      = 5;
  unsigned int max_first_step_adaptations = 10;
};



ProblemParameters::ProblemParameters()
  : ParameterAcceptor("Adaptive polytopal solver/")
{
  enter_subsection("Problem");
  {
    add_parameter("Dimension",
                  dimension,
                  "2: the unit square. 3: the unit cube.",
                  this->prm,
                  Patterns::Integer(2, 3));
    add_parameter("Problem",
                  problem,
                  "'singular': the Poisson problem with a point singularity. "
                  "'constant': the Poisson problem with f = 1, which has no "
                  "exact solution to compare with. 'polynomial': the Poisson "
                  "problem with u = x(1 - x) y(1 - y)(1 + x + 2 y^2), times "
                  "z(1 - z) on the unit cube. 'heat': the heat equation with "
                  "a moving peak.");
    add_parameter("Number of subdivisions",
                  n_subdivisions,
                  "The initial mesh divides the unit square or cube into "
                  "this many cells per direction.");
    add_parameter("Mesh file",
                  mesh_file,
                  "If not empty, a two-dimensional gmsh file whose mesh, "
                  "refined globally, is the initial mesh instead of the unit "
                  "square, for 'singular' and 'constant'. Its domain must be "
                  "a rectangle: the window of the point singularity is fitted "
                  "to it, and the singular point is the vertex of the file "
                  "nearest its centre.");
    add_parameter("Mesh file refinements",
                  n_mesh_file_refinements,
                  "The global refinements of the mesh file.");
    add_parameter("Degree", degree, "The polynomial degree of the elements.");
  }
  leave_subsection();
  enter_subsection("Adaptivity");
  {
    add_parameter("Max cells per polytope",
                  max_cells_per_polytope,
                  "The largest number of initial cells in a polytope. It "
                  "bounds the initial polytopes and every merge.");
    add_parameter("Level rule",
                  level_rule,
                  "'multiple': as many levels per adaptation as the error "
                  "indicator predicts. 'single': one level.");
    add_parameter("Coarsening fraction",
                  coarsening_fraction,
                  "For the rule 'single': coarsen at or under this fraction "
                  "of the polytope tolerance.");
    add_parameter("Tolerance",
                  tolerance,
                  "The tolerance for the error estimator of the Poisson "
                  "problem. The loop stops once the error estimator is at or "
                  "under it.");
    add_parameter("Number of cycles",
                  n_cycles,
                  "The largest number of adaptive cycles of the Poisson "
                  "problem.");
    add_parameter("Initial partition",
                  initial_partition,
                  "Which rank builds the R-tree over an initial cell. "
                  "'p4est': the rank that owns it on the p4est mesh. "
                  "'metis': the rank METIS assigns it to; on more than one "
                  "rank this requires solving on a partitioned copy.");
  }
  leave_subsection();
  enter_subsection("Quadrature-free integration");
  {
    add_parameter("Use quadrature-free integration",
                  quadrature_free,
                  "Assemble the volume and face matrices without quadrature.");
  }
  leave_subsection();
  enter_subsection("Repartition");
  {
    add_parameter("Solve on a partitioned copy",
                  repartition,
                  "Solve on a copy of the mesh on which every polytope lies "
                  "on one rank.");
    add_parameter("Partition the copy by ParMETIS",
                  use_parmetis,
                  "Partition the copy by ParMETIS on the graph of the "
                  "polytopes. Otherwise every polytope goes to the rank whose "
                  "R-tree holds its initial cells.");
  }
  leave_subsection();
  enter_subsection("Time stepping");
  {
    add_parameter("Peaks per direction",
                  n_peaks_per_direction,
                  "The unit square is divided into this many squares per "
                  "direction, and each holds a moving peak scaled to it.");
    add_parameter("Final time", final_time, "The final time T.");
    add_parameter("Initial step length",
                  initial_dt,
                  "The length of the intervals of the initial uniform "
                  "subdivision of the time interval. Steps are not bisected "
                  "below 1/1024 of it.");
    add_parameter("Maximal step length",
                  max_dt,
                  "Steps are not merged beyond this length.");
    add_parameter("Space tolerance",
                  space_tolerance,
                  "The tolerance for the error estimator in space, shared by "
                  "the current polytopes.");
    add_parameter("Time tolerance",
                  time_tolerance,
                  "The tolerance for the error indicator in time.");
    add_parameter("Time doubling fraction",
                  time_doubling_fraction,
                  "Steps whose error indicator in time is under this "
                  "fraction of the time tolerance are merged with the next "
                  "interval.");
    add_parameter("Initial adaptations",
                  n_initial_adaptations,
                  "Adaptations of the mesh of the initial value.");
    add_parameter("Max first step adaptations",
                  max_first_step_adaptations,
                  "The largest number of mesh adaptations in the loop of the "
                  "first step.");
  }
  leave_subsection();
}



/**
 * $u = \sin(\pi x)\sin(\pi y) r^{2/3}$, $r$ the distance to the centre of
 * the square. On a rectangle $[a, b]$ other than the unit square the window
 * is $\prod_d \sin(\pi (x_d - a_d)/(b_d - a_d))$, which vanishes on its
 * boundary, and $r$ is the distance to a given singular point.
 */
template <int dim>
class SingularSolution : public Function<dim>
{
public:
  /**
   * The unit square or cube, with the singular point at its centre.
   */
  SingularSolution();

  /**
   * The rectangle @p domain, with the singular point @p singular_point.
   */
  SingularSolution(const BoundingBox<dim> &domain,
                   const Point<dim>       &singular_point);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  /**
   * Value and Laplacian at @p p.
   */
  void
  evaluate(const Point<dim> &p, double &u, double &laplacian) const;

private:
  static constexpr double alpha = 2. / 3.;

  /**
   * The lower corner of the rectangle, the wave number $\pi/(b_d - a_d)$ of
   * the window in each direction, and the singular point.
   */
  Point<dim>     lower_corner;
  Tensor<1, dim> wave_number;
  Point<dim>     singular_point;
};



template <int dim>
SingularSolution<dim>::SingularSolution()
{
  for (unsigned int direction = 0; direction < dim; ++direction)
    {
      wave_number[direction]    = numbers::PI;
      singular_point[direction] = 0.5;
    }
}



template <int dim>
SingularSolution<dim>::SingularSolution(const BoundingBox<dim> &domain,
                                        const Point<dim>       &singular_point)
  : lower_corner(domain.get_boundary_points().first)
  , singular_point(singular_point)
{
  for (unsigned int direction = 0; direction < dim; ++direction)
    wave_number[direction] = numbers::PI / domain.side_length(direction);
}



template <int dim>
void
SingularSolution<dim>::evaluate(const Point<dim> &p,
                                double           &u,
                                double           &laplacian) const
{
  // The window sin(pi x) sin(pi y), which vanishes on the boundary.
  Tensor<1, dim> phase;
  for (unsigned int direction = 0; direction < dim; ++direction)
    phase[direction] =
      wave_number[direction] * (p[direction] - lower_corner[direction]);
  double window = 1.;
  for (unsigned int direction = 0; direction < dim; ++direction)
    window *= std::sin(phase[direction]);
  Tensor<1, dim> grad_window;
  for (unsigned int direction = 0; direction < dim; ++direction)
    {
      grad_window[direction] =
        wave_number[direction] * std::cos(phase[direction]);
      for (unsigned int other_direction = 0; other_direction < dim;
           ++other_direction)
        if (other_direction != direction)
          grad_window[direction] *= std::sin(phase[other_direction]);
    }
  const double laplacian_window = -wave_number.norm_square() * window;

  const Tensor<1, dim> offset = p - singular_point;
  const double         r_sq   = offset.norm_square();
  // The singular point itself: u vanishes there, and the Laplacian, which
  // is infinite, is set to zero. No quadrature point lies on it, since it
  // is a vertex of the mesh.
  if (r_sq == 0.)
    {
      u         = 0.;
      laplacian = 0.;
      return;
    }
  const double r             = std::sqrt(r_sq);
  const double r_alpha       = std::pow(r, alpha);
  const double r_alpha_minus = std::pow(r, alpha - 2.);
  u                          = window * r_alpha;
  laplacian = laplacian_window * r_alpha +
              2. * alpha * r_alpha_minus * (grad_window * offset) +
              window * alpha * (alpha + dim - 2) * r_alpha_minus;
}



template <int dim>
double
SingularSolution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double u, laplacian;
  evaluate(p, u, laplacian);
  return u;
}



/**
 * $f = -\Delta u$ for SingularSolution.
 */
template <int dim>
class SingularSource : public Function<dim>
{
public:
  /**
   * The source of @p solution.
   */
  SingularSource(const SingularSolution<dim> &solution);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
  const SingularSolution<dim> solution;
};



template <int dim>
SingularSource<dim>::SingularSource(const SingularSolution<dim> &solution)
  : solution(solution)
{}



template <int dim>
double
SingularSource<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double u, laplacian;
  solution.evaluate(p, u, laplacian);
  return -laplacian;
}



/**
 * $u = x(1-x)\,y(1-y)(1 + x + 2y^2)$ on the unit square, times $z(1-z)$ on the
 * unit cube, or its right-hand side $f = -\Delta u$ if @p right_hand_side, as
 * sums of monomials. Both are polynomials, so that quadrature-free
 * integration also assembles the load vector.
 */
template <int dim>
QuadratureFree::PolynomialFunction<dim>
polynomial_function(const bool right_hand_side)
{
  // The terms a x^i y^j of u and of f on the square.
  using Exponents = std::vector<std::pair<unsigned int, unsigned int>>;
  const Exponents solution_exponents = {
    {1, 1}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 1}, {3, 2}};
  const std::vector<double> solution_coefficients = {
    1., -1., 2., -2., -2., 2., -1., 1.};
  const Exponents source_exponents = {
    {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 0}};
  const std::vector<double> source_coefficients = {
    4., -4., 2., -6., 18., 12., -24., -2.};

  std::vector<std::array<unsigned int, dim>> exponents;
  std::vector<double>                        coefficients;
  const auto add_terms = [&](const Exponents           &exponents_2d,
                             const std::vector<double> &coefficients_2d,
                             const unsigned int         z_exponent,
                             const double               factor) {
    for (unsigned int i = 0; i < exponents_2d.size(); ++i)
      {
        std::array<unsigned int, dim> exponent{};
        exponent[0] = exponents_2d[i].first;
        exponent[1] = exponents_2d[i].second;
        if constexpr (dim == 3)
          exponent[2] = z_exponent;
        exponents.push_back(exponent);
        coefficients.push_back(factor * coefficients_2d[i]);
      }
  };

  if constexpr (dim == 2)
    {
      if (right_hand_side)
        add_terms(source_exponents, source_coefficients, 0, 1.);
      else
        add_terms(solution_exponents, solution_coefficients, 0, 1.);
    }
  else
    {
      // On the cube u_3 = u_2 z(1 - z), so f_3 = f_2 z(1 - z) + 2 u_2.
      const auto &exponents_2d =
        right_hand_side ? source_exponents : solution_exponents;
      const auto &coefficients_2d =
        right_hand_side ? source_coefficients : solution_coefficients;
      add_terms(exponents_2d, coefficients_2d, 1, 1.);
      add_terms(exponents_2d, coefficients_2d, 2, -1.);
      if (right_hand_side)
        add_terms(solution_exponents, solution_coefficients, 0, 2.);
    }
  return QuadratureFree::PolynomialFunction<dim>(exponents, coefficients);
}



/**
 * $u = A\sin(\pi x)\sin(\pi y)\exp(-k|x - c(t)|^2)$, its centre moving at
 * unit speed from $(0.3, 0.3)$ to $(0.7, 0.7)$, where it stops.
 *
 * With $n$ peaks per direction the unit square is divided into $n \times n$
 * squares, and each holds this function scaled to it:
 * $u = A\sin(n\pi x)\sin(n\pi y)\sum_i \exp(-k n^2|x - c_i(t)|^2)$. The
 * centre $c_i$ moves from $0.3$ to $0.7$ of its square along the diagonal, at
 * speed $1/n$, so that all peaks stop at the same time as the single one. The
 * window vanishes on the boundary of every square, so $u$ is smooth.
 */
template <int dim>
class MovingPeak : public Function<dim>
{
public:
  MovingPeak(const unsigned int n_peaks_per_direction = 1);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  /**
   * Value, Laplacian and time derivative at @p p.
   */
  void
  evaluate(const Point<dim> &p,
           double           &u,
           double           &laplacian,
           double           &time_derivative) const;

private:
  /**
   * The rate $k$ of the Gaussian, whose width is about $1/\sqrt{k}$, and the
   * amplitude $A$.
   */
  static constexpr double k         = 400.;
  static constexpr double amplitude = 5.;

  /**
   * The number $n$ of peaks per direction.
   */
  const unsigned int n_peaks_per_direction;
};



template <int dim>
MovingPeak<dim>::MovingPeak(const unsigned int n_peaks_per_direction)
  : n_peaks_per_direction(n_peaks_per_direction)
{}



template <int dim>
void
MovingPeak<dim>::evaluate(const Point<dim> &p,
                          double           &u,
                          double           &laplacian,
                          double           &time_derivative) const
{
  // The window sin(n pi x) sin(n pi y), which vanishes on the boundary of
  // every square.
  const double n_pi   = n_peaks_per_direction * numbers::PI;
  double       window = 1.;
  for (unsigned int direction = 0; direction < dim; ++direction)
    window *= std::sin(n_pi * p[direction]);
  Tensor<1, dim> grad_window;
  for (unsigned int direction = 0; direction < dim; ++direction)
    {
      grad_window[direction] = n_pi * std::cos(n_pi * p[direction]);
      for (unsigned int other_direction = 0; other_direction < dim;
           ++other_direction)
        if (other_direction != direction)
          grad_window[direction] *= std::sin(n_pi * p[other_direction]);
    }
  const double laplacian_window = -dim * n_pi * n_pi * window;
  const double k_n = k * n_peaks_per_direction * n_peaks_per_direction;

  // The peaks of the square containing p and of its neighbours; the others
  // are below exp(-k (0.3 + 1)^2) ~ 1e-294 there.
  std::array<int, dim> square;
  for (unsigned int direction = 0; direction < dim; ++direction)
    square[direction] = std::min(static_cast<int>(n_peaks_per_direction) - 1,
                                 static_cast<int>(std::floor(
                                   n_peaks_per_direction * p[direction])));
  u               = 0.;
  laplacian       = 0.;
  time_derivative = 0.;
  std::array<int, dim> shift;
  shift.fill(-1);
  while (true)
    {
      bool inside = true;
      for (unsigned int direction = 0; direction < dim; ++direction)
        inside = inside && square[direction] + shift[direction] >= 0 &&
                 square[direction] + shift[direction] <
                   static_cast<int>(n_peaks_per_direction);
      if (inside)
        {
          Point<dim> start, end;
          for (unsigned int direction = 0; direction < dim; ++direction)
            {
              const int corner = square[direction] + shift[direction];
              start[direction] = (corner + 0.3) / n_peaks_per_direction;
              end[direction]   = (corner + 0.7) / n_peaks_per_direction;
            }
          // The single peak's path takes 0.4 sqrt(dim); every path takes the
          // same time.
          const double     length = start.distance(end) * n_peaks_per_direction;
          const double     fraction = std::min(1., this->get_time() / length);
          const Point<dim> centre   = start + fraction * (end - start);
          const Tensor<1, dim> velocity =
            (fraction < 1.) ? Tensor<1, dim>((end - start) / length) :
                              Tensor<1, dim>();

          // The Gaussian exp(-k n^2 |x - c(t)|^2).
          const Tensor<1, dim> offset        = p - centre;
          const double         r_sq          = offset.norm_square();
          const double         gaussian      = std::exp(-k_n * r_sq);
          const Tensor<1, dim> grad_gaussian = -2. * k_n * gaussian * offset;
          const double         laplacian_gaussian =
            (4. * k_n * k_n * r_sq - 2. * dim * k_n) * gaussian;

          const double u_peak = amplitude * window * gaussian;
          u += u_peak;
          laplacian += amplitude * (laplacian_window * gaussian +
                                    2. * (grad_window * grad_gaussian) +
                                    window * laplacian_gaussian);
          time_derivative += u_peak * 2. * k_n * (offset * velocity);
        }
      unsigned int direction = 0;
      while (direction < dim && shift[direction] == 1)
        shift[direction++] = -1;
      if (direction == dim)
        break;
      ++shift[direction];
    }
}



template <int dim>
double
MovingPeak<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double u, laplacian, time_derivative;
  evaluate(p, u, laplacian, time_derivative);
  return u;
}



/**
 * $f = u_t - \Delta u$ for MovingPeak, or $-\Delta u$ for the initial value
 * (its elliptic projection).
 */
template <int dim>
class MovingPeakSource : public Function<dim>
{
public:
  MovingPeakSource(const bool         stationary,
                   const unsigned int n_peaks_per_direction);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
  const bool              stationary;
  mutable MovingPeak<dim> solution;
};



template <int dim>
MovingPeakSource<dim>::MovingPeakSource(
  const bool         stationary,
  const unsigned int n_peaks_per_direction)
  : stationary(stationary)
  , solution(n_peaks_per_direction)
{}



template <int dim>
double
MovingPeakSource<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double u, laplacian, time_derivative;
  solution.set_time(this->get_time());
  solution.evaluate(p, u, laplacian, time_derivative);
  return stationary ? -laplacian : time_derivative - laplacian;
}



/**
 * Add the SIPG interface terms of one face, by quadrature, to the four blocks:
 * @p M11 couples the basis functions of this polytope with themselves, @p M12
 * this polytope with the neighbour, @p M21 the neighbour with this polytope and
 * @p M22 the neighbour with itself. The values and gradients on the
 * neighbour's side are given by @p neighbor_value and @p neighbor_gradient,
 * the quadrature weights by @p weights. In the loop, v0 and v1 are values and
 * g0 and g1 normal derivatives on this side (0) and the neighbour's (1), of
 * basis function i or j.
 */
template <int dim, typename ValueFunction, typename GradientFunction>
void
assemble_interface_terms(FullMatrix<double>        &M11,
                         FullMatrix<double>        &M12,
                         FullMatrix<double>        &M21,
                         FullMatrix<double>        &M22,
                         const FEValuesBase<dim>   &fe_faces0,
                         const ValueFunction       &neighbor_value,
                         const GradientFunction    &neighbor_gradient,
                         const std::vector<double> &weights,
                         const double               sigma)
{
  const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
  const unsigned int                 n_dofs  = M11.m();
  for (const unsigned int q : fe_faces0.quadrature_point_indices())
    for (unsigned int i = 0; i < n_dofs; ++i)
      for (unsigned int j = 0; j < n_dofs; ++j)
        {
          const double v0_i = fe_faces0.shape_value(i, q);
          const double v0_j = fe_faces0.shape_value(j, q);
          const double v1_i = neighbor_value(i, q);
          const double v1_j = neighbor_value(j, q);
          const double g0_i = fe_faces0.shape_grad(i, q) * normals[q];
          const double g0_j = fe_faces0.shape_grad(j, q) * normals[q];
          const double g1_i = neighbor_gradient(i, q) * normals[q];
          const double g1_j = neighbor_gradient(j, q) * normals[q];
          M11(i, j) +=
            (-0.5 * g0_i * v0_j - 0.5 * g0_j * v0_i + sigma * v0_i * v0_j) *
            weights[q];
          M12(i, j) +=
            (0.5 * g0_i * v1_j - 0.5 * g1_j * v0_i - sigma * v0_i * v1_j) *
            weights[q];
          M21(i, j) +=
            (-0.5 * g1_i * v0_j + 0.5 * g0_j * v1_i - sigma * v1_i * v0_j) *
            weights[q];
          M22(i, j) +=
            (0.5 * g1_i * v1_j + 0.5 * g1_j * v1_i + sigma * v1_i * v1_j) *
            weights[q];
        }
}



/**
 * The four SIPG blocks of one face, as in assemble_interface_terms(), from the
 * face matrices of QuadratureFree::interface_matrices(): @p face_mass(I, J) is
 * the integral of $\phi_I \phi_J$ and @p face_flux(I, J) that of
 * $(\nabla\phi_I \cdot n) \phi_J$, the indices running over the basis
 * functions of the polytope and then of its neighbour. The form is symmetric,
 * so @p M21 is the transpose of @p M12.
 */
void
interface_blocks(const FullMatrix<double> &face_mass,
                 const FullMatrix<double> &face_flux,
                 const double              sigma,
                 FullMatrix<double>       &M11,
                 FullMatrix<double>       &M12,
                 FullMatrix<double>       &M21,
                 FullMatrix<double>       &M22)
{
  const unsigned int n_dofs = M11.m();
  for (unsigned int i = 0; i < n_dofs; ++i)
    for (unsigned int j = 0; j < n_dofs; ++j)
      {
        M11(i, j) = -0.5 * face_flux(i, j) - 0.5 * face_flux(j, i) +
                    sigma * face_mass(i, j);
        M12(i, j) = 0.5 * face_flux(i, n_dofs + j) -
                    0.5 * face_flux(n_dofs + j, i) -
                    sigma * face_mass(i, n_dofs + j);
        M22(i, j) = 0.5 * face_flux(n_dofs + i, n_dofs + j) +
                    0.5 * face_flux(n_dofs + j, n_dofs + i) +
                    sigma * face_mass(n_dofs + i, n_dofs + j);
      }
  M21.copy_transposed(M12);
}



template <int dim>
class AdaptivePolytopalProblem
{
public:
  AdaptivePolytopalProblem(const ProblemParameters &parameters,
                           const MPI_Comm           comm);

  void
  run();

private:
  /**
   * The adaptive cycles of the Poisson problem, with the point singularity
   * or with f = 1.
   */
  void
  run_poisson();

  /**
   * The initial value and the time steps of the heat equation.
   */
  void
  run_heat();

  /**
   * The initial mesh, the unit square divided into the given number of cells
   * per direction or the refined mesh of the file, and the polytopes at the
   * start.
   */
  void
  make_fine_grid();

  /**
   * The handler on the p4est mesh or on the partitioned copy, and the data
   * of the last step on its cells.
   */
  void
  setup_agglomerated_problem();

  /**
   * Assemble the step of length @p dt, or with @p dt zero the stationary
   * problem with the source @p rhs_function.
   */
  void
  assemble_system(const Function<dim> &rhs_function, const double dt);

  /**
   * Conjugate gradients with algebraic multigrid, and the solution with the
   * degrees of freedom of the neighbouring polytopes on other ranks.
   */
  void
  solve();

  /**
   * The error indicators of the solution, sent to the p4est cells; for the
   * heat equation also the polynomials and the discrete elliptic operator of
   * the solution, and for a step the error indicator in time.
   */
  void
  estimate(const Function<dim> &rhs_function, const double dt);

  /**
   * Compute the initial value of the heat equation on the current mesh.
   */
  void
  solve_initial_value();

  /**
   * Refine and coarsen the mesh once, carrying the data of the heat
   * equation.
   */
  void
  adapt_mesh(const bool allow_refinement, const bool allow_coarsening);

  /**
   * The $L^2$ error against @p exact_solution.
   */
  double
  compute_l2_error(const Function<dim> &exact_solution) const;

  /**
   * Write the solution and the polytopes to adaptive_polytopal_NNNN.pvtu,
   * one piece per rank, and add it at @p at_time to the series
   * adaptive_polytopal.pvd. In ParaView, colour by polytope_colour to see the
   * polytopes.
   */
  void
  output_results(const unsigned int counter, const double at_time);

  const ProblemParameters &parameters;

  const MPI_Comm comm;

  /**
   * Output on rank 0 only.
   */
  ConditionalOStream pcout;

  /**
   * The p4est triangulation. It is never repartitioned, so that every cell
   * keeps its rank, as AdaptivePolytopalMesh requires, and
   * CellPolynomialTransfer finds the data of every cell on its rank.
   */
  parallel::distributed::Triangulation<dim> tria;

  /**
   * For a mesh read from a file: the bounding box of the domain and the
   * singular point of the point singularity.
   */
  BoundingBox<dim> domain;
  Point<dim>       singular_point;

  /**
   * The mapping of the cells, for GridTools::Cache.
   */
  MappingQ1<dim> mapping;

  /**
   * The discontinuous element of degree $p$ on the polytopes.
   */
  FE_AggloDGP<dim> fe_dg;

  /**
   * The penalty constant of the SIPG form, $C_\sigma (p + 1)(p + d)$ with
   * $C_\sigma = \sqrt{2}/3$, so that the penalty of a square of side $s$ at
   * $p = 4$ is $20\sqrt{2}/s$. On the polytopes that the R-trees in the rank
   * regions of METIS form, this is about 1.5 times the smallest constant that
   * keeps their diagonal blocks positive definite.
   */
  const double penalty_constant;

  /**
   * The polytopes and their adaptation.
   */
  PolytopeAdaptivity::AdaptivePolytopalMesh<dim> polytopal_mesh;

  /**
   * The copy of the mesh on which the system is assembled and solved, if
   * the parameters ask for it.
   */
  PolytopeRepartition::PartitionedCopy<dim> partitioned_copy;

  /**
   * Carries the solution of the last step and its operator through the
   * changes of the mesh.
   */
  TimeAdaptivity::CellPolynomialTransfer<dim> polynomial_transfer;

  /**
   * The quadrature-free assembly, if the parameters ask for it.
   */
  std::unique_ptr<QuadratureFree::PolytopeIntegrator<dim>> integrator;

  /**
   * The cache of the system mesh and the AgglomerationHandler on its
   * polytopes, both rebuilt for every solution.
   */
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<AgglomerationHandler<dim>> ah;

  /**
   * The system and its solution, with the locally owned degrees of freedom.
   */
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  system_rhs;
  TrilinosWrappers::MPI::Vector  solution;

  /**
   * The solution with the degrees of freedom of the neighbouring polytopes on
   * other ranks, see PolytopeAdaptivity::locally_relevant_polytope_dofs().
   */
  TrilinosWrappers::MPI::Vector locally_relevant_solution;

  /**
   * The solution of the last accepted step and its discrete elliptic
   * operator, as CellPolynomial data on the p4est cells, and the same for
   * the most recent solution.
   */
  std::vector<std::vector<double>> old_solution;
  std::vector<std::vector<double>> old_operator;
  std::vector<std::vector<double>> new_solution;
  std::vector<std::vector<double>> new_operator;

  /**
   * The old solution and operator on the cells of the system mesh, the
   * triangulation on which the system is assembled and solved: the p4est
   * triangulation, or the partitioned copy.
   */
  std::vector<std::vector<double>> old_solution_on_system_mesh;
  std::vector<std::vector<double>> old_operator_on_system_mesh;

  /**
   * The error indicators on the active cells of the system mesh: each of the
   * $n_P$ cells of a polytope $P$ holds $\eta_P / \sqrt{n_P}$.
   */
  Vector<float> indicators_on_system_mesh;

  /**
   * The source $f = u_t - \Delta u$ of the time steps of the heat equation,
   * and $-\Delta u(0)$ of its initial value.
   */
  MovingPeakSource<dim> heat_source;
  MovingPeakSource<dim> initial_source;

  /**
   * The error indicators, stored on the active cells of the p4est
   * triangulation: each of the $n_P$ cells of a polytope $P$ holds
   * $\eta_P / \sqrt{n_P}$.
   */
  Vector<float> cell_indicators;

  /**
   * The error estimator, the largest error indicator of a polytope, and the
   * error indicator in time of the most recent solution.
   */
  double error_estimator;
  double max_polytope_indicator;
  double time_indicator;

  /**
   * The tolerance of a polytope: the tolerance of the error estimator
   * shared by the current polytopes.
   */
  double polytope_tolerance;

  /**
   * The time of the last accepted step.
   */
  double time;

  /**
   * The time and the name of every output so far, for the .pvd series.
   */
  std::vector<std::pair<double, std::string>> times_and_names;
};



template <int dim>
AdaptivePolytopalProblem<dim>::AdaptivePolytopalProblem(
  const ProblemParameters &parameters,
  const MPI_Comm           communicator)
  : parameters(parameters)
  , comm(communicator)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
  , tria(comm,
         Triangulation<dim>::limit_level_difference_at_vertices,
         parallel::distributed::Triangulation<dim>::no_automatic_repartitioning)
  , fe_dg(parameters.degree)
  , penalty_constant(std::sqrt(2.) / 3. * (parameters.degree + 1) *
                     (parameters.degree + dim))
  , polytopal_mesh(
      tria,
      parameters.degree,
      [&]() {
        PolytopeAdaptivity::AdaptivityParameters data;
        data.max_cells_per_polytope = parameters.max_cells_per_polytope;
        data.coarsening_fraction    = parameters.coarsening_fraction;
        AssertThrow(parameters.level_rule == "multiple" ||
                      parameters.level_rule == "single",
                    ExcMessage("The level rule is 'multiple' or 'single'."));
        data.level_rule = (parameters.level_rule == "multiple") ?
                            PolytopeAdaptivity::LevelRule::multiple_levels :
                            PolytopeAdaptivity::LevelRule::single_level;
        AssertThrow(parameters.initial_partition == "p4est" ||
                      parameters.initial_partition == "metis",
                    ExcMessage("The initial partition is 'p4est' or "
                               "'metis'."));
        AssertThrow(parameters.initial_partition == "p4est" ||
                      parameters.repartition ||
                      Utilities::MPI::n_mpi_processes(communicator) == 1,
                    ExcMessage("With the initial partition 'metis' a "
                               "polytope may lie on several ranks of the "
                               "p4est mesh, so the system must be solved on "
                               "a partitioned copy."));
        data.initial_partition = (parameters.initial_partition == "metis") ?
                                   PolytopeAdaptivity::InitialPartition::metis :
                                   PolytopeAdaptivity::InitialPartition::p4est;
        return data;
      }())
  , partitioned_copy(tria)
  , polynomial_transfer(tria)
  , heat_source(false, parameters.n_peaks_per_direction)
  , initial_source(true, parameters.n_peaks_per_direction)
  , error_estimator(0.)
  , max_polytope_indicator(0.)
  , time_indicator(0.)
  , polytope_tolerance(-1.)
  , time(0.)
{
  AssertThrow(parameters.problem == "singular" ||
                parameters.problem == "constant" ||
                parameters.problem == "polynomial" ||
                parameters.problem == "heat",
              ExcMessage(
                "The problem is 'singular', 'constant', 'polynomial' or "
                "'heat'."));
  AssertThrow(parameters.mesh_file.empty() ||
                (dim == 2 && (parameters.problem == "singular" ||
                              parameters.problem == "constant")),
              ExcMessage("A mesh file is read only in two dimensions, for "
                         "'singular' and 'constant'."));
  if (parameters.quadrature_free)
    integrator = std::make_unique<QuadratureFree::PolytopeIntegrator<dim>>(
      fe_dg, parameters.problem == "heat");
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::make_fine_grid()
{
  if (parameters.mesh_file.empty())
    GridGenerator::subdivided_hyper_cube(tria,
                                         parameters.n_subdivisions,
                                         0.,
                                         1.);
  else
    {
      // The initial cells are the coarse cells of the p4est triangulation,
      // so the refined mesh of the file is flattened to its active cells.
      Triangulation<dim> file_tria;
      GridIn<dim>        grid_in(file_tria);
      std::ifstream      file(parameters.mesh_file);
      AssertThrow(file, ExcFileNotOpen(parameters.mesh_file));
      grid_in.read_msh(file);
      domain = GridTools::compute_bounding_box(file_tria);
      // The singular point is the vertex of the file nearest the centre of
      // the domain, so that no quadrature point lies on it and it stays a
      // vertex under refinement.
      singular_point = file_tria.get_vertices()[GridTools::find_closest_vertex(
        file_tria, domain.center())];
      file_tria.refine_global(parameters.n_mesh_file_refinements);
      GridGenerator::flatten_triangulation(file_tria, tria);
      pcout << "mesh " << parameters.mesh_file << ": " << tria.n_global_active_cells()
            << " initial cells in [" << domain.get_boundary_points().first
            << "] x [" << domain.get_boundary_points().second
            << "], singular point " << singular_point << std::endl;
    }
  polytopal_mesh.initialize();
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::setup_agglomerated_problem()
{
  ah.reset();
  cached_tria.reset();
  if (parameters.repartition)
    {
      partitioned_copy.reinit(polytopal_mesh.get_polytopes(),
                              polytopal_mesh.get_polytope_indices(),
                              polytopal_mesh.get_polytope_index_offsets(),
                              fe_dg.n_dofs_per_cell(),
                              parameters.use_parmetis ?
                                std::vector<unsigned int>() :
                                polytopal_mesh.get_tree_ranks());
      cached_tria = std::make_unique<GridTools::Cache<dim>>(
        partitioned_copy.get_triangulation(), mapping);
    }
  else
    cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  for (const auto &polytope : parameters.repartition ?
                                partitioned_copy.get_polytopes() :
                                polytopal_mesh.get_polytopes())
    ah->define_agglomerate(polytope);
  const unsigned int n_q_points = fe_dg.degree + 1;
  ah->initialize_fe_values(QGauss<dim>(n_q_points),
                           update_values | update_gradients |
                             update_JxW_values | update_quadrature_points,
                           QGauss<dim - 1>(n_q_points));
  ah->distribute_agglomerated_dofs(fe_dg);

  if (parameters.problem == "heat")
    {
      // Before the initial value there is no old solution yet.
      old_solution.resize(tria.n_active_cells());
      old_operator.resize(tria.n_active_cells());
      if (parameters.repartition)
        {
          partitioned_copy.p4est_to_copy(old_solution,
                                         old_solution_on_system_mesh);
          partitioned_copy.p4est_to_copy(old_operator,
                                         old_operator_on_system_mesh);
        }
      else
        {
          old_solution_on_system_mesh = old_solution;
          old_operator_on_system_mesh = old_operator;
        }
    }
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::assemble_system(
  const Function<dim> &rhs_function,
  const double         dt)
{
  TrilinosWrappers::SparsityPattern sparsity;
  ah->create_agglomeration_sparsity_pattern(sparsity);
  system_matrix.reinit(sparsity);
  system_rhs.reinit(ah->agglo_dh.locally_owned_dofs(), comm);

  // A step solves (M + dt A) U = M U_old + dt F: the bilinear form is scaled
  // by the step length.
  const bool   time_step = dt > 0.;
  const double scale     = time_step ? dt : 1.;

  // Without quadrature, the load vector of a polynomial right-hand side
  // comes from the moments as well.
  const auto *polynomial_rhs =
    dynamic_cast<const QuadratureFree::PolynomialFunction<dim> *>(
      &rhs_function);
  const bool polynomial_load = integrator && polynomial_rhs;
  std::vector<FullMatrix<double>> stiffness, mass;
  std::vector<Vector<double>>     load;
  if (integrator)
    integrator->volume_matrices(*ah,
                                stiffness,
                                time_step ? &mass : nullptr,
                                polynomial_load ? polynomial_rhs : nullptr,
                                polynomial_load ? &load : nullptr);

  const unsigned int n_dofs = fe_dg.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(n_dofs, n_dofs);
  Vector<double>     cell_rhs(n_dofs);
  FullMatrix<double> M11(n_dofs, n_dofs), M12(n_dofs, n_dofs),
    M21(n_dofs, n_dofs), M22(n_dofs, n_dofs);
  FullMatrix<double>                   face_mass, face_flux;
  std::vector<types::global_dof_index> local_dof_indices(n_dofs);
  std::vector<types::global_dof_index> neighbor_dof_indices(n_dofs);
  std::vector<double>                  rhs_values, weights;
  std::vector<unsigned int>            cell_of_point;

  for (const auto &polytope : ah->polytope_iterators())
    {
      if (!polytope->is_locally_owned())
        continue;
      cell_matrix = 0.;
      cell_rhs    = 0.;
      if (polynomial_load)
        cell_rhs = load[polytope->index()];
      else
        {
          const auto &agglo_values = ah->reinit(polytope);
          const auto &points       = agglo_values.get_quadrature_points();
          rhs_values.resize(points.size());
          rhs_function.value_list(points, rhs_values);

          // The quadrature points of the polytope come cell by cell, in the
          // order of get_agglomerate(); the old solution is a polynomial on
          // every cell.
          cell_of_point.clear();
          if (time_step)
            {
              const auto        &cells    = polytope->get_agglomerate();
              const unsigned int per_cell = points.size() / cells.size();
              for (const auto &cell : cells)
                cell_of_point.insert(cell_of_point.end(),
                                     per_cell,
                                     cell->active_cell_index());
            }

          for (const unsigned int q : agglo_values.quadrature_point_indices())
            {
              const double old_value =
                time_step ?
                  TimeAdaptivity::CellPolynomial::evaluate<dim>(
                    old_solution_on_system_mesh[cell_of_point[q]], points[q]) :
                  0.;
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  if (!integrator)
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      {
                        cell_matrix(i, j) +=
                          scale * agglo_values.shape_grad(i, q) *
                          agglo_values.shape_grad(j, q) * agglo_values.JxW(q);
                        if (time_step)
                          cell_matrix(i, j) += agglo_values.shape_value(i, q) *
                                               agglo_values.shape_value(j, q) *
                                               agglo_values.JxW(q);
                      }
                  cell_rhs(i) += agglo_values.shape_value(i, q) *
                                 (scale * rhs_values[q] + old_value) *
                                 agglo_values.JxW(q);
                }
            }
        }
      if (integrator)
        {
          cell_matrix = stiffness[polytope->index()];
          cell_matrix *= scale;
          if (time_step)
            cell_matrix.add(1., mass[polytope->index()]);
        }

      polytope->get_dof_indices(local_dof_indices);

      for (unsigned int f = 0; f < polytope->n_faces(); ++f)
        {
          if (polytope->at_boundary(f))
            {
              // Nitsche's method for u = 0, with the penalty of
              // PolytopeAdaptivity::boundary_penalty().
              if (integrator)
                {
                  integrator->face_matrices(
                    *ah, polytope, f, face_mass, face_flux);
                  const double sigma =
                    PolytopeAdaptivity::boundary_penalty<dim>(
                      penalty_constant, polytope->get_bounding_box());
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      cell_matrix(i, j) +=
                        scale * (-face_flux(i, j) - face_flux(j, i) +
                                 sigma * face_mass(i, j));
                  continue;
                }
              const auto  &fe_face = ah->reinit(polytope, f);
              const auto  &normals = fe_face.get_normal_vectors();
              const double sigma   = PolytopeAdaptivity::boundary_penalty<dim>(
                penalty_constant, polytope->get_bounding_box());
              for (const unsigned int q : fe_face.quadrature_point_indices())
                for (unsigned int i = 0; i < n_dofs; ++i)
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    cell_matrix(i, j) +=
                      scale *
                      (-fe_face.shape_value(i, q) * fe_face.shape_grad(j, q) *
                         normals[q] -
                       fe_face.shape_grad(i, q) * normals[q] *
                         fe_face.shape_value(j, q) +
                       sigma * fe_face.shape_value(i, q) *
                         fe_face.shape_value(j, q)) *
                      fe_face.JxW(q);
              continue;
            }

          const auto &neighbor = polytope->neighbor(f);
          if (!(polytope->id() < neighbor->id()))
            continue;
          const unsigned int neighbor_face =
            polytope->neighbor_of_agglomerated_neighbor(f);
          neighbor->get_dof_indices(neighbor_dof_indices);
          const double sigma = PolytopeAdaptivity::interior_penalty<dim>(
            penalty_constant,
            polytope->get_bounding_box(),
            neighbor->get_bounding_box());
          M11 = 0.;
          M12 = 0.;
          M21 = 0.;
          M22 = 0.;
          if (integrator)
            {
              integrator->face_matrices(*ah, polytope, f, face_mass, face_flux);
              interface_blocks(face_mass, face_flux, sigma, M11, M12, M21, M22);
              M11 *= scale;
              M12 *= scale;
              M21 *= scale;
              M22 *= scale;
            }
          else
            {
              const auto &fe_faces =
                ah->reinit_interface(polytope, neighbor, f, neighbor_face);
              const auto &fe_faces0 = fe_faces.first;
              if (neighbor->is_locally_owned())
                {
                  const auto &fe_faces1 = fe_faces.second;
                  weights.resize(fe_faces0.n_quadrature_points);
                  for (const unsigned int q :
                       fe_faces0.quadrature_point_indices())
                    weights[q] = scale * fe_faces1.JxW(q);
                  assemble_interface_terms<dim>(
                    M11,
                    M12,
                    M21,
                    M22,
                    fe_faces0,
                    [&](const unsigned int j, const unsigned int q) {
                      return fe_faces1.shape_value(j, q);
                    },
                    [&](const unsigned int j, const unsigned int q) {
                      return fe_faces1.shape_grad(j, q);
                    },
                    weights,
                    sigma);
                }
              else
                {
                  // The neighbour's rank sent its shape values, gradients and
                  // weights on this face.
                  const types::subdomain_id rank = neighbor->subdomain_id();
                  const auto &recv_values        = ah->recv_values.at(rank).at(
                    {neighbor->id(), neighbor_face});
                  const auto &recv_gradients = ah->recv_gradients.at(rank).at(
                    {neighbor->id(), neighbor_face});
                  const auto &recv_jxws =
                    ah->recv_jxws.at(rank).at({neighbor->id(), neighbor_face});
                  weights.resize(recv_jxws.size());
                  for (unsigned int q = 0; q < recv_jxws.size(); ++q)
                    weights[q] = scale * recv_jxws[q];
                  assemble_interface_terms<dim>(
                    M11,
                    M12,
                    M21,
                    M22,
                    fe_faces0,
                    [&](const unsigned int j, const unsigned int q) {
                      return recv_values[j][q];
                    },
                    [&](const unsigned int j, const unsigned int q) {
                      return recv_gradients[j][q];
                    },
                    weights,
                    sigma);
                }
            }
          system_matrix.add(local_dof_indices, M11);
          system_matrix.add(local_dof_indices, neighbor_dof_indices, M12);
          system_matrix.add(neighbor_dof_indices, local_dof_indices, M21);
          system_matrix.add(neighbor_dof_indices, M22);
        }
      system_matrix.add(local_dof_indices, cell_matrix);
      system_rhs.add(local_dof_indices, cell_rhs);
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::solve()
{
  // Algebraic multigrid with one smoother block per polytope.
  TrilinosWrappers::PreconditionAMG amg;
  PolytopeAdaptivity::initialize_polytope_amg(amg, system_matrix, *ah);
  solution.reinit(ah->agglo_dh.locally_owned_dofs(), comm);
  ReductionControl                        solver_control(10000, 1e-13, 1e-7);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, amg);

  locally_relevant_solution.reinit(
    ah->agglo_dh.locally_owned_dofs(),
    PolytopeAdaptivity::locally_relevant_polytope_dofs(*ah),
    comm);
  locally_relevant_solution = solution;
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::estimate(const Function<dim> &rhs_function,
                                        const double         dt)
{
  const bool time_step = dt > 0.;
  const std::function<
    double(const typename Triangulation<dim>::active_cell_iterator &,
           const Point<dim> &,
           const double)>
    time_derivative =
      [&](const typename Triangulation<dim>::active_cell_iterator &cell,
          const Point<dim>                                        &point,
          const double                                             u_h) {
        return -(u_h - TimeAdaptivity::CellPolynomial::evaluate<dim>(
                         old_solution_on_system_mesh[cell->active_cell_index()],
                         point)) /
               dt;
      };
  PolytopeAdaptivity::compute_residual_indicators(
    *ah,
    locally_relevant_solution,
    rhs_function,
    penalty_constant,
    indicators_on_system_mesh,
    time_step ? time_derivative : decltype(time_derivative)());

  double sum_sq = 0., largest = 0.;
  for (const auto &polytope : ah->polytope_iterators())
    if (polytope->is_locally_owned())
      {
        double polytope_sq = 0.;
        for (const auto &cell : polytope->get_agglomerate())
          polytope_sq += indicators_on_system_mesh[cell->active_cell_index()] *
                         indicators_on_system_mesh[cell->active_cell_index()];
        sum_sq += polytope_sq;
        largest = std::max(largest, std::sqrt(polytope_sq));
      }
  error_estimator        = std::sqrt(Utilities::MPI::sum(sum_sq, comm));
  max_polytope_indicator = Utilities::MPI::max(largest, comm);
  if (parameters.repartition)
    partitioned_copy.copy_to_p4est(indicators_on_system_mesh, cell_indicators);
  else
    cell_indicators = indicators_on_system_mesh;

  if (parameters.problem != "heat")
    return;
  std::vector<std::vector<double>> solution_on_system_mesh,
    operator_on_system_mesh;
  TimeAdaptivity::polytope_polynomials(*ah, solution, solution_on_system_mesh);
  TimeAdaptivity::compute_discrete_operator(*ah,
                                            solution,
                                            rhs_function,
                                            time_step ?
                                              &old_solution_on_system_mesh :
                                              nullptr,
                                            dt,
                                            operator_on_system_mesh);
  time_indicator =
    time_step ?
      TimeAdaptivity::compute_time_indicator(*ah,
                                             heat_source,
                                             time,
                                             dt,
                                             operator_on_system_mesh,
                                             old_operator_on_system_mesh,
                                             parameters.final_time) :
      0.;
  if (parameters.repartition)
    {
      partitioned_copy.copy_to_p4est(solution_on_system_mesh, new_solution);
      partitioned_copy.copy_to_p4est(operator_on_system_mesh, new_operator);
    }
  else
    {
      new_solution = solution_on_system_mesh;
      new_operator = operator_on_system_mesh;
    }
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::solve_initial_value()
{
  setup_agglomerated_problem();
  assemble_system(initial_source, 0.);
  solve();
  estimate(initial_source, 0.);
  old_solution = new_solution;
  old_operator = new_operator;
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::adapt_mesh(const bool allow_refinement,
                                          const bool allow_coarsening)
{
  // The handler reinitializes itself when its triangulation changes, so it
  // goes before the mesh does.
  ah.reset();
  cached_tria.reset();
  const bool heat    = parameters.problem == "heat";
  const bool adapted = polytopal_mesh.refine_and_coarsen(
    cell_indicators,
    polytope_tolerance,
    [&]() {
      if (heat)
        polynomial_transfer.prepare_for_coarsening_and_refinement(
          {&old_solution, &old_operator});
    },
    [&]() {
      if (heat)
        polynomial_transfer.unpack();
    },
    allow_refinement,
    allow_coarsening);
  AssertThrow(adapted,
              ExcMessage("The finest refinement level of p4est is reached."));
}



template <int dim>
double
AdaptivePolytopalProblem<dim>::compute_l2_error(
  const Function<dim> &exact_solution) const
{
  std::vector<double> u_values;
  double              error_sq = 0.;
  for (const auto &polytope : ah->polytope_iterators())
    {
      if (!polytope->is_locally_owned())
        continue;
      const auto &fe_values = ah->reinit(polytope);
      const auto &points    = fe_values.get_quadrature_points();
      u_values.resize(points.size());
      fe_values.get_function_values(solution, u_values);
      for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          const double error = u_values[q] - exact_solution.value(points[q]);
          error_sq += error * error * fe_values.JxW(q);
        }
    }
  return std::sqrt(Utilities::MPI::sum(error_sq, comm));
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::output_results(const unsigned int counter,
                                              const double       at_time)
{
  // The solution, interpolated into the discontinuous elements of the same
  // degree on the cells.
  TrilinosWrappers::MPI::Vector interpolated_solution;
  PolyUtils::interpolate_to_fine_grid(*ah, interpolated_solution, solution);

  // For every cell of the mesh of the system: the global polytope index of
  // its polytope, a colour that differs between neighbouring polytopes, the
  // rank, the refinement level, and the error indicator of its polytope.
  const unsigned int n_cells = ah->get_triangulation().n_active_cells();
  Vector<float>      polytope_index(n_cells), polytope_colour(n_cells),
    rank(n_cells), level(n_cells), indicator(n_cells);
  const unsigned int my_rank           = Utilities::MPI::this_mpi_process(comm);
  std::uint64_t      n_local_polytopes = 0;
  for (const auto &polytope : ah->polytope_iterators())
    if (polytope->is_locally_owned())
      ++n_local_polytopes;
  std::uint64_t global_index = 0;
  {
    const auto counts = Utilities::MPI::all_gather(comm, n_local_polytopes);
    for (unsigned int r = 0; r < my_rank; ++r)
      global_index += counts[r];
  }
  for (const auto &polytope : ah->polytope_iterators())
    if (polytope->is_locally_owned())
      {
        const auto &cells        = polytope->get_agglomerate();
        double      indicator_sq = 0.;
        for (const auto &cell : cells)
          indicator_sq += indicators_on_system_mesh[cell->active_cell_index()] *
                          indicators_on_system_mesh[cell->active_cell_index()];
        for (const auto &cell : cells)
          {
            const unsigned int index = cell->active_cell_index();
            polytope_index[index]    = global_index;
            // Multiples of the golden ratio spread consecutive indices
            // evenly over [0, 1).
            polytope_colour[index] = std::fmod(global_index * 0.6180339887, 1.);
            rank[index]            = my_rank;
            level[index]           = cell->level();
            indicator[index]       = std::sqrt(indicator_sq);
          }
        ++global_index;
      }

  DataOut<dim> data_out;
  data_out.attach_dof_handler(ah->output_dh);
  data_out.add_data_vector(interpolated_solution, "u");
  data_out.add_data_vector(polytope_index,
                           "polytope",
                           DataOut<dim>::type_cell_data);
  data_out.add_data_vector(polytope_colour,
                           "polytope_colour",
                           DataOut<dim>::type_cell_data);
  data_out.add_data_vector(rank, "rank", DataOut<dim>::type_cell_data);
  data_out.add_data_vector(level, "level", DataOut<dim>::type_cell_data);
  data_out.add_data_vector(indicator,
                           "error_indicator",
                           DataOut<dim>::type_cell_data);
  data_out.build_patches(mapping, ah->get_fe().degree);
  const std::string pvtu = data_out.write_vtu_with_pvtu_record(
    "./", "adaptive_polytopal", counter, comm, 4);

  // The record of all outputs so far, which ParaView opens as one series.
  if (my_rank == 0)
    {
      times_and_names.emplace_back(at_time, pvtu);
      std::ofstream pvd("adaptive_polytopal.pvd");
      DataOutBase::write_pvd_record(pvd, times_and_names);
    }
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::run_poisson()
{
  const bool constant   = parameters.problem == "constant";
  const bool polynomial = parameters.problem == "polynomial";
  std::array<unsigned int, dim> zero_exponents;
  zero_exponents.fill(0);
  const QuadratureFree::PolynomialFunction<dim> constant_source({zero_exponents},
                                                                {1.});
  const QuadratureFree::PolynomialFunction<dim> polynomial_source =
    polynomial_function<dim>(true);
  const QuadratureFree::PolynomialFunction<dim> polynomial_solution =
    polynomial_function<dim>(false);
  const SingularSolution<dim> singular_solution =
    parameters.mesh_file.empty() ?
      SingularSolution<dim>() :
      SingularSolution<dim>(domain, singular_point);
  const SingularSource<dim> singular_source(singular_solution);
  const Function<dim>        &source =
    constant   ? static_cast<const Function<dim> &>(constant_source) :
    polynomial ? static_cast<const Function<dim> &>(polynomial_source) :
                 static_cast<const Function<dim> &>(singular_source);
  const Function<dim> &exact_solution =
    polynomial ? static_cast<const Function<dim> &>(polynomial_solution) :
                 static_cast<const Function<dim> &>(singular_solution);
  pcout << "cycle  polytopes      dofs   estimator       L2 error        "
           "time [s]"
        << std::endl;
  Timer timer;
  for (unsigned int cycle = 0; cycle < parameters.n_cycles; ++cycle)
    {
      setup_agglomerated_problem();
      assemble_system(source, 0.);
      solve();
      // The error before the estimate, which initializes the handler's
      // FEValues with more quadrature points. f = 1 has no exact solution.
      std::ostringstream l2_error;
      if (constant)
        l2_error << "-";
      else
        l2_error << compute_l2_error(exact_solution);
      estimate(source, 0.);
      pcout << std::setw(5) << cycle << std::setw(11)
            << polytopal_mesh.n_global_polytopes() << std::setw(10)
            << ah->n_dofs() << "   " << std::left << std::setw(16)
            << error_estimator << std::setw(16) << l2_error.str() << std::right
            << Utilities::MPI::max(timer.wall_time(), comm) << std::endl;
      // Uncomment to write the solution and the polytopes for ParaView,
      // outside the timing.
      // timer.stop();
      // output_results(cycle, cycle);
      // timer.start();
      if (error_estimator <= parameters.tolerance ||
          cycle + 1 == parameters.n_cycles)
        break;
      polytope_tolerance =
        polytopal_mesh.polytope_tolerance(parameters.tolerance);
      adapt_mesh(true, true);
    }
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::run_heat()
{
  initial_source.set_time(0.);
  for (unsigned int adaptation = 0;; ++adaptation)
    {
      solve_initial_value();
      pcout << "initial value: polytopes "
            << polytopal_mesh.n_global_polytopes() << ", dofs " << ah->n_dofs()
            << ", estimator " << error_estimator << std::endl;
      if (adaptation == parameters.n_initial_adaptations)
        break;
      polytope_tolerance =
        polytopal_mesh.polytope_tolerance(parameters.space_tolerance);
      adapt_mesh(true, true);
    }
  // From the first step on the polytope tolerance is a constant: its value
  // on the mesh of the initial value.
  polytope_tolerance =
    polytopal_mesh.polytope_tolerance(parameters.space_tolerance);

  TimeAdaptivity::StepControlParameters step_parameters;
  step_parameters.initial_dt             = parameters.initial_dt;
  step_parameters.max_dt                 = parameters.max_dt;
  step_parameters.min_dt                 = parameters.initial_dt / 1024.;
  step_parameters.time_tolerance         = parameters.time_tolerance;
  step_parameters.time_doubling_fraction = parameters.time_doubling_fraction;
  step_parameters.max_first_step_adaptations =
    parameters.max_first_step_adaptations;
  TimeAdaptivity::StepController controller(step_parameters);
  // Uncomment these two lines and the call in the accepted step below to
  // write the solution and the polytopes of every accepted step for
  // ParaView.
  // output_results(0, 0.);
  // unsigned int n_steps = 0;

  pcout << "      t        dt  polytopes      dofs   L2 error" << std::endl;
  MovingPeak<dim> exact_solution(parameters.n_peaks_per_direction);
  while (time < parameters.final_time - 1e-12)
    {
      const double dt = controller.step_length();
      heat_source.set_time(time + dt);
      setup_agglomerated_problem();
      assemble_system(heat_source, dt);
      solve();
      // The error before the estimate, which initializes the handler's
      // FEValues with more quadrature points.
      exact_solution.set_time(time + dt);
      const double l2_error = compute_l2_error(exact_solution);
      estimate(heat_source, dt);

      switch (controller.decide(max_polytope_indicator,
                                polytope_tolerance,
                                time_indicator))
        {
          case TimeAdaptivity::StepDecision::solve_again:
            break;
          case TimeAdaptivity::StepDecision::adapt_and_solve_again:
            adapt_mesh(true, true);
            if (!controller.first_step_done())
              solve_initial_value();
            break;
          case TimeAdaptivity::StepDecision::merge_and_restart:
            adapt_mesh(false, true);
            solve_initial_value();
            break;
          case TimeAdaptivity::StepDecision::accept:
            time += dt;
            pcout << std::setw(7) << time << std::setw(10) << dt
                  << std::setw(11) << polytopal_mesh.n_global_polytopes()
                  << std::setw(10) << ah->n_dofs() << "   " << l2_error
                  << std::endl;
            old_solution = new_solution;
            old_operator = new_operator;
            // output_results(++n_steps, time);
            break;
        }
    }
}



template <int dim>
void
AdaptivePolytopalProblem<dim>::run()
{
  make_fine_grid();
  pcout << "Running with " << Utilities::MPI::n_mpi_processes(comm)
        << " MPI ranks." << std::endl;
  if (parameters.problem == "heat")
    run_heat();
  else
    run_poisson();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const std::string                parameter_file =
    (argc > 1) ? argv[1] : "adaptive_polytopal.prm";
  ProblemParameters parameters;
  ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
  if (parameters.dimension == 2)
    {
      AdaptivePolytopalProblem<2> problem(parameters, MPI_COMM_WORLD);
      problem.run();
    }
  else
    {
      AdaptivePolytopalProblem<3> problem(parameters, MPI_COMM_WORLD);
      problem.run();
    }
}
