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


// The adaptive Poisson problem of adaptive_poisson.cc, with the singular point
// moved to (1/4, 1/4), solved on a copy of the mesh partitioned by polytopes.
//
// The p4est triangulation is never repartitioned, so that every polytope stays
// on one rank, and refinement towards the singular point piles up the degrees
// of freedom on the rank that owns it. The singular point is off the centre of
// the square: at the centre, the space-filling curve of p4est splits the
// refined region evenly between 2 or 4 ranks, and nothing piles up.
//
// Every cycle the mesh is copied into a
// parallel::fullydistributed::Triangulation partitioned by ParMETIS on the
// graph of the polytopes (PolytopeRepartition::PartitionedCopy). The system is
// assembled and solved and the error is estimated there, and the error
// indicators are sent back to the p4est cells for the next adaptation. The
// program prints the largest over mean number of degrees of freedom per rank
// on both meshes.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
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
#include <ml_MultiLevelPreconditioner.h>
#include <poly_utils.h>
#include <polytope_adaptivity.h>
#include <polytope_repartition.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>

using namespace dealii;



/**
 * The exact solution $u = \sin(\pi x) \sin(\pi y) r^{2/3}$ and its
 * derivatives, $r$ the distance to the singular point (1/4, 1/4). The singular
 * point is a vertex of the mesh, so no quadrature point lies on it.
 */
template <int dim>
class SingularSolution : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;

  /**
   * Value, gradient and Laplacian at @p p.
   */
  void
  evaluate(const Point<dim> &p,
           double           &u,
           Tensor<1, dim>   &grad,
           double           &laplacian) const;

private:
  static constexpr double alpha = 2. / 3.;
};



template <int dim>
void
SingularSolution<dim>::evaluate(const Point<dim> &p,
                                double           &u,
                                Tensor<1, dim>   &grad,
                                double           &laplacian) const
{
  // The window sin(pi x) sin(pi y), which vanishes on the boundary.
  double window = 1.;
  for (unsigned int direction = 0; direction < dim; ++direction)
    window *= std::sin(numbers::PI * p[direction]);
  Tensor<1, dim> grad_window;
  for (unsigned int direction = 0; direction < dim; ++direction)
    {
      grad_window[direction] =
        numbers::PI * std::cos(numbers::PI * p[direction]);
      for (unsigned int other_direction = 0; other_direction < dim;
           ++other_direction)
        if (other_direction != direction)
          grad_window[direction] *= std::sin(numbers::PI * p[other_direction]);
    }
  const double laplacian_window = -dim * numbers::PI * numbers::PI * window;

  Point<dim> centre;
  for (unsigned int direction = 0; direction < dim; ++direction)
    centre[direction] = 0.25;
  const Tensor<1, dim> offset = p - centre;
  const double         r_sq   = offset.norm_square();
  // The singular point itself: u vanishes there, and the gradient and the
  // Laplacian, which are infinite, are set to zero. No quadrature point
  // lies on it, since it is a vertex of the mesh.
  if (r_sq == 0.)
    {
      u         = 0.;
      grad      = Tensor<1, dim>();
      laplacian = 0.;
      return;
    }
  const double r             = std::sqrt(r_sq);
  const double r_alpha       = std::pow(r, alpha);
  const double r_alpha_minus = std::pow(r, alpha - 2.);

  u         = window * r_alpha;
  grad      = r_alpha * grad_window + window * alpha * r_alpha_minus * offset;
  laplacian = laplacian_window * r_alpha +
              2. * alpha * r_alpha_minus * (grad_window * offset) +
              window * alpha * (alpha + dim - 2) * r_alpha_minus;
}



template <int dim>
double
SingularSolution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double         u, laplacian;
  Tensor<1, dim> grad;
  evaluate(p, u, grad, laplacian);
  return u;
}



template <int dim>
Tensor<1, dim>
SingularSolution<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  double         u, laplacian;
  Tensor<1, dim> grad;
  evaluate(p, u, grad, laplacian);
  return grad;
}



/**
 * The right-hand side $f = -\Delta u$.
 */
template <int dim>
class SingularRightHandSide : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
  const SingularSolution<dim> solution;
};



template <int dim>
double
SingularRightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double         u, laplacian;
  Tensor<1, dim> grad;
  solution.evaluate(p, u, grad, laplacian);
  return -laplacian;
}



/**
 * Add the SIPG interface terms of one face to the four blocks: @p M11 couples
 * the basis functions of this polytope with themselves, @p M12 this polytope
 * with the neighbour, @p M21 the neighbour with this polytope and @p M22 the
 * neighbour with itself. The values and gradients on the neighbour's side are
 * given by @p neighbor_value and @p neighbor_gradient, the quadrature weights
 * by @p weights. In the loop, v0 and v1 are values and g0 and g1 normal
 * derivatives on this side (0) and the neighbour's (1), of basis function i or
 * j.
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
    {
      const Tensor<1, dim> &normal = normals[q];
      for (unsigned int i = 0; i < n_dofs; ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            const double v0_i = fe_faces0.shape_value(i, q);
            const double v0_j = fe_faces0.shape_value(j, q);
            const double v1_i = neighbor_value(i, q);
            const double v1_j = neighbor_value(j, q);
            const double g0_i = fe_faces0.shape_grad(i, q) * normal;
            const double g0_j = fe_faces0.shape_grad(j, q) * normal;
            const double g1_i = neighbor_gradient(i, q) * normal;
            const double g1_j = neighbor_gradient(j, q) * normal;
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
}



template <int dim>
class RepartitionedPoissonProblem
{
public:
  RepartitionedPoissonProblem(
    const unsigned int                              n_subdivisions,
    const unsigned int                              degree,
    const unsigned int                              n_cycles,
    const double                                    relative_tolerance,
    const PolytopeAdaptivity::AdaptivityParameters &parameters,
    const MPI_Comm                                  comm);

  void
  run();

private:
  /**
   * The initial mesh, the unit square divided into n_subdivisions cells per
   * direction, and the polytopes at the start.
   */
  void
  make_fine_grid();

  /**
   * The AgglomerationHandler on the current polytopes, with its degrees of
   * freedom. It is rebuilt in every cycle.
   */
  void
  setup_agglomerated_problem();

  /**
   * The SIPG system on the polytopes.
   */
  void
  assemble_system();

  /**
   * Conjugate gradients with algebraic multigrid, and the solution with the
   * degrees of freedom of the neighbouring polytopes on other ranks.
   */
  void
  solve();

  /**
   * The errors against the exact solution, and the error indicators.
   */
  void
  estimate_and_compute_errors();

  /**
   * Write the solution and the polytopes to repartitioned_poisson_NNNN.pvtu,
   * one piece per rank, and add it at @p at_time to the series
   * repartitioned_poisson.pvd. In ParaView, colour by polytope_colour to see
   * the polytopes.
   */
  void
  output_results(const unsigned int counter, const double at_time);

  const MPI_Comm comm;

  /**
   * Cells per direction of the initial mesh.
   */
  const unsigned int n_subdivisions;

  /**
   * Number of cycles of solving, estimating and adapting.
   */
  const unsigned int n_cycles;

  /**
   * The tolerance of the error estimator as a fraction of the error estimator
   * of the first cycle.
   */
  const double relative_tolerance;

  /**
   * Output on rank 0 only.
   */
  ConditionalOStream pcout;

  /**
   * The p4est triangulation. It is never repartitioned, so that every polytope
   * stays on one rank.
   */
  parallel::distributed::Triangulation<dim> tria;

  /**
   * The mapping of the cells, for GridTools::Cache.
   */
  MappingQ1<dim> mapping;

  /**
   * The discontinuous element of degree $p$ on the polytopes.
   */
  FE_AggloDGP<dim> fe_dg;

  /**
   * The penalty constant of the SIPG form, $2 p (p + 1)$.
   */
  const double penalty_constant;

  /**
   * The polytopes and their adaptation.
   */
  PolytopeAdaptivity::AdaptivePolytopalMesh<dim> polytopal_mesh;

  /**
   * The copy of the mesh on which the system is assembled and solved, rebuilt
   * in every cycle.
   */
  PolytopeRepartition::PartitionedCopy<dim> partitioned_copy;

  /**
   * The cache of the copy and the AgglomerationHandler on the polytopes of
   * the copy, both rebuilt in every cycle.
   */
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<AgglomerationHandler<dim>> ah;

  /**
   * The SIPG system and its solution, with the locally owned degrees of
   * freedom.
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
   * The exact solution and the right-hand side $f = -\Delta u$.
   */
  SingularSolution<dim>      exact_solution;
  SingularRightHandSide<dim> rhs_function;

  /**
   * The error indicators on the active cells of the p4est triangulation and
   * of the copy: each of the $n_P$ cells of a polytope $P$ holds
   * $\eta_P / \sqrt{n_P}$.
   */
  Vector<float> cell_indicators;
  Vector<float> copy_indicators;

  /**
   * Largest over mean number of degrees of freedom per rank, had the system
   * been solved on the p4est triangulation.
   */
  double p4est_imbalance;

  /**
   * The error estimator, and its tolerance, a fraction of the error
   * estimator of the first cycle.
   */
  double error_estimator;
  double tolerance;

  /**
   * The errors against the exact solution.
   */
  double l2_error;
  double h1_error;

  /**
   * The time and the name of every output so far, for the .pvd series.
   */
  std::vector<std::pair<double, std::string>> times_and_names;
};



template <int dim>
RepartitionedPoissonProblem<dim>::RepartitionedPoissonProblem(
  const unsigned int                              n_subdivisions,
  const unsigned int                              degree,
  const unsigned int                              n_cycles,
  const double                                    relative_tolerance,
  const PolytopeAdaptivity::AdaptivityParameters &parameters,
  const MPI_Comm                                  communicator)
  : comm(communicator)
  , n_subdivisions(n_subdivisions)
  , n_cycles(n_cycles)
  , relative_tolerance(relative_tolerance)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
  , tria(comm,
         Triangulation<dim>::limit_level_difference_at_vertices,
         parallel::distributed::Triangulation<dim>::no_automatic_repartitioning)
  , fe_dg(degree)
  , penalty_constant(2. * degree * (degree + 1))
  , polytopal_mesh(tria, degree, parameters)
  , partitioned_copy(tria)
  , p4est_imbalance(1.)
  , error_estimator(0.)
  , tolerance(-1.)
  , l2_error(0.)
  , h1_error(0.)
{}



template <int dim>
void
RepartitionedPoissonProblem<dim>::make_fine_grid()
{
  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions, 0., 1.);
  polytopal_mesh.initialize();
}



template <int dim>
void
RepartitionedPoissonProblem<dim>::setup_agglomerated_problem()
{
  // The imbalance of the degrees of freedom per rank, had the system been
  // solved on the p4est triangulation.
  const double local_dofs =
    double(polytopal_mesh.get_polytopes().size()) * fe_dg.n_dofs_per_cell();
  p4est_imbalance = Utilities::MPI::max(local_dofs, comm) /
                    (Utilities::MPI::sum(local_dofs, comm) /
                     Utilities::MPI::n_mpi_processes(comm));

  ah.reset();
  cached_tria.reset();
  partitioned_copy.reinit(polytopal_mesh.get_polytopes(),
                          fe_dg.n_dofs_per_cell());
  cached_tria = std::make_unique<GridTools::Cache<dim>>(
    partitioned_copy.get_triangulation(), mapping);
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  for (const auto &polytope : partitioned_copy.get_polytopes())
    ah->define_agglomerate(polytope);
  const unsigned int n_q_points = 2 * fe_dg.degree + 1;
  ah->initialize_fe_values(QGauss<dim>(n_q_points),
                           update_values | update_gradients |
                             update_JxW_values | update_quadrature_points,
                           QGauss<dim - 1>(n_q_points));
  ah->distribute_agglomerated_dofs(fe_dg);
}



template <int dim>
void
RepartitionedPoissonProblem<dim>::assemble_system()
{
  TrilinosWrappers::SparsityPattern sparsity;
  ah->create_agglomeration_sparsity_pattern(sparsity);
  system_matrix.reinit(sparsity);
  const IndexSet &locally_owned_dofs = ah->agglo_dh.locally_owned_dofs();
  system_rhs.reinit(locally_owned_dofs, comm);

  const unsigned int n_dofs = fe_dg.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(n_dofs, n_dofs);
  Vector<double>     cell_rhs(n_dofs);
  FullMatrix<double> M11(n_dofs, n_dofs), M12(n_dofs, n_dofs),
    M21(n_dofs, n_dofs), M22(n_dofs, n_dofs);
  std::vector<types::global_dof_index> local_dof_indices(n_dofs);
  std::vector<types::global_dof_index> neighbor_dof_indices(n_dofs);
  std::vector<double>                  rhs_values;
  std::vector<double>                  weights;

  for (const auto &polytope : ah->polytope_iterators())
    {
      if (!polytope->is_locally_owned())
        continue;
      cell_matrix              = 0.;
      cell_rhs                 = 0.;
      const auto &agglo_values = ah->reinit(polytope);
      rhs_values.resize(agglo_values.n_quadrature_points);
      rhs_function.value_list(agglo_values.get_quadrature_points(), rhs_values);
      for (const unsigned int q : agglo_values.quadrature_point_indices())
        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            for (unsigned int j = 0; j < n_dofs; ++j)
              cell_matrix(i, j) += agglo_values.shape_grad(i, q) *
                                   agglo_values.shape_grad(j, q) *
                                   agglo_values.JxW(q);
            cell_rhs(i) += agglo_values.shape_value(i, q) * rhs_values[q] *
                           agglo_values.JxW(q);
          }

      // The measure as the sum of the quadrature weights: measure() of a
      // cell loses all digits on cells much smaller than their distance to
      // the origin.
      double measure = 0.;
      for (const unsigned int q : agglo_values.quadrature_point_indices())
        measure += agglo_values.JxW(q);
      const double h = polytope->diameter();
      polytope->get_dof_indices(local_dof_indices);

      for (unsigned int f = 0; f < polytope->n_faces(); ++f)
        {
          if (polytope->at_boundary(f))
            {
              // Nitsche's method for u = 0. The penalty must dominate the two
              // consistency terms, which grow like |F|/|P|, the length of the
              // boundary face over the area of the polytope. The usual 1/h
              // assumes |F|/|P| ~ 1/h, which fails for a long polytope along
              // the boundary, so the penalty uses max(1/h, |F|/|P|) to keep the
              // matrix positive definite.
              const auto &fe_face      = ah->reinit(polytope, f);
              const auto &normals      = fe_face.get_normal_vectors();
              double      face_measure = 0.;
              for (const unsigned int q : fe_face.quadrature_point_indices())
                face_measure += fe_face.JxW(q);
              const double sigma =
                penalty_constant * std::max(1. / h, face_measure / measure);
              for (const unsigned int q : fe_face.quadrature_point_indices())
                for (unsigned int i = 0; i < n_dofs; ++i)
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    cell_matrix(i, j) +=
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
          const auto &fe_faces =
            ah->reinit_interface(polytope, neighbor, f, neighbor_face);
          const auto &fe_faces0 = fe_faces.first;
          neighbor->get_dof_indices(neighbor_dof_indices);
          M11 = 0.;
          M12 = 0.;
          M21 = 0.;
          M22 = 0.;
          if (neighbor->is_locally_owned())
            {
              const auto  &fe_faces1 = fe_faces.second;
              const double sigma =
                penalty_constant / std::min(h, neighbor->diameter());
              weights.resize(fe_faces0.n_quadrature_points);
              for (const unsigned int q : fe_faces0.quadrature_point_indices())
                weights[q] = fe_faces1.JxW(q);
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
              // The neighbour lives on another rank, which sent its shape
              // values, gradients and weights on this face. diameter() is
              // not available for it, its bounding-box diagonal is.
              const types::subdomain_id rank = neighbor->subdomain_id();
              const auto               &recv_values =
                ah->recv_values.at(rank).at({neighbor->id(), neighbor_face});
              const auto &recv_gradients =
                ah->recv_gradients.at(rank).at({neighbor->id(), neighbor_face});
              const auto &recv_jxws =
                ah->recv_jxws.at(rank).at({neighbor->id(), neighbor_face});
              const auto corners =
                neighbor->get_bounding_box().get_boundary_points();
              const double sigma =
                penalty_constant /
                std::min(h, (corners.second - corners.first).norm());
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
                recv_jxws,
                sigma);
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
RepartitionedPoissonProblem<dim>::solve()
{
  // deal.II's AMG for an elliptic problem: smoothed aggregation, one V-cycle,
  // two Chebyshev sweeps and KLU on the coarsest level. Its finest level is
  // smoothed instead by one block-Jacobi sweep with one block per polytope.
  // The polytopes of a rank hold consecutive degrees of freedom, so the blocks
  // are those of the linear partition. Below the finest level aggregation
  // destroys the polytope blocks, so Chebyshev stays there.
  const auto range      = system_matrix.local_range();
  const int  local_rows = static_cast<int>(range.second - range.first);
  const int  n_blocks = local_rows / static_cast<int>(fe_dg.n_dofs_per_cell());

  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.aggregation_threshold = 0.02;
  Teuchos::ParameterList              parameters;
  std::unique_ptr<Epetra_MultiVector> constant_modes;
  amg_data.set_parameters(parameters,
                          constant_modes,
                          system_matrix.trilinos_matrix());
  Teuchos::ParameterList &finest =
    parameters.sublist("smoother: list (level 0)");
  finest.set("smoother: type", "IFPACK");
  finest.set("smoother: ifpack type", "block relaxation stand-alone");
  finest.set("smoother: sweeps", 1);
  Teuchos::ParameterList &ifpack = finest.sublist("smoother: ifpack list");
  ifpack.set("relaxation: type", "Jacobi");
  ifpack.set("relaxation: sweeps", 1);
  ifpack.set("partitioner: type", "linear");
  ifpack.set("partitioner: local parts", n_blocks);

  TrilinosWrappers::PreconditionAMG amg;
  amg.initialize(system_matrix, parameters);
  solution.reinit(ah->agglo_dh.locally_owned_dofs(), comm);
  ReductionControl                        solver_control(10000, 1e-13, 1e-7);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, amg);

  const IndexSet locally_relevant_dofs =
    PolytopeAdaptivity::locally_relevant_polytope_dofs(*ah);
  locally_relevant_solution.reinit(ah->agglo_dh.locally_owned_dofs(),
                                   locally_relevant_dofs,
                                   comm);
  locally_relevant_solution = solution;
}



template <int dim>
void
RepartitionedPoissonProblem<dim>::estimate_and_compute_errors()
{
  // The errors against the exact solution.
  double                      l2_sq = 0., h1_sq = 0.;
  std::vector<double>         u_values;
  std::vector<Tensor<1, dim>> u_gradients;
  for (const auto &polytope : ah->polytope_iterators())
    {
      if (!polytope->is_locally_owned())
        continue;
      const auto &fe_values = ah->reinit(polytope);
      const auto &points    = fe_values.get_quadrature_points();
      u_values.resize(points.size());
      u_gradients.resize(points.size());
      fe_values.get_function_values(solution, u_values);
      fe_values.get_function_gradients(solution, u_gradients);
      for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          const double error = u_values[q] - exact_solution.value(points[q]);
          const Tensor<1, dim> grad_error =
            u_gradients[q] - exact_solution.gradient(points[q]);
          l2_sq += error * error * fe_values.JxW(q);
          h1_sq += grad_error.norm_square() * fe_values.JxW(q);
        }
    }
  l2_error = std::sqrt(Utilities::MPI::sum(l2_sq, comm));
  h1_error = std::sqrt(Utilities::MPI::sum(h1_sq, comm));

  PolytopeAdaptivity::compute_residual_indicators(*ah,
                                                  locally_relevant_solution,
                                                  rhs_function,
                                                  penalty_constant,
                                                  copy_indicators);
  partitioned_copy.copy_to_p4est(copy_indicators, cell_indicators);
  double sum_sq = 0.;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      sum_sq += cell_indicators[cell->active_cell_index()] *
                cell_indicators[cell->active_cell_index()];
  error_estimator = std::sqrt(Utilities::MPI::sum(sum_sq, comm));
  // The tolerance is a fraction of the error estimator of the first cycle.
  if (tolerance < 0.)
    tolerance = relative_tolerance * error_estimator;
}



template <int dim>
void
RepartitionedPoissonProblem<dim>::output_results(const unsigned int counter,
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
          indicator_sq += copy_indicators[cell->active_cell_index()] *
                          copy_indicators[cell->active_cell_index()];
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
    "./", "repartitioned_poisson", counter, comm, 4);

  // The record of all outputs so far, which ParaView opens as one series.
  if (my_rank == 0)
    {
      times_and_names.emplace_back(at_time, pvtu);
      std::ofstream pvd("repartitioned_poisson.pvd");
      DataOutBase::write_pvd_record(pvd, times_and_names);
    }
}



template <int dim>
void
RepartitionedPoissonProblem<dim>::run()
{
  make_fine_grid();
  pcout << "Running with " << Utilities::MPI::n_mpi_processes(comm)
        << " MPI ranks." << std::endl;
  pcout << "cycle  polytopes      dofs   estimator       L2 error        "
           "H1 error        imbalance p4est / copy"
        << std::endl;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      setup_agglomerated_problem();
      assemble_system();
      solve();
      estimate_and_compute_errors();
      pcout << std::setw(5) << cycle << std::setw(11)
            << polytopal_mesh.n_global_polytopes() << std::setw(10)
            << ah->n_dofs() << "   " << std::left << std::setw(16)
            << error_estimator << std::setw(16) << l2_error << std::setw(16)
            << h1_error << p4est_imbalance << " / "
            << partitioned_copy.dof_imbalance() << std::right << std::endl;
      // Uncomment to write the solution and the polytopes for ParaView.
      // output_results(cycle, cycle);
      if (cycle + 1 == n_cycles)
        break;
      if (!polytopal_mesh.refine_and_coarsen(
            cell_indicators, polytopal_mesh.polytope_tolerance(tolerance)))
        {
          pcout << "The finest refinement level of p4est is reached."
                << std::endl;
          break;
        }
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;

  PolytopeAdaptivity::AdaptivityParameters parameters;
  parameters.max_cells_per_polytope = 64;
  parameters.level_rule = PolytopeAdaptivity::LevelRule::multiple_levels;

  const unsigned int             n_subdivisions     = 64;
  const unsigned int             degree             = 4;
  const unsigned int             n_cycles           = 20;
  const double                   relative_tolerance = 1e-5;
  RepartitionedPoissonProblem<2> problem(
    n_subdivisions, degree, n_cycles, relative_tolerance, parameters, comm);
  problem.run();
}
