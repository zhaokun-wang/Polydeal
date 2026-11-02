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


// Quadrature-free assembly of the SIPG discretization of
//   -Delta u = f in Omega = (0,1)^2,
//          u = 0 on the boundary,
// with u = sin(pi x) sin(pi y), on polytopes of an R-tree agglomeration.
//
// For every degree the system is assembled twice, by quadrature over the
// cells of every polytope and without quadrature
// (QuadratureFree::PolytopeIntegrator): volume terms from the moments of the
// polytopes, face terms from the moments of the interfaces. The load vector
// is integrated by quadrature in both. The program prints the relative
// difference of the two matrices, the assembly times, and the error of the
// solution of the quadrature-free system.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

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
#include <deal.II/numerics/rtree.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>
#include <quadrature_free_integration.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>

using namespace dealii;



/**
 * The exact solution $u = \sin(\pi x) \sin(\pi y)$.
 */
template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};



template <int dim>
double
Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double value = 1.;
  for (unsigned int direction = 0; direction < dim; ++direction)
    value *= std::sin(numbers::PI * p[direction]);
  return value;
}



template <int dim>
Tensor<1, dim>
Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> gradient;
  for (unsigned int direction = 0; direction < dim; ++direction)
    {
      gradient[direction] = numbers::PI * std::cos(numbers::PI * p[direction]);
      for (unsigned int other_direction = 0; other_direction < dim;
           ++other_direction)
        if (other_direction != direction)
          gradient[direction] *= std::sin(numbers::PI * p[other_direction]);
    }
  return gradient;
}



/**
 * The right-hand side $f = -\Delta u = d \pi^2 u$.
 */
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
  const Solution<dim> solution;
};



template <int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return dim * numbers::PI * numbers::PI * solution.value(p);
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
class QuadratureFreePoissonProblem
{
public:
  QuadratureFreePoissonProblem(const unsigned int n_global_refinements,
                               const unsigned int extraction_level,
                               const MPI_Comm     comm);

  void
  run(const std::vector<unsigned int> &degrees);

private:
  /**
   * The mesh, the unit square refined n_global_refinements times, and the
   * polytopes on it.
   */
  void
  make_fine_grid();

  /**
   * The AgglomerationHandler on the polytopes, with the degrees of freedom of
   * @p fe_dg.
   */
  void
  setup_agglomerated_problem(const FE_AggloDGP<dim> &fe_dg);

  /**
   * Assemble the system by quadrature, or without it if @p integrator is
   * given.
   */
  void
  assemble_system(const FE_AggloDGP<dim>                        &fe_dg,
                  const QuadratureFree::PolytopeIntegrator<dim> *integrator,
                  TrilinosWrappers::SparseMatrix                &matrix);

  /**
   * Conjugate gradients with algebraic multigrid on the system with the
   * matrix @p matrix.
   */
  void
  solve(const TrilinosWrappers::SparseMatrix &matrix);

  /**
   * The $L^2$ error of the solution against the exact solution.
   */
  double
  compute_l2_error() const;

  /**
   * Write the solution and the polytopes to quadrature_free_poisson_NNNN.pvtu,
   * one piece per rank, and add it at @p at_time to the series
   * quadrature_free_poisson.pvd. In ParaView, colour by polytope_colour to see
   * the polytopes.
   */
  void
  output_results(const unsigned int counter, const double at_time);

  const MPI_Comm comm;

  /**
   * Global refinements of the unit square.
   */
  const unsigned int n_global_refinements;

  /**
   * The level of the R-tree whose nodes are the polytopes.
   */
  const unsigned int extraction_level;

  /**
   * Output on rank 0 only.
   */
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> tria;

  /**
   * The mapping of the cells, for GridTools::Cache.
   */
  MappingQ1<dim> mapping;

  /**
   * The polytopes of this rank, those of one extraction level of an R-tree.
   */
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    polytopes;

  /**
   * The cache of the triangulation and the AgglomerationHandler on the
   * polytopes, both rebuilt for every degree.
   */
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<AgglomerationHandler<dim>> ah;

  /**
   * The sparsity pattern of the system, the load vector, which is integrated
   * by quadrature in both assemblies, and the solution of the quadrature-free
   * system.
   */
  TrilinosWrappers::SparsityPattern sparsity;
  TrilinosWrappers::MPI::Vector     system_rhs;
  TrilinosWrappers::MPI::Vector     solution;

  /**
   * The exact solution and the right-hand side.
   */
  Solution<dim>      exact_solution;
  RightHandSide<dim> rhs_function;

  /**
   * The time and the name of every output so far, for the .pvd series.
   */
  std::vector<std::pair<double, std::string>> times_and_names;
};



template <int dim>
QuadratureFreePoissonProblem<dim>::QuadratureFreePoissonProblem(
  const unsigned int n_global_refinements,
  const unsigned int extraction_level,
  const MPI_Comm     communicator)
  : comm(communicator)
  , n_global_refinements(n_global_refinements)
  , extraction_level(extraction_level)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
  , tria(comm)
{}



template <int dim>
void
QuadratureFreePoissonProblem<dim>::make_fine_grid()
{
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(n_global_refinements);

  // The polytopes are those of one extraction level of an R-tree over the
  // locally owned cells, split into their connected components.
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim);
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
    boxes;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      boxes.emplace_back(mapping.get_bounding_box(cell), cell);
  auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  CellsAgglomerator<dim, decltype(tree)> agglomerator{tree, extraction_level};
  polytopes = agglomerator.extract_agglomerates();
}



template <int dim>
void
QuadratureFreePoissonProblem<dim>::setup_agglomerated_problem(
  const FE_AggloDGP<dim> &fe_dg)
{
  ah.reset();
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  ah          = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  for (const auto &polytope : polytopes)
    ah->define_agglomerate(polytope);
  const unsigned int n_q_points = 2 * fe_dg.degree + 1;
  ah->initialize_fe_values(QGauss<dim>(n_q_points),
                           update_values | update_gradients |
                             update_JxW_values | update_quadrature_points,
                           QGauss<dim - 1>(n_q_points));
  ah->distribute_agglomerated_dofs(fe_dg);
  ah->create_agglomeration_sparsity_pattern(sparsity);
}



template <int dim>
void
QuadratureFreePoissonProblem<dim>::assemble_system(
  const FE_AggloDGP<dim>                        &fe_dg,
  const QuadratureFree::PolytopeIntegrator<dim> *integrator,
  TrilinosWrappers::SparseMatrix                &matrix)
{
  matrix.reinit(sparsity);
  system_rhs.reinit(ah->agglo_dh.locally_owned_dofs(), comm);
  const double penalty_constant = 2. * fe_dg.degree * (fe_dg.degree + 1);

  std::vector<FullMatrix<double>> stiffness;
  if (integrator != nullptr)
    integrator->volume_matrices(*ah, stiffness);

  const unsigned int n_dofs = fe_dg.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(n_dofs, n_dofs);
  Vector<double>     cell_rhs(n_dofs);
  FullMatrix<double> M11(n_dofs, n_dofs), M12(n_dofs, n_dofs),
    M21(n_dofs, n_dofs), M22(n_dofs, n_dofs);
  FullMatrix<double>                   face_mass, face_flux;
  std::vector<types::global_dof_index> local_dof_indices(n_dofs);
  std::vector<types::global_dof_index> neighbor_dof_indices(n_dofs);
  std::vector<double>                  rhs_values, weights;

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
            if (integrator == nullptr)
              for (unsigned int j = 0; j < n_dofs; ++j)
                cell_matrix(i, j) += agglo_values.shape_grad(i, q) *
                                     agglo_values.shape_grad(j, q) *
                                     agglo_values.JxW(q);
            cell_rhs(i) += agglo_values.shape_value(i, q) * rhs_values[q] *
                           agglo_values.JxW(q);
          }
      if (integrator != nullptr)
        cell_matrix = stiffness[polytope->index()];

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
              if (integrator != nullptr)
                {
                  const double face_measure = integrator->face_matrices(
                    *ah, polytope, f, face_mass, face_flux);
                  const double sigma =
                    penalty_constant * std::max(1. / h, face_measure / measure);
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      cell_matrix(i, j) += -face_flux(i, j) - face_flux(j, i) +
                                           sigma * face_mass(i, j);
                  continue;
                }
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
          neighbor->get_dof_indices(neighbor_dof_indices);
          double h_neighbor;
          if (neighbor->is_locally_owned())
            h_neighbor = neighbor->diameter();
          else
            {
              const auto corners =
                neighbor->get_bounding_box().get_boundary_points();
              h_neighbor = (corners.second - corners.first).norm();
            }
          const double sigma = penalty_constant / std::min(h, h_neighbor);
          M11                = 0.;
          M12                = 0.;
          M21                = 0.;
          M22                = 0.;
          if (integrator != nullptr)
            {
              integrator->face_matrices(*ah, polytope, f, face_mass, face_flux);
              interface_blocks(face_mass, face_flux, sigma, M11, M12, M21, M22);
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
                  const types::subdomain_id rank = neighbor->subdomain_id();
                  const auto &recv_values        = ah->recv_values.at(rank).at(
                    {neighbor->id(), neighbor_face});
                  const auto &recv_gradients = ah->recv_gradients.at(rank).at(
                    {neighbor->id(), neighbor_face});
                  const auto &recv_jxws =
                    ah->recv_jxws.at(rank).at({neighbor->id(), neighbor_face});
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
            }
          matrix.add(local_dof_indices, M11);
          matrix.add(local_dof_indices, neighbor_dof_indices, M12);
          matrix.add(neighbor_dof_indices, local_dof_indices, M21);
          matrix.add(neighbor_dof_indices, M22);
        }
      matrix.add(local_dof_indices, cell_matrix);
      system_rhs.add(local_dof_indices, cell_rhs);
    }
  matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void
QuadratureFreePoissonProblem<dim>::solve(
  const TrilinosWrappers::SparseMatrix &matrix)
{
  TrilinosWrappers::PreconditionAMG                 amg;
  TrilinosWrappers::PreconditionAMG::AdditionalData data;
  data.elliptic = true;
  amg.initialize(matrix, data);
  solution.reinit(ah->agglo_dh.locally_owned_dofs(), comm);
  ReductionControl                        control(10000, 1e-14, 1e-10);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
  solver.solve(matrix, solution, system_rhs, amg);
}



template <int dim>
double
QuadratureFreePoissonProblem<dim>::compute_l2_error() const
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
QuadratureFreePoissonProblem<dim>::output_results(const unsigned int counter,
                                                  const double       at_time)
{
  // The solution, interpolated into the discontinuous elements of the same
  // degree on the cells.
  TrilinosWrappers::MPI::Vector interpolated_solution;
  PolyUtils::interpolate_to_fine_grid(*ah, interpolated_solution, solution);

  // For every cell of the mesh of the system: the global polytope index of
  // its polytope, a colour that differs between neighbouring polytopes, the
  // rank, the refinement level.
  const unsigned int n_cells = ah->get_triangulation().n_active_cells();
  Vector<float>      polytope_index(n_cells), polytope_colour(n_cells),
    rank(n_cells), level(n_cells);
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
        const auto &cells = polytope->get_agglomerate();
        for (const auto &cell : cells)
          {
            const unsigned int index = cell->active_cell_index();
            polytope_index[index]    = global_index;
            // Multiples of the golden ratio spread consecutive indices
            // evenly over [0, 1).
            polytope_colour[index] = std::fmod(global_index * 0.6180339887, 1.);
            rank[index]            = my_rank;
            level[index]           = cell->level();
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
  data_out.build_patches(mapping, ah->get_fe().degree);
  const std::string pvtu = data_out.write_vtu_with_pvtu_record(
    "./", "quadrature_free_poisson", counter, comm, 4);

  // The record of all outputs so far, which ParaView opens as one series.
  if (my_rank == 0)
    {
      times_and_names.emplace_back(at_time, pvtu);
      std::ofstream pvd("quadrature_free_poisson.pvd");
      DataOutBase::write_pvd_record(pvd, times_and_names);
    }
}



template <int dim>
void
QuadratureFreePoissonProblem<dim>::run(const std::vector<unsigned int> &degrees)
{
  make_fine_grid();
  pcout << "Running with " << Utilities::MPI::n_mpi_processes(comm)
        << " MPI ranks, " << Utilities::MPI::sum(polytopes.size(), comm)
        << " polytopes." << std::endl;
  pcout << "degree    dofs   |A_qf - A|/|A|  quadrature [s]  "
           "quadrature-free [s]  L2 error"
        << std::endl;
  for (const unsigned int degree : degrees)
    {
      const FE_AggloDGP<dim> fe_dg(degree);
      setup_agglomerated_problem(fe_dg);

      TrilinosWrappers::SparseMatrix matrix_quadrature, matrix_quadrature_free;
      Timer                          timer;
      timer.start();
      assemble_system(fe_dg, nullptr, matrix_quadrature);
      timer.stop();
      const double time_quadrature =
        Utilities::MPI::max(timer.wall_time(), comm);

      // The expansions of the products of the basis functions depend only on
      // the element, and are computed once per degree, outside the timing.
      const QuadratureFree::PolytopeIntegrator<dim> integrator(fe_dg);
      timer.restart();
      assemble_system(fe_dg, &integrator, matrix_quadrature_free);
      timer.stop();
      const double time_quadrature_free =
        Utilities::MPI::max(timer.wall_time(), comm);

      solve(matrix_quadrature_free);
      const double l2_error = compute_l2_error();

      const double norm = matrix_quadrature.frobenius_norm();
      matrix_quadrature.add(-1., matrix_quadrature_free);
      pcout << std::setw(6) << degree << std::setw(8) << ah->n_dofs() << "   "
            << std::left << std::setw(15)
            << matrix_quadrature.frobenius_norm() / norm << std::setw(16)
            << time_quadrature << std::setw(21) << time_quadrature_free
            << l2_error << std::right << std::endl;
      // Uncomment to write the solution and the polytopes for ParaView.
      // output_results(degree, degree);
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const unsigned int               n_global_refinements = 6;
  const unsigned int               extraction_level     = 3;
  QuadratureFreePoissonProblem<2>  problem(n_global_refinements,
                                          extraction_level,
                                          MPI_COMM_WORLD);
  problem.run({1, 2, 3, 4, 5, 6});
}
