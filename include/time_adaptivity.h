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


#ifndef time_adaptivity_h
#define time_adaptivity_h


#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/cell_data_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/adaptation_strategies.h>

#include <agglomeration_handler.h>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>


/**
 * Adaptive time stepping for the backward Euler discretization of
 * $u_t - \Delta u = f$ on adaptive polytopal meshes, following A. Cangiani,
 * E.H. Georgoulis, S. Metcalfe, "Adaptive discontinuous Galerkin methods for
 * nonstationary convection-diffusion problems", IMA J. Numer. Anal. 34
 * (2014): the error indicator in time (7.1), built on the elliptic
 * reconstruction, and the step and mesh control of their Algorithm 1.
 */
namespace dealii::TimeAdaptivity
{
  /**
   * Polynomials stored on cells, which carry a discrete function from one
   * mesh to the next. The polytopes change with the mesh, the cells do not,
   * so every cell stores the polynomial of the polytope it belonged to. The
   * polynomial is given in the basis of FE_AggloDGP: the complete Legendre
   * basis on the bounding box of that polytope, mapped to the unit box. The
   * data of a cell is one vector
   * [number of coefficients, lower corner of the box, upper corner of the
   * box, coefficients]; an empty vector stands for the zero function.
   *
   * The polynomial of a cell lives on the bounding box of an old polytope,
   * which the AgglomerationHandler of the new mesh does not know, so the
   * functions here evaluate the basis themselves.
   */
  namespace CellPolynomial
  {
    /**
     * The degree of the complete polynomial space with @p n_coefficients
     * basis functions.
     */
    template <int dim>
    unsigned int
    degree_of(const unsigned int n_coefficients)
    {
      unsigned int degree = 0;
      while (PolynomialSpace<dim>::n_polynomials(degree + 1) < n_coefficients)
        ++degree;
      AssertThrow(PolynomialSpace<dim>::n_polynomials(degree + 1) ==
                    n_coefficients,
                  ExcInternalError());
      return degree;
    }



    /**
     * The number of coefficients of the polynomial @p data, which is not
     * empty.
     */
    inline unsigned int
    n_coefficients_of(const std::vector<double> &data)
    {
      return static_cast<unsigned int>(data[0] + 0.5);
    }



    /**
     * The point @p point mapped from the box of the polynomial @p data to the
     * unit box.
     */
    template <int dim>
    Point<dim>
    to_unit_box(const std::vector<double> &data, const Point<dim> &point)
    {
      Point<dim> unit;
      for (unsigned int direction = 0; direction < dim; ++direction)
        unit[direction] = (point[direction] - data[1 + direction]) /
                          (data[1 + dim + direction] - data[1 + direction]);
      return unit;
    }



    /**
     * The complete Legendre basis with @p n_coefficients basis functions. It
     * is built once per number of coefficients and kept.
     */
    template <int dim>
    const PolynomialSpace<dim> &
    basis(const unsigned int n_coefficients)
    {
      static std::map<unsigned int, std::unique_ptr<PolynomialSpace<dim>>>
            spaces;
      auto &space = spaces[n_coefficients];
      if (!space)
        space = std::make_unique<PolynomialSpace<dim>>(
          Polynomials::Legendre::generate_complete_basis(
            degree_of<dim>(n_coefficients)));
      return *space;
    }



    /**
     * The data of the polynomial with coefficients @p coefficients on the box
     * @p box.
     */
    template <int dim>
    std::vector<double>
    pack(const BoundingBox<dim> &box, const std::vector<double> &coefficients)
    {
      std::vector<double> data(1 + 2 * dim + coefficients.size());
      data[0] = coefficients.size();
      for (unsigned int direction = 0; direction < dim; ++direction)
        {
          data[1 + direction] = box.get_boundary_points().first[direction];
          data[1 + dim + direction] =
            box.get_boundary_points().second[direction];
        }
      std::copy(coefficients.begin(),
                coefficients.end(),
                data.begin() + 1 + 2 * dim);
      return data;
    }



    /**
     * The value at @p point of the polynomial @p data; zero for empty data.
     */
    template <int dim>
    double
    evaluate(const std::vector<double> &data, const Point<dim> &point)
    {
      if (data.empty())
        return 0.;
      const unsigned int n_coefficients = n_coefficients_of(data);

      // The values of the basis functions at the point, kept between calls.
      // Only the values are requested; PolynomialSpace::evaluate() skips the
      // empty vectors of derivatives.
      static thread_local std::vector<double>         values;
      static thread_local std::vector<Tensor<1, dim>> gradients;
      static thread_local std::vector<Tensor<2, dim>> hessians;
      static thread_local std::vector<Tensor<3, dim>> third_derivatives;
      static thread_local std::vector<Tensor<4, dim>> fourth_derivatives;
      values.resize(n_coefficients);
      basis<dim>(n_coefficients)
        .evaluate(to_unit_box(data, point),
                  values,
                  gradients,
                  hessians,
                  third_derivatives,
                  fourth_derivatives);
      double value = 0.;
      for (unsigned int i = 0; i < n_coefficients; ++i)
        value += data[1 + 2 * dim + i] * values[i];
      return value;
    }



    /**
     * The polynomial of a cell @p parent whose children p4est coarsens, from
     * the polynomials @p children of the children: their $L^2$ projection
     * onto the polynomials of their largest degree on the bounding box of
     * @p parent. The children are active descendants of a refined initial
     * cell, each a polytope of its own, so their polynomials differ.
     */
    template <int dim>
    std::vector<double>
    coarsen(const typename Triangulation<dim>::cell_iterator &parent,
            const std::vector<std::vector<double>>           &children)
    {
      if (children.empty() || children[0].empty())
        return {};

      // The space of the result: the largest number of coefficients of the
      // children, on the bounding box of the parent.
      unsigned int n_coefficients = 0;
      for (const auto &child : children)
        if (!child.empty())
          n_coefficients = std::max(n_coefficients, n_coefficients_of(child));
      std::vector<double> data =
        pack<dim>(parent->bounding_box(),
                  std::vector<double>(n_coefficients, 0.));
      const unsigned int degree = degree_of<dim>(n_coefficients);

      // The mass matrix and the right-hand side of the projection,
      // integrated child by child with the polynomial of each child.
      const QGauss<dim>     quadrature(2 * degree + 1);
      const MappingQ1<dim>  mapping;
      const FE_Nothing<dim> fe_nothing;
      FEValues<dim>         fe_values(mapping,
                              fe_nothing,
                              quadrature,
                              update_quadrature_points | update_JxW_values);
      FullMatrix<double>    mass(n_coefficients, n_coefficients);
      Vector<double>        rhs(n_coefficients), coefficients(n_coefficients);
      std::vector<double>   values(n_coefficients);
      std::vector<Tensor<1, dim>> gradients;
      std::vector<Tensor<2, dim>> hessians;
      std::vector<Tensor<3, dim>> third_derivatives;
      std::vector<Tensor<4, dim>> fourth_derivatives;
      for (unsigned int child = 0;
           child < children.size() && child < parent->n_children();
           ++child)
        {
          fe_values.reinit(parent->child(child));
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              const Point<dim> &point = fe_values.quadrature_point(q);
              const double      value = evaluate<dim>(children[child], point);
              basis<dim>(n_coefficients)
                .evaluate(to_unit_box(data, point),
                          values,
                          gradients,
                          hessians,
                          third_derivatives,
                          fourth_derivatives);
              for (unsigned int i = 0; i < n_coefficients; ++i)
                {
                  rhs(i) += values[i] * value * fe_values.JxW(q);
                  for (unsigned int j = 0; j < n_coefficients; ++j)
                    mass(i, j) += values[i] * values[j] * fe_values.JxW(q);
                }
            }
        }
      mass.gauss_jordan();
      mass.vmult(coefficients, rhs);
      for (unsigned int i = 0; i < n_coefficients; ++i)
        data[1 + 2 * dim + i] = coefficients(i);
      return data;
    }
  } // namespace CellPolynomial



  /**
   * Carry fields of CellPolynomial data, one entry per active cell, through
   * refinement and coarsening of a p4est triangulation. The children of a
   * refined cell keep its polynomial. A cell that replaces coarsened cells
   * gets CellPolynomial::coarsen() of their polynomials.
   *
   * All cells of a polytope hold the same data, so the data is stored once
   * per distinct value in a table of rows on this rank, a row holding the
   * data of every field. Only the row of every cell is moved, by one
   * parallel::distributed::CellDataTransfer. This requires that the
   * triangulation is not repartitioned, so that every cell stays on the rank
   * of its row: it has to be built with
   * parallel::distributed::Triangulation::no_automatic_repartitioning, and
   * unpack() checks it.
   */
  template <int dim>
  class CellPolynomialTransfer
  {
  public:
    /**
     * Constructor.
     */
    CellPolynomialTransfer(
      const parallel::distributed::Triangulation<dim> &tria);

    /**
     * Prepare the fields @p fields_to_carry for the next
     * execute_coarsening_and_refinement(). Call after the refinement flags
     * are set.
     */
    void
    prepare_for_coarsening_and_refinement(
      const std::vector<std::vector<std::vector<double>> *> &fields_to_carry);

    /**
     * Write the fields on the new cells.
     */
    void
    unpack();

  private:
    /**
     * The value moved for row @p row: the row and this rank together, so that
     * unpack() can check that the cell did not move to another rank.
     */
    double
    encode(const std::size_t row) const;

    /**
     * The row of a moved value, after checking that it belongs to this rank.
     */
    std::size_t
    decode(const double value) const;

    /**
     * The triangulation whose cells carry the fields.
     */
    const parallel::distributed::Triangulation<dim> &tria;

    /**
     * The fields being carried.
     */
    std::vector<std::vector<std::vector<double>> *> fields;

    /**
     * The table: rows[row][field] is the data of a field.
     */
    std::vector<std::vector<std::vector<double>>> rows;

    /**
     * The encoded row of every active cell, which the transfer moves.
     */
    Vector<double> row_of_cell;

    /**
     * The transfer of row_of_cell.
     */
    std::unique_ptr<
      parallel::distributed::CellDataTransfer<dim, dim, Vector<double>>>
      transfer;
  };



  template <int dim>
  CellPolynomialTransfer<dim>::CellPolynomialTransfer(
    const parallel::distributed::Triangulation<dim> &tria)
    : tria(tria)
  {}



  template <int dim>
  double
  CellPolynomialTransfer<dim>::encode(const std::size_t row) const
  {
    const unsigned int n_ranks =
      Utilities::MPI::n_mpi_processes(tria.get_mpi_communicator());
    const unsigned int my_rank =
      Utilities::MPI::this_mpi_process(tria.get_mpi_communicator());
    return double(row * n_ranks + my_rank);
  }



  template <int dim>
  std::size_t
  CellPolynomialTransfer<dim>::decode(const double value) const
  {
    const unsigned int n_ranks =
      Utilities::MPI::n_mpi_processes(tria.get_mpi_communicator());
    const unsigned int my_rank =
      Utilities::MPI::this_mpi_process(tria.get_mpi_communicator());
    const std::size_t encoded = static_cast<std::size_t>(value + 0.5);
    AssertThrow(encoded % n_ranks == my_rank,
                ExcMessage("A cell moved to another rank. The triangulation "
                           "must not be repartitioned."));
    return encoded / n_ranks;
  }



  template <int dim>
  void
  CellPolynomialTransfer<dim>::prepare_for_coarsening_and_refinement(
    const std::vector<std::vector<std::vector<double>> *> &fields_to_carry)
  {
    fields = fields_to_carry;

    // The table, one row per distinct data of the locally owned cells, and
    // the row of every cell.
    rows.clear();
    row_of_cell.reinit(tria.n_active_cells());
    std::map<std::vector<std::vector<double>>, std::size_t> row_of_data;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<std::vector<double>> data;
          for (const auto *field : fields)
            data.push_back((*field)[cell->active_cell_index()]);
          const auto inserted = row_of_data.emplace(data, rows.size());
          if (inserted.second)
            rows.push_back(std::move(data));
          row_of_cell[cell->active_cell_index()] =
            encode(inserted.first->second);
        }

    // Refinement copies the row to the children. Coarsening adds a row with
    // the coarsened data of every field.
    transfer = std::make_unique<
      parallel::distributed::CellDataTransfer<dim, dim, Vector<double>>>(
      tria,
      false,
      &AdaptationStrategies::Refinement::preserve<dim, dim, double>,
      [this](const typename Triangulation<dim>::cell_iterator &parent,
             const std::vector<double> &children_values) -> double {
        std::vector<std::vector<double>> data(fields.size());
        for (unsigned int field = 0; field < fields.size(); ++field)
          {
            std::vector<std::vector<double>> children;
            for (const double value : children_values)
              children.push_back(rows[decode(value)][field]);
            data[field] = CellPolynomial::coarsen<dim>(parent, children);
          }
        rows.push_back(std::move(data));
        return encode(rows.size() - 1);
      });
    transfer->prepare_for_coarsening_and_refinement(row_of_cell);
  }



  template <int dim>
  void
  CellPolynomialTransfer<dim>::unpack()
  {
    Vector<double> new_row_of_cell(tria.n_active_cells());
    transfer->unpack(new_row_of_cell);
    for (unsigned int field = 0; field < fields.size(); ++field)
      {
        std::vector<std::vector<double>> carried(tria.n_active_cells());
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            carried[cell->active_cell_index()] =
              rows[decode(new_row_of_cell[cell->active_cell_index()])][field];
        fields[field]->swap(carried);
      }
    transfer.reset();
    rows.clear();
  }



  /**
   * The polynomial of every locally owned polytope of @p ah given by
   * @p solution, written as CellPolynomial data into @p cell_data for every
   * cell of the polytope. The coefficients are those of @p solution, since
   * CellPolynomial uses the basis of FE_AggloDGP.
   */
  template <int dim>
  void
  polytope_polynomials(const AgglomerationHandler<dim>     &ah,
                       const TrilinosWrappers::MPI::Vector &solution,
                       std::vector<std::vector<double>>    &cell_data)
  {
    const unsigned int n_dofs = ah.get_fe().dofs_per_cell;
    cell_data.assign(ah.get_triangulation().n_active_cells(), {});
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    std::vector<double>                  coefficients(n_dofs);
    for (const auto &polytope : ah.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          polytope->get_dof_indices(dof_indices);
          for (unsigned int k = 0; k < n_dofs; ++k)
            coefficients[k] = solution(dof_indices[k]);
          const auto data =
            CellPolynomial::pack<dim>(ah.get_local_bboxes()[polytope->index()],
                                      coefficients);
          for (const auto &cell : polytope->get_agglomerate())
            cell_data[cell->active_cell_index()] = data;
        }
  }



  /**
   * The discrete elliptic operator of the solution, $A u_h \in P_p(P)$ on
   * every locally owned polytope $P$ of @p ah, defined by
   * $(A u_h, v) = (f - (u_h - u^{n-1})/\Delta t, v)$ for all
   * $v \in P_p(P)$ (Cangiani, Georgoulis, Metcalfe, Def. 6.1 and Remark 6.2),
   * written as CellPolynomial data into @p cell_data. Without
   * @p old_solution, or with $\Delta t = 0$, it is the projection of $f$.
   *
   * The mass matrix of a polytope that fills little of its bounding box is
   * badly conditioned at higher degree; it is inverted through its singular
   * value decomposition, dropping singular values below $10^{-12}$ of the
   * largest.
   *
   * The handler's FEValues are initialized here with values, on quadratures
   * of $2p+2$ points per direction.
   */
  template <int dim>
  void
  compute_discrete_operator(
    AgglomerationHandler<dim>              &ah,
    const TrilinosWrappers::MPI::Vector    &solution,
    const Function<dim>                    &rhs_function,
    const std::vector<std::vector<double>> *old_solution,
    const double                            dt,
    std::vector<std::vector<double>>       &cell_data)
  {
    const unsigned int n_dofs = ah.get_fe().dofs_per_cell;
    const unsigned int n_q_1d = 2 * ah.get_fe().degree + 2;
    ah.initialize_fe_values(QGauss<dim>(n_q_1d),
                            update_values | update_quadrature_points |
                              update_JxW_values,
                            QGauss<dim - 1>(n_q_1d));
    const unsigned int  n_q_cell  = QGauss<dim>(n_q_1d).size();
    const bool          time_step = old_solution != nullptr && dt > 0.;
    std::vector<double> rhs_values, u_values;

    cell_data.assign(ah.get_triangulation().n_active_cells(), {});
    for (const auto &polytope : ah.polytope_iterators())
      {
        if (!polytope->is_locally_owned())
          continue;

        // f and u_h at the points of the polytope.
        const auto &fe_values = ah.reinit(polytope);
        const auto &points    = fe_values.get_quadrature_points();
        rhs_values.resize(points.size());
        rhs_function.value_list(points, rhs_values);
        if (time_step)
          {
            u_values.resize(points.size());
            fe_values.get_function_values(solution, u_values);
          }

        // The mass matrix, and the right-hand side: the integrals of the
        // source $f - (u_h - u^{n-1})/\Delta t$ times every basis function. The
        // old solution is a polynomial on every cell, so the loop runs over the
        // cells of the polytope, whose points the handler lists one cell after
        // the other in the order of get_agglomerate().
        FullMatrix<double> mass(n_dofs, n_dofs);
        Vector<double>     rhs(n_dofs);
        unsigned int       q = 0;
        for (const auto &cell : polytope->get_agglomerate())
          for (const unsigned int end = q + n_q_cell; q < end; ++q)
            {
              double source = rhs_values[q];
              if (time_step)
                source -=
                  (u_values[q] -
                   CellPolynomial::evaluate<dim>(
                     (*old_solution)[cell->active_cell_index()], points[q])) /
                  dt;
              const double jxw = fe_values.JxW(q);
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  rhs(i) += source * fe_values.shape_value(i, q) * jxw;
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    mass(i, j) += fe_values.shape_value(i, q) *
                                  fe_values.shape_value(j, q) * jxw;
                }
            }
        AssertDimension(q, points.size());

        // The coefficients of $A u_h$: the inverse of the mass matrix applied
        // to the right-hand side, through its singular value decomposition
        // $M = U \Sigma U^T$, with the small singular values dropped.
        LAPACKFullMatrix<double> mass_svd(n_dofs, n_dofs);
        mass_svd = mass;
        mass_svd.compute_svd();
        const auto         &u       = mass_svd.get_svd_u();
        const double        largest = mass_svd.singular_value(0);
        std::vector<double> operator_coefficients(n_dofs, 0.);
        for (unsigned int k = 0; k < n_dofs; ++k)
          {
            const double sigma = mass_svd.singular_value(k);
            if (sigma <= 1e-12 * largest)
              break;
            double component = 0.;
            for (unsigned int i = 0; i < n_dofs; ++i)
              component += u(i, k) * rhs(i);
            for (unsigned int i = 0; i < n_dofs; ++i)
              operator_coefficients[i] += u(i, k) * component / sigma;
          }
        const auto data =
          CellPolynomial::pack<dim>(ah.get_local_bboxes()[polytope->index()],
                                    operator_coefficients);
        for (const auto &cell : polytope->get_agglomerate())
          cell_data[cell->active_cell_index()] = data;
      }
  }



  /**
   * The error indicator in time (7.1) of Cangiani, Georgoulis and Metcalfe
   * for the step from @p t_old to $t_{\text{old}} + \Delta t$, with
   * $\alpha_T = 1$:
   * @f[
   *   \eta_T^2 = \min(1, T) \int_{t_{\text{old}}}^{t_{\text{old}}+\Delta t}
   *     \| f(t) - f^{n+1} + \ell(t) (A^{n+1} - A^n) \|^2 \, dt,
   * @f]
   * $\ell$ the linear function that is one at $t_{\text{old}}$ and zero at
   * the new time, $A^{n+1}$ and $A^n$ the discrete elliptic operators of the
   * new and the old solution (compute_discrete_operator()) on the cells of
   * @p ah, and $T$ the final time @p final_time. The integral in time is by
   * three-point Gauss, in space by $2p+2$ points per direction on the cells.
   * @p rhs_function is evaluated at the times of the rule and left at the new
   * time.
   *
   * The handler's FEValues are initialized here with values, on quadratures
   * of $2p+2$ points per direction.
   */
  template <int dim>
  double
  compute_time_indicator(AgglomerationHandler<dim>              &ah,
                         Function<dim>                          &rhs_function,
                         const double                            t_old,
                         const double                            dt,
                         const std::vector<std::vector<double>> &operator_new,
                         const std::vector<std::vector<double>> &operator_old,
                         const double                            final_time)
  {
    const unsigned int n_q_1d = 2 * ah.get_fe().degree + 2;
    ah.initialize_fe_values(QGauss<dim>(n_q_1d),
                            update_values | update_quadrature_points |
                              update_JxW_values,
                            QGauss<dim - 1>(n_q_1d));
    const unsigned int               n_q_cell = QGauss<dim>(n_q_1d).size();
    const QGauss<1>                  in_time(3);
    std::vector<std::vector<double>> f_at(in_time.size());
    std::vector<double>              f_new;
    double                           sum_sq = 0.;
    for (const auto &polytope : ah.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          // f at the points of the polytope, at the new time and at the
          // points of the rule in time.
          const auto &fe_values = ah.reinit(polytope);
          const auto &points    = fe_values.get_quadrature_points();
          f_new.resize(points.size());
          rhs_function.set_time(t_old + dt);
          rhs_function.value_list(points, f_new);
          for (unsigned int time_point = 0; time_point < in_time.size();
               ++time_point)
            {
              rhs_function.set_time(t_old + in_time.point(time_point)[0] * dt);
              f_at[time_point].resize(points.size());
              rhs_function.value_list(points, f_at[time_point]);
            }

          // The integrand in time at every point. The operators are
          // polynomials on every cell, so the loop runs over the cells of the
          // polytope, whose points the handler lists one cell after the other
          // in the order of get_agglomerate().
          unsigned int q = 0;
          for (const auto &cell : polytope->get_agglomerate())
            {
              const unsigned int cell_index = cell->active_cell_index();
              for (const unsigned int end = q + n_q_cell; q < end; ++q)
                {
                  const double jump =
                    CellPolynomial::evaluate<dim>(operator_new[cell_index],
                                                  points[q]) -
                    CellPolynomial::evaluate<dim>(operator_old[cell_index],
                                                  points[q]);
                  double time_integral = 0.;
                  for (unsigned int time_point = 0; time_point < in_time.size();
                       ++time_point)
                    {
                      const double ell = 1. - in_time.point(time_point)[0];
                      const double residual =
                        f_at[time_point][q] - f_new[q] + ell * jump;
                      time_integral +=
                        residual * residual * in_time.weight(time_point) * dt;
                    }
                  sum_sq += time_integral * fe_values.JxW(q);
                }
            }
          AssertDimension(q, points.size());
        }
    rhs_function.set_time(t_old + dt);
    return std::sqrt(
      std::min(1., final_time) *
      Utilities::MPI::sum(sum_sq,
                          ah.get_triangulation().get_mpi_communicator()));
  }



  /**
   * Parameters of StepController.
   */
  struct StepControlParameters
  {
    /**
     * The initial step length.
     */
    double initial_dt = 0.01;

    /**
     * Steps whose error indicator in time is over this tolerance are halved.
     * Zero switches the control of the step length off.
     */
    double time_tolerance = 0.;

    /**
     * Steps whose error indicator in time is under this fraction of the time
     * tolerance are doubled; zero never doubles, as in the experiments of
     * the paper.
     */
    double time_doubling_fraction = 0.;

    /**
     * Steps are not doubled beyond this length.
     */
    double max_dt = std::numeric_limits<double>::max();

    /**
     * Steps are not halved below this length.
     */
    double min_dt = 0.;

    /**
     * Largest number of mesh adaptations in the loop of the first step.
     */
    unsigned int max_first_step_adaptations = 10;
  };



  /**
   * The decision of StepController::decide() after a step has been solved.
   */
  enum class StepDecision
  {
    /**
     * Solve the step again on the same mesh, with the new length.
     */
    solve_again,
    /**
     * Refine and coarsen the mesh once, then solve the step again. In the
     * loop of the first step the initial value is computed again on the new
     * mesh first.
     */
    adapt_and_solve_again,
    /**
     * End of the loop of the first step (line 22 of Algorithm 1): adapt the
     * mesh once more without refinement, since no polytope is over the
     * tolerance any more, so that only merging remains. Then compute the
     * initial value again on the new mesh and solve the first step again.
     */
    merge_and_restart,
    /**
     * Accept the step.
     */
    accept
  };



  /**
   * The step and mesh control of Algorithm 1 of Cangiani, Georgoulis and
   * Metcalfe.
   *
   * The first step is repeated, halving or doubling its length and adapting
   * the mesh, with the initial value computed again on every new mesh. This
   * goes on until the error indicator in time is within its tolerance and
   * the largest error indicator of a polytope within the polytope
   * tolerance. Then the mesh is coarsened once and the time stepping starts.
   * Every later step is first brought within the time tolerance on the mesh
   * of the last step. Then the mesh is refined and coarsened once, and the
   * step is solved again and accepted.
   */
  class StepController
  {
  public:
    /**
     * Constructor.
     */
    StepController(const StepControlParameters &parameters);

    /**
     * The length of the next step to solve.
     */
    double
    step_length() const;

    /**
     * Whether the loop of the first step is over.
     */
    bool
    first_step_done() const;

    /**
     * The decision after a step has been solved, from the largest error
     * indicator of a polytope @p max_indicator against the polytope tolerance
     * @p polytope_tolerance, and from the error indicator in time
     * @p time_indicator.
     */
    StepDecision
    decide(const double max_indicator,
           const double polytope_tolerance,
           const double time_indicator);

  private:
    /**
     * The parameters of the control.
     */
    const StepControlParameters parameters;

    /**
     * The length of the next step to solve.
     */
    double dt;

    /**
     * Whether the loop of the first step, lines 5-22 of Algorithm 1, is
     * running.
     */
    bool in_first_step;

    /**
     * Whether the current step was halved already: a halved step is not
     * doubled again.
     */
    bool halved;

    /**
     * Whether the step being solved comes after a mesh adaptation of the
     * main loop, and is then accepted.
     */
    bool adapted;

    /**
     * Number of mesh adaptations in the loop of the first step so far.
     */
    unsigned int n_first_step_adaptations;
  };



  inline StepController::StepController(const StepControlParameters &parameters)
    : parameters(parameters)
    , dt(parameters.initial_dt)
    , in_first_step(true)
    , halved(false)
    , adapted(false)
    , n_first_step_adaptations(0)
  {}



  inline double
  StepController::step_length() const
  {
    return dt;
  }



  inline bool
  StepController::first_step_done() const
  {
    return !in_first_step;
  }



  inline StepDecision
  StepController::decide(const double max_indicator,
                         const double polytope_tolerance,
                         const double time_indicator)
  {
    // Whether the step is to be halved or may be doubled. A step is not
    // halved below the shortest length, not doubled beyond the longest, and
    // not doubled after it was halved.
    const bool step_control_on = parameters.time_tolerance > 0.;
    const bool time_over       = step_control_on &&
                           time_indicator > parameters.time_tolerance &&
                           dt > parameters.min_dt;
    const bool time_under = step_control_on &&
                            time_indicator < parameters.time_doubling_fraction *
                                               parameters.time_tolerance &&
                            !halved && 2. * dt <= parameters.max_dt;

    if (in_first_step)
      {
        // Lines 5-21: the step length and the mesh change together until
        // both error indicators are within their tolerances.
        const bool space_over =
          max_indicator > polytope_tolerance &&
          n_first_step_adaptations < parameters.max_first_step_adaptations;
        if (time_over)
          {
            dt *= 0.5;
            halved = true;
          }
        else if (time_under)
          dt *= 2.;
        if (space_over)
          {
            ++n_first_step_adaptations;
            return StepDecision::adapt_and_solve_again;
          }
        if (time_over || time_under)
          return StepDecision::solve_again;
        in_first_step = false;
        halved        = false;
        return StepDecision::merge_and_restart;
      }

    if (adapted)
      {
        // Line 41: solved again on the adapted mesh, accepted.
        adapted = false;
        halved  = false;
        return StepDecision::accept;
      }
    if (time_over)
      {
        // Lines 26-31.
        dt *= 0.5;
        halved = true;
        return StepDecision::solve_again;
      }
    if (time_under)
      {
        // Lines 32-36.
        dt *= 2.;
        return StepDecision::solve_again;
      }
    // Line 40: adapt the mesh once and solve the step again.
    adapted = true;
    return StepDecision::adapt_and_solve_again;
  }
} // namespace dealii::TimeAdaptivity

#endif
