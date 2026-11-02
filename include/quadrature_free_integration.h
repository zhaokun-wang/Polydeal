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


#ifndef quadrature_free_integration_h
#define quadrature_free_integration_h


#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/householder.h>
#include <deal.II/lac/vector.h>

#include <agglomeration_handler.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>


/**
 * Assembly of the SIPG matrices on polytopes without quadrature, following
 * P.F. Antonietti, P. Houston, G. Pennesi, "Fast numerical integration on
 * polytopic meshes with applications to discontinuous Galerkin finite element
 * methods", J. Sci. Comput. 77 (2018), with the optimisations of
 * T. Radley, P. Houston, M.E. Hubbard, Math. Eng. 6 (2024).
 *
 * Every entry of the volume stiffness and mass matrices is a linear
 * combination of the moments $\int_P t^\alpha$ of the polytope, $t$ the
 * centred coordinates $2\hat x - 1 \in [-1,1]^d$ of its bounding box. The
 * coefficients depend only on the element and are fitted once per degree.
 * The moments need no quadrature: a monomial is a homogeneous function, so
 * the divergence theorem turns its integral over the polytope into a sum of
 * integrals over the faces, each weighted by the distance of the face from
 * an origin. The same step turns a face integral into edge integrals and an
 * edge integral into the values at its two vertices.
 *
 * The interface terms use that every basis function of FE_AggloDGP is a
 * product of one-dimensional polynomials, one per coordinate direction. The
 * polynomials of both sides are rewritten in the coordinates of the
 * interface and multiplied direction by direction. An entry is then a linear
 * combination of the moments of the interface, which the same recursion
 * computes.
 */
namespace dealii::QuadratureFree
{
  /**
   * All multi-indices of total degree up to @p max_degree, ordered by degree,
   * and within one degree with the first exponent decreasing.
   */
  template <int dim>
  const std::vector<std::array<unsigned int, dim>> &
  multi_indices(const unsigned int max_degree)
  {
    static std::map<unsigned int, std::vector<std::array<unsigned int, dim>>>
               cache;
    const auto cached = cache.find(max_degree);
    if (cached != cache.end())
      return cached->second;

    std::vector<std::array<unsigned int, dim>> indices;
    std::array<unsigned int, dim>              index;
    index.fill(0);
    const auto generate = [&](const auto        &self,
                              const unsigned int direction,
                              const unsigned int remaining) -> void {
      if (direction == dim - 1)
        {
          index[direction] = remaining;
          indices.push_back(index);
          return;
        }
      for (int exponent = remaining; exponent >= 0; --exponent)
        {
          index[direction] = exponent;
          self(self, direction + 1, remaining - exponent);
        }
    };
    for (unsigned int degree = 0; degree <= max_degree; ++degree)
      generate(generate, 0, degree);
    return cache.emplace(max_degree, std::move(indices)).first->second;
  }



  /**
   * Monomial expansion, in the centred coordinates $t = 2\hat x - 1$ of the
   * reference cell, of the products of two basis functions of an element
   * ($\phi_i \phi_j$, @p derivative_order 0) or of two components of their
   * gradients ($\partial_{d_1} \phi_i \, \partial_{d_2} \phi_j$,
   * @p derivative_order 1, derivatives with respect to $\hat x$). A product
   * has degree $2p - 2\,$@p derivative_order. It is sampled at Gauss points
   * with FiniteElement::shape_value() or FiniteElement::shape_grad() and
   * fitted in that degree by least squares, which is exact up to rounding.
   */
  template <int dim, int derivative_order>
  class BasisProductExpansion
  {
    static_assert(derivative_order == 0 || derivative_order == 1,
                  "Products of values or of gradients.");

  public:
    /**
     * Constructor. Fit the products for the element @p fe.
     */
    BasisProductExpansion(const FiniteElement<dim> &fe);

    /**
     * Coefficients of the products, [component][i][j][monomial]; the
     * component is $d_1 d + d_2$ for gradients, 0 for values.
     */
    const std::vector<std::vector<std::vector<double>>> &
    get_coefficients(const unsigned int component) const;

    /**
     * Number of monomials of the expansion.
     */
    unsigned int
    n_monomials() const;

  private:
    /**
     * Number of monomials of the expansion.
     */
    unsigned int n_monomials_of_expansion;

    /**
     * The coefficients, [component][i][j][monomial].
     */
    std::vector<std::vector<std::vector<std::vector<double>>>> coefficients;
  };



  template <int dim, int derivative_order>
  BasisProductExpansion<dim, derivative_order>::BasisProductExpansion(
    const FiniteElement<dim> &fe)
  {
    const unsigned int degree       = (2 * fe.degree >= 2 * derivative_order) ?
                                        2 * fe.degree - 2 * derivative_order :
                                        0;
    const auto        &indices      = multi_indices<dim>(degree);
    const unsigned int n_dofs       = fe.n_dofs_per_cell();
    const unsigned int n_components = (derivative_order == 0) ? 1 : dim * dim;
    n_monomials_of_expansion        = indices.size();
    coefficients.assign(
      n_components,
      std::vector<std::vector<std::vector<double>>>(
        n_dofs,
        std::vector<std::vector<double>>(
          n_dofs, std::vector<double>(n_monomials_of_expansion, 0.))));

    const QGauss<dim>              quadrature(degree + 2);
    const std::vector<Point<dim>> &points   = quadrature.get_points();
    const unsigned int             n_points = points.size();

    // Least squares by Householder QR: the Vandermonde matrix is badly
    // conditioned, and the normal equations would square that.
    FullMatrix<double> vandermonde(n_points, n_monomials_of_expansion);
    for (unsigned int q = 0; q < n_points; ++q)
      for (unsigned int monomial = 0; monomial < n_monomials_of_expansion;
           ++monomial)
        {
          double value = 1.;
          for (unsigned int direction = 0; direction < dim; ++direction)
            value *= std::pow(2. * points[q][direction] - 1.,
                              indices[monomial][direction]);
          vandermonde(q, monomial) = value;
        }
    Householder<double> qr(vandermonde);

    std::vector<Vector<double>> samples(n_components, Vector<double>(n_points));
    Vector<double>              fitted(n_monomials_of_expansion);
    for (unsigned int i = 0; i < n_dofs; ++i)
      for (unsigned int j = 0; j < n_dofs; ++j)
        {
          for (unsigned int q = 0; q < n_points; ++q)
            if constexpr (derivative_order == 0)
              samples[0][q] =
                fe.shape_value(i, points[q]) * fe.shape_value(j, points[q]);
            else
              {
                const Tensor<1, dim> grad_i = fe.shape_grad(i, points[q]);
                const Tensor<1, dim> grad_j = fe.shape_grad(j, points[q]);
                for (unsigned int d1 = 0; d1 < dim; ++d1)
                  for (unsigned int d2 = 0; d2 < dim; ++d2)
                    samples[d1 * dim + d2][q] = grad_i[d1] * grad_j[d2];
              }
          for (unsigned int component = 0; component < n_components;
               ++component)
            {
              qr.least_squares(fitted, samples[component]);
              for (unsigned int monomial = 0;
                   monomial < n_monomials_of_expansion;
                   ++monomial)
                coefficients[component][i][j][monomial] = fitted[monomial];
            }
        }
  }



  template <int dim, int derivative_order>
  const std::vector<std::vector<std::vector<double>>> &
  BasisProductExpansion<dim, derivative_order>::get_coefficients(
    const unsigned int component) const
  {
    return coefficients[component];
  }



  template <int dim, int derivative_order>
  unsigned int
  BasisProductExpansion<dim, derivative_order>::n_monomials() const
  {
    return n_monomials_of_expansion;
  }



  /**
   * The boundary faces of the union of @p cells, each with the centre of the
   * cell it belongs to, which orients its outward normal. A face is on the
   * boundary if the cell on its other side is not one of @p cells; across a
   * hanging node every subface is tested against its own neighbour, so the
   * faces always close up.
   */
  template <int dim, typename CellContainer>
  std::vector<std::pair<typename Triangulation<dim>::face_iterator, Point<dim>>>
  boundary_faces_of_cells(const CellContainer &cells)
  {
    std::set<CellId> inside;
    for (const auto &cell : cells)
      inside.insert(cell->id());

    std::vector<
      std::pair<typename Triangulation<dim>::face_iterator, Point<dim>>>
      faces;
    for (const auto &cell : cells)
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            faces.emplace_back(cell->face(f), cell->center());
          else if (cell->face(f)->has_children())
            {
              for (unsigned int subface = 0;
                   subface < cell->face(f)->n_children();
                   ++subface)
                if (inside.count(
                      cell->neighbor_child_on_subface(f, subface)->id()) == 0)
                  faces.emplace_back(cell->face(f)->child(subface),
                                     cell->center());
            }
          else if (inside.count(cell->neighbor(f)->id()) == 0)
            faces.emplace_back(cell->face(f), cell->center());
        }
    return faces;
  }



  namespace internal
  {
    /**
     * A table of the multi-indices of degree up to some maximum, indexed as
     * in multi_indices(), for the recursion of the moments. The recursion
     * needs, for every multi-index $\alpha$, the moment of $\alpha - e_j$,
     * that is, of $\alpha$ with the exponent in direction $j$ lowered by one.
     * That multi-index comes earlier in the list, and lowered gives its index.
     */
    template <int dim>
    struct MultiIndexTable
    {
      /**
       * exponents[alpha] is the multi-index with index alpha.
       */
      std::vector<std::array<unsigned int, dim>> exponents;

      /**
       * lowered[alpha][j] is the index of $\alpha - e_j$, the multi-index
       * alpha with its exponent in direction j lowered by one, and -1 where
       * that exponent is zero.
       */
      std::vector<std::array<int, dim>> lowered;

      /**
       * degree[alpha] is $|\alpha|$.
       */
      std::vector<unsigned int> degree;
    };



    /**
     * The MultiIndexTable of degree @p max_degree, built once and kept.
     */
    template <int dim>
    const MultiIndexTable<dim> &
    multi_index_table(const unsigned int max_degree)
    {
      static std::map<unsigned int, MultiIndexTable<dim>> cache;
      const auto cached = cache.find(max_degree);
      if (cached != cache.end())
        return cached->second;
      const auto &indices = multi_indices<dim>(max_degree);
      std::map<std::array<unsigned int, dim>, int> index_of;
      for (unsigned int alpha = 0; alpha < indices.size(); ++alpha)
        index_of[indices[alpha]] = alpha;
      MultiIndexTable<dim> index_table;
      index_table.exponents = indices;
      index_table.lowered.resize(indices.size());
      index_table.degree.assign(indices.size(), 0);
      for (unsigned int alpha = 0; alpha < indices.size(); ++alpha)
        for (unsigned int direction = 0; direction < dim; ++direction)
          {
            index_table.degree[alpha] += indices[alpha][direction];
            index_table.lowered[alpha][direction] = -1;
            if (indices[alpha][direction] > 0)
              {
                auto lowered_index = indices[alpha];
                --lowered_index[direction];
                index_table.lowered[alpha][direction] =
                  index_of.at(lowered_index);
              }
          }
      return cache.emplace(max_degree, std::move(index_table)).first->second;
    }



    /**
     * The moments of a vertex, $v^\alpha$ for every $\alpha$ of
     * @p index_table. Each is the moment of a lowered multi-index times one
     * coordinate of the vertex.
     */
    template <int dim>
    void
    vertex_moments(const Point<dim>           &vertex,
                   const MultiIndexTable<dim> &index_table,
                   std::vector<double>        &moments)
    {
      moments.resize(index_table.degree.size());
      moments[0] = 1.;
      for (unsigned int alpha = 1; alpha < moments.size(); ++alpha)
        for (unsigned int direction = 0; direction < dim; ++direction)
          if (index_table.lowered[alpha][direction] >= 0)
            {
              moments[alpha] = moments[index_table.lowered[alpha][direction]] *
                               vertex[direction];
              break;
            }
    }



    /**
     * Close one step of the recursion of Antonietti, Houston and Pennesi, in
     * the notation of Radley, Houston and Hubbard (Algorithm 1). There, a
     * facet is any of the polytope itself, its faces, their edges and their
     * vertices, of dimension $k$ from $d$ down to 0, and the facets $G$ of a
     * facet $F$ are those of dimension $k - 1$ on its boundary.
     *
     * For a facet $F$ of dimension $k$ = @p facet_dimension with local origin
     * $x_F$ = @p origin in its affine hull: on entry @p moments holds
     * $\sum_G \text{dist}(G, x_F) I(G,\alpha)$ over the facets $G$ of $F$, on
     * exit $I(F, \alpha) = \int_F x^\alpha$,
     * @f[
     *   I(F, \alpha) = \frac{1}{k + |\alpha|} \Big( \sum_G \text{dist}(G, x_F)
     *     I(G, \alpha) + \sum_j \alpha_j (x_F)_j I(F, \alpha - e_j) \Big).
     * @f]
     */
    template <int dim>
    void
    close_recursion(const unsigned int          facet_dimension,
                    const Point<dim>           &origin,
                    const MultiIndexTable<dim> &index_table,
                    std::vector<double>        &moments)
    {
      for (unsigned int alpha = 0; alpha < moments.size(); ++alpha)
        {
          double value = moments[alpha];
          for (unsigned int direction = 0; direction < dim; ++direction)
            if (index_table.lowered[alpha][direction] >= 0)
              value += index_table.exponents[alpha][direction] *
                       origin[direction] *
                       moments[index_table.lowered[alpha][direction]];
          moments[alpha] =
            value / (facet_dimension + index_table.degree[alpha]);
        }
    }



    /**
     * The moments of the edge from @p a to @p b. The boundary of an edge is
     * its two end points. With @p a as the origin, only @p b has a non-zero
     * distance $|b - a|$ from it, and the moment of a point is the value of
     * the monomial there. Return the length.
     */
    template <int dim>
    double
    edge_moments(const Point<dim>           &a,
                 const Point<dim>           &b,
                 const MultiIndexTable<dim> &index_table,
                 std::vector<double>        &moments)
    {
      vertex_moments(b, index_table, moments);
      const double length = (b - a).norm();
      for (double &value : moments)
        value *= length;
      close_recursion<dim>(1, a, index_table, moments);
      return length;
    }



    /**
     * Facets closer than this to the local origin (in the coordinates of the
     * recursion) pass through it and are skipped.
     */
    constexpr double prune_tolerance = 1e-13;



    /**
     * Binomial coefficient $\binom{n}{k}$, $n < 64$.
     */
    inline double
    binomial(const unsigned int n, const unsigned int k)
    {
      static const auto table = [] {
        std::vector<std::array<double, 64>> pascal(64);
        for (unsigned int row = 0; row < 64; ++row)
          {
            pascal[row].fill(0.);
            pascal[row][0] = 1.;
            for (unsigned int column = 1; column <= row; ++column)
              pascal[row][column] =
                pascal[row - 1][column - 1] +
                (column < row ? pascal[row - 1][column] : 0.);
          }
        return pascal;
      }();
      return table[n][k];
    }



    /**
     * Substitute $t = \text{scaling}\, u + \text{shift}$ into the polynomial
     * $\sum_k c_k t^k$ of degree @p degree with the @p coefficients $c_k$, and
     * write the coefficients of the result as a polynomial in $u$ to
     * @p substituted:
     * @f[
     *   \text{substituted}_m = \text{scaling}^m \sum_{k \ge m} c_k
     *     \binom{k}{m} \text{shift}^{k - m}.
     * @f]
     */
    inline void
    substitute_affine(const double      *coefficients,
                      const unsigned int degree,
                      const double       scaling,
                      const double       shift,
                      double            *substituted)
    {
      double scaling_power[64], shift_power[64];
      scaling_power[0] = shift_power[0] = 1.;
      for (unsigned int power = 1; power <= degree; ++power)
        {
          scaling_power[power] = scaling_power[power - 1] * scaling;
          shift_power[power]   = shift_power[power - 1] * shift;
        }
      for (unsigned int exponent = 0; exponent <= degree; ++exponent)
        {
          double value = 0.;
          for (unsigned int power = exponent; power <= degree; ++power)
            value += coefficients[power] * binomial(power, exponent) *
                     shift_power[power - exponent];
          substituted[exponent] = value * scaling_power[exponent];
        }
    }
  } // namespace internal



  /**
   * The moments $\int_P t^\alpha \, d\hat x$, $|\alpha| \le$ @p max_degree,
   * of the union $P$ of @p cells, $t = 2\hat x - 1$ and $\hat x$ the unit
   * coordinates of the bounding box @p box. The faces must be planar.
   *
   * The recursion runs from the boundary faces of the cells
   * (boundary_faces_of_cells()) with the first vertex as the origin, so that
   * every face through it is skipped, and computes each edge (and in 3D each
   * vertex) once.
   */
  template <int dim, typename CellContainer>
  std::vector<double>
  moments_of_cells(const CellContainer    &cells,
                   const BoundingBox<dim> &box,
                   const unsigned int      max_degree)
  {
    static_assert(dim == 2 || dim == 3, "Implemented for dim = 2 and 3.");
    const auto &index_table    = internal::multi_index_table<dim>(max_degree);
    const unsigned int n_terms = index_table.degree.size();

    // The moments of every edge, computed once and kept by the global indices
    // of its two vertices, since an edge is shared by the faces around it.
    std::map<std::pair<unsigned int, unsigned int>, std::vector<double>>
               edge_cache;
    const auto edge = [&](unsigned int index_a,
                          Point<dim>   a,
                          unsigned int index_b,
                          Point<dim>   b) -> const std::vector<double> & {
      if (index_b < index_a)
        {
          std::swap(index_a, index_b);
          std::swap(a, b);
        }
      auto it = edge_cache.find({index_a, index_b});
      if (it == edge_cache.end())
        {
          std::vector<double> moments;
          internal::edge_moments<dim>(a, b, index_table, moments);
          it = edge_cache
                 .emplace(std::make_pair(index_a, index_b), std::move(moments))
                 .first;
        }
      return it->second;
    };
    // The centred coordinates t = 2 x_hat - 1 of the bounding box.
    const auto to_centred = [&](const Point<dim> &point) {
      Point<dim> t = box.real_to_unit(point);
      for (unsigned int direction = 0; direction < dim; ++direction)
        t[direction] = 2. * t[direction] - 1.;
      return t;
    };

    // The recursion over the boundary faces, with the first vertex met as
    // the origin. A face contributes its own moments times its distance from
    // the origin along its outward normal; faces through the origin are
    // skipped.
    std::vector<double> moments(n_terms, 0.);
    bool                have_origin = false;
    Point<dim>          origin;
    for (const auto &[face, owner_centre] : boundary_faces_of_cells<dim>(cells))
      {
        const Point<dim> owner  = to_centred(owner_centre);
        const Point<dim> centre = to_centred(face->center());
        std::array<Point<dim>, GeometryInfo<dim>::vertices_per_face> vertex;
        for (unsigned int i = 0; i < vertex.size(); ++i)
          vertex[i] = to_centred(face->vertex(i));
        if (!have_origin)
          {
            origin      = vertex[0];
            have_origin = true;
          }

        Tensor<1, dim> normal;
        if constexpr (dim == 2)
          {
            normal[0] = vertex[1][1] - vertex[0][1];
            normal[1] = vertex[0][0] - vertex[1][0];
          }
        else
          normal =
            cross_product_3d(vertex[1] - vertex[0], vertex[2] - vertex[0]);
        normal /= normal.norm();
        if (normal * (centre - owner) < 0)
          normal *= -1.;

        const double distance = normal * (vertex[0] - origin);
        if (std::abs(distance) < internal::prune_tolerance)
          continue;

        if constexpr (dim == 2)
          {
            const auto &face_moments = edge(face->vertex_index(0),
                                            vertex[0],
                                            face->vertex_index(1),
                                            vertex[1]);
            for (unsigned int alpha = 0; alpha < n_terms; ++alpha)
              moments[alpha] += distance * face_moments[alpha];
          }
        else
          {
            // The face as a facet with origin vertex[0]: its four edges in
            // cyclic order (deal.II numbers the vertices of a quad
            // lexicographically), the two through vertex[0] skipped.
            const std::array<unsigned int, 4> cycle = {{0, 1, 3, 2}};
            std::vector<double>               face_moments(n_terms, 0.);
            for (unsigned int edge_number = 0; edge_number < 4; ++edge_number)
              {
                const unsigned int i = cycle[edge_number],
                                   j = cycle[(edge_number + 1) % 4];
                const Point<3> middle =
                  vertex[i] + 0.5 * (vertex[j] - vertex[i]);
                Tensor<1, 3> edge_normal =
                  cross_product_3d(vertex[j] - vertex[i], normal);
                edge_normal /= edge_normal.norm();
                if (edge_normal * (middle - centre) < 0)
                  edge_normal *= -1.;
                const double edge_distance =
                  edge_normal * (vertex[i] - vertex[0]);
                if (std::abs(edge_distance) < internal::prune_tolerance)
                  continue;
                const auto &moments_of_edge = edge(face->vertex_index(i),
                                                   vertex[i],
                                                   face->vertex_index(j),
                                                   vertex[j]);
                for (unsigned int alpha = 0; alpha < n_terms; ++alpha)
                  face_moments[alpha] += edge_distance * moments_of_edge[alpha];
              }
            internal::close_recursion<3>(2,
                                         vertex[0],
                                         index_table,
                                         face_moments);
            for (unsigned int alpha = 0; alpha < n_terms; ++alpha)
              moments[alpha] += distance * face_moments[alpha];
          }
      }
    internal::close_recursion<dim>(dim, origin, index_table, moments);
    // d x_hat = 2^-dim dt.
    for (double &value : moments)
      value /= (1 << dim);
    return moments;
  }



  /**
   * The basis of FE_AggloDGP as products of one-dimensional polynomials, one
   * per coordinate direction,
   * $\phi_i(t) = c_0^{-(d-1)} B_{i_1}(t_1) \cdots B_{i_d}(t_d)$ in the
   * centred coordinates $t$ of the bounding box, $c_0 = \phi_0$ the constant
   * function. The face terms need this form: the bounding boxes of the two
   * sides of an interface are axis-aligned, so the change from the
   * coordinates of a side to those of the interface acts on each direction
   * alone, and so does it on each factor $B_{i_d}(t_d)$.
   */
  template <int dim>
  struct SeparableBasis
  {
    /**
     * The degree $p$ of the element.
     */
    unsigned int degree = 0;

    /**
     * Number of basis functions.
     */
    unsigned int n_dofs = 0;

    /**
     * The factor $c_0^{-(d-1)}$ in front of the product.
     */
    double prefactor = 1.;

    /**
     * exponents[i][d] is $i_d$, the index of the one-dimensional polynomial
     * of basis function i in direction d.
     */
    std::vector<std::array<unsigned int, dim>> exponents;

    /**
     * coefficients_1d[n][k] is the coefficient of $t^k$ in $B_n$.
     */
    std::vector<std::vector<double>> coefficients_1d;
  };



  /**
   * The separable form of the element @p fe, fitted from the values of its
   * basis functions and checked against them. Throw if the element is not a
   * product of one-dimensional polynomials with a constant first function.
   */
  template <int dim>
  SeparableBasis<dim>
  separable_basis(const FiniteElement<dim> &fe)
  {
    const unsigned int  degree = fe.degree;
    SeparableBasis<dim> basis;
    basis.degree = degree;
    basis.n_dofs = fe.n_dofs_per_cell();

    // Two points inside the reference cell, away from any symmetry, at which
    // the element is checked below. The first function must be the constant
    // c_0.
    Point<dim> check_point_a, check_point_b;
    for (unsigned int direction = 0; direction < dim; ++direction)
      {
        check_point_a[direction] = 0.13 + 0.1 * direction;
        check_point_b[direction] = 0.71 - 0.05 * direction;
      }
    const double constant_value = fe.shape_value(0, check_point_a);
    AssertThrow(std::abs(fe.shape_value(0, check_point_b) - constant_value) <
                  1e-12 * std::abs(constant_value),
                ExcMessage("The first basis function is not constant."));
    basis.prefactor = std::pow(constant_value, -static_cast<double>(dim - 1));

    // The exponents of every basis function: fit it in the monomials of
    // degree p. Its exponent in direction d is the largest exponent of t_d in
    // the fit.
    {
      const auto        &indices = multi_indices<dim>(degree);
      const QGauss<dim>  points(degree + 2);
      FullMatrix<double> vandermonde(points.size(), indices.size());
      for (unsigned int q = 0; q < points.size(); ++q)
        for (unsigned int monomial = 0; monomial < indices.size(); ++monomial)
          {
            double value = 1.;
            for (unsigned int direction = 0; direction < dim; ++direction)
              value *= std::pow(2. * points.point(q)[direction] - 1.,
                                indices[monomial][direction]);
            vandermonde(q, monomial) = value;
          }
      Householder<double> qr(vandermonde);
      Vector<double>      samples(points.size()), fitted(indices.size());
      basis.exponents.resize(basis.n_dofs);
      for (unsigned int i = 0; i < basis.n_dofs; ++i)
        {
          for (unsigned int q = 0; q < points.size(); ++q)
            samples[q] = fe.shape_value(i, points.point(q));
          qr.least_squares(fitted, samples);
          double largest = 0.;
          for (unsigned int monomial = 0; monomial < indices.size(); ++monomial)
            largest = std::max(largest, std::abs(fitted[monomial]));
          basis.exponents[i].fill(0);
          for (unsigned int monomial = 0; monomial < indices.size(); ++monomial)
            if (std::abs(fitted[monomial]) > 1e-8 * largest)
              for (unsigned int direction = 0; direction < dim; ++direction)
                basis.exponents[i][direction] =
                  std::max(basis.exponents[i][direction],
                           indices[monomial][direction]);
        }
    }

    // B_n by a fit along a line t_2 = ... = const of the function with
    // exponents (n, 0, ..., 0): a fit of degree p, far better conditioned
    // than reading B_n off the fit of degree 2p.
    basis.coefficients_1d.assign(degree + 1,
                                 std::vector<double>(degree + 1, 0.));
    std::vector<bool>  found(degree + 1, false);
    const QGauss<1>    line(degree + 2);
    FullMatrix<double> vandermonde(line.size(), degree + 1);
    for (unsigned int q = 0; q < line.size(); ++q)
      for (unsigned int power = 0; power <= degree; ++power)
        vandermonde(q, power) = std::pow(2. * line.point(q)[0] - 1., power);
    Householder<double> qr(vandermonde);
    Vector<double>      samples(line.size()), fitted(degree + 1);
    for (unsigned int i = 0; i < basis.n_dofs; ++i)
      {
        bool along_first = true;
        for (unsigned int direction = 1; direction < dim; ++direction)
          along_first = along_first && basis.exponents[i][direction] == 0;
        if (!along_first)
          continue;
        const unsigned int index_1d = basis.exponents[i][0];
        found[index_1d]             = true;
        for (unsigned int q = 0; q < line.size(); ++q)
          {
            Point<dim> point = check_point_a;
            point[0]         = line.point(q)[0];
            samples[q]       = fe.shape_value(i, point);
          }
        qr.least_squares(fitted, samples);
        for (unsigned int power = 0; power <= degree; ++power)
          basis.coefficients_1d[index_1d][power] = fitted[power];
      }
    for (unsigned int index_1d = 0; index_1d <= degree; ++index_1d)
      AssertThrow(found[index_1d],
                  ExcMessage(
                    "No basis function with exponents (n, 0, ..., 0)."));

    // Check the product form against the element at the two points.
    const auto evaluate_1d = [&](const unsigned int index_1d, const double t) {
      double value = 0.;
      for (int power = degree; power >= 0; --power)
        value = value * t + basis.coefficients_1d[index_1d][power];
      return value;
    };
    double worst = 0., largest = 0.;
    for (const auto &point : {check_point_a, check_point_b})
      for (unsigned int i = 0; i < basis.n_dofs; ++i)
        {
          double value = basis.prefactor;
          for (unsigned int direction = 0; direction < dim; ++direction)
            value *= evaluate_1d(basis.exponents[i][direction],
                                 2. * point[direction] - 1.);
          const double reference = fe.shape_value(i, point);
          largest                = std::max(largest, std::abs(reference));
          worst                  = std::max(worst, std::abs(value - reference));
        }
    AssertThrow(worst <= 1e-9 * largest,
                ExcMessage("The product form does not reproduce the element."));
    return basis;
  }



  /**
   * A piece of an interface: a straight segment in 2D, a planar
   * quadrilateral in 3D (vertices in deal.II's face order), with the unit
   * normal pointing out of the first side.
   */
  template <int dim>
  struct InterfacePiece
  {
    /**
     * The vertices of the piece.
     */
    std::array<Point<dim>, GeometryInfo<dim>::vertices_per_face> vertex;

    /**
     * The unit normal pointing out of the first side.
     */
    Tensor<1, dim> normal;

    /**
     * A unit normal, of either orientation.
     */
    Tensor<1, dim>
    unit_normal() const
    {
      Tensor<1, dim> result;
      if constexpr (dim == 2)
        {
          result[0] = vertex[1][1] - vertex[0][1];
          result[1] = vertex[0][0] - vertex[1][0];
        }
      else
        result = cross_product_3d(vertex[1] - vertex[0], vertex[2] - vertex[0]);
      return result / result.norm();
    }

    /**
     * Length, or area.
     */
    double
    measure() const
    {
      if constexpr (dim == 2)
        return (vertex[1] - vertex[0]).norm();
      else
        return 0.5 *
               cross_product_3d(vertex[3] - vertex[0], vertex[2] - vertex[1])
                 .norm();
    }
  };



  /**
   * The face matrices of one interface,
   * @f[
   *   M_{IJ} = \int_F \phi_I \phi_J, \qquad
   *   F_{IJ} = \int_F (\nabla\phi_I \cdot n) \phi_J,
   * @f]
   * $I$ and $J$ running over the basis functions of the first side and then
   * of the second (if @p bases has two entries), $n$ the normal out of the
   * first side, from the separable bases and bounding boxes of the sides and
   * the @p pieces of the interface.
   *
   * The one-dimensional polynomials of both sides are rewritten in the
   * coordinates $u$ of the interface, centred on its bounding box and scaled
   * to it, so that $|u_d| \le 1$ on the interface. A direction in which the
   * interface does not extend has $u_d = 0$ on it, and drops out. The
   * polynomials are multiplied direction by direction, and an entry is the
   * sum of the coefficients of the product times the moments of the
   * interface in $u$, computed by the recursion of moments_of_cells() and
   * summed one direction at a time.
   */
  template <int dim>
  void
  interface_matrices(const std::vector<const SeparableBasis<dim> *> &bases,
                     const std::vector<BoundingBox<dim>>            &boxes,
                     const std::vector<InterfacePiece<dim>>         &pieces,
                     FullMatrix<double>                             &face_mass,
                     FullMatrix<double>                             &face_flux)
  {
    static_assert(dim == 2 || dim == 3, "Implemented for dim = 2 and 3.");

    // The rows and columns of the sides: those of side s start at offset[s].
    const unsigned int          n_sides    = bases.size();
    std::array<unsigned int, 3> offset     = {{0, 0, 0}};
    unsigned int                max_degree = 0;
    for (unsigned int side = 0; side < n_sides; ++side)
      {
        offset[side + 1] = offset[side] + bases[side]->n_dofs;
        max_degree       = std::max(max_degree, bases[side]->degree);
      }
    AssertThrow(2 * max_degree < 63, ExcMessage("The degree is too high."));
    // The largest degree of a product of two basis functions, and the number
    // of exponents 0, ..., max_product_degree in one direction.
    const unsigned int max_product_degree = 2 * max_degree;
    const unsigned int n_exponents        = max_product_degree + 1;

    // 1. The coordinates u of the interface: centred on its bounding box and
    // scaled to it, so that |u_d| <= 1 on the interface. A direction in which
    // the interface does not extend is flat, and has u_d = 0 there.
    Point<dim> face_lo = pieces.front().vertex[0], face_hi = face_lo;
    for (const auto &piece : pieces)
      for (const auto &point : piece.vertex)
        for (unsigned int direction = 0; direction < dim; ++direction)
          {
            face_lo[direction] = std::min(face_lo[direction], point[direction]);
            face_hi[direction] = std::max(face_hi[direction], point[direction]);
          }
    double extent = 0.;
    for (unsigned int direction = 0; direction < dim; ++direction)
      extent = std::max(extent, face_hi[direction] - face_lo[direction]);
    std::array<double, dim> lo_u, length_u;
    std::array<bool, dim>   flat;
    for (unsigned int direction = 0; direction < dim; ++direction)
      {
        flat[direction] =
          face_hi[direction] - face_lo[direction] <= 1e-12 * extent;
        length_u[direction] =
          flat[direction] ? extent : face_hi[direction] - face_lo[direction];
        lo_u[direction] = 0.5 * (face_lo[direction] + face_hi[direction]) -
                          0.5 * length_u[direction];
      }
    const auto to_u = [&](const Point<dim> &point) {
      Point<dim> u;
      for (unsigned int direction = 0; direction < dim; ++direction)
        u[direction] =
          flat[direction] ?
            0. :
            2. * (point[direction] - lo_u[direction]) / length_u[direction] -
              1.;
      return u;
    };

    // 2. The moments of the interface in u, by the recursion of
    // moments_of_cells() applied to every piece. Table 0 holds the plain
    // moments, table 1 + d those weighted by the normal component n_d. The
    // tables are dense: the moment of exponents (k_0, ..., k_{dim-1}) is at
    // sum_d k_d n_exponents^d.
    const auto &index_table =
      internal::multi_index_table<dim>(max_product_degree);
    const unsigned int n_terms    = index_table.degree.size();
    unsigned int       table_size = 1;
    for (unsigned int direction = 0; direction < dim; ++direction)
      table_size *= n_exponents;
    thread_local std::vector<double> moments[dim + 1];
    for (auto &table : moments)
      table.assign(table_size, 0.);
    thread_local std::vector<double> piece_moments, edge_moments;
    piece_moments.resize(n_terms);
    edge_moments.resize(n_terms);
    // The moments of the edge from a to b with origin a, in u.
    const auto moments_of_edge =
      [&](const Point<dim> &a, const Point<dim> &b, std::vector<double> &edge) {
        const double length = (b - a).norm();
        edge[0]             = length;
        for (unsigned int alpha = 1; alpha < n_terms; ++alpha)
          for (unsigned int direction = 0; direction < dim; ++direction)
            if (index_table.lowered[alpha][direction] >= 0)
              {
                edge[alpha] =
                  edge[index_table.lowered[alpha][direction]] * b[direction];
                break;
              }
        internal::close_recursion<dim>(1, a, index_table, edge);
        return length;
      };
    for (const auto &piece : pieces)
      {
        std::array<Point<dim>, GeometryInfo<dim>::vertices_per_face> vertex_u;
        for (unsigned int vertex = 0; vertex < vertex_u.size(); ++vertex)
          vertex_u[vertex] = to_u(piece.vertex[vertex]);
        double measure_u = 0.;
        if constexpr (dim == 2)
          measure_u = moments_of_edge(vertex_u[0], vertex_u[1], piece_moments);
        else
          {
            // A quadrilateral: its four edges in cyclic order, with the first
            // vertex as the origin, so that the two edges through it are
            // skipped.
            const Tensor<1, 3> cross =
              cross_product_3d(vertex_u[3] - vertex_u[0],
                               vertex_u[2] - vertex_u[1]);
            measure_u                   = 0.5 * cross.norm();
            const Tensor<1, 3> normal_u = cross / cross.norm();
            const Point<3>     centroid =
              vertex_u[0] + 0.25 * ((vertex_u[1] - vertex_u[0]) +
                                    (vertex_u[2] - vertex_u[0]) +
                                    (vertex_u[3] - vertex_u[0]));
            std::fill(piece_moments.begin(), piece_moments.end(), 0.);
            const std::array<unsigned int, 4> cycle = {{0, 1, 3, 2}};
            for (unsigned int edge = 0; edge < 4; ++edge)
              {
                const unsigned int i = cycle[edge], j = cycle[(edge + 1) % 4];
                Tensor<1, 3>       edge_normal =
                  cross_product_3d(vertex_u[j] - vertex_u[i], normal_u);
                edge_normal /= edge_normal.norm();
                if (edge_normal *
                      (vertex_u[i] + 0.5 * (vertex_u[j] - vertex_u[i]) -
                       centroid) <
                    0)
                  edge_normal *= -1.;
                const double distance =
                  edge_normal * (vertex_u[i] - vertex_u[0]);
                if (std::abs(distance) < internal::prune_tolerance)
                  continue;
                moments_of_edge(vertex_u[i], vertex_u[j], edge_moments);
                for (unsigned int alpha = 0; alpha < n_terms; ++alpha)
                  piece_moments[alpha] += distance * edge_moments[alpha];
              }
            internal::close_recursion<dim>(2,
                                           vertex_u[0],
                                           index_table,
                                           piece_moments);
          }
        // From the measure in u back to the measure in x.
        const double jacobian = piece.measure() / measure_u;
        for (unsigned int alpha = 0; alpha < n_terms; ++alpha)
          {
            unsigned int index = 0;
            for (unsigned int direction = dim; direction-- > 0;)
              index =
                index * n_exponents + index_table.exponents[alpha][direction];
            moments[0][index] += jacobian * piece_moments[alpha];
            for (unsigned int direction = 0; direction < dim; ++direction)
              moments[1 + direction][index] +=
                jacobian * piece.normal[direction] * piece_moments[alpha];
          }
      }

    // 3. The one-dimensional polynomials B_n of every side, per direction,
    // written as polynomials in u_d: their values, and their derivatives
    // with respect to x_d. value[slot(side, d, n) + k] is the coefficient of
    // u_d^k.
    const unsigned int               n_coefficients_1d = max_degree + 1;
    thread_local std::vector<double> value, gradient;
    value.assign(n_sides * dim * n_coefficients_1d * n_coefficients_1d, 0.);
    gradient.assign(n_sides * dim * n_coefficients_1d * n_coefficients_1d, 0.);
    const auto slot = [&](const unsigned int side,
                          const unsigned int direction,
                          const unsigned int index_1d) {
      return ((side * dim + direction) * n_coefficients_1d + index_1d) *
             n_coefficients_1d;
    };
    for (unsigned int side = 0; side < n_sides; ++side)
      {
        const unsigned int degree = bases[side]->degree;
        for (unsigned int direction = 0; direction < dim; ++direction)
          {
            // The centred coordinate t_d of the box of the side as an affine
            // function of u_d: t_d = scaling u_d + shift.
            const double lo_side =
              boxes[side].get_boundary_points().first[direction];
            const double length_side = boxes[side].side_length(direction);
            const double scaling     = length_u[direction] / length_side;
            const double shift       = (2. * (lo_u[direction] - lo_side) +
                                  length_u[direction] - length_side) /
                                 length_side;
            for (unsigned int index_1d = 0; index_1d <= degree; ++index_1d)
              {
                const double *coefficients =
                  bases[side]->coefficients_1d[index_1d].data();
                internal::substitute_affine(
                  coefficients,
                  degree,
                  scaling,
                  shift,
                  &value[slot(side, direction, index_1d)]);
                // d/dx_d = (2 / length_side) d/dt_d.
                double derivative[64];
                for (unsigned int power = 0; power < degree; ++power)
                  derivative[power] =
                    (power + 1) * coefficients[power + 1] * 2. / length_side;
                derivative[degree] = 0.;
                internal::substitute_affine(
                  derivative,
                  degree,
                  scaling,
                  shift,
                  &gradient[slot(side, direction, index_1d)]);
              }
          }
      }

    face_mass.reinit(offset[n_sides], offset[n_sides]);
    face_flux.reinit(offset[n_sides], offset[n_sides]);

    // 4. The directions in which the interface extends, and the terms to
    // compute. Term 0 is the mass. Term 1 + e is the part of the flux with
    // d/dx_e, which vanishes when n_e does on every piece.
    std::array<unsigned int, dim> active{};
    unsigned int                  n_active = 0;
    for (unsigned int direction = 0; direction < dim; ++direction)
      if (!flat[direction])
        active[n_active++] = direction;
    AssertThrow(n_active > 0, ExcMessage("The interface is degenerate."));
    std::array<bool, dim + 1> term_present;
    term_present[0] = true;
    for (unsigned int direction = 0; direction < dim; ++direction)
      {
        term_present[1 + direction] = false;
        for (const auto &piece : pieces)
          term_present[1 + direction] =
            term_present[1 + direction] || piece.normal[direction] != 0.;
      }
    std::array<unsigned int, dim> stride;
    stride[0] = 1;
    for (unsigned int direction = 1; direction < dim; ++direction)
      stride[direction] = stride[direction - 1] * n_exponents;

    // 5. The entries, for every pair of sides. A basis function is a product
    // of one-dimensional polynomials, so the integral of a product of two of
    // them is a contraction of the moments with one product of
    // one-dimensional polynomials per direction, done direction by direction.
    thread_local std::vector<double> products_value, products_gradient, partial,
      partial2, entry_mass, entry_flux, single;
    thread_local std::vector<char> partial2_done;
    for (unsigned int side_a = 0; side_a < n_sides; ++side_a)
      for (unsigned int side_b = 0; side_b < n_sides; ++side_b)
        {
          const auto &basis_a = *bases[side_a], &basis_b = *bases[side_b];
          const unsigned int degree_a = basis_a.degree,
                             degree_b = basis_b.degree;
          // The length of a product of two one-dimensional polynomials, and
          // the number of pairs (n, m) of one-dimensional polynomials.
          const unsigned int product_length = degree_a + degree_b + 1;
          const unsigned int n_pairs        = (degree_a + 1) * (degree_b + 1);

          // 5a. The products of one-dimensional polynomials per direction and
          // pair (n, m): of two values, and of a derivative and a value.
          products_value.assign(dim * n_pairs * product_length, 0.);
          products_gradient.assign(dim * n_pairs * product_length, 0.);
          for (unsigned int direction = 0; direction < dim; ++direction)
            for (unsigned int index_a = 0; index_a <= degree_a; ++index_a)
              for (unsigned int index_b = 0; index_b <= degree_b; ++index_b)
                {
                  const double *value_a =
                    &value[slot(side_a, direction, index_a)];
                  const double *gradient_a =
                    &gradient[slot(side_a, direction, index_a)];
                  const double *value_b =
                    &value[slot(side_b, direction, index_b)];
                  const unsigned int pair = index_a * (degree_b + 1) + index_b;
                  double            *product_value =
                    &products_value[(direction * n_pairs + pair) *
                                    product_length];
                  double *product_gradient =
                    &products_gradient[(direction * n_pairs + pair) *
                                       product_length];
                  for (unsigned int i = 0; i <= degree_a; ++i)
                    for (unsigned int j = 0; j <= degree_b; ++j)
                      {
                        product_value[i + j] += value_a[i] * value_b[j];
                        product_gradient[i + j] += gradient_a[i] * value_b[j];
                      }
                }
          // The pair of one-dimensional polynomials of the basis functions i
          // of side a and j of side b in direction d.
          const auto pair_of = [&](const unsigned int i,
                                   const unsigned int j,
                                   const unsigned int direction) {
            return basis_a.exponents[i][direction] * (degree_b + 1) +
                   basis_b.exponents[j][direction];
          };

          // 5b. The contraction, term by term.
          entry_mass.assign(basis_a.n_dofs * basis_b.n_dofs, 0.);
          entry_flux.assign(basis_a.n_dofs * basis_b.n_dofs, 0.);
          for (unsigned int term = 0; term <= dim; ++term)
            {
              if (!term_present[term])
                continue;
              const std::vector<double> &moment_table = moments[term];
              // The product in direction d for a pair: with the derivative in
              // the direction of the term, of the values otherwise.
              const auto product = [&](const unsigned int direction,
                                       const unsigned int pair) {
                return (term == 1 + direction) ?
                         &products_gradient[(direction * n_pairs + pair) *
                                            product_length] :
                         &products_value[(direction * n_pairs + pair) *
                                         product_length];
              };
              std::vector<double> &entry =
                (term == 0) ? entry_mass : entry_flux;
              // In a flat direction u_d = 0, so only the constant coefficient
              // of the product counts.
              const auto flat_factor = [&](const unsigned int i,
                                           const unsigned int j) {
                double factor = 1.;
                for (unsigned int direction = 0; direction < dim; ++direction)
                  if (flat[direction])
                    factor *= product(direction, pair_of(i, j, direction))[0];
                return factor;
              };
              const unsigned int first_active = active[0];
              if (n_active == 1)
                {
                  // One direction: contract with the moments in it, once per
                  // pair.
                  single.assign(n_pairs, 0.);
                  for (unsigned int pair = 0; pair < n_pairs; ++pair)
                    {
                      const double *coefficients = product(first_active, pair);
                      double        sum          = 0.;
                      for (unsigned int exponent_0 = 0;
                           exponent_0 < product_length;
                           ++exponent_0)
                        sum += coefficients[exponent_0] *
                               moment_table[exponent_0 * stride[first_active]];
                      single[pair] = sum;
                    }
                  for (unsigned int i = 0; i < basis_a.n_dofs; ++i)
                    for (unsigned int j = 0; j < basis_b.n_dofs; ++j)
                      entry[i * basis_b.n_dofs + j] +=
                        flat_factor(i, j) * single[pair_of(i, j, first_active)];
                }
              else if (n_active == 2)
                {
                  // Two directions: contract the first direction once per
                  // pair, then the second per entry.
                  const unsigned int second_active = active[1];
                  partial.assign(n_pairs * product_length, 0.);
                  for (unsigned int pair = 0; pair < n_pairs; ++pair)
                    {
                      const double *coefficients = product(first_active, pair);
                      double *contracted = &partial[pair * product_length];
                      for (unsigned int exponent_0 = 0;
                           exponent_0 < product_length;
                           ++exponent_0)
                        if (coefficients[exponent_0] != 0.)
                          for (unsigned int exponent_1 = 0;
                               exponent_1 < product_length &&
                               exponent_0 + exponent_1 <= max_product_degree;
                               ++exponent_1)
                            contracted[exponent_1] +=
                              coefficients[exponent_0] *
                              moment_table[exponent_0 * stride[first_active] +
                                           exponent_1 * stride[second_active]];
                    }
                  for (unsigned int i = 0; i < basis_a.n_dofs; ++i)
                    for (unsigned int j = 0; j < basis_b.n_dofs; ++j)
                      {
                        const double *contracted =
                          &partial[pair_of(i, j, first_active) *
                                   product_length];
                        const double *coefficients =
                          product(second_active, pair_of(i, j, second_active));
                        double sum = 0.;
                        for (unsigned int exponent_1 = 0;
                             exponent_1 < product_length;
                             ++exponent_1)
                          sum +=
                            coefficients[exponent_1] * contracted[exponent_1];
                        entry[i * basis_b.n_dofs + j] +=
                          flat_factor(i, j) * sum;
                      }
                }
              else
                {
                  // Three directions: contract the first direction once per
                  // pair, the second once per pair of pairs, and the third
                  // per entry.
                  partial.assign(n_pairs * product_length * product_length, 0.);
                  for (unsigned int pair = 0; pair < n_pairs; ++pair)
                    {
                      const double *coefficients = product(0, pair);
                      double       *contracted =
                        &partial[pair * product_length * product_length];
                      for (unsigned int exponent_0 = 0;
                           exponent_0 < product_length;
                           ++exponent_0)
                        if (coefficients[exponent_0] != 0.)
                          for (unsigned int exponent_1 = 0;
                               exponent_1 < product_length &&
                               exponent_0 + exponent_1 <= max_product_degree;
                               ++exponent_1)
                            for (unsigned int exponent_2 = 0;
                                 exponent_2 < product_length &&
                                 exponent_0 + exponent_1 + exponent_2 <=
                                   max_product_degree;
                                 ++exponent_2)
                              contracted[exponent_1 * product_length +
                                         exponent_2] +=
                                coefficients[exponent_0] *
                                moment_table[exponent_0 * stride[0] +
                                             exponent_1 * stride[1] +
                                             exponent_2 * stride[2]];
                    }
                  partial2.resize(n_pairs * n_pairs * product_length);
                  partial2_done.assign(n_pairs * n_pairs, 0);
                  for (unsigned int i = 0; i < basis_a.n_dofs; ++i)
                    for (unsigned int j = 0; j < basis_b.n_dofs; ++j)
                      {
                        const unsigned int pair0 = pair_of(i, j, 0);
                        const unsigned int pair1 = pair_of(i, j, 1);
                        double            *contracted2 =
                          &partial2[(pair0 * n_pairs + pair1) * product_length];
                        if (!partial2_done[pair0 * n_pairs + pair1])
                          {
                            partial2_done[pair0 * n_pairs + pair1] = 1;
                            std::fill(contracted2,
                                      contracted2 + product_length,
                                      0.);
                            const double *coefficients = product(1, pair1);
                            const double *contracted =
                              &partial[pair0 * product_length * product_length];
                            for (unsigned int exponent_1 = 0;
                                 exponent_1 < product_length;
                                 ++exponent_1)
                              if (coefficients[exponent_1] != 0.)
                                for (unsigned int exponent_2 = 0;
                                     exponent_2 + exponent_1 <=
                                       max_product_degree &&
                                     exponent_2 < product_length;
                                     ++exponent_2)
                                  contracted2[exponent_2] +=
                                    coefficients[exponent_1] *
                                    contracted[exponent_1 * product_length +
                                               exponent_2];
                          }
                        const double *coefficients =
                          product(2, pair_of(i, j, 2));
                        double sum = 0.;
                        for (unsigned int exponent_2 = 0;
                             exponent_2 < product_length;
                             ++exponent_2)
                          sum +=
                            coefficients[exponent_2] * contracted2[exponent_2];
                        entry[i * basis_b.n_dofs + j] += sum;
                      }
                }
            }

          // 5c. The entries of the pair of sides, with the prefactors of the
          // two bases.
          const double prefactor = basis_a.prefactor * basis_b.prefactor;
          for (unsigned int i = 0; i < basis_a.n_dofs; ++i)
            for (unsigned int j = 0; j < basis_b.n_dofs; ++j)
              {
                face_mass(offset[side_a] + i, offset[side_b] + j) =
                  prefactor * entry_mass[i * basis_b.n_dofs + j];
                face_flux(offset[side_a] + i, offset[side_b] + j) =
                  prefactor * entry_flux[i * basis_b.n_dofs + j];
              }
        }
  }



  /**
   * Quadrature-free assembly for FE_AggloDGP of one degree on the polytopes
   * of an AgglomerationHandler.
   */
  template <int dim>
  class PolytopeIntegrator
  {
  public:
    /**
     * Constructor. Expand the products of the gradients of the basis
     * functions of @p fe, and with @p with_mass the products of their values
     * for the mass matrix as well.
     */
    PolytopeIntegrator(const FiniteElement<dim> &fe,
                       const bool                with_mass = false);

    /**
     * The volume stiffness matrices $\int_P \nabla\phi_i \cdot \nabla\phi_j$
     * of all locally owned polytopes of @p ah, indexed by
     * polytope->index(), and, if @p mass is given and the integrator was
     * built with the mass, their mass matrices. All polytopes are done by one
     * matrix-matrix product of the fitted coefficients and the moments.
     */
    void
    volume_matrices(const AgglomerationHandler<dim> &ah,
                    std::vector<FullMatrix<double>> &stiffness,
                    std::vector<FullMatrix<double>> *mass = nullptr) const;

    /**
     * The face matrices (see interface_matrices()) on face @p f of
     * @p polytope: of the polytope alone on the boundary, of the polytope and
     * its neighbour otherwise. A face split by hanging nodes is taken from
     * its finer side. Return the measure of the face.
     */
    template <typename PolytopeIterator>
    double
    face_matrices(const AgglomerationHandler<dim> &ah,
                  const PolytopeIterator          &polytope,
                  const unsigned int               f,
                  FullMatrix<double>              &face_mass,
                  FullMatrix<double>              &face_flux) const;

  private:
    /**
     * The degree of the element.
     */
    const unsigned int degree;

    /**
     * The expansion of the products of the gradients.
     */
    const BasisProductExpansion<dim, 1> stiffness_expansion;

    /**
     * The expansion of the products of the values, if the mass is needed.
     */
    std::unique_ptr<const BasisProductExpansion<dim, 0>> mass_expansion;

    /**
     * The basis as a tensor product, for the face terms.
     */
    const SeparableBasis<dim> separable;

    /**
     * The coefficients of the stiffness, one row per pair (i, j) and one
     * column per pair (direction, monomial).
     */
    FullMatrix<double> stiffness_coefficients;

    /**
     * The coefficients of the mass matrix, one row per pair (i, j).
     */
    FullMatrix<double> mass_coefficients;
  };



  template <int dim>
  PolytopeIntegrator<dim>::PolytopeIntegrator(const FiniteElement<dim> &fe,
                                              const bool with_mass)
    : degree(fe.degree)
    , stiffness_expansion(fe)
    , mass_expansion(with_mass ?
                       std::make_unique<BasisProductExpansion<dim, 0>>(fe) :
                       nullptr)
    , separable(separable_basis(fe))
  {
    // The stiffness needs only the products of a derivative with the
    // derivative in the same direction. Their coefficients are laid out as a
    // matrix with one row per pair (i, j), so that volume_matrices() gets all
    // polytopes from one matrix-matrix product.
    const unsigned int n_dofs      = fe.n_dofs_per_cell();
    const unsigned int n_monomials = stiffness_expansion.n_monomials();
    stiffness_coefficients.reinit(n_dofs * n_dofs, dim * n_monomials);
    for (unsigned int direction = 0; direction < dim; ++direction)
      {
        const auto &fitted =
          stiffness_expansion.get_coefficients(direction * dim + direction);
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            for (unsigned int monomial = 0; monomial < n_monomials; ++monomial)
              stiffness_coefficients(i * n_dofs + j,
                                     direction * n_monomials + monomial) =
                fitted[i][j][monomial];
      }
    if (mass_expansion)
      {
        const unsigned int n_mass_monomials = mass_expansion->n_monomials();
        mass_coefficients.reinit(n_dofs * n_dofs, n_mass_monomials);
        const auto &fitted = mass_expansion->get_coefficients(0);
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            for (unsigned int monomial = 0; monomial < n_mass_monomials;
                 ++monomial)
              mass_coefficients(i * n_dofs + j, monomial) =
                fitted[i][j][monomial];
      }
  }



  template <int dim>
  void
  PolytopeIntegrator<dim>::volume_matrices(
    const AgglomerationHandler<dim> &ah,
    std::vector<FullMatrix<double>> &stiffness,
    std::vector<FullMatrix<double>> *mass) const
  {
    Assert(mass == nullptr || mass_expansion,
           ExcMessage("The integrator was built without the mass."));
    const unsigned int n_dofs       = separable.n_dofs;
    const unsigned int n_monomials  = stiffness_expansion.n_monomials();
    const bool         compute_mass = mass != nullptr;
    // The mass needs the moments up to degree 2p; the stiffness uses the
    // first of them, up to degree 2p - 2 (multi_indices() orders by degree).
    const unsigned int moment_degree =
      compute_mass ? 2 * degree : (degree >= 1 ? 2 * degree - 2 : 0);
    const auto &boxes = ah.get_local_bboxes();

    // The moments of every locally owned polytope.
    std::vector<unsigned int>        polytope_index;
    std::vector<std::vector<double>> moments;
    for (const auto &polytope : ah.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          polytope_index.push_back(polytope->index());
          moments.push_back(moments_of_cells<dim>(polytope->get_agglomerate(),
                                                  boxes[polytope->index()],
                                                  moment_degree));
        }
    const unsigned int n_polytopes = polytope_index.size();
    unsigned int       size        = 0;
    for (const unsigned int index : polytope_index)
      size = std::max(size, index + 1);

    // Every column of the product is the matrix of one polytope, flattened.
    // The columns of the right factor are the moments of the polytopes, times
    // |B| / h_d^2 for the stiffness and times |B| for the mass, with B the
    // bounding box and h_d its side lengths.
    FullMatrix<double> scaled_moments(dim * n_monomials, n_polytopes);
    for (unsigned int column = 0; column < n_polytopes; ++column)
      {
        const BoundingBox<dim> &box = boxes[polytope_index[column]];
        for (unsigned int d = 0; d < dim; ++d)
          {
            const double scale =
              box.volume() / (box.side_length(d) * box.side_length(d));
            for (unsigned int k = 0; k < n_monomials; ++k)
              scaled_moments(d * n_monomials + k, column) =
                moments[column][k] * scale;
          }
      }
    FullMatrix<double> products(n_dofs * n_dofs, n_polytopes);
    stiffness_coefficients.mmult(products, scaled_moments);
    stiffness.resize(size);
    for (unsigned int column = 0; column < n_polytopes; ++column)
      {
        FullMatrix<double> &matrix = stiffness[polytope_index[column]];
        matrix.reinit(n_dofs, n_dofs);
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            matrix(i, j) = products(i * n_dofs + j, column);
      }

    if (compute_mass)
      {
        const unsigned int n_mass_monomials = mass_expansion->n_monomials();
        FullMatrix<double> volume_moments(n_mass_monomials, n_polytopes);
        for (unsigned int column = 0; column < n_polytopes; ++column)
          for (unsigned int k = 0; k < n_mass_monomials; ++k)
            volume_moments(k, column) =
              moments[column][k] * boxes[polytope_index[column]].volume();
        products.reinit(n_dofs * n_dofs, n_polytopes);
        mass_coefficients.mmult(products, volume_moments);
        mass->resize(size);
        for (unsigned int column = 0; column < n_polytopes; ++column)
          {
            FullMatrix<double> &matrix = (*mass)[polytope_index[column]];
            matrix.reinit(n_dofs, n_dofs);
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int j = 0; j < n_dofs; ++j)
                matrix(i, j) = products(i * n_dofs + j, column);
          }
      }
  }



  template <int dim>
  template <typename PolytopeIterator>
  double
  PolytopeIntegrator<dim>::face_matrices(const AgglomerationHandler<dim> &ah,
                                         const PolytopeIterator &polytope,
                                         const unsigned int      f,
                                         FullMatrix<double>     &face_mass,
                                         FullMatrix<double> &face_flux) const
  {
    const bool   boundary = polytope->at_boundary(f);
    const CellId this_id  = polytope->id();
    const CellId other_id = boundary ? this_id : polytope->neighbor(f)->id();

    // The pieces of the face: the faces of the cells of this polytope on it,
    // and, across a hanging node, the finer faces of the neighbour's cells,
    // with the normal turned to point out of this polytope.
    std::vector<InterfacePiece<dim>> pieces;
    const auto                       add =
      [&](const auto &face, const Point<dim> &owner_centre, const bool flip) {
        InterfacePiece<dim> piece;
        for (unsigned int vertex = 0; vertex < piece.vertex.size(); ++vertex)
          piece.vertex[vertex] = face->vertex(vertex);
        piece.normal = piece.unit_normal();
        if ((piece.normal * (face->center() - owner_centre) < 0) != flip)
          piece.normal *= -1.;
        pieces.push_back(piece);
      };
    const auto &interface = ah.get_interface();
    for (const auto &[cell, face_index] : interface.at({this_id, other_id}))
      if (!cell->face(face_index)->has_children())
        add(cell->face(face_index), cell->center(), false);
    if (!boundary)
      for (const auto &[cell, face_index] : interface.at({other_id, this_id}))
        if (cell->neighbor_is_coarser(face_index))
          add(cell->face(face_index), cell->center(), true);

    const BoundingBox<dim> &box = ah.get_local_bboxes()[polytope->index()];
    if (boundary)
      interface_matrices<dim>(
        {&separable}, {box}, pieces, face_mass, face_flux);
    else
      {
        const auto            &neighbor = polytope->neighbor(f);
        const BoundingBox<dim> neighbor_box =
          neighbor->is_locally_owned() ?
            ah.get_local_bboxes()[neighbor->index()] :
            neighbor->get_bounding_box();
        interface_matrices<dim>({&separable, &separable},
                                {box, neighbor_box},
                                pieces,
                                face_mass,
                                face_flux);
      }
    double measure = 0.;
    for (const auto &piece : pieces)
      measure += piece.measure();
    return measure;
  }
} // namespace dealii::QuadratureFree

#endif
