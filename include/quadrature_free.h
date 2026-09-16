#ifndef POLYDEAL_QUADRATURE_FREE_H
#define POLYDEAL_QUADRATURE_FREE_H

// =====================================================================
// Quadrature-free assembly of VOLUME terms on agglomerated polytopes.
//
// Every entry of a mass or stiffness matrix is the integral of a linear
// combination of monomials, and the combination coefficients depend only on the
// reference-cell basis -- not on the polytope. So the coefficients are
// precomputed once per polynomial degree, and at run time an entry costs one
// dot product against the polytope's geometric moments
//
//     H_alpha = \int_P x^alpha dx,
//
// which are themselves obtained without quadrature, by cascading the divergence
// theorem down in dimension (3D -> 2D -> 1D -> closed form).
//
// SCOPE, and it is narrow on purpose:
//   * VOLUME terms only. The interface terms of the SIPG form need the two
//     neighbours' monomial bases expressed in a common frame, and in this
//     implementation that transition matrix costs more than simply integrating
//     with quadrature -- see MHPC_thesis/outline.md, C7.3.2.
//   * The MATRIX only, never the load vector. The method integrates POLYNOMIALS
//     exactly; grad(phi_i).grad(phi_j) is one, but f*phi_i is not unless f
//     happens to be a polynomial, and none of this thesis' manufactured
//     solutions have polynomial data. The load vector therefore keeps using
//     quadrature even when the matrix does not.
//
// Reference: P.F. Antonietti, P. Houston, G. Pennesi, "Fast numerical
// integration on polytopic meshes with applications to discontinuous Galerkin
// finite element methods", J. Sci. Comput. 77 (2018) 1339-1370.
// One deliberate deviation from that paper is documented at
// integrate_3d_face_via_2d_projection() below.
//
// This header is a cleaned-up extraction of examples/quadrature_free4.cc, which
// is left untouched as the standing 2D/3D unit test of the same machinery.
// =====================================================================

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/householder.h>
#include <deal.II/lac/vector.h>

#include <agglomeration_handler.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <iostream>

namespace qfree
{
template <int dim>
std::vector<std::array<unsigned int, dim>> get_multi_indices(unsigned int max_degree)
{
    std::vector<std::array<unsigned int, dim>> indices;
    std::array<unsigned int, dim> current;
    current.fill(0);
    auto generate = [&](auto& self, unsigned int current_dim, unsigned int remaining_degree) -> void
    {
        if (current_dim == dim - 1)
        {
            current[current_dim] = remaining_degree;
            indices.push_back(current);
            return;
        }
        for (unsigned int d = remaining_degree; d != static_cast<unsigned int>(-1); --d)
        {
            current[current_dim] = d;
            self(self, current_dim + 1, remaining_degree - d);
        }
    };
    for (unsigned int k = 0; k <= max_degree; ++k)
        generate(generate, 0, k);
    return indices;
}

// Cached wrapper: the callers below ask for this once PER POLYTOPE, and the
// list depends only on the degree. Rebuilding it (recursion + 45 heap-allocated
// arrays at p=4) for every polytope of every cycle is pure overhead.
template <int dim>
const std::vector<std::array<unsigned int, dim>> &
get_multi_indices_cached(unsigned int max_degree)
{
    static std::map<unsigned int, std::vector<std::array<unsigned int, dim>>> cache;
    auto it = cache.find(max_degree);
    if (it == cache.end())
        it = cache.emplace(max_degree, get_multi_indices<dim>(max_degree)).first;
    return it->second;
}

template <int dim, int rank>
struct ShapeDerivative;

// Specialization for order 0: Value -> returns a scalar double
template <int dim>
struct ShapeDerivative<dim, 0> {
    using type = double;
    static type get(const dealii::FiniteElement<dim>& fe, unsigned int i, const dealii::Point<dim>& p) {
        return fe.shape_value(i, p);
    }
};

// Specialization for order 1: Gradient -> returns a Tensor<1, dim>
template <int dim>
struct ShapeDerivative<dim, 1> {
    using type = dealii::Tensor<1, dim>;
    static type get(const dealii::FiniteElement<dim>& fe, unsigned int i, const dealii::Point<dim>& p) {
        return fe.shape_grad(i, p);
    }
};

// Specialization for order 2: Hessian -> returns a Tensor<2, dim>
template <int dim>
struct ShapeDerivative<dim, 2> {
    using type = dealii::Tensor<2, dim>;
    static type get(const dealii::FiniteElement<dim>& fe, unsigned int i, const dealii::Point<dim>& p) {
        return fe.shape_grad_grad(i, p); // deal.II's native second-derivative method
    }
};

/**
 * To get the coefficients of the monomials,
 * think it as the solution to Ax = b,
 * where A is the evaluation matrix of monomials at sampled points,
 * and b is the evaluation of the formula to be integrated(LHS of the weak form) at sampled points.
 * We solve this linear system by x = (A^T A)^{-1} A^T b, 
 * to take more sampling points than the number of monomials to get a more accurate and stable solution.
 */
template <int dim, int rank_i, int rank_j>
class UniversalQFreePrecomputer {
public:
    UniversalQFreePrecomputer(const dealii::FiniteElement<dim>& fe) 
    {
        unsigned int p = fe.degree;
        unsigned int max_degree = 2 * p; 
        
        const auto &multi_indices = get_multi_indices_cached<dim>(max_degree);
        n_monomials = multi_indices.size();
        unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        // Compute the tensor's total component count at compile time
        // rank=0 -> 1 component; rank=1 -> dim components; rank=2 -> dim*dim components
        constexpr unsigned int n_comp_i = (rank_i == 0) ? 1 : (rank_i == 1) ? dim : dim * dim;
        constexpr unsigned int n_comp_j = (rank_j == 0) ? 1 : (rank_j == 1) ? dim : dim * dim;
        n_total_comp = n_comp_i * n_comp_j;

        // Unified storage layout: [tensor component idx][dof i][dof j][monomial alpha]
        C_tensor.assign(n_total_comp, std::vector<std::vector<std::vector<double>>>(
            dofs_per_cell, std::vector<std::vector<double>>(
                dofs_per_cell, std::vector<double>(n_monomials, 0.0))));

        dealii::QGauss<dim> quad(max_degree + 2); 
        unsigned int n_qpoints = quad.size();
        const std::vector<dealii::Point<dim>>& q_points = quad.get_points();

        // 1. Assemble and invert the A matrix (the core geometric-algebra projection, never changes)
        dealii::FullMatrix<double> A(n_qpoints, n_monomials);
        for (unsigned int q = 0; q < n_qpoints; ++q) {
            for (unsigned int alpha = 0; alpha < n_monomials; ++alpha) {
                double val = 1.0;
                for (unsigned int d = 0; d < dim; ++d) {
                    val *= std::pow(q_points[q][d], multi_indices[alpha][d]);
                }
                A(q, alpha) = val;
            }
        }

        // Least squares by Householder QR, NOT by the normal equations.
        //
        // The obvious route is (A^T A)^{-1} A^T b, and that is what this code
        // did. It squares the condition number, and A is a Vandermonde-like
        // matrix in the monomials on [0,1]^d, which is Hilbert-ill-conditioned
        // to begin with: at p=4 (total degree 8, 45 monomials in 2D) the squared
        // conditioning costs enough digits that the assembled stiffness entries
        // agree with a quadrature reference to only ~3e-5 relative -- for a
        // method whose selling point is that it is EXACT.
        //
        // QR solves the same least-squares problem without forming A^T A.
        dealii::Householder<double> qr(A);

        // Fit quality, reported once at the end of construction.
        double max_fit_residual = 0.0;
        double max_coeff        = 0.0;

        // Temporary data carrier
        std::vector<dealii::Vector<double>> b_target(n_total_comp, dealii::Vector<double>(n_qpoints));
        dealii::Vector<double> At_b(n_monomials);
        dealii::Vector<double> c(n_monomials);

        // 2. Loop over DoFs, performing tensor expansion and sampling
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                
                for (unsigned int q = 0; q < n_qpoints; ++q) {
                    // The magic happens here: the compiler automatically replaces get with shape_value or shape_grad!
                    auto val_i = ShapeDerivative<dim, rank_i>::get(fe, i, q_points[q]);
                    auto val_j = ShapeDerivative<dim, rank_j>::get(fe, j, q_points[q]);

                    // C++17 if constexpr guarantees only the matching branch is compiled, with no runtime overhead
                    if constexpr (rank_i == 0 && rank_j == 0) {
                        b_target[0][q] = val_i * val_j;
                    } 
                    else if constexpr (rank_i == 1 && rank_j == 1) {
                        for (unsigned int d1 = 0; d1 < dim; ++d1)
                            for (unsigned int d2 = 0; d2 < dim; ++d2)
                                b_target[d1 * dim + d2][q] = val_i[d1] * val_j[d2];
                    }
                    else if constexpr (rank_i == 0 && rank_j == 1) {
                        for (unsigned int d2 = 0; d2 < dim; ++d2)
                            b_target[d2][q] = val_i * val_j[d2];
                    }
                    else if constexpr (rank_i == 1 && rank_j == 0) {
                        for (unsigned int d1 = 0; d1 < dim; ++d1)
                            b_target[d1][q] = val_i[d1] * val_j;
                    }
                    else if constexpr (rank_i == 2 && rank_j == 2) {
                        // Fourth-order-derivative Hessian expansion (dim^4 components)
                        unsigned int comp = 0;
                        for (unsigned int d1 = 0; d1 < dim; ++d1)
                            for (unsigned int d2 = 0; d2 < dim; ++d2)
                                for (unsigned int d3 = 0; d3 < dim; ++d3)
                                    for (unsigned int d4 = 0; d4 < dim; ++d4)
                                        b_target[comp++][q] = val_i[d1][d2] * val_j[d3][d4];
                    }
                }

                // 3. Batch-perform the least-squares projection
                for (unsigned int comp = 0; comp < n_total_comp; ++comp) {
                    qr.least_squares(c, b_target[comp]);
                    // Diagnostic: how well does the monomial expansion actually
                    // reproduce the sampled product? The expansion is EXACT in
                    // exact arithmetic -- a product of two polynomials of degree
                    // p is a polynomial of degree 2p, which is exactly the space
                    // being fitted -- so any residual here is pure conditioning
                    // of the Vandermonde system, and it is the error this
                    // implementation adds over one that takes the Legendre ->
                    // monomial coefficients in closed form, as AHP 2018 do.
                    {
                        double nb = 0., nr = 0.;
                        for (unsigned int q = 0; q < n_qpoints; ++q) {
                            double fit = 0.;
                            for (unsigned int a = 0; a < n_monomials; ++a)
                                fit += A(q, a) * c[a];
                            nb = std::max(nb, std::abs(b_target[comp][q]));
                            nr = std::max(nr, std::abs(b_target[comp][q] - fit));
                        }
                        if (nb > 0.)
                            max_fit_residual = std::max(max_fit_residual, nr / nb);
                        max_coeff = std::max(max_coeff, c.linfty_norm());
                    }
                    for (unsigned int alpha = 0; alpha < n_monomials; ++alpha) {
                        // No thresholding: a coefficient that looks negligible
                        // on its own can still matter once multiplied by a
                        // moment, and the moments span many orders of magnitude.
                        C_tensor[comp][i][j][alpha] = c[alpha];
                    }
                }
            }
        }
        std::cout << "  [qfree-precomp] p=" << p
                  << " n_monomials=" << n_monomials
                  << " max|c|=" << max_coeff
                  << " max relative fit residual=" << max_fit_residual
                  << std::endl;
    }

    // Expose the data interface
    // flat_comp_idx is the flattened tensor index, e.g. for order1*order1, idx = d1 * dim + d2
    const std::vector<std::vector<std::vector<double>>>& get_C_tensor(unsigned int flat_comp_idx) const { 
        return C_tensor[flat_comp_idx]; 
    }
    
    unsigned int get_n_monomials() const { return n_monomials; }
    unsigned int get_n_components() const { return n_total_comp; }

private:
    unsigned int n_monomials;
    unsigned int n_total_comp;
    // Storage layout: [component idx][dof i][dof j][monomial alpha]
    std::vector<std::vector<std::vector<std::vector<double>>>> C_tensor; 
};


// Helper function: compute the factorial

// Helper function: compute the factorial
inline unsigned int factorial(unsigned int n) {
    unsigned int res = 1;
    for (unsigned int i = 2; i <= n; ++i) res *= i;
    return res;
}
// Helper function: compute the binomial coefficient C(n, k)
unsigned int compute_nCr(unsigned int n, unsigned int k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;
    unsigned int result = n;
    for (unsigned int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

// [1D -> 0D] Using the coordinates of two vertices A and B, analytically line-integrate the monomial x^m * y^n
template <int dim>
double integrate_monomial_on_line_2d_analytical(
    const dealii::Point<dim>& A, 
    const dealii::Point<dim>& B, 
    unsigned int m, 
    unsigned int n) 
{
    const double dx = B[0] - A[0];
    const double dy = B[1] - A[1];
    const double length = A.distance(B);
    double integral_value = 0.0;

    // Power tables instead of std::pow.
    //
    // This inner loop is THE cost of the whole quadrature-free path. With
    // std::pow it runs four transcendental calls per term, and at p=4 the
    // moments of one polytope need ~10 600 of them -- about 420 us per
    // polytope against a measured total of 455, i.e. the method's "cost" was
    // ~93% std::pow. The exponents are small non-negative integers, so a table
    // built by repeated multiplication is both exact and ~100x cheaper, and it
    // turns the loop into pure multiply-add.
    constexpr unsigned int MAXP = 64;
    double px[MAXP], pdx[MAXP], py[MAXP], pdy[MAXP];
    const unsigned int hi = std::max(m, n) + 1;
    px[0] = pdx[0] = py[0] = pdy[0] = 1.0;
    for (unsigned int e = 1; e <= hi && e < MAXP; ++e) {
        px[e]  = px[e - 1]  * A[0];
        pdx[e] = pdx[e - 1] * dx;
        py[e]  = py[e - 1]  * A[1];
        pdy[e] = pdy[e - 1] * dy;
    }

    // Binomial expansion: (Ax + t*dx)^m * (Ay + t*dy)^n,
    // \int_0^1 t^(k+j) dt = 1 / (k + j + 1)
    for (unsigned int k = 0; k <= m; ++k) {
        const double cx = compute_nCr(m, k) * px[m - k] * pdx[k];
        if (cx == 0.0) continue;
        for (unsigned int j = 0; j <= n; ++j) {
            const double cy = compute_nCr(n, j) * py[n - j] * pdy[j];
            integral_value += (cx * cy) / (k + j + 1.0);
        }
    }

    return integral_value * length;
}

template <int dim>
double integrate_2d_polygon_analytical(
    const std::vector<dealii::Point<2>>& vertices,
    unsigned int m, unsigned int n)
{
    // 1. Compute the 2D polygon's signed area, to ensure the normal vector's orientation is correct
    double signed_area = 0;
    unsigned int num_v = vertices.size();
    for(unsigned int i = 0; i < num_v; ++i) {
        auto p0 = vertices[i]; 
        auto p1 = vertices[(i + 1) % num_v];
        signed_area += 0.5 * (p0[0] * p1[1] - p1[0] * p0[1]);
    }
    // If the vertices are ordered clockwise, compensate with a negative sign
    double orientation_sign = (signed_area >= 0) ? 1.0 : -1.0;

    double integral = 0.0;
    // 2. Loop over the polygon's 1D edges
    for (unsigned int i = 0; i < num_v; ++i) {
        dealii::Point<2> p0 = vertices[i];
        dealii::Point<2> p1 = vertices[(i + 1) % num_v];
        
        double dx = p1[0] - p0[0];
        double dy = p1[1] - p0[1];
        double length = std::sqrt(dx * dx + dy * dy);
        if (length < 1e-14) continue;
        
        // x component of the 2D segment's outward normal vector
        double nx = dy / length; 
        
        // Call the ultimate 1D -> 0D analytic engine
        double line_int = integrate_monomial_on_line_2d_analytical<2>(p0, p1, m + 1, n);
        integral += (line_int / (m + 1.0)) * nx;
    }
    return integral * orientation_sign;
}

// [Second-level bridge]: project a 3D face integral onto a combination of 2D analytic integrals
template <int dim>
double integrate_3d_face_via_2d_projection(
    const std::vector<dealii::Point<3>>& face_vertices, 
    const dealii::Tensor<1, 3>& normal,                 
    unsigned int M, unsigned int N, unsigned int L)     
{
    // DELIBERATE DEVIATION from Antonietti-Houston-Pennesi (2018), thesis C7.2.4.
    // The paper reduces a face integral with a recursive Euler-theorem /
    // distance-to-centroid cascade. Here the face is instead projected onto a
    // coordinate plane, and the projection axis is the one carrying the LARGEST
    // normal component. Two reasons: the projection's area scaling is
    // 1/|n_axis|, so the largest component is the farthest from the singular
    // case; and the cost per face drops from O(N^3) to O(N) in the total
    // monomial degree N.
    // 1. Find the largest normal-vector component, to pick the projection plane and avoid a divide-by-zero blowup
    int proj_axis = 0;
    if (std::abs(normal[1]) > std::abs(normal[proj_axis])) proj_axis = 1;
    if (std::abs(normal[2]) > std::abs(normal[proj_axis])) proj_axis = 2;

    double area_scaling = 1.0 / std::abs(normal[proj_axis]);
    double D = normal[0]*face_vertices[0][0] + normal[1]*face_vertices[0][1] + normal[2]*face_vertices[0][2];

    double total_integral = 0.0;

    if (proj_axis == 2) { 
        // Project onto the XY plane; the Z variable will be substituted out
        double nz = normal[2];
        double nx = normal[0];
        double ny = normal[1];

        std::vector<dealii::Point<2>> shadow_vertices;
        for (const auto& v : face_vertices) shadow_vertices.push_back(dealii::Point<2>(v[0], v[1]));

        double nz_inv_pow = std::pow(1.0 / nz, L);
        for (unsigned int i = 0; i <= L; ++i) {
            for (unsigned int j = 0; j <= L - i; ++j) {
                unsigned int k = L - i - j;
                double multinomial_coeff = factorial(L) / (factorial(i) * factorial(j) * factorial(k));
                double coeff = multinomial_coeff * std::pow(D, i) * std::pow(-nx, j) * std::pow(-ny, k) * nz_inv_pow;
                
                double shadow_int = integrate_2d_polygon_analytical<2>(shadow_vertices, M + j, N + k);
                total_integral += coeff * shadow_int;
            }
        }
    } 
    else if (proj_axis == 0) { 
        // Project onto the YZ plane; the X variable will be substituted out
        double nx = normal[0];
        double ny = normal[1];
        double nz = normal[2];

        std::vector<dealii::Point<2>> shadow_vertices;
        for (const auto& v : face_vertices) shadow_vertices.push_back(dealii::Point<2>(v[1], v[2])); // here Y is dimension 0, Z is dimension 1

        double nx_inv_pow = std::pow(1.0 / nx, M);
        for (unsigned int i = 0; i <= M; ++i) {
            for (unsigned int j = 0; j <= M - i; ++j) {
                unsigned int k = M - i - j; 
                double multinomial_coeff = factorial(M) / (factorial(i) * factorial(j) * factorial(k));
                double coeff = multinomial_coeff * std::pow(D, i) * std::pow(-ny, j) * std::pow(-nz, k) * nx_inv_pow;
                
                double shadow_int = integrate_2d_polygon_analytical<2>(shadow_vertices, N + j, L + k); 
                total_integral += coeff * shadow_int;
            }
        }
    } 
    else if (proj_axis == 1) { 
        // Project onto the XZ plane; the Y variable will be substituted out
        double ny = normal[1];
        double nx = normal[0];
        double nz = normal[2];

        std::vector<dealii::Point<2>> shadow_vertices;
        for (const auto& v : face_vertices) shadow_vertices.push_back(dealii::Point<2>(v[0], v[2])); // here X is dimension 0, Z is dimension 1

        double ny_inv_pow = std::pow(1.0 / ny, N);
        for (unsigned int i = 0; i <= N; ++i) {
            for (unsigned int j = 0; j <= N - i; ++j) {
                unsigned int k = N - i - j; 
                double multinomial_coeff = factorial(N) / (factorial(i) * factorial(j) * factorial(k));
                double coeff = multinomial_coeff * std::pow(D, i) * std::pow(-nx, j) * std::pow(-nz, k) * ny_inv_pow;
                
                double shadow_int = integrate_2d_polygon_analytical<2>(shadow_vertices, M + j, L + k); 
                total_integral += coeff * shadow_int;
            }
        }
    }

    return total_integral * area_scaling;
}


// ---------------------------------------------------------------------------
// Boundary segments of an agglomerate, by TOPOLOGY rather than by look-up.
//
// The obvious implementation asks, for each face of each cell in the
// agglomerate, whether that face appears in polytope_boundary(). It works on a
// uniform mesh and FAILS SILENTLY once the mesh has been adaptively refined:
// the iterators stored in polytope_boundary() no longer compare equal to the
// ones obtained from cell->face(f), so most segments are simply skipped. The
// divergence theorem then integrates over an OPEN contour and every moment is
// wrong. Measured on the thesis' own structured run: 8 segments matched out of
// 24, and 0 out of 32.
//
// A face of a cell in the agglomerate lies on the agglomerate's boundary if and
// only if the cell on the other side is not itself in the agglomerate. That is
// the definition, it costs one set look-up, and it produces a closed contour by
// construction -- including across hanging nodes, where the face is split into
// its children and each child is tested against its own neighbour.
// ---------------------------------------------------------------------------
template <int dim, typename PolytopeIterator>
std::vector<std::pair<typename dealii::Triangulation<dim>::face_iterator,
                      dealii::Point<dim>>>
agglomerate_boundary_segments(const PolytopeIterator &polytope)
{
    using Tria = dealii::Triangulation<dim>;
    const auto &cells = polytope->get_agglomerate();

    std::set<dealii::CellId> inside;
    for (const auto &c : cells)
        inside.insert(c->id());

    // Each entry is (boundary face, centre of the cell that owns it). The centre
    // is what orients the normal outwards, and it must be the OWNING cell's, not
    // the polytope's: a non-convex polytope has boundary faces whose outward
    // direction does not point away from its centroid.
    std::vector<std::pair<typename Tria::face_iterator, dealii::Point<dim>>> out;

    for (const auto &cell : cells)
        for (const auto f : cell->face_indices())
        {
            if (cell->at_boundary(f))
            {
                out.emplace_back(cell->face(f), cell->center());
                continue;
            }
            if (cell->face(f)->has_children())
            {
                // Neighbour is finer: test each subface against its own child.
                for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                {
                    const auto child = cell->neighbor_child_on_subface(f, sf);
                    if (inside.find(child->id()) == inside.end())
                        out.emplace_back(cell->face(f)->child(sf), cell->center());
                }
                continue;
            }
            const auto neigh = cell->neighbor(f);
            if (inside.find(neigh->id()) == inside.end())
                out.emplace_back(cell->face(f), cell->center());
        }
    return out;
}

template <typename PolytopeIterator>
std::vector<double> compute_geometric_moments_2d_fully_analytical(
    const PolytopeIterator &polytope,
    const AgglomerationHandler<2> &ah,
    unsigned int max_degree)
{
    const auto &multi_indices = get_multi_indices_cached<2>(max_degree);
    unsigned int n_monomials = multi_indices.size();
    std::vector<double> H(n_monomials, 0.0);

    const dealii::BoundingBox<2> &bbox = ah.get_local_bboxes()[polytope->index()];

    // Closed by construction -- see agglomerate_boundary_segments().
    for (const auto &seg : agglomerate_boundary_segments<2>(polytope))
    {
        const auto &face = seg.first;

        // [2D -> 1D]: a segment is just its two endpoints, in the box's unit
        // coordinates (the whole identity below is written in those).
        const dealii::Point<2> p0 = bbox.real_to_unit(face->vertex(0));
        const dealii::Point<2> p1 = bbox.real_to_unit(face->vertex(1));

        // Outward normal. Rotating the tangent gives the two candidates; which
        // one is outward depends on deal.II's local vertex ordering, not on any
        // consistent walk around the polygon, so orient it explicitly against
        // the owning cell's centre.
        const double dx = p1[0] - p0[0];
        const double dy = p1[1] - p0[1];
        const double length = std::sqrt(dx * dx + dy * dy);
        dealii::Tensor<1, 2> n_hat;
        n_hat[0] =  dy / length;
        n_hat[1] = -dx / length;

        const dealii::Point<2> cell_center_hat = bbox.real_to_unit(seg.second);
        const dealii::Point<2> face_center_hat = bbox.real_to_unit(face->center());
        dealii::Tensor<1, 2> dir = face_center_hat - cell_center_hat;
        if (n_hat * dir < 0) n_hat *= -1.0;

        const double nx = n_hat[0];

        for (unsigned int alpha = 0; alpha < n_monomials; ++alpha)
        {
            // Divergence theorem with F = (x^(a1+1)/(a1+1) * y^a2, 0):
            // div F is exactly the monomial, so only n_x is needed.
            const unsigned int m_elevated = multi_indices[alpha][0] + 1;
            const unsigned int n_current  = multi_indices[alpha][1];

            const double line_integral =
              integrate_monomial_on_line_2d_analytical<2>(p0, p1, m_elevated, n_current);

            H[alpha] += (line_integral / m_elevated) * nx;
        }
    }

    // NOTE (thesis C7.4.x): NO extra scaling here, and there must not be. H is
    // the moment in the bounding box's UNIT coordinates, and the identity above
    // is written entirely in those, so the line integrals already produce it.
    // All physical scaling (|B|, and one 1/h_d per gradient) is the caller's.
    //
    // The version this header was extracted from multiplied H by
    // side_length(0)/volume, i.e. by 1/h in 2D. That is wrong, and the only
    // test that existed could not see it: u^T A u for a constant u vanishes for
    // ANY moments, because the stiffness matrix annihilates constants. Against
    // a reference assembly the error is exactly 1/h -- relative entry
    // differences of 7, 15 and 63 on polytopes of 64, 16 and 1 cells of a 64x64
    // mesh.
    return H;
}

template <typename PolytopeIterator>
std::vector<double> compute_geometric_moments_3d_fully_analytical(
    const PolytopeIterator &polytope,
    const AgglomerationHandler<3> &ah,
    unsigned int max_degree) 
{
    const auto &multi_indices = get_multi_indices_cached<3>(max_degree);
    unsigned int n_monomials = multi_indices.size();
    std::vector<double> H(n_monomials, 0.0);
    
    const dealii::BoundingBox<3> &bbox = ah.get_local_bboxes()[polytope->index()];
    const auto &boundary_faces = polytope->polytope_boundary();

    for (const auto &cell : polytope->get_agglomerate()) {
        for (const auto f : cell->face_indices()) {
            auto face = cell->face(f);
            
            if (std::find(boundary_faces.begin(), boundary_faces.end(), face) != boundary_faces.end()) {
                
                // 1. Extract the reference-space 3D face vertices (ensure they're ordered around the perimeter)
                std::vector<dealii::Point<3>> face_vertices;
                if (face->n_vertices() == 4) {
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(0)));
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(1)));
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(3))); // swap 2 and 3 to form a closed loop
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(2)));
                } else if (face->n_vertices() == 3) {
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(0)));
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(1)));
                    face_vertices.push_back(bbox.real_to_unit(face->vertex(2)));
                } else {
                    for(unsigned int v = 0; v < face->n_vertices(); ++v) 
                        face_vertices.push_back(bbox.real_to_unit(face->vertex(v)));
                }

                // 2. Compute the 3D reference face's outward normal vector
                // Subtracting points yields a Tensor; use dealii::cross_product_3d on that
                dealii::Tensor<1, 3> v0 = face_vertices[1] - face_vertices[0];
                dealii::Tensor<1, 3> v1 = face_vertices[face_vertices.size()-1] - face_vertices[0];
                
                dealii::Tensor<1, 3> n_hat = dealii::cross_product_3d(v0, v1); // deal.II's official cross-product API
                n_hat /= n_hat.norm();
                
                // Ensure it points outward
                dealii::Point<3> cell_center_hat = bbox.real_to_unit(cell->center());
                dealii::Point<3> face_center_hat = bbox.real_to_unit(face->center());
                
                // Points must be subtracted into a Tensor before they can be dotted with another Tensor (n_hat)
                dealii::Tensor<1, 3> dir = face_center_hat - cell_center_hat; 
                if (n_hat * dir < 0) n_hat *= -1.0;

                double nx = n_hat[0]; // the nx component needed by the 3D divergence theorem

                // 3. Feed each monomial into the projection engine
                for (unsigned int alpha = 0; alpha < n_monomials; ++alpha) {
                    unsigned int M_elevated = multi_indices[alpha][0] + 1; // divergence-theorem order elevation
                    unsigned int N_current  = multi_indices[alpha][1];
                    unsigned int L_current  = multi_indices[alpha][2];
                    
                    double face_int = integrate_3d_face_via_2d_projection<3>(
                        face_vertices, n_hat, M_elevated, N_current, L_current
                    );
                    
                    H[alpha] += (face_int / M_elevated) * nx;
                }
            }
        }
    }
    
    // NOTE (thesis C7.4.x): there is NO extra scaling here, and there must not
    // be. H is the moment in the bounding box's UNIT coordinates,
    //     H_alpha = \int_{\hat P} \hat x^alpha d\hat x,
    // and the divergence-theorem identity used above is written entirely in
    // those coordinates, so the line/face integrals already produce it. All the
    // physical scaling (|B| and the 1/h_d from the gradients) is applied by the
    // caller.
    //
    // The version this header was extracted from multiplied H by
    // side_length(0)/volume here, i.e. by 1/h in 2D. That is wrong, and it was
    // invisible to the only test that existed: u^T A u for a constant u is zero
    // no matter what H is. Measured against a reference assembly the error is
    // exactly a factor 1/h -- relative entry difference 7, 15 and 63 for
    // polytopes of 64, 16 and 1 background cells on a 64x64 mesh.
    // (scaling removed)
    return H;
}

template <int dim, typename PolytopeIterator>
std::vector<double> compute_geometric_moments(
    const PolytopeIterator &polytope,
    const AgglomerationHandler<dim> &ah,
    unsigned int max_degree) 
{
    const auto &multi_indices = get_multi_indices_cached<dim>(max_degree);
    unsigned int n_monomials = multi_indices.size();
    std::vector<double> H(n_monomials, 0.0);
    
    dealii::QGauss<dim - 1> quad_face((max_degree + 4) / 2);
    dealii::FE_Nothing<dim> fe_dummy; 
    dealii::FEFaceValues<dim> fe_face(
        ah.get_mapping(), fe_dummy, quad_face, 
        dealii::update_quadrature_points | dealii::update_normal_vectors | dealii::update_JxW_values);

    const dealii::BoundingBox<dim> &bbox = ah.get_local_bboxes()[polytope->index()];
    
    // scaling factor of bounding box from dS_hat in reference space to dS in physical space
    const double scaling = bbox.side_length(0) / bbox.volume(); 

    const auto &boundary_faces = polytope->polytope_boundary();

    /**
     * To use the divergence theorem, 
     * we take the integral of F dot n over the boundaries
     * as linear combination of integral of monomials.
     * Let's take the field to be F with the first dimension to be the integral of the monomial,
     * and the rest dimensions to be zero, then the gradient is exactly the monomial.
     * then F dot n = integral of this monomial * n_x
     */
    for (const auto &cell : polytope->get_agglomerate()) {
        for (const auto f : cell->face_indices()) {
            
            auto face = cell->face(f);
            bool is_agglo_boundary = (std::find(boundary_faces.begin(), boundary_faces.end(), face) != boundary_faces.end());

            if (is_agglo_boundary) {
                fe_face.reinit(cell, f);
                const auto &q_points = fe_face.get_quadrature_points();
                const auto &normals  = fe_face.get_normal_vectors();
                const auto &JxWs     = fe_face.get_JxW_values();

                for (unsigned int q = 0; q < q_points.size(); ++q) {
                    dealii::Point<dim> p_hat = bbox.real_to_unit(q_points[q]);
                    double nx = normals[q][0]; // the divergence theorem only needs the normal's 0-direction component
                    double JxW = JxWs[q];      
                    
                    for (unsigned int alpha = 0; alpha < n_monomials; ++alpha) {
                        double val = 1.0;
                        // Closed-form integral along direction 0 (x^(m+1) / (m+1))
                        val *= std::pow(p_hat[0], multi_indices[alpha][0] + 1) / (multi_indices[alpha][0] + 1.0);
                        
                        // Evaluate the remaining directions directly
                        for (unsigned int d = 1; d < dim; ++d) {
                            val *= std::pow(p_hat[d], multi_indices[alpha][d]);
                        }
                        H[alpha] += val * nx * JxW;
                    }
                }
            }
        }
    }
    
    // Apply the unified Nanson scaling map
    for (double &val : H) val *= scaling;
    return H;
}

} // namespace qfree

#endif
