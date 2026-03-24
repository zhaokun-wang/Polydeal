#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_base.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/meshworker/scratch_data.h>
#include <agglomeration_handler.h>
#ifndef dealii_fe_agglovalues_h
#define dealii_fe_agglovalues_h

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  /**
   * Helper class to reinit finite element spaces on polytopal cells.
   */
  template <int, int>
  class AgglomerationValuesBaseImplementation;
} // namespace internal

template <int dim, int spacedim = dim>
class FEAggloValues
{
  friend class internal::AgglomerationValuesBaseImplementation<dim, spacedim>;
  using ScratchData = MeshWorker::ScratchData<dim, spacedim>;
public:
  
    /**
   *
   * Set the degree of the quadrature formula to be used and the proper flags
   * for the FEValues object on the agglomerated cell.
   */
  FEAggloValues(
    const AgglomerationHandler<dim, spacedim>  &agglo_handler,
    const Quadrature<dim>                      &cell_quadrature = QGauss<dim>(1),
    const UpdateFlags                          &flags           = UpdateFlags::update_default,
    const Quadrature<dim - 1>                  &face_quadrature = QGauss<dim - 1>(1),
    const UpdateFlags                          &face_flags      = UpdateFlags::update_default);

  /**
   * Overload for hp::FECollection.
   */
  FEAggloValues(
    const AgglomerationHandler<dim, spacedim>  &agglo_handler,
    const hp::QCollection<dim>                 &cell_qcollection = hp::QCollection<dim>(QGauss<dim>(1)),
    const UpdateFlags                          &flags            = UpdateFlags::update_default,
    const hp::QCollection<dim - 1>             &face_qcollection = hp::QCollection<dim - 1>(QGauss<dim - 1>(1)),
    const UpdateFlags                          &face_flags       = UpdateFlags::update_default);

  inline const FiniteElement<dim, spacedim> &
  get_fe() const;

  inline const Mapping<dim> &
  get_mapping() const;

  using AgglomerationContainer =
  typename AgglomerationAccessor<dim, spacedim>::AgglomerationContainer;
  /**
   * Return the agglomerated quadrature for the given agglomeration. This
   * amounts to loop over all cells in an agglomeration and collecting together
   * all the rules.
   */
  Quadrature<dim>
  agglomerated_quadrature(
    const AgglomerationContainer &cells,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const;

  /**
   * Construct a finite element space on the agglomeration.
   */
  const FEValues<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope) const;

  /**
   * For a given polytope and face index, initialize shape functions, normals
   * and quadratures rules to integrate there.
   */
  const FEValuesBase<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope,
         const unsigned int                          face_index) const;

  /**
   * Helper function to call reinit on a master cell.
   */
  const FEValuesBase<dim, spacedim> &
  reinit_master(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              face_number,
    std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
      &agglo_isv_ptr) const;
  
  /**
   *
   * Return a pair of FEValuesBase object reinited from the two sides of the
   * agglomeration.
   */
  std::pair<const FEValuesBase<dim, spacedim> &,
            const FEValuesBase<dim, spacedim> &>
  reinit_interface(const AgglomerationIterator<dim, spacedim> &polytope_in,
                   const AgglomerationIterator<dim, spacedim> &neigh_polytope,
                   const unsigned int                          local_in,
                   const unsigned int local_outside) const;
private:
  const AgglomerationHandler<dim, spacedim> &agglomeration_handler;

  /**
   * Use this in reinit(cell) for  (non-agglomerated, standard)  cells,
   * and return the result of scratch.reinit(cell) for cells
   */
  mutable std::unique_ptr<ScratchData> standard_scratch;

  /**
   * Fill this up in reinit(cell), for agglomerated cells, using the custom
   * quadrature, and return the result of
   * scratch.reinit(cell);
   */
  mutable std::unique_ptr<ScratchData> agglomerated_scratch;


  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv_neigh;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv_bdary;

  UpdateFlags agglomeration_flags;

  const UpdateFlags internal_agglomeration_flags =
    update_values | update_gradients | update_JxW_values |
    update_quadrature_points;

  UpdateFlags agglomeration_face_flags;

  const UpdateFlags internal_agglomeration_face_flags =
    update_quadrature_points | update_normal_vectors | update_values |
    update_gradients | update_JxW_values | update_inverse_jacobians;

  Quadrature<dim> agglomeration_quad;

  Quadrature<dim - 1> agglomeration_face_quad;

    /**
   * Dummy FE_Nothing
   */
  FE_Nothing<dim, spacedim> dummy_fe;

  /**
   * Dummy FEValues, needed for cell quadratures.
   */
  std::unique_ptr<FEValues<dim, spacedim>> no_values;

  /**
   * Dummy FEFaceValues, needed for face quadratures.
   */
  std::unique_ptr<FEFaceValues<dim, spacedim>> no_face_values;

  // Stores quadrature rules; these QCollections should have the same size as
  // hp_fe_collection
  hp::QCollection<dim>     agglomeration_quad_collection;
  hp::QCollection<dim - 1> agglomeration_face_quad_collection;

  hp::MappingCollection<dim>
    mapping_collection; // Contains only one mapping object
  hp::FECollection<dim, spacedim>
    dummy_fe_collection; // Similar to dummy_fe, but as an FECollection
                         // containing only dummy_fe
  // Note: The above two variables provide an hp::FECollection interface but
  // actually contain only one element each.

  // Analogous to no_values and no_face_values, but used when different cells
  // employ different FEs or quadratures
  std::unique_ptr<hp::FEValues<dim, spacedim>>     hp_no_values;
  std::unique_ptr<hp::FEFaceValues<dim, spacedim>> hp_no_face_values;
};

template <int dim, int spacedim>
Quadrature<dim>
FEAggloValues<dim, spacedim>::agglomerated_quadrature(
  const typename FEAggloValues<dim, spacedim>::AgglomerationContainer
    &cells,
  const typename Triangulation<dim, spacedim>::active_cell_iterator
    &master_cell) const
{
  // const auto &is_master_cell   = agglomeration_handler.is_master_cell;
  const auto &is_hp_collection = agglomeration_handler.is_hp_collection;
  const auto &agglo_dh         = agglomeration_handler.agglo_dh;
  const auto &bboxes           = agglomeration_handler.bboxes;
  const auto &master2polygon   = agglomeration_handler.master2polygon;
  Assert(agglomeration_handler.is_master_cell(master_cell),
         ExcMessage("This must be a master cell."));

  std::vector<Point<dim>> vec_pts;
  std::vector<double>     vec_JxWs;

  if (!is_hp_collection)
    {
      // Original version: handle case without hp::FECollection
      for (const auto &dummy_cell : cells)
        {
          no_values->reinit(dummy_cell);
          auto q_points    = no_values->get_quadrature_points(); // real qpoints
          const auto &JxWs = no_values->get_JxW_values();

          std::transform(q_points.begin(),
                         q_points.end(),
                         std::back_inserter(vec_pts),
                         [&](const Point<spacedim> &p) { return p; });
          std::transform(JxWs.begin(),
                         JxWs.end(),
                         std::back_inserter(vec_JxWs),
                         [&](const double w) { return w; });
        }
    }
  else
    {
      // Handle the hp::FECollection case
      const auto &master_cell_as_dh_iterator =
        master_cell->as_dof_handler_iterator(agglo_dh);
      for (const auto &dummy_cell : cells)
        {
          // The following verbose call is necessary to handle cases where
          // different slave cells on different polytopes use different
          // quadrature rules. If the hp::QCollection contains multiple
          // elements, calling hp_no_values->reinit(dummy_cell) won't work
          // because it cannot infer the correct quadrature rule. By explicitly
          // passing the active FE index as q_index, and setting mapping_index
          // and fe_index to 0, we ensure that the dummy cell uses the same
          // quadrature rule as its corresponding master cell. This assumes a
          // one-to-one correspondence between hp::QCollection and
          // hp::FECollection, which is the convention in deal.II. However, this
          // implementation does not support cases where hp::QCollection and
          // hp::FECollection have different sizes.
          // TODO: Refactor the architecture to better handle numerical
          // integration for hp::QCollection.
          hp_no_values->reinit(dummy_cell,
                               master_cell_as_dh_iterator->active_fe_index(),
                               0,
                               0);
          auto q_points = hp_no_values->get_present_fe_values()
                            .get_quadrature_points(); // real qpoints
          const auto &JxWs =
            hp_no_values->get_present_fe_values().get_JxW_values();

          std::transform(q_points.begin(),
                         q_points.end(),
                         std::back_inserter(vec_pts),
                         [&](const Point<spacedim> &p) { return p; });
          std::transform(JxWs.begin(),
                         JxWs.end(),
                         std::back_inserter(vec_JxWs),
                         [&](const double w) { return w; });
        }
    }

  // Map back each point in real space by using the map associated to the
  // bounding box.
  std::vector<Point<dim>> unit_points(vec_pts.size());
  const auto             &bbox =
    bboxes[master2polygon.at(master_cell->active_cell_index())];
  unit_points.reserve(vec_pts.size());

  for (unsigned int i = 0; i < vec_pts.size(); i++)
    unit_points[i] = bbox.real_to_unit(vec_pts[i]);

  return Quadrature<dim>(unit_points, vec_JxWs);
}

template <int dim, int spacedim>
inline const FiniteElement<dim, spacedim> &
FEAggloValues<dim, spacedim>::get_fe() const
{
  return agglomeration_handler.get_fe();
}

template <int dim, int spacedim>
inline const Mapping<dim> &
FEAggloValues<dim, spacedim>::get_mapping() const
{
  return agglomeration_handler.get_mapping();
}

template <int dim, int spacedim>
FEAggloValues<dim, spacedim>::FEAggloValues(
  const AgglomerationHandler<dim, spacedim>  &agglo_handler,
  const Quadrature<dim>                      &cell_quadrature,
  const UpdateFlags                          &flags,
  const Quadrature<dim - 1>                  &face_quadrature,
  const UpdateFlags                          &face_flags)
  : agglomeration_handler(agglo_handler)
{
  agglomeration_quad       = cell_quadrature;
  agglomeration_flags      = flags;
  agglomeration_face_quad  = face_quadrature;
  agglomeration_face_flags = face_flags | internal_agglomeration_face_flags;


  no_values =
    std::make_unique<FEValues<dim>>(agglomeration_handler.get_mapping(),
                                    dummy_fe,
                                    agglomeration_quad,
                                    update_quadrature_points |
                                      update_JxW_values); // only for quadrature
  no_face_values = std::make_unique<FEFaceValues<dim>>(
    agglomeration_handler.get_mapping(),
    dummy_fe,
    agglomeration_face_quad,
    update_quadrature_points | update_JxW_values |
      update_normal_vectors); // only for quadrature
}

template <int dim, int spacedim>
FEAggloValues<dim, spacedim>::FEAggloValues(
  const AgglomerationHandler<dim, spacedim>  &agglo_handler,
  const hp::QCollection<dim>                 &cell_qcollection,
  const UpdateFlags                          &flags,
  const hp::QCollection<dim - 1>             &face_qcollection,
  const UpdateFlags                          &face_flags)
  : agglomeration_handler(agglo_handler)
{
  agglomeration_quad_collection      = cell_qcollection;
  agglomeration_flags                = flags;
  agglomeration_face_quad_collection = face_qcollection;
  agglomeration_face_flags           = face_flags | internal_agglomeration_face_flags;

  mapping_collection  = hp::MappingCollection<dim>(agglomeration_handler.get_mapping());
  dummy_fe_collection = hp::FECollection<dim, spacedim>(dummy_fe);
  hp_no_values        = std::make_unique<hp::FEValues<dim>>(
    mapping_collection,
    dummy_fe_collection,
    agglomeration_quad_collection,
    update_quadrature_points | update_JxW_values); // only for quadrature

  hp_no_face_values = std::make_unique<hp::FEFaceValues<dim>>(
    mapping_collection,
    dummy_fe_collection,
    agglomeration_face_quad_collection,
    update_quadrature_points | update_JxW_values |
      update_normal_vectors); // only for quadrature
}

template <int dim, int spacedim>
const FEValues<dim, spacedim> &
FEAggloValues<dim, spacedim>::reinit(
  const AgglomerationIterator<dim, spacedim> &polytope) const
{
  // Assert(euler_mapping,
  //        ExcMessage("The mapping describing the physical element stemming
  //        from "
  //                   "agglomeration has not been set up."));
  const auto &agglo_dh         = agglomeration_handler.agglo_dh;
  const auto &is_hp_collection = agglomeration_handler.is_hp_collection;
  const auto &box_mapping      = agglomeration_handler.box_mapping;
  const auto &fe_collection    = agglomeration_handler.fe_collection;
  const auto &deal_cell        = polytope->as_dof_handler_iterator(agglo_dh);

  // First check if the polytope is made just by a single cell. If so, use
  // classical FEValues
  // if (polytope->n_background_cells() == 1)
  //   return standard_scratch->reinit(deal_cell);

  const auto &agglo_cells = polytope->get_agglomerate();

  // Quadrature<dim> agglo_quad = agglomeration_handler.agglomerated_quadrature(agglo_cells, deal_cell);
  Quadrature<dim> agglo_quad = this->agglomerated_quadrature(agglo_cells, deal_cell);
  if (!is_hp_collection)
    {
      // Original version: handle case without hp::FECollection
      agglomerated_scratch = std::make_unique<ScratchData>(*box_mapping,
                                                           fe_collection[0],
                                                           agglo_quad,
                                                           agglomeration_flags);
    }
  else
    {
      // Handle the hp::FECollection case
      agglomerated_scratch = std::make_unique<ScratchData>(*(agglomeration_handler.box_mapping),
                                                           polytope->get_fe(),
                                                           agglo_quad,
                                                           agglomeration_flags);
    }
  return agglomerated_scratch->reinit(deal_cell);
}



template <int dim, int spacedim>
const FEValuesBase<dim, spacedim> &
FEAggloValues<dim, spacedim>::reinit_master(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              face_index,
  std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    &agglo_isv_ptr) const
{
  return internal::AgglomerationValuesBaseImplementation<dim, spacedim>::
    reinit_master(cell, face_index, agglo_isv_ptr, agglomeration_handler, *this);
}



template <int dim, int spacedim>
const FEValuesBase<dim, spacedim> &
FEAggloValues<dim, spacedim>::reinit(
  const AgglomerationIterator<dim, spacedim> &polytope,
  const unsigned int                          face_index) const
{
  // Assert(euler_mapping,
  //        ExcMessage("The mapping describing the physical element stemming
  //        from "
  //                   "agglomeration has not been set up."));
  const auto &agglo_dh  = agglomeration_handler.agglo_dh;
  const auto &deal_cell = polytope->as_dof_handler_iterator(agglo_dh);
  Assert(agglomeration_handler.is_master_cell(deal_cell), ExcMessage("This should be true."));

  return internal::AgglomerationValuesBaseImplementation<dim, spacedim>::
    reinit_master(deal_cell, face_index, agglomerated_isv_bdary, agglomeration_handler, *this);
}



template <int dim, int spacedim>
std::pair<const FEValuesBase<dim, spacedim> &,
          const FEValuesBase<dim, spacedim> &>
FEAggloValues<dim, spacedim>::reinit_interface(
  const AgglomerationIterator<dim, spacedim> &polytope_in,
  const AgglomerationIterator<dim, spacedim> &neigh_polytope,
  const unsigned int                          local_in,
  const unsigned int                          local_neigh) const
{
  const auto &agglo_dh         = agglomeration_handler.agglo_dh;
  // const auto &is_hp_collection = agglomeration_handler.is_hp_collection;
  const auto &box_mapping      = agglomeration_handler.box_mapping;
  const auto &fe               = agglomeration_handler.fe;
  // const auto &fe_collection    = agglomeration_handler.fe_collection;
  const auto &bboxes           = agglomeration_handler.bboxes;
  const auto &master2polygon   = agglomeration_handler.master2polygon;
  const auto &recv_qpoints     = agglomeration_handler.recv_qpoints;
  const auto &recv_normals     = agglomeration_handler.recv_normals;
  const auto &recv_jxws        = agglomeration_handler.recv_jxws;
  // If current and neighboring polytopes are both locally owned, then compute
  // the jump in the classical way without needing information about ghosted
  // entities.
  if (polytope_in->is_locally_owned() && neigh_polytope->is_locally_owned())
    {
      const auto &cell_in = polytope_in->as_dof_handler_iterator(agglo_dh);
      const auto &neigh_cell =
        neigh_polytope->as_dof_handler_iterator(agglo_dh);

      const auto &fe_in =
        internal::AgglomerationValuesBaseImplementation<dim, spacedim>::
          reinit_master(cell_in, local_in, agglomerated_isv, agglomeration_handler, *this);
      const auto &fe_out =
        internal::AgglomerationValuesBaseImplementation<dim, spacedim>::
          reinit_master(neigh_cell, local_neigh, agglomerated_isv_neigh, agglomeration_handler, *this);
      std::pair<const FEValuesBase<dim, spacedim> &,
                const FEValuesBase<dim, spacedim> &>
        my_p(fe_in, fe_out);

      return my_p;
    }
  else
    {
      Assert((polytope_in->is_locally_owned() &&
              !neigh_polytope->is_locally_owned()),
             ExcInternalError());

      const auto &cell = polytope_in->as_dof_handler_iterator(agglo_dh);
      const auto &bbox = bboxes[master2polygon.at(cell->active_cell_index())];
      // const double bbox_measure = bbox.volume();

      const unsigned int neigh_rank = neigh_polytope->subdomain_id();
      const CellId      &neigh_id   = neigh_polytope->id();

      // Retrieve qpoints,JxWs, normals sent previously from the neighboring
      // rank.
      const std::vector<Point<spacedim>> &real_qpoints =
        recv_qpoints.at(neigh_rank).at({neigh_id, local_neigh});

      const auto &JxWs = recv_jxws.at(neigh_rank).at({neigh_id, local_neigh});

      std::vector<Tensor<1, spacedim>> normals =
        recv_normals.at(neigh_rank).at({neigh_id, local_neigh});

      // Apply the necessary scalings due to the bbox.
      std::vector<Point<spacedim>> final_unit_q_points;
      std::transform(real_qpoints.begin(),
                     real_qpoints.end(),
                     std::back_inserter(final_unit_q_points),
                     [&](const Point<spacedim> &p) {
                       return bbox.real_to_unit(p);
                     });

      // std::vector<double>         scale_factors(final_unit_q_points.size());
      // std::vector<double>         scaled_weights(final_unit_q_points.size());
      // std::vector<Tensor<1, dim>> scaled_normals(final_unit_q_points.size());

      // Since we received normal vectors from a neighbor, we have to swap
      // the
      // // sign of the vector in order to have outward normals.
      // for (unsigned int q = 0; q < final_unit_q_points.size(); ++q)
      //   {
      //     for (unsigned int direction = 0; direction < spacedim; ++direction)
      //       scaled_normals[q][direction] =
      //         normals[q][direction] * (bbox.side_length(direction));

      //     scaled_normals[q] *= -1;

      //     scaled_weights[q] =
      //       (JxWs[q] * scaled_normals[q].norm()) / bbox_measure;
      //     scaled_normals[q] /= scaled_normals[q].norm();
      //   }
      for (unsigned int q = 0; q < final_unit_q_points.size(); ++q)
        normals[q] *= -1;


      NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(
        final_unit_q_points, JxWs, normals);

      agglomerated_isv =
        std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
          *box_mapping, *fe, surface_quad, agglomeration_face_flags);


      agglomerated_isv->reinit(cell);

      std::pair<const FEValuesBase<dim, spacedim> &,
                const FEValuesBase<dim, spacedim> &>
        my_p(*agglomerated_isv, *agglomerated_isv);

      return my_p;
    }
}


namespace internal
{
  template <int dim, int spacedim>
  class AgglomerationValuesBaseImplementation
  {
  public:
    static const FEValuesBase<dim, spacedim> &
    reinit_master(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const unsigned int face_index,
      std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
                                                &agglo_isv_ptr,
      const AgglomerationHandler<dim, spacedim> &handler,
      const FEAggloValues<dim, spacedim> &fe_values)
    {
      Assert(handler.is_master_cell(cell),
              ExcMessage("This cell must be a master one."));

      AgglomerationIterator<dim, spacedim> it{cell, &handler};
      const auto &neigh_polytope = it->neighbor(face_index);

      const CellId polytope_in_id = cell->id();

      // Retrieve the bounding box of the agglomeration
      const auto &bbox =
        handler.bboxes[handler.master2polygon.at(cell->active_cell_index())];

      CellId polytope_out_id;
      if (neigh_polytope.state() == IteratorState::valid)
        polytope_out_id = neigh_polytope->id();
      else
        polytope_out_id = polytope_in_id; // on the boundary. Same id

      const auto &common_face = handler.polytope_cache.interface.at(
        {polytope_in_id, polytope_out_id});

      std::vector<Point<spacedim>> final_unit_q_points;
      std::vector<double>          final_weights;
      std::vector<Tensor<1, dim>>  final_normals;

      if (!handler.is_hp_collection)
        {
          // Original version: handle case without hp::FECollection
          const unsigned int expected_qpoints =
            common_face.size() * fe_values.agglomeration_face_quad.size();
          final_unit_q_points.reserve(expected_qpoints);
          final_weights.reserve(expected_qpoints);
          final_normals.reserve(expected_qpoints);


          for (const auto &[deal_cell, local_face_idx] : common_face)
            {
              fe_values.no_face_values->reinit(deal_cell, local_face_idx);

              const auto &q_points =
                fe_values.no_face_values->get_quadrature_points();
              const auto &JxWs = fe_values.no_face_values->get_JxW_values();
              const auto &normals =
                fe_values.no_face_values->get_normal_vectors();

              const unsigned int n_qpoints_agglo = q_points.size();

              for (unsigned int q = 0; q < n_qpoints_agglo; ++q)
                {
                  final_unit_q_points.push_back(
                    bbox.real_to_unit(q_points[q]));
                  final_weights.push_back(JxWs[q]);
                  final_normals.push_back(normals[q]);
                }
            }
        }
      else
        {
          // Handle the hp::FECollection case
          unsigned int higher_order_quad_index = cell->active_fe_index();
          if (neigh_polytope.state() == IteratorState::valid)
            if (fe_values
                  .agglomeration_face_quad_collection[cell->active_fe_index()]
                  .size() <
                fe_values
                  .agglomeration_face_quad_collection[neigh_polytope
                                                        ->active_fe_index()]
                  .size())
              higher_order_quad_index = neigh_polytope->active_fe_index();

          const unsigned int expected_qpoints =
            common_face.size() *
            fe_values
              .agglomeration_face_quad_collection[higher_order_quad_index]
              .size();
          final_unit_q_points.reserve(expected_qpoints);
          final_weights.reserve(expected_qpoints);
          final_normals.reserve(expected_qpoints);

          for (const auto &[deal_cell, local_face_idx] : common_face)
            {
              fe_values.hp_no_face_values->reinit(
                deal_cell, local_face_idx, higher_order_quad_index, 0, 0);

              const auto &q_points =
                fe_values.hp_no_face_values->get_present_fe_values()
                  .get_quadrature_points();
              const auto &JxWs =
                fe_values.hp_no_face_values->get_present_fe_values()
                  .get_JxW_values();
              const auto &normals =
                fe_values.hp_no_face_values->get_present_fe_values()
                  .get_normal_vectors();

              const unsigned int n_qpoints_agglo = q_points.size();

              for (unsigned int q = 0; q < n_qpoints_agglo; ++q)
                {
                  final_unit_q_points.push_back(
                    bbox.real_to_unit(q_points[q]));
                  final_weights.push_back(JxWs[q]);
                  final_normals.push_back(normals[q]);
                }
            }
        }


      NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(
        final_unit_q_points, final_weights, final_normals);

      if (!handler.is_hp_collection)
        {
          agglo_isv_ptr =
            std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
              *(handler.box_mapping),
              *(handler.fe),
              surface_quad,
              fe_values.agglomeration_face_flags);
        }
      else
        {
          agglo_isv_ptr =
            std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
              *(handler.box_mapping),
              cell->get_fe(),
              surface_quad,
              fe_values.agglomeration_face_flags);
        }

      agglo_isv_ptr->reinit(cell);

      return *agglo_isv_ptr;
    }
  };  
}

DEAL_II_NAMESPACE_CLOSE
#endif