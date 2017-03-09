#ifndef PIXELCLASSIFICATION_TASK_H
#define PIXELCLASSIFICATION_TASK_H

#include "ilastik-backend/utils/blocking.h"
#include "ilastik-backend/utils/feature_calculator.h"
#include "ilastik-backend/utils/random_forest_reader.h"

namespace ilastikbackend
{
namespace tasks
{

/**
 * A pixel classification task can compute features; and train, or predict a random forest.
 * It is used as computational backend from a python microservice.
 */
template<int DIM, typename IN_TYPE, typename OUT_TYPE>
class pixel_classification_task
{
public:
    // typedefs
    using selected_features_type = std::vector<std::pair<std::string, double>>;
    using coordinate = vigra::TinyVector<int64_t, DIM>;
    using multichannel_coordinate = vigra::TinyVector<int64_t, DIM+1>;

public:
    // API
    void configure_dataset_size(coordinate begin, coordinate end, coordinate blockShape)
    {
        blocking_ = utils::Blocking<DIM>(begin, end, blockShape);
    }

    void configure_selected_features(selected_features_type features)
    {
        selected_features_ = features;
        feature_calculator_ = utils::FeatureCalculator<DIM, OUT_TYPE>(features);
        size_t num_feature_channels = feature_calculator_.get_feature_size();
        vigra::TinyVector<float, 3> halo = feature_calculator_.getHaloShape();
    }

private:
    // members
    utils::Blocking<DIM> blocking_;
    selected_features_type selected_features_;
    vigra::TinyVector<int64_t, 3> halo_size_;
    utils::FeatureCalculator<DIM, OUT_TYPE> feature_calculator_;
};


} // namespace tasks
} // namespace ilastikbackend

#endif // PIXELCLASSIFICATION_TASK_H
