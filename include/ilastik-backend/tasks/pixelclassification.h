#ifndef PIXELCLASSIFICATION_TASK_H
#define PIXELCLASSIFICATION_TASK_H

#include <memory>

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
 *
 * IN_TYPE is the raw data input type
 * OUT_TYPE is used for the features as well as the predictions, and should be float or double
 */
template<int DIM, typename IN_TYPE, typename OUT_TYPE>
class pixel_classification_task
{
public:
    // typedefs
    using selected_features_type = std::vector<std::pair<std::string, OUT_TYPE>>;
    using coordinate = vigra::TinyVector<int64_t, DIM>;
    using multichannel_coordinate = vigra::TinyVector<int64_t, DIM+1>;
    using raw_array_type = vigra::MultiArrayView<DIM, IN_TYPE>;
    using features_array_type = vigra::MultiArrayView<DIM+1, OUT_TYPE>;
    using predictions_array_type = features_array_type;
    using feature_calculator_t =  utils::FeatureCalculator<DIM, OUT_TYPE>;

public:
    pixel_classification_task():
        is_cache_valid_(true)
    { }

    // API
    void configure_dataset_size(utils::Blocking<DIM> blocking)
    {
        blocking_ = blocking;
        is_cache_valid_ = false;
    }

    void configure_selected_features(selected_features_type features)
    {
        selected_features_ = features;
        feature_calculator_ = std::make_shared<feature_calculator_t>(features);
        size_t num_feature_channels = feature_calculator_->get_feature_size();
        halo_size_ = feature_calculator_->getHaloShape();
        is_cache_valid_ = false;
    }

    void load_random_forest(const std::string& filename, const std::string& path_in_file)
    {
        // TODO implement me
    }

    void save_random_forest(const std::string& filename, const std::string& path_in_file)
    {
        // TODO implement me
    }

    features_array_type compute_features_of_block(size_t blockIndex, const raw_array_type& raw_data)
    {
        // check preconditions
        if(selected_features_.empty())
            throw std::runtime_error("No feature selection provided yet, cannot compute features!");

        const utils::BlockWithHalo<DIM>& blockWithHalo = blocking_.getBlockWithHalo(blockIndex, halo_size_);
        if(raw_data.shape() != blockWithHalo.outerBlock().shape())
            throw std::runtime_error("Provided raw data block does not have the required shape!");

        // ------------------------------------------------------------
        // compute features
        vigra::MultiArray<DIM, OUT_TYPE> converted_raw_data(raw_data);
        vigra::MultiArray<DIM+1, OUT_TYPE> out_array;
        feature_calculator_->calculate(converted_raw_data, out_array);

        // cut away the halo
        const utils::Block<DIM>& localCore  = blockWithHalo.innerBlockLocal();
        const coordinate& localBegin = localCore.begin();
        const coordinate& localShape = localCore.shape();

        multichannel_coordinate coreBegin;
        multichannel_coordinate coreShape;
        for(int d = 0; d < DIM; d++){
            coreBegin[d] = localBegin[d];
            coreShape[d]  = localShape[d];
        }
        coreBegin[DIM] = 0;
        coreShape[DIM] = feature_calculator_->get_feature_size();

        return out_array.subarray(coreBegin, coreShape);
    }

    predictions_array_type predict_for_block(size_t blockIndex, const features_array_type& feature_data)
    {
        // preconditions
        if(random_forest_vector_.empty())
            throw std::runtime_error("No random forest loaded, cannot predict");

        size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
        size_t num_required_features = random_forest_vector_[0].feature_count();

        if(num_required_features != feature_data.shape(DIM))
            throw std::runtime_error("Provided number of features did not match the one required by the random forest");

        // ------------------------------------------------------------
        // transform to a feature view for prediction

        size_t pixel_count = 1;
        for (size_t dim = 0; dim < DIM; dim++) {
          pixel_count *= feature_data.shape(dim);
        }
        vigra::MultiArrayView<2, OUT_TYPE> feature_view(vigra::Shape2(pixel_count, num_required_features), feature_data.data());

        // out array for the predictions
        vigra::MultiArray<2, OUT_TYPE> prediction_map(vigra::Shape2(pixel_count, num_pixel_classification_labels));

        // loop over all random forests for prediction probabilities
        for(size_t rf = 0; rf < random_forest_vector_.size(); rf++)
        {
            vigra::MultiArray<2, OUT_TYPE> prediction_temp(pixel_count, num_pixel_classification_labels);
            random_forest_vector_[rf].predictProbabilities(feature_view, prediction_temp);
            prediction_map += prediction_temp;
        }

        // divide probs by num random forests
        prediction_map /= random_forest_vector_.size();

        auto prediction_map_shape = feature_data.shape();
        prediction_map_shape[DIM] = num_pixel_classification_labels; // has DIM+1 entries, so DIM is last one

        return predictions_array_type(prediction_map_shape, prediction_map.data());
    }

    utils::Block<DIM> get_required_raw_roi_for_feature_computation_of_block(size_t blockIndex)
    {
        if(selected_features_.empty())
            throw std::runtime_error("No feature selection provided yet, cannot compute halo");
        if(blocking_.roiBegin() == blocking_.roiEnd())
            throw std::runtime_error("Blocking is not specified yet");

        return blocking_.getBlockWithHalo(blockIndex, halo_size_).outerBlock();
    }

    utils::Blocking<DIM> get_blocking() const
    { return blocking_; }


    bool is_cache_valid()
    { return is_cache_valid_; }

    coordinate get_halo_size()
    { return halo_size_; }

private:
    // members
    utils::Blocking<DIM> blocking_;
    selected_features_type selected_features_;
    coordinate halo_size_;
    std::shared_ptr<feature_calculator_t> feature_calculator_;
    bool is_cache_valid_;
    utils::RandomForestVectorType random_forest_vector_;
};


} // namespace tasks
} // namespace ilastikbackend

#endif // PIXELCLASSIFICATION_TASK_H
