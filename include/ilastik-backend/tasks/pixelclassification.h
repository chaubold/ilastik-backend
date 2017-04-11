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

// helper to get from 5D to DIMs
template<int DIM, typename TYPE>
struct adjust_5d_block_for_dims
{
    vigra::MultiArrayView<DIM, TYPE> operator()(const vigra::MultiArrayView<5, TYPE>& data) const
    {
        throw std::runtime_error("adjust_5d_block_for_dims::operator() not implemented for these template params");
    }
};

template<typename TYPE>
struct adjust_5d_block_for_dims<2, TYPE>
{
    vigra::MultiArrayView<2, TYPE> operator()(const vigra::MultiArrayView<5, TYPE>& data) const
    {
        vigra::MultiArrayView<3, TYPE> squeezed_converted_raw_data(data.template bind<4>(0).template bind<0>(0));
        return vigra::MultiArrayView<2, TYPE>(squeezed_converted_raw_data.template bind<2>(0));
    }
};

template<typename TYPE>
struct adjust_5d_block_for_dims<3, TYPE>
{
    vigra::MultiArrayView<3, TYPE> operator()(const vigra::MultiArrayView<5, TYPE>& data) const
    {
        return data.template bind<4>(0).template bind<0>(0);
    }
};

// helper to get from DIMs to 5D
template<int DIM, typename TYPE>
struct adjust_dims_to_5d_block
{
    vigra::MultiArrayView<5, TYPE> operator()(const vigra::MultiArrayView<DIM, TYPE>& data) const
    {
        throw std::runtime_error("adjust_dims_to_5d_block::operator() not implemented for these template params");
    }
};

template<typename TYPE>
struct adjust_dims_to_5d_block<3, TYPE>
{
    vigra::MultiArrayView<5, TYPE> operator()(const vigra::MultiArrayView<3, TYPE>& data) const
    {
        return data.insertSingletonDimension(2).insertSingletonDimension(0);
    }
};

template<typename TYPE>
struct adjust_dims_to_5d_block<4, TYPE>
{
    vigra::MultiArrayView<5, TYPE> operator()(const vigra::MultiArrayView<4, TYPE>& data) const
    {
        return data.insertSingletonDimension(0);
    }
};

/**
 * A pixel classification task can compute features; and train, or predict a random forest.
 * It is used as computational backend from a python microservice.
 *
 * DIM is 2 or 3 for the dimensionality of the data per timestep. Data is assumed to also have a time (axis=0) and channel (axis=-1) dimension
 * IN_TYPE is the raw data input type
 * OUT_TYPE is used for the features as well as the predictions, and should be float or double
 */
template<int DIM, typename IN_TYPE, typename OUT_TYPE>
class pixel_classification_task
{
public:
    // typedefs
    using selected_features_type = std::vector<std::pair<std::string, OUT_TYPE>>;
    using coordinate = vigra::TinyVector<int64_t, 5>;
    using raw_array_type = vigra::MultiArrayView<5, IN_TYPE>;
    using features_array_type = vigra::MultiArrayView<5, OUT_TYPE>;
    using predictions_array_type = features_array_type;
    using feature_calculator_t =  utils::FeatureCalculator<DIM, OUT_TYPE>;

public:
    pixel_classification_task():
        is_cache_valid_(true)
    { }

    // API
    void configure_dataset_size(utils::Blocking<5> blocking)
    {
        blocking_ = blocking;
        is_cache_valid_ = false;
    }

    void configure_selected_features(selected_features_type features)
    {
        selected_features_ = features;
        feature_calculator_ = std::make_shared<feature_calculator_t>(features);
        vigra::TinyVector<int64_t, DIM> per_frame_halo = feature_calculator_->getHaloShape();
        if(DIM==2)
            halo_size_ = coordinate(0, per_frame_halo[0], per_frame_halo[1], 0, 0);
        else
            halo_size_ = coordinate(0, per_frame_halo[0], per_frame_halo[1], per_frame_halo[2], 0);

        is_cache_valid_ = false;
    }

    void load_random_forest(const std::string& filename, const std::string& path_in_file, size_t num_zeros_in_forest_name)
    {
        if(!utils::get_rfs_from_file(random_forest_vector_, filename, path_in_file, num_zeros_in_forest_name))
            throw std::runtime_error("Error when loading random forest!");
    }

    void save_random_forest(const std::string& filename, const std::string& path_in_file, size_t num_zeros_in_forest_name) const
    {
        // TODO implement me
    }

    features_array_type compute_features_of_block(size_t blockIndex, const raw_array_type& raw_data) const
    {
        // check preconditions
        if(selected_features_.empty())
            throw std::runtime_error("No feature selection provided yet, cannot compute features!");

        const utils::BlockWithHalo<5>& blockWithHalo = blocking_.getBlockWithHalo(blockIndex, halo_size_);
        if(raw_data.shape() != blockWithHalo.outerBlock().shape())
            throw std::runtime_error("Provided raw data block does not have the required shape!");

        if(raw_data.shape(0) != 1)
            throw std::runtime_error("Can only compute features per time frame!");
        if(raw_data.shape(4) != 1)
            throw std::runtime_error("Cannot work with multi-channel images yet!");
        if(DIM==2 && raw_data.shape(3) != 1)
            throw std::runtime_error("When using 2D pixel classification you cannot pass 3D blocks!");

        // ------------------------------------------------------------
        // compute features

        vigra::MultiArray<5, OUT_TYPE> converted_raw_data(raw_data);
        std::cout << "Got raw data of shape " << raw_data.shape() << std::endl;
        vigra::MultiArrayView<DIM, OUT_TYPE> dim_adjusted_raw_data(adjust_5d_block_for_dims<DIM, OUT_TYPE>()(converted_raw_data));
        std::cout << "Adjusted raw data shape to " << dim_adjusted_raw_data.shape() << std::endl;

        vigra::MultiArray<DIM+1, OUT_TYPE> out_array;
        feature_calculator_->calculate(dim_adjusted_raw_data, out_array);
        std::cout << "Done computing features " << std::endl;

        // cut away the halo
        const utils::Block<5>& localCore  = blockWithHalo.innerBlockLocal();
        const coordinate& localBegin = localCore.begin();
        const coordinate& localShape = localCore.shape();

        vigra::TinyVector<int64_t, DIM+1> coreBegin;
        vigra::TinyVector<int64_t, DIM+1> coreShape;
        for(int d = 0; d < DIM; d++){
            coreBegin[d] = localBegin[d + 1]; // skip time dimension in localBegin and localShape!
            coreShape[d]  = localShape[d + 1];
        }
        coreBegin[DIM] = 0;
        coreShape[DIM] = feature_calculator_->get_feature_size();

        vigra::MultiArray<DIM+1, OUT_TYPE> cropped_features(out_array.subarray(coreBegin, coreBegin + coreShape));
        std::cout << "resulting features have shape" << cropped_features.shape() << std::endl;

        return adjust_dims_to_5d_block<DIM+1, OUT_TYPE>()(cropped_features);
    }

    const size_t get_num_features() const
    {
        return feature_calculator_->get_feature_size();
    }

    const size_t get_num_classes() const
    {
        if(random_forest_vector_.empty())
            throw std::runtime_error("No random forest loaded, don't know number of classes yet");
        return random_forest_vector_[0].class_count();
    }

    predictions_array_type predict_for_block(const features_array_type& feature_data) const
    {
        // preconditions
        if(random_forest_vector_.empty())
            throw std::runtime_error("No random forest loaded, cannot predict");

        size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
        size_t num_required_features = random_forest_vector_[0].feature_count();

        if(num_required_features != feature_data.shape(4))
            throw std::runtime_error("Provided number of features did not match the one required by the random forest");

        // ------------------------------------------------------------
        // transform to a feature view for prediction

        size_t pixel_count = 1;
        for (size_t dim = 0; dim < 4; dim++) // num pixels is not dependent on the number of features, so only sum coordinate axes
        {
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
        prediction_map_shape[4] = num_pixel_classification_labels;

        return predictions_array_type(prediction_map_shape, prediction_map.data());
    }

    utils::Block<5> get_required_raw_roi_for_feature_computation_of_block(size_t blockIndex) const
    {
        if(selected_features_.empty())
            throw std::runtime_error("No feature selection provided yet, cannot compute halo");
        if(blocking_.roiBegin() == blocking_.roiEnd())
            throw std::runtime_error("Blocking is not specified yet");

        return blocking_.getBlockWithHalo(blockIndex, halo_size_).outerBlock();
    }

    utils::Blocking<5> get_blocking() const
    { return blocking_; }

    bool is_cache_valid() const
    { return is_cache_valid_; }

    coordinate get_halo_size() const
    { return halo_size_; }

private:
    // members
    utils::Blocking<5> blocking_;
    selected_features_type selected_features_;
    coordinate halo_size_;
    std::shared_ptr<feature_calculator_t> feature_calculator_;
    bool is_cache_valid_;
    utils::RandomForestVectorType random_forest_vector_;
};


} // namespace tasks
} // namespace ilastikbackend

#endif // PIXELCLASSIFICATION_TASK_H
