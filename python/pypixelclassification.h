#ifndef PYPIXELCLASSIFICATION_H
#define PYPIXELCLASSIFICATION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vigraconverter.h"
#include "pyblocking.h"

#include "ilastik-backend/tasks/pixelclassification.h"

using coordinate_type = int64_t;
using coordinate_array = pybind11::array_t<coordinate_type, pybind11::array::f_style | pybind11::array::forcecast>;

// ------------------------------------------------------------------------------------
template<int DIM, typename IN_TYPE, typename OUT_TYPE>
class PyPixelClassification{
public:
    using np_raw_array = pybind11::array_t<IN_TYPE, pybind11::array::f_style | pybind11::array::forcecast>;
    using np_features_array = pybind11::array_t<OUT_TYPE, pybind11::array::f_style | pybind11::array::forcecast>;
    using np_predictions_array = np_features_array;
    using selected_features_type = std::vector<std::pair<std::string, OUT_TYPE>>;

public:
    PyPixelClassification(){}

    void configure_dataset_size(PyBlocking<5> blocking)
    {
        pixelclassification_.configure_dataset_size(blocking.getBlocking());
    }

    void configure_selected_features(selected_features_type features)
    {
        pixelclassification_.configure_selected_features(features);
    }

    void load_random_forest(const std::string& filename, const std::string& path_in_file, size_t num_zeros_in_forest_name)
    {
        pixelclassification_.load_random_forest(filename, path_in_file, num_zeros_in_forest_name);
    }

    void save_random_forest(const std::string& filename, const std::string& path_in_file, size_t num_zeros_in_forest_name)
    {
        pixelclassification_.save_random_forest(filename, path_in_file, num_zeros_in_forest_name);
    }

    np_features_array compute_features_of_block(size_t blockIndex, const np_raw_array& raw_data)
    {
        auto vigra_raw = numpy_to_vigra<5, IN_TYPE>(raw_data);
        vigra::MultiArrayView<5, OUT_TYPE> result;
        {
            pybind11::gil_scoped_release release;
            result = pixelclassification_.compute_features_of_block(blockIndex, vigra_raw);
        }
        return vigra_to_numpy<5, OUT_TYPE>(result);
    }

    np_predictions_array predict_for_block(const np_features_array& feature_data)
    {
        auto vigra_features = numpy_to_vigra<5, OUT_TYPE>(feature_data);
        vigra::MultiArrayView<5, OUT_TYPE> result;
        {
            pybind11::gil_scoped_release release;
            result = pixelclassification_.predict_for_block(vigra_features);
        }
        return vigra_to_numpy<5, OUT_TYPE>(result);
    }

    PyBlock<5> get_required_raw_roi_for_feature_computation_of_block(size_t blockIndex)
    {
        return PyBlock<5>(pixelclassification_.get_required_raw_roi_for_feature_computation_of_block(blockIndex));
    }

    PyBlocking<5> get_blocking() const
    { return PyBlocking<5>(pixelclassification_.get_blocking()); }

    bool is_cache_valid()
    { return pixelclassification_.is_cache_valid(); }

    coordinate_array get_halo_size()
    { return tiny_vector_to_numpy<5, coordinate_type>(pixelclassification_.get_halo_size()); }

    const size_t get_num_features() const
    { return pixelclassification_.get_num_features(); }

    const size_t get_num_classes() const
    { return pixelclassification_.get_num_classes(); }

private:
    ilastikbackend::tasks::pixel_classification_task<DIM, IN_TYPE, OUT_TYPE> pixelclassification_;
};


// ------------------------------------------------------------------------------------
template<int DIM, typename IN_TYPE, typename OUT_TYPE>
void export_pixel_classificationT(pybind11::module& m, const std::string& input_type_str)
{
    const auto dim_type_str = std::string("_") + std::to_string(DIM) + std::string("d_") + input_type_str;

    const auto blocking_class_name = std::string("PixelClassification") + dim_type_str;
    pybind11::class_<PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>>(m, blocking_class_name.c_str())
        .def(pybind11::init<>())
        .def("configureDatasetSize",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::configure_dataset_size)
        .def("configureSelectedFeatures",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::configure_selected_features)
        .def("loadRandomForest",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::load_random_forest)
        .def("saveRandomForest",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::save_random_forest)
        .def("computeFeaturesOfBlock",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::compute_features_of_block)
        .def("computePredictionsOfBlock",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::predict_for_block)
        .def("getRequiredRawRoiForFeatureComputationOfBlock", &PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::get_required_raw_roi_for_feature_computation_of_block)
        .def_property_readonly("blocking",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::get_blocking)
        .def_property_readonly("numberOfFeatures",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::get_num_features)
        .def_property_readonly("numberOfClasses",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::get_num_classes)
        .def_property_readonly("cacheValid",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::is_cache_valid)
        .def_property_readonly("haloSize",&PyPixelClassification<DIM, IN_TYPE, OUT_TYPE>::get_halo_size)
    ;


}

#endif // PYPIXELCLASSIFICATION_H
