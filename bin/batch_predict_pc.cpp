#include <iostream>
#include <tbb/flow_graph.h>
#include <assert.h>

#include <vigra/multi_array_chunked_hdf5.hxx>

#include "ilastik-backend/flowgraph/jobdata.h"
#include "ilastik-backend/flowgraph/multiinoutnode.h"
#include "ilastik-backend/operators/baseoperator.h"
#include "ilastik-backend/operators/feature_computation_operator.h"
#include "ilastik-backend/operators/random_forest_prediction_operator.h"
#include "ilastik-backend/utils/random_forest_reader.h"
#include "ilastik-backend/utils/blocking.h"
#include "ilastik-backend/utils/feature_calculator.h"
#include "ilastik-backend/operators/hdf5_output_operator.h"

using namespace tbb;
using namespace tbb::flow;
using namespace ilastikbackend;

using feature_computer3u8f = operators::feature_computation_operator<3, uint8_t, float>;
using feature_computer3u8f_node = flowgraph::single_inout_node< feature_computer3u8f::in_job_type, tuple<feature_computer3u8f::out_job_type> >;
using random_forest_predictor3f = operators::random_forest_prediction_operator<3, float>;
using random_forest_predictor3f_node = flowgraph::single_inout_node< random_forest_predictor3f::in_job_type, tuple<random_forest_predictor3f::out_job_type> >;
using hdf5_output = operators::hdf5_output_operator<4, float>;
using hdf5_output_node = flowgraph::single_inout_node< hdf5_output::in_job_type, tuple<hdf5_output::in_job_type>>;

int main()
{
    std::cout << "Starting..." << std::endl;
    types::set_of_cancelled_job_ids cancelled_job_ids;

    // load Random forests
    const std::string rf_filename = "./testPC.ilp";
    const std::string rf_path = "/PixelClassification/ClassifierForests/Forest";
//    const std::string rf_filename = "/Users/chaubold/Desktop/ilastik_hackathon/hackathon_flyem_forest.h5";
//    const std::string rf_path = "/Forest";
    util::RandomForestVectorType rf_vector;
    util::get_rfs_from_file(rf_vector, rf_filename, rf_path, 4);

    // read raw data (chunked and cached)
    const std::string raw_file_name = "./testraw.h5";
    const std::string dataset_name = "/exported_data";
//    const std::string raw_file_name = "/Users/chaubold/Desktop/ilastik_hackathon/data_200_8bit_crop.h5";
//    const std::string dataset_name = "/volume/data";
    vigra::HDF5File in_hdf5_file(raw_file_name, vigra::HDF5File::ReadOnly);
    vigra::ChunkedArrayHDF5<3, uint8_t> in_data(in_hdf5_file, dataset_name, vigra::HDF5File::ReadOnly);
    vigra::TinyVector<int64_t, 3> blockShape(64, 64, 64);
    vigra::TinyVector<int64_t, 3> coordBegin(0.0, 0.0, 0.0);
    vigra::TinyVector<int64_t, 3> coordEnd = in_data.shape();
//    vigra::TinyVector<int64_t, 3> coordBegin(0, 0, 0);
//    vigra::TinyVector<int64_t, 3> coordEnd(200, 300, 300);
    utils::Blocking<3> blocking(coordBegin, coordEnd, blockShape);

    // TODO: read selected features
    feature_computer3u8f::selected_features_type selected_features = {std::make_pair("GaussianSmoothing", 1.0f), std::make_pair("GaussianSmoothing", 3.5f)};
//    feature_computer3u8f::selected_features_type selected_features = {
//        std::make_tuple("GaussianSmoothing", 0.3f),
//        std::make_tuple("GaussianSmoothing", 1.0f),
//        std::make_tuple("GaussianSmoothing", 1.6f),
//        std::make_tuple("GaussianSmoothing", 3.5f),
//        std::make_tuple("GaussianSmoothing", 5.0f),
//        std::make_tuple("GaussianSmoothing", 10.0f),
//        std::make_tuple("LaplacianOfGaussian", 1.0f),
//        std::make_tuple("LaplacianOfGaussian", 3.5f),
//        std::make_tuple("LaplacianOfGaussian", 10.0f),
//        std::make_tuple("GaussianGradientMagnitude", 1.6f),
//        std::make_tuple("GaussianGradientMagnitude", 5.0f),
//        std::make_tuple("HessianOfGaussianEigenvalues", 1.6f),
//        std::make_tuple("HessianOfGaussianEigenvalues", 5.0f)
//    };
    util::FeatureCalculator<3, float> feature_calculator(selected_features);
    size_t num_feature_channels = feature_calculator.get_feature_size();
    vigra::TinyVector<float, 3> halo = feature_calculator.getHaloShape();
    std::cout << "using halo of size: " << halo << std::endl;

    // reserve output
    const std::string out_file_name = "./out.h5";
    vigra::HDF5File out_hdf5_file(out_file_name, vigra::HDF5File::New);
    auto in_shape = in_data.shape();
    auto out_shape = util::append_to_shape<3>(in_shape, num_feature_channels); // TODO: make this dependent on the selected features!
    std::cout << "Trying to set up output file " << out_file_name << " with dataset " << dataset_name << " and shape " << out_shape << std::endl;
    vigra::TinyVector<int64_t, 4> chunkSize(64.0, 64.0, 64.0, 1);
    vigra::ChunkedArrayHDF5<4, float> out_data(out_hdf5_file, dataset_name, vigra::HDF5File::New, out_shape, chunkSize);


    // set up graph
    graph g;
    feature_computer3u8f_node feature_computer(g, std::make_shared<feature_computer3u8f>(cancelled_job_ids, selected_features, halo, blocking));
    random_forest_predictor3f_node rf_predictor(g, std::make_shared<random_forest_predictor3f>(rf_vector, cancelled_job_ids));
    hdf5_output_node hdf5_writer(g, std::make_shared<hdf5_output>(cancelled_job_ids, out_data, blocking), serial);

    make_edge(get<feature_computer3u8f::OUT_FEATURES>(feature_computer.output_ports()), rf_predictor);
    make_edge(get<random_forest_predictor3f::OUT_PREDICTION>(rf_predictor.output_ports()), hdf5_writer);

    // process blocks
    std::cout << "found a dataset of shape " << in_data.shape() << " and " << blocking.numberOfBlocks() << " blocks" << std::endl;
    for (size_t i = 0; i < blocking.numberOfBlocks(); ++i)
    {
        utils::BlockWithHalo<3> blockWithHalo = blocking.getBlockWithHalo(i, halo);
        const auto & outerBlock = blockWithHalo.outerBlock();
        const auto & outerBlockShape = outerBlock.shape();
        vigra::MultiArray<3, uint8_t> raw_block(outerBlockShape);
        std::cout << "Reading block ID " << i << " from " << outerBlock.begin() << " and size " << outerBlockShape << std::endl;
        in_data.checkoutSubarray(outerBlock.begin(), raw_block);
        feature_computer.try_put(feature_computer3u8f::in_job_type(i, raw_block));
    }
    g.wait_for_all();

    std::cout << "done processing" << std::endl;
    return 0;
}
