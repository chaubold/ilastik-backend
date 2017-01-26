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
    util::RandomForestVectorType rf_vector;
    util::get_rfs_from_file(rf_vector, rf_filename, rf_path, 4);

    // read raw data (chunked and cached)
    const std::string raw_file_name = "./testraw.h5";
    const std::string dataset_name = "/exported_data";
    vigra::HDF5File in_hdf5_file(raw_file_name, vigra::HDF5File::ReadOnly);
    vigra::ChunkedArrayHDF5<3, uint8_t> in_data(in_hdf5_file, dataset_name, vigra::HDF5File::ReadOnly);
    vigra::TinyVector<int64_t, 3> blockShape(64, 64, 64);
    vigra::TinyVector<int64_t, 3> coordBegin(0.0, 0.0, 0.0);
    utils::Blocking<3> blocking(coordBegin, in_data.shape(), blockShape);

    // reserve output
    const std::string out_file_name = "./out.h5";
    vigra::HDF5File out_hdf5_file(out_file_name, vigra::HDF5File::New);
    auto in_shape = in_data.shape();
    auto out_shape = util::append_to_shape<3>(in_shape, 2); // TODO: make this dependent on the selected features!
    std::cout << "Trying to set up output file " << out_file_name << " with dataset " << dataset_name << " and shape " << out_shape << std::endl;
    vigra::TinyVector<int64_t, 4> chunkSize(128.0, 128.0, 128.0, 1);
    vigra::ChunkedArrayHDF5<4, float> out_data(out_hdf5_file, dataset_name, vigra::HDF5File::New, out_shape, chunkSize);

    // TODO: read selected features
    feature_computer3u8f::selected_features_type selected_features = {std::make_pair("GaussianSmoothing", 1.0f), std::make_pair("GaussianSmoothing", 3.5f)};

    // set up graph
    graph g;
    feature_computer3u8f_node feature_computer(g, std::make_shared<feature_computer3u8f>(cancelled_job_ids, selected_features));
    random_forest_predictor3f_node rf_predictor(g, std::make_shared<random_forest_predictor3f>(rf_vector, cancelled_job_ids));
    hdf5_output_node hdf5_writer(g, std::make_shared<hdf5_output>(cancelled_job_ids, out_data, blocking), serial);

    make_edge(get<feature_computer3u8f::OUT_FEATURES>(feature_computer.output_ports()), rf_predictor);
    make_edge(get<random_forest_predictor3f::OUT_PREDICTION>(rf_predictor.output_ports()), hdf5_writer);

    // process blocks
    std::cout << "found a dataset of shape " << in_data.shape() << " and " << blocking.numberOfBlocks() << " blocks" << std::endl;
    for (size_t i = 0; i < blocking.numberOfBlocks(); ++i)
//    for (size_t i = 0; i < 1; ++i)
    {
        // TODO: get halo!
        utils::Block<3> block = blocking.getBlock(i);
        vigra::MultiArray<3, uint8_t> raw_block(block.shape());
        in_data.checkoutSubarray(block.begin(), raw_block);
        feature_computer.try_put(feature_computer3u8f::in_job_type(i, raw_block));
    }
    g.wait_for_all();

    std::cout << "done processing" << std::endl;
    return 0;
}
