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

using namespace tbb;
using namespace tbb::flow;
using namespace ilastikbackend;

using feature_computer3u8f = operators::feature_computation_operator<3, uint8_t, float>;
using feature_computer3u8f_node = flowgraph::single_inout_node< feature_computer3u8f::in_job_type, tuple<feature_computer3u8f::out_job_type> >;
using random_forest_predictor3f = operators::random_forest_prediction_operator<3, float>;
using random_forest_predictor3f_node = flowgraph::single_inout_node< random_forest_predictor3f::in_job_type, tuple<random_forest_predictor3f::out_job_type> >;

int main()
{
    std::cout << "Starting..." << std::endl;
    types::set_of_cancelled_job_ids cancelled_job_ids;

    // load Random forests
    const std::string rf_filename = "./testPC.ilp";
    const std::string rf_path = "/PixelClassification/ClassifierForests/Forest";
    util::RandomForestVectorType rf_vector;
    util::get_rfs_from_file(rf_vector, rf_filename, rf_path, 4);

    // TODO: read selected features
    feature_computer3u8f::selected_features_type selected_features = {std::make_pair("GaussianSmoothing", 1.0f), std::make_pair("GaussianSmoothing", 3.5f)};

    // set up graph
    graph g;
    feature_computer3u8f_node feature_computer(g, std::make_shared<feature_computer3u8f>(cancelled_job_ids, selected_features));
    random_forest_predictor3f_node rf_predictor(g, std::make_shared<random_forest_predictor3f>(rf_vector, cancelled_job_ids));
    make_edge(get<feature_computer3u8f::OUT_FEATURES>(feature_computer.output_ports()), rf_predictor);

    // read raw data (chunked and cached)
    const std::string raw_file_name = "./testraw.h5";
    const std::string dataset_name = "exported_data";
    vigra::HDF5File hdf5_file(raw_file_name, vigra::HDF5File::ReadOnly);
    vigra::ChunkedArrayHDF5<3, uint8_t> in_data(hdf5_file, dataset_name, vigra::HDF5File::ReadOnly);

    // process blocks
    vigra::TinyVector<int64_t, 3> blockShape(64, 64, 64);
    vigra::TinyVector<int64_t, 3> coordBegin(0.0, 0.0, 0.0);
    utils::Blocking<3> blocking(coordBegin, in_data.shape(), blockShape);
    std::cout << "found a dataset of shape " << in_data.shape() << " and " << blocking.numberOfBlocks() << " blocks" << std::endl;
    for (size_t i = 0; i < blocking.numberOfBlocks(); ++i)
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
