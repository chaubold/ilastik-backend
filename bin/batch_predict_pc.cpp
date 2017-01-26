#include <iostream>
#include <tbb/flow_graph.h>
#include <assert.h>

#include "ilastik-backend/flowgraph/jobdata.h"
#include "ilastik-backend/flowgraph/multiinoutnode.h"
#include "ilastik-backend/operators/baseoperator.h"
#include "ilastik-backend/operators/feature_computation_operator.h"
#include "ilastik-backend/operators/random_forest_prediction_operator.h"
#include "ilastik-backend/utils/random_forest_reader.h"

using namespace tbb;
using namespace tbb::flow;
using namespace ilastikbackend;

using feature_computer3u8f = operators::feature_computation_operator<3, uint8_t, float>;
using feature_computer3u8f_node = flowgraph::multi_inout_node< tuple<feature_computer3u8f::in_job_type>, tuple<feature_computer3u8f::out_job_type> >;
using random_forest_predictor3f = operators::random_forest_prediction_operator<3, float>;
using random_forest_predictor3f_node = flowgraph::multi_inout_node< tuple<random_forest_predictor3f::in_job_type>, tuple<random_forest_predictor3f::out_job_type> >;

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
    feature_computer3u8f::selected_features_type selected_features = {std::make_pair("Gaussian Smoothing", 1.0f), std::make_pair("Gaussian Smoothing", 3.5f)};

    // set up graph
    graph g;
    feature_computer3u8f_node feature_computer(g, std::make_shared<feature_computer3u8f>(cancelled_job_ids, selected_features));
    random_forest_predictor3f_node rf_predictor(g, std::make_shared<random_forest_predictor3f>(rf_vector, cancelled_job_ids));
    make_edge(get<feature_computer3u8f::OUT_FEATURES>(feature_computer.output_ports()), get<random_forest_predictor3f::IN_FEATURES>(rf_predictor.input_ports()));

    // process blocks
//    for (int i = 1; i <= 10; ++i)
//    {
//        input.try_put(data_type(i, i));
//    }
    g.wait_for_all();

    std::cout << "done processing" << std::endl;
    return 0;
}
