#ifndef RANDOM_FOREST_TRAINING_OPERATOR_H
#define RANDOM_FOREST_TRAINING_OPERATOR_H

#include <tuple>
#include <assert.h>

#include <tbb/task.h>
#include <tbb/flow_graph.h>

#include <vigra/multi_array.hxx>

#include "ilastik-backend/flowgraph/jobdata.h"
#include "ilastik-backend/operators/baseoperator.h"
#include "ilastik-backend/utils/random_forest_reader.h"

namespace ilastikbackend
{
    namespace operators
    {
        template<int DIM, typename DATATYPE>
        class random_forest_training_operator : public base_operator< tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM+1, DATATYPE>>>, tbb::flow::tuple<flowgraph::job_data<utils::RandomForestType>> >
        {
        public:
            using in_array_type = vigra::MultiArray<DIM+1, DATATYPE>;
            using in_job_type = flowgraph::job_data<in_array_type>;
            using out_job_type = flowgraph::job_data<utils::RandomForestType>;
            using base_type = base_operator<tbb::flow::tuple<in_job_type>, tbb::flow::tuple<out_job_type> >;

            // definition of enums to use for the slots
            static constexpr size_t IN_FEATURES = 0;
            static constexpr size_t OUT_PREDICTION = 0;

        public:
            // API
            random_forest_training_operator(
                const types::set_of_cancelled_job_ids& setOfCancelledJobIds
            ):
                base_type(setOfCancelledJobIds),
            {
                std::cout << "Constructing RF training operator" << std::endl;
            }

            virtual tbb::flow::tuple<out_job_type> executeImpl(const tbb::flow::tuple<in_job_type>& in) const
            {
                std::cout << "Training for job " << std::get<IN_FEATURES>(in).job_id << std::endl;
                size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
                size_t num_required_features = random_forest_vector_[0].feature_count();
                in_array_type in_array = *(std::get<IN_FEATURES>(in).data);

                size_t pixel_count = 1;
                for (size_t dim = 0; dim < DIM; dim++) {
                  pixel_count *= in_array.shape(dim);
                }

                std::cout << "Have data of shape " << in_array.shape() << " and need " << num_required_features << " features" << std::endl;
                assert(num_required_features == in_array.shape(DIM));

                // transform to a feature view for prediction
                vigra::MultiArrayView<2, DATATYPE> feature_view(vigra::Shape2(pixel_count, num_required_features), in_array.data());

                assert(features.shape(0) == labels.shape(0));
                std::cout << "random_forest2_training from " << features.shape(0)
                          << " samples and " << features.shape(1) << " features" << std::endl;

                // train
                rf.learn(feature_view, labels_view);

                std::cout << "done training for job " << std::get<IN_FEATURES>(in).job_id << std::endl;
                return tbb::flow::tuple<out_job_type>(out_job_type(std::get<IN_FEATURES>(in).job_id));
            }
        };
    } // namespace operators
} // namespace ilastik_backend

#endif // RANDOM_FOREST_TRAINING_OPERATOR_H
