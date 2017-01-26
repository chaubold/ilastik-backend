#ifndef RANDOM_FOREST_PREDICTION_OPERATOR_HXX
#define RANDOM_FOREST_PREDICTION_OPERATOR_HXX

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
        class random_forest_prediction_operator : public base_operator< tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM+1, DATATYPE>>>, tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM+1, DATATYPE>>> >
        {
        public:
            using in_array_type = vigra::MultiArray<DIM+1, DATATYPE>;
            using out_array_type = vigra::MultiArray<DIM+1, DATATYPE>;
            using in_job_type = flowgraph::job_data<in_array_type>;
            using out_job_type = flowgraph::job_data<out_array_type>;
            using base_type = base_operator<tbb::flow::tuple<in_job_type>, tbb::flow::tuple<out_job_type> >;

            // definition of enums to use for the slots
            static constexpr size_t IN_FEATURES = 0;
            static constexpr size_t OUT_PREDICTION = 0;

        public:
            // API
            random_forest_prediction_operator(
                const util::RandomForestVectorType& random_forest_vector,
                const types::set_of_cancelled_job_ids& setOfCancelledJobIds
            ):
                base_type(setOfCancelledJobIds),
                random_forest_vector_(random_forest_vector)
            {
                std::cout << "Constructing RF prediction operator with " << random_forest_vector.size() << " RFs" << std::endl;
                assert(random_forest_vector_.size() > 0);
            }

            virtual tbb::flow::tuple<out_job_type> executeImpl(const tbb::flow::tuple<in_job_type>& in) const
            {
                std::cout << "Predicting for job " << std::get<IN_FEATURES>(in).job_id << std::endl;
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

                // out array for the predictions
                vigra::MultiArray<2, DATATYPE> prediction_map_view(vigra::Shape2(pixel_count, num_pixel_classification_labels));

                // loop over all random forests for prediction probabilities
                std::cout << "\tPredict RFs" << std::endl;
                for(size_t rf = 0; rf < random_forest_vector_.size(); rf++)
                {
                    std::cout << "\tjob " << std::get<IN_FEATURES>(in).job_id << " step " << rf << std::endl;
                    vigra::MultiArray<2, DATATYPE> prediction_temp(pixel_count, num_pixel_classification_labels);
                    random_forest_vector_[rf].predictProbabilities(feature_view, prediction_temp);
                    prediction_map_view += prediction_temp;
                }

                // divide probs by num random forests
                prediction_map_view /= random_forest_vector_.size();

                auto prediction_map_shape = in_array.shape();
                prediction_map_shape[DIM] = num_pixel_classification_labels; // has DIM+1 entries, do DIM is last one

                vigra::MultiArrayView<DIM+1, DATATYPE> prediction_map_image_view(prediction_map_shape, prediction_map_view.data());

                std::cout << "done predicting for job " << std::get<IN_FEATURES>(in).job_id << std::endl;
                return tbb::flow::tuple<out_job_type>(out_job_type(std::get<IN_FEATURES>(in).job_id));
            }

        private:
            const util::RandomForestVectorType& random_forest_vector_;
        };
    } // namespace operators
} // namespace ilastik_backend

#endif // RANDOM_FOREST_PREDICTION_OPERATOR_HXX
