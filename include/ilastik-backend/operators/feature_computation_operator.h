#ifndef FEATURE_COMPUTATION_OPERATOR_H
#define FEATURE_COMPUTATION_OPERATOR_H

#include <iostream>
#include <vigra/multi_array.hxx>
#include "ilastik-backend/flowgraph/jobdata.h"
#include "ilastik-backend/operators/baseoperator.h"
#include "ilastik-backend/utils/feature_calculator.h"

namespace ilastikbackend
{
    namespace operators
    {

        template<int DIM, typename IN_TYPE, typename OUT_TYPE>
        class feature_computation_operator : public base_operator<tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM, IN_TYPE>>>, tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM+1, OUT_TYPE>>> >
        {
        public:
            using in_array_type = vigra::MultiArray<DIM, IN_TYPE>;
            using out_array_type = vigra::MultiArray<DIM+1, OUT_TYPE>;
            using in_job_type = flowgraph::job_data<in_array_type>;
            using out_job_type = flowgraph::job_data<out_array_type>;
            using base_type = base_operator<tbb::flow::tuple<in_job_type>, tbb::flow::tuple<out_job_type> >;
            using selected_features_type = std::vector<std::pair<std::string, OUT_TYPE>>;

            // definition of enums to use for the slots
            static constexpr size_t IN_RAW = 0;
            static constexpr size_t OUT_FEATURES = 0;
        public:
            feature_computation_operator(const types::set_of_cancelled_job_ids& setOfCancelledJobIds,
                                         const selected_features_type& selected_features):
                base_type(setOfCancelledJobIds),
                selected_features_(selected_features)
            {
                std::cout << "Setting up feature computation operator with " << selected_features.size() << " selected features" << std::endl;
            }

            virtual tbb::flow::tuple<out_job_type> executeImpl(const tbb::flow::tuple<in_job_type>& in) const
            {
                std::cout << "Computing features for job" << std::get<IN_RAW>(in).job_id << std::endl;

                const in_array_type& in_array = (*std::get<IN_RAW>(in).data);
                vigra::MultiArray<DIM, OUT_TYPE> in_array_out_type(in_array);

                out_array_type out_array;
                util::FeatureCalculator<DIM, OUT_TYPE> feature_calculator(selected_features_);
                feature_calculator.calculate(in_array_out_type, out_array);

                return tbb::flow::tuple<out_job_type>(out_job_type(std::get<OUT_FEATURES>(in).job_id));
            }

        private:
            // members
            selected_features_type selected_features_;
        };
    }
}

#endif // FEATURE_COMPUTATION_OPERATOR_H
