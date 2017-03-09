#ifndef HDF5_OUTPUT_OPERATOR_H
#define HDF5_OUTPUT_OPERATOR_H

#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/multi_array_chunked_hdf5.hxx>

#include "ilastik-backend/flowgraph/jobdata.h"
#include "ilastik-backend/operators/baseoperator.h"
#include "ilastik-backend/utils/blocking.h"
#include "ilastik-backend/utils/random_forest_reader.h"

namespace ilastikbackend
{
    namespace operators
    {

        template<int DIM, typename DATATYPE>
        class hdf5_output_operator : public base_operator<tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM, DATATYPE>>>, tbb::flow::tuple<flowgraph::job_data<vigra::MultiArray<DIM, DATATYPE>>> >
        {
        public:
            using in_array_type = vigra::MultiArray<DIM, DATATYPE>;
            using in_job_type = flowgraph::job_data<in_array_type>;
            using base_type = base_operator<tbb::flow::tuple<in_job_type>, tbb::flow::tuple<in_job_type> >;
            using selected_features_type = std::vector<std::pair<std::string, DATATYPE>>;
            using HDF5Array = vigra::ChunkedArrayHDF5<DIM, DATATYPE>;

            // definition of enums to use for the slots
            static constexpr size_t IN_DATA = 0;
        public:
            hdf5_output_operator(const types::set_of_cancelled_job_ids& setOfCancelledJobIds,
                                 HDF5Array& out_array,
                                 const utils::Blocking<DIM-1>& blocking):
                base_type(setOfCancelledJobIds),
                out_array_(out_array),
                blocking_(blocking)
            {
                std::cout << "Setting up HDF5 output" << std::endl;
            }

            virtual tbb::flow::tuple<in_job_type> executeImpl(const tbb::flow::tuple<in_job_type>& in) const
            {
                types::job_id_type job_id = std::get<IN_DATA>(in).job_id;
                std::cout << "saving result for job" << job_id << std::endl;

                const in_array_type& in_array = (*std::get<IN_DATA>(in).data);
                auto block_start = utils::append_to_shape<DIM-1>(blocking_.getBlock(job_id).begin(), 0);
                std::cout << "saving block from " << block_start << " with shape " << in_array.shape() << " into array " << out_array_.shape() << std::endl;
                try{
                    out_array_.commitSubarray(block_start, in_array);
                }
                catch(std::exception& e)
                {
                    std::cout << "Error: " << e.what() << std::endl;
                }

                return tbb::flow::tuple<in_job_type>(in_job_type(job_id));
            }

        private:
            // members
            HDF5Array& out_array_;
            const utils::Blocking<DIM-1>& blocking_;
        };
    }
}

#endif // HDF5_OUTPUT_OPERATOR_H
