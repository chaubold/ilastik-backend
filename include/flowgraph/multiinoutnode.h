#ifndef _FLOWGRAPH_MULTIINOUTNODE_H_
#define _FLOWGRAPH_MULTIINOUTNODE_H_

#include <iostream>
#include <tbb/flow_graph.h>
#include "types.h"
#include "operators/baseoperator.h"
#include "flowgraph/jobdata.h"

namespace ilastikbackend
{
    namespace flowgraph
    {

        /**
         * Unpacked join node takes every individual input type as template parameter, not just a tuple of them.
         * This is to make unwrapping the key matching of the job Id easier.
         *
         * Some docs on tuple unpacking: http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
         */
        template<typename ...INS>
        class unpacked_join_node : public tbb::flow::join_node< tbb::flow::tuple<INS...>, tbb::flow::key_matching<types::JobIdType> >
        {
        public:
            using base_type = tbb::flow::join_node< tbb::flow::tuple<INS...>, tbb::flow::key_matching<types::JobIdType> >;
        public:
            unpacked_join_node(tbb::flow::graph& graph):
                base_type(graph, JobDataIdExtractor<INS>()...)
            {}
        };

        /**
         * The output_setter struct is a helper to place all values from the result tuple
         * on their respective output slots of the multi_inout_node.
         */
        template<typename IN, typename OUT, size_t N>
        struct output_setter
        {
            void operator()(IN& result, OUT& output_ports)
            {
                // set output
                std::cout << "Setting output: " << N << std::endl;
                tbb::flow::get<N>(output_ports).try_put(tbb::flow::get<N>(result));

                // recursive call
                output_setter<IN, OUT, N-1> os;
                os(result, output_ports);
            }
        };

        template<typename IN, typename OUT>
        struct output_setter<IN, OUT, 0>
        {
            void operator()(IN& result, OUT& output_ports)
            {
                // set output witout recursive call
                std::cout << "Setting output: 0"<< std::endl;
                tbb::flow::get<0>(output_ports).try_put(tbb::flow::get<0>(result));
            }
        };

        /**
        * A multi_inout_node is a flow graph node that requires several inputs which may have come from separate branches, and produces several outputs.
        * The inputs are first collected by a join node and then processed by a function node, which is here always an instance of an ilastikbackend::operators::BaseOperator
        *
        * See https://software.intel.com/en-us/node/589717 for an example
        */
        template<typename IN, typename OUT>
        class multi_inout_node : public tbb::flow::composite_node<IN, OUT>
        {
        public:
            // typedefs
            using input_join_node = tbb::flow::join_node< IN, tbb::flow::key_matching<types::JobIdType> >;
            using operator_multifunction_node = tbb::flow::multifunction_node<typename input_join_node::output_type, OUT>;
            using base_type = tbb::flow::composite_node<IN, OUT>;

        public:
            // API
            multi_inout_node(tbb::flow::graph& graph, std::shared_ptr<operators::base_operator<IN, OUT> > baseOp):
                base_type(graph),
                input_join_node_(graph,
                                JobDataIdExtractor<JobData<int> >(),
                                JobDataIdExtractor<JobData<int> >()),
                function_node_(graph, tbb::flow::unlimited, [baseOp](const IN& tupleIn, typename operator_multifunction_node::output_ports_type &output_ports) -> void {
                    OUT result = baseOp->execute(tupleIn);
                    output_setter<OUT, typename operator_multifunction_node::output_ports_type, std::tuple_size<OUT>::value - 1> os;
                    os(result, output_ports);
                }),
                operator_(baseOp)
            {
                // fix constructor call to input_join_node through something like this?
                // http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
                tbb::flow::make_edge(input_join_node_, function_node_);
                typename base_type::input_ports_type input_tuple(tbb::flow::input_port<0>(input_join_node_), tbb::flow::input_port<1>(input_join_node_));
                typename base_type::output_ports_type output_tuple(tbb::flow::output_port<0>(function_node_));
                base_type::set_external_ports(input_tuple, output_tuple);
            }

            ~multi_inout_node()
            {
            }

        private:
            input_join_node input_join_node_;
            operator_multifunction_node function_node_;
            std::shared_ptr<operators::base_operator<IN, OUT> > operator_;
        };

    }
}

#endif // _FLOWGRAPH_MULTIINOUTNODE_H_
