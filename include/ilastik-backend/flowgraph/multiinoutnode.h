#ifndef _FLOWGRAPH_MULTIINOUTNODE_H_
#define _FLOWGRAPH_MULTIINOUTNODE_H_

#include <iostream>
#include <tbb/flow_graph.h>
#include "ilastik-backend/types.h"
#include "ilastik-backend/operators/baseoperator.h"
#include "ilastik-backend/flowgraph/jobdata.h"

namespace ilastikbackend
{
    namespace flowgraph
    {

        /**
         * Unpacked join node takes every individual input type as template parameter, not just a tuple of them.
         * This is to make unwrapping the key matching of the job Id easier.
         *
         * Some docs on tuple unpacking: http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
         *
         * Also note the fancy trick applied in tbb::flow: define class with one template type, specialize it to be a tuple, but then take
         * the tuple's types as template parameter pack!
         */
        template<typename InputTuple> class unpacked_join_node;

        template<typename ...INS>
        class unpacked_join_node<tbb::flow::tuple<INS...> > : public tbb::flow::join_node< tbb::flow::tuple<INS...>, tbb::flow::key_matching<types::job_id_type> >
        {
        public:
            using base_type = tbb::flow::join_node< tbb::flow::tuple<INS...>, tbb::flow::key_matching<types::job_id_type> >;
        public:
            unpacked_join_node(tbb::flow::graph& graph):
                base_type(graph, job_data_id_extractor<INS>()...)
            {}
        };

        /**
         * The output_setter struct is a helper to place all values from the result tuple
         * on their respective output slots of the multi_inout_node.
         */
        template<typename IN, typename OUT, size_t N>
        struct output_setter
        {
            void operator()(IN& result, OUT& output_ports) const
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
            void operator()(IN& result, OUT& output_ports) const
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
            using input_join_node = unpacked_join_node<IN>;
            using operator_multifunction_node = tbb::flow::multifunction_node<typename input_join_node::output_type, OUT>;
            using base_type = tbb::flow::composite_node<IN, OUT>;

        public:
            // API
            multi_inout_node(tbb::flow::graph& graph, std::shared_ptr<operators::base_operator<IN, OUT> > baseOp):
                base_type(graph),
                input_join_node_(graph),
                function_node_(graph, tbb::flow::unlimited, [baseOp](const IN& tupleIn, typename operator_multifunction_node::output_ports_type &output_ports) -> void {
                    OUT result = baseOp->execute(tupleIn);
                    output_setter<OUT, typename operator_multifunction_node::output_ports_type, std::tuple_size<OUT>::value - 1> os;
                    os(result, output_ports);
                }),
                operator_(baseOp)
            {
                tbb::flow::make_edge(input_join_node_, function_node_);

                // http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
                std::cout << "Found " << std::tuple_size<IN>::value << " inputs" << std::endl;
                std::cout << "Found " << std::tuple_size<OUT>::value << " outputs" << std::endl;
                typename base_type::input_ports_type input_tuple = get_input_tuple(std::make_index_sequence<std::tuple_size<IN>::value>());
                typename base_type::output_ports_type output_tuple = get_output_tuple(std::make_index_sequence<std::tuple_size<OUT>::value>());
                base_type::set_external_ports(input_tuple, output_tuple);
            }

            ~multi_inout_node()
            {
            }

        private:
            template<size_t... INDICES>
            typename base_type::input_ports_type get_input_tuple(std::index_sequence<INDICES...>)
            {
                return typename base_type::input_ports_type(tbb::flow::input_port<INDICES>(input_join_node_)...);
            }

            template<size_t... INDICES>
            typename base_type::output_ports_type get_output_tuple(std::index_sequence<INDICES...>)
            {
                return typename base_type::output_ports_type(tbb::flow::output_port<INDICES>(function_node_)...);
            }

        private:
            input_join_node input_join_node_;
            operator_multifunction_node function_node_;
            std::shared_ptr<operators::base_operator<IN, OUT> > operator_;
        };

        /**
        * A single_inout_node is a flow graph node that requires a *single* input, and produces several outputs.
        * It encapsulates a function node that calls an instance of an ilastikbackend::operators::BaseOperator.
        *
        * See https://software.intel.com/en-us/node/589717 for an example
        */
        template<typename IN, typename OUT>
        class single_inout_node : public tbb::flow::multifunction_node<IN, OUT>
        {
        public:
            // typedefs
            using base_type = tbb::flow::multifunction_node<IN, OUT>;

        public:
            // API
            single_inout_node(tbb::flow::graph& graph, std::shared_ptr<operators::base_operator<tbb::flow::tuple<IN>, OUT> > baseOp):
                base_type(graph, tbb::flow::unlimited, [baseOp](const IN& in, typename base_type::output_ports_type &output_ports) -> void {
                    OUT result = baseOp->execute(std::make_tuple(in));
                    output_setter<OUT, typename base_type::output_ports_type, std::tuple_size<OUT>::value - 1> os;
                    os(result, output_ports);
                }),
                operator_(baseOp)
            {
            }

        private:
            std::shared_ptr<operators::base_operator<tbb::flow::tuple<IN>, OUT> > operator_;
        };


    }
}

#endif // _FLOWGRAPH_MULTIINOUTNODE_H_
