#ifndef _FLOWGRAPH_MULTIINOUTNODE_H_
#define _FLOWGRAPH_MULTIINOUTNODE_H_

#include <tbb/flow_graph.h>
#include "types.h"
#include "operators/baseoperator.h"
#include "flowgraph/jobdata.h"

namespace ilastikbackend {
    namespace flowgraph {

        /**
        * A MultiInOutNode is a flow graph node that requires several inputs which may have come from separate branches, and produces several outputs.
        * The inputs are first collected by a join node and then processed by a function node, which is here always an instance of an ilastikbackend::operators::BaseOperator
        */
        template<typename OUT, typename ...INS>
        class MultiInOutNode
        {
        public:
            MultiInOutNode(tbb::flow::graph& graph, const std::shared_ptr<operators::BaseOperator<OUT, INS...> >& baseOp, INS... in):
                inputCollector_(graph,
                                JobDataIdExtractor<INS>(in)...),
                functionNode_(graph, [baseOp](INS... otherIn) -> OUT {baseOp->execute(otherIn...);})
            {}

            ~MultiInOutNode()
            {
            }
        
        public:
            enum InputSlots {InA=0, InB};
            enum OutputSlots {OutA=0, OutB};

        private:
            tbb::flow::join_node< INS..., tbb::flow::tag_matching > inputCollector_;
            tbb::flow::multifunction_node<INS..., OUT> functionNode_;
            std::shared_ptr<operators::BaseOperator<OUT, INS...> > operator_;
        };

    }
}

#endif // _FLOWGRAPH_MULTIINOUTNODE_H_
