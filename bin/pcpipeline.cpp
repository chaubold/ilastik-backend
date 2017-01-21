#include <iostream>
#include <tbb/flow_graph.h>

#include "flowgraph/jobdata.h"
#include "flowgraph/multiinoutnode.h"
#include "operators/baseoperator.h"

/**
 * Find TBB examples here:
 * https://software.intel.com/en-us/blogs/2011/09/09/a-feature-detection-example-using-the-intel-threading-building-blocks-flow-graph
 */

using namespace tbb;
using namespace tbb::flow;
using namespace ilastikbackend;

using data_type = flowgraph::JobData<int>;

struct square
{
    data_type operator()(data_type v)
    {
        if(v.data)
            return data_type(v.jobId, *v.data * *v.data); // *v.data dereferences the optional!
        else
            return data_type(v.jobId);
    }
};

struct cube
{
    data_type operator()(data_type v)
    {
        if(v.data)
            return data_type(v.jobId, *v.data * *v.data * *v.data);
        else
            return data_type(v.jobId);
    }
};

class sum
{
    int &my_sum;

  public:
    sum(int &s) : my_sum(s) {}
    int operator()(data_type v)
    {
        if(v.data)
            my_sum += *v.data;

        return my_sum;
    }
};

class test_operator : public operators::base_operator<tuple<data_type, data_type>, tuple<data_type> >
{
public:
    test_operator(const types::SetOfCancelledJobIds& setOfCancelledJobIds):
        base_operator<tuple<data_type, data_type>, tuple<data_type> >(setOfCancelledJobIds)
    {
    }

    virtual tuple<data_type> executeImpl(const tuple<data_type,data_type>& a)
    {
        std::cout << "Combining jobs of ids " << get<0>(a).jobId << " and " << get<1>(a).jobId << std::endl;
        if(get<0>(a).data && get<1>(a).data)
        {
            return tuple<data_type>(data_type(get<0>(a).jobId, *get<0>(a).data * *get<1>(a).data));
        }

        return tuple<data_type>(data_type(get<0>(a).jobId));
    }
};

int main()
{
    int result = 0;
    types::SetOfCancelledJobIds cancelledJobIds;

    graph g;
    broadcast_node<data_type> input(g);
    function_node<data_type, data_type> squarer(g, unlimited, square());
    function_node<data_type, data_type> cuber(g, unlimited, cube());
    join_node<tuple<data_type, data_type>, queueing> join(g);
    function_node<data_type, int> summer(g, serial, sum(result));
    flowgraph::multi_inout_node<tuple<data_type, data_type>, tuple<data_type> > multi_inout_tester(g, std::make_shared<test_operator>(cancelledJobIds));

    make_edge(input, squarer);
    make_edge(input, cuber);
    make_edge(squarer, get<0>(multi_inout_tester.input_ports()));
    make_edge(cuber, get<1>(multi_inout_tester.input_ports()));
    make_edge(multi_inout_tester, summer);

    for (int i = 1; i <= 10; ++i)
    {
        input.try_put(data_type(i, i));
    }
    g.wait_for_all();

    printf("Final result is %d\n", result);
    return 0;
}
