#include <iostream>
#include <tbb/flow_graph.h>
#include <assert.h>

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

using data_type = flowgraph::job_data<int>;

struct square
{
    data_type operator()(data_type v)
    {
        if(v.data)
            return data_type(v.job_id, *v.data * *v.data); // *v.data dereferences the optional!
        else
            return data_type(v.job_id);
    }
};

struct cube
{
    data_type operator()(data_type v)
    {
        if(v.data)
            return data_type(v.job_id, *v.data * *v.data * *v.data);
        else
            return data_type(v.job_id);
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

class test_operator : public operators::base_operator<tuple<data_type, data_type, data_type>, tuple<data_type> >
{
public:
    test_operator(const types::set_of_cancelled_job_ids& setOfCancelledJobIds):
        base_operator<tuple<data_type, data_type, data_type>, tuple<data_type> >(setOfCancelledJobIds)
    {
    }

    virtual tuple<data_type> executeImpl(const tuple<data_type,data_type, data_type>& a)
    {
        std::cout << "Combining jobs of ids " << get<0>(a).job_id << " and " << get<1>(a).job_id << std::endl;
        assert(get<0>(a).job_id == get<1>(a).job_id);

        if(get<0>(a).data && get<1>(a).data)
        {
            return tuple<data_type>(data_type(get<0>(a).job_id, *get<0>(a).data * *get<1>(a).data));
        }

        return tuple<data_type>(data_type(get<0>(a).job_id));
    }
};

int main()
{
    int result = 0;
    types::set_of_cancelled_job_ids cancelled_job_ids;

    graph g;
    broadcast_node<data_type> input(g);
    function_node<data_type, data_type> squarer(g, unlimited, square());
    function_node<data_type, data_type> cuber(g, unlimited, cube());
    function_node<data_type, int> summer(g, serial, sum(result));
    flowgraph::multi_inout_node<tuple<data_type, data_type, data_type>, tuple<data_type> > multi_inout_tester(g, std::make_shared<test_operator>(cancelled_job_ids));

    make_edge(input, squarer);
    make_edge(input, cuber);
    make_edge(squarer, get<0>(multi_inout_tester.input_ports()));
    make_edge(cuber, get<1>(multi_inout_tester.input_ports()));
    make_edge(input, get<2>(multi_inout_tester.input_ports()));
    make_edge(multi_inout_tester, summer);

    for (int i = 1; i <= 10; ++i)
    {
        input.try_put(data_type(i, i));
    }
    g.wait_for_all();

    printf("Final result is %d\n", result);
    return 0;
}
