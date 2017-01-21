#ifndef _OPERATORS_BASEOPERATOR_H_
#define _OPERATORS_BASEOPERATOR_H_

#include "types.h"
#include <tbb/flow_graph.h>

namespace ilastikbackend
{
    namespace operators
    {
        template<typename IN, typename OUT>
        class base_operator
        {
        public:
            base_operator(const types::set_of_cancelled_job_ids& setOfCancelledJobIds);
            virtual ~base_operator();
        
            OUT execute(const IN& in);
            virtual OUT executeImpl(const IN& in) = 0;
        private:
            const types::set_of_cancelled_job_ids& setOfCancelledJobIds_;
        };
    
        template<typename IN, typename OUT>
        base_operator<IN, OUT>::base_operator(const types::set_of_cancelled_job_ids& setOfCancelledJobIds):
            setOfCancelledJobIds_(setOfCancelledJobIds)
        {
        }

        template<typename IN, typename OUT>
        base_operator<IN, OUT>::~base_operator()
        {
        }

        template<typename IN, typename OUT>
        OUT base_operator<IN, OUT>::execute(const IN& in)
        {
            // TODO: if any of the incomings is cancelled, pass on cancel messages everywhere
            // TODO: if(setOfCancelledJobIds_.count(jobId) > 0): cancelled! pass on cancel messages everywhere.
            return executeImpl(in);
        }
    }
}

#endif // _OPERATORS_BASEOPERATOR_H_
