#ifndef _OPERATORS_BASEOPERATOR_H_
#define _OPERATORS_BASEOPERATOR_H_

#include "types.h"

namespace ilastikbackend
{
    namespace operators
    {
        template<typename OUT, typename ...INS>
        class BaseOperator
        {
        public:
            BaseOperator(const types::SetOfCancelledJobIds& setOfCancelledJobIds);
            virtual ~BaseOperator();
        
            OUT execute(INS... in);
            virtual OUT executeImpl(INS... in) = 0;
        private:
            const types::SetOfCancelledJobIds& setOfCancelledJobIds_;
        };
    
        template<typename OUT, typename ...INS>
        BaseOperator<OUT, INS...>::BaseOperator(const types::SetOfCancelledJobIds& setOfCancelledJobIds)
        {
        }

        template<typename OUT, typename ...INS>
        BaseOperator<OUT, INS...>::~BaseOperator()
        {
        }

        template<typename OUT, typename ...INS>
        OUT BaseOperator<OUT, INS...>::execute(INS... in)
        {
            // TODO: if any of the incomings is cancelled, pass on cancel messages everywhere
            // TODO: if(setOfCancelledJobIds_.count(jobId) > 0): cancelled! pass on cancel messages everywhere.
            return executeImpl(in...);
        }
    }
}

#endif // _OPERATORS_BASEOPERATOR_H_
