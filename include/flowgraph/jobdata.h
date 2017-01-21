#ifndef _FLOWGRAPH_JOBDATA_H_
#define _FLOWGRAPH_JOBDATA_H_

#include "types.h"

namespace ilastikbackend
{
    namespace flowgraph
    {
        /**
        * JobData encapsulates all data that is passed along edges of the flow graph.
        * It contains the jobId, which is needed to group information from different slots by job for the next task.
        * The data is stored as optional type, which is present ONLY IF the job was NOT cancelled.
        */
        template<typename T>
        struct JobData
        {
            types::JobIdType jobId;
            ilastikbackend::types::Optional<T> data;

            /// empty constructor
            JobData():
                jobId(0),
                data()
            {}

            /// constructor without data
            JobData(types::JobIdType jobId):
                jobId(jobId)
            {}

            /// constructor with jobid and data
            JobData(types::JobIdType jobId, T data):
              jobId(jobId),
              data(data)
            {}
        };

        /**
        Functor needed by the MultiInOutNode's join_node to extract the key per input.
        */
        template<typename T>
        struct JobDataIdExtractor
        {
            types::JobIdType operator()(const T& jd)
            {
                return jd.jobId;
            }
        };
    }
}

#endif // _FLOWGRAPH_JOBDATA_H_
