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
        struct job_data
        {
            types::job_id_type job_id;
            ilastikbackend::types::optional<T> data;

            /// empty constructor
            job_data():
                job_id(0),
                data()
            {}

            /// constructor without data
            job_data(types::job_id_type jobId):
                job_id(jobId)
            {}

            /// constructor with jobid and data
            job_data(types::job_id_type jobId, T data):
              job_id(jobId),
              data(data)
            {}
        };

        /**
        Functor needed by the MultiInOutNode's join_node to extract the key per input.
        */
        template<typename T>
        struct job_data_id_extractor
        {
            types::job_id_type operator()(const T& jd)
            {
                return jd.job_id;
            }
        };
    }
}

#endif // _FLOWGRAPH_JOBDATA_H_
