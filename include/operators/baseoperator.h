#ifndef _OPERATORS_BASEOPERATOR_H_
#define _OPERATORS_BASEOPERATOR_H_

#include "types.h"
#include <tbb/flow_graph.h>

namespace ilastikbackend
{
    namespace operators
    {

        // -----------------------------------------------------------------------------------------------
        /**
          * Helper to check every element of the incoming parameter tuple for cancelled jobs.
          * First, all elements of the tuple are checked whether the data is empty. If that
          * is not the case, we query the set of cancelled job ids.
          */
        template<typename IN, size_t N>
        struct cancellation_checker
        {
            bool operator()(const IN& params, const types::set_of_cancelled_job_ids& set_of_cancelled_job_ids) const
            {
                // if this index of the tuple was cancelled, return false
                if(!tbb::flow::get<N>(params).data)
                    return false;

                // else check the remaining tuple elements
                cancellation_checker<IN, N-1> cc;
                return cc(params, set_of_cancelled_job_ids);
            }
        };

        template<typename IN>
        struct cancellation_checker<IN, 0>
        {
            bool operator()(const IN& params, const types::set_of_cancelled_job_ids& set_of_cancelled_job_ids) const
            {
                if(!tbb::flow::get<0>(params).data)
                    return false;

                // if the last tuple element was not cancelled, check the set_of_cancelled_job_ids
                types::job_id_type job_id = tbb::flow::get<0>(params).job_id;
                return (set_of_cancelled_job_ids.count(job_id) > 0);
            }
        };

        // -----------------------------------------------------------------------------------------------
        /**
         * empty_tuple_builder helps to set up a tuple of empty results in case a job was cancelled
         */
        template<typename InputTuple> struct empty_tuple_builder;

        template<typename ...INS>
        struct empty_tuple_builder<tbb::flow::tuple<INS...>>
        {
            using tuple_type = typename tbb::flow::tuple<INS...>;

            tuple_type operator()(types::job_id_type job_id) const
            {
                return tuple_type(INS()...);
            }
        };

        // -----------------------------------------------------------------------------------------------
        /**
         * The base operator handles job cancellation. If the job needs to be computed,
         * it calls executeImpl that is implemented by derived classes.
         *
         * TODO: check that IN or OUT are not empty, or things might break...
         */
        template<typename IN, typename OUT>
        class base_operator
        {
        public:
            base_operator(const types::set_of_cancelled_job_ids& set_of_cancelled_job_ids);
            virtual ~base_operator();
        
            OUT execute(const IN& in) const;
            virtual OUT executeImpl(const IN& in) const = 0;

        protected:
            /**
             * @return True if this jobId is now contained in the setOfCanneledJobIds
             */
            bool cancelled(types::job_id_type job_id) const;

        private:
            const types::set_of_cancelled_job_ids& set_of_cancelled_job_ids_;
        };
    
        // -----------------------------------------------------------------------------------------------
        // Implementation
        // -----------------------------------------------------------------------------------------------

        template<typename IN, typename OUT>
        base_operator<IN, OUT>::base_operator(const types::set_of_cancelled_job_ids& set_of_cancelled_job_ids):
            set_of_cancelled_job_ids_(set_of_cancelled_job_ids)
        {
        }

        template<typename IN, typename OUT>
        base_operator<IN, OUT>::~base_operator()
        {
        }

        template<typename IN, typename OUT>
        OUT base_operator<IN, OUT>::execute(const IN& in) const
        {
            // if any of the incomings is cancelled, pass on cancel messages everywhere
            // if(set_of_cancelled_job_ids_.count(jobId) > 0): cancelled! pass on cancel messages everywhere.
            cancellation_checker<IN, std::tuple_size<IN>::value - 1 > cc;
            bool cancelled = cc(in, set_of_cancelled_job_ids_);

            if(cancelled)
            {
                types::job_id_type job_id = tbb::flow::get<0>(in).job_id;
                std::cout << "Found cancelled job: " << job_id << std::endl;
                empty_tuple_builder<OUT> etb;
                return etb(job_id);
            }
            else
                return executeImpl(in);
        }

        template<typename IN, typename OUT>
        bool base_operator<IN, OUT>::cancelled(types::job_id_type job_id) const
        {
            return (set_of_cancelled_job_ids_.count(job_id) > 0);
        }
    } // namespace operator
} // namespace ilastik_backend

#endif // _OPERATORS_BASEOPERATOR_H_
