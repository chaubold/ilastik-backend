#ifndef _TYPES_H_
#define _TYPES_H_

#ifdef CPP17
    #include <experimental/optional>
#else
    #include <boost/optional>
#endif

#include <tbb/concurrent_unordered_set.h>

namespace ilastikbackend {
namespace types {

/// Optionals are variables that must not have a value set.
#ifdef CPP17
    template<typename T>
    using Optional = std::experimental::optional<T>;
#else
    template<typename T>
    using Optional = boost::optional<T>;
#endif

/// We use size_t to assign a unique ID to every job (e.g. processing a block)
using JobIdType = size_t;

/// We use a concurrently accessible set to store all job ids that are cancelled
using SetOfCancelledJobIds = tbb::concurrent_unordered_set<JobIdType>;
} // types
} // ilastikbackend

#endif // _TYPES_H_
