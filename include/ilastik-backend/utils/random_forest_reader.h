#ifndef _RANDOM_FOREST_LOADER_HXX_
#define _RANDOM_FOREST_LOADER_HXX_

#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/multi_array.hxx>
#include <hdf5_hl.h>

namespace ilastikbackend
{
    namespace util
    {
        typedef size_t LabelType;
        typedef vigra::RandomForest<LabelType> RandomForestType;
        typedef std::vector<RandomForestType> RandomForestVectorType;

        /**
         * @brief Read the random forests from the hdf5 files.
         *
         * WARNING: this shows some warnings on the command line because we try to read one more
         *          tree than is available. But that seems to be the easiest option to get all RFs in the group.
         *
         */
        bool get_rfs_from_file(
            RandomForestVectorType& rfs,
            const std::string& fn,
            const std::string& path_in_file = "PixelClassification/ClassifierForests/Forest",
            int n_leading_zeros = 4);

        /**
         * helper for extending shape dimensions
         */
        template<unsigned int N>
        typename vigra::MultiArrayShape<N+1>::type append_to_shape(
          const typename vigra::MultiArrayShape<N>::type& shape,
          const size_t value)
        {
          typename vigra::MultiArrayShape<N+1>::type ret;
          for (size_t i = 0; i < N; i++) {
            ret[i] = shape[i];
          }
          ret[N] = value;
          return ret;
        }
    }
}

#endif // _RANDOM_FOREST_LOADER_HXX_
