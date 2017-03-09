#ifndef _FEATURE_CALCULATOR_H_
#define _FEATURE_CALCULATOR_H_

#include <map>

#include <vigra/multi_array.hxx> /* for MultiArray */
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_tensorutilities.hxx>
#include <vigra/tinyvector.hxx>

#include "ilastik-backend/utils/random_forest_reader.h" // for append_to_shape

namespace ilastikbackend
{
    namespace utils
    {
        template<int N, typename DataType>
        class FeatureCalculator
        {
        public:
            // Type for reading the feature configuration
            typedef std::vector<std::pair<std::string, DataType> > StringDataPairVectorType;
            typedef std::map<std::string, std::string> StringStringMapType;
        public:
            FeatureCalculator(
                    const StringDataPairVectorType& feature_scales,
                    DataType window_size = 3.5);
            FeatureCalculator(
                    const StringDataPairVectorType& feature_scales,
                    const vigra::TinyVector<DataType, N> image_scales,
                    DataType window_size = 3.5);
            size_t get_feature_size(const std::string& feature_name) const;
            size_t get_feature_size() const;
            int calculate(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArray<N+1, DataType>& features);

            vigra::TinyVector<float, N> getHaloShape();
        private:
            int calculate_gaussian_smoothing(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArrayView<N+1, DataType>& features,
                    DataType feature_scale) const;
            int calculate_laplacian_of_gaussian(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArrayView<N+1, DataType>& features,
                    DataType feature_scale) const;
            int calculate_gaussian_gradient_magnitude(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArrayView<N+1, DataType>& features,
                    DataType feature_scale);
            int calculate_difference_of_gaussians(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArrayView<N+1, DataType>& features,
                    DataType feature_scale);
            int calculate_structure_tensor_eigenvalues(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArrayView<N+1, DataType>& features,
                    DataType feature_scale);
            int calculate_hessian_of_gaussian_eigenvalues(
                    const vigra::MultiArrayView<N, DataType>& image,
                    vigra::MultiArrayView<N+1, DataType>& features,
                    DataType feature_scale);

            // temporary storage for feature computation allocated only once
            vigra::MultiArray<N, DataType> feature_temp_;
            vigra::MultiArray<N, vigra::TinyVector<DataType, (N*(N+1))/2> > tensor_temp_;
            vigra::MultiArray<N, vigra::TinyVector<DataType, N> > eigenvalue_temp_;

            const StringDataPairVectorType& feature_scales_;
            DataType window_size_;
            std::map<std::string, size_t> feature_sizes_;
            vigra::ConvolutionOptions<N> conv_options_;
        };

        template<int N, typename DataType>
        vigra::TinyVector<float, N> FeatureCalculator<N, DataType>::getHaloShape() {
            vigra::TinyVector<float, N> ret;
            size_t halo_size = 0;
            for (
                 typename StringDataPairVectorType::const_iterator it = feature_scales_.begin();
                 it != feature_scales_.end();
                 it++
                 )
            {
                halo_size = std::max(size_t(round(3.5 * it->second)), halo_size);
            }

            std::fill(ret.begin(), ret.end(), halo_size);
            return ret;
        }

        ////
        //// class FeatureCalculator
        ////
        template<int N, typename DataType>
        FeatureCalculator<N, DataType>::FeatureCalculator(
                const StringDataPairVectorType& feature_scales,
                DataType window_size) :
            feature_scales_(feature_scales),
            window_size_(window_size)
        {
            // initialize the feature dimension map
            feature_sizes_["GaussianSmoothing"] = 1;
            feature_sizes_["LaplacianOfGaussian"] = 1;
            feature_sizes_["StructureTensorEigenvalues"] = N;
            feature_sizes_["HessianOfGaussianEigenvalues"] = N;
            feature_sizes_["GaussianGradientMagnitude"] = 1;
            feature_sizes_["DifferenceOfGaussians"] = 1;
            // set the filter window size
            conv_options_.filterWindowSize(window_size);
        }

        template<int N, typename DataType>
        FeatureCalculator<N, DataType>::FeatureCalculator(
                const StringDataPairVectorType& feature_scales,
                const vigra::TinyVector<DataType, N> image_scales,
                DataType window_size) :
            FeatureCalculator(feature_scales, window_size)
        {
            conv_options_.stepSize(image_scales);
        }

        template<int N, typename DataType>
        size_t FeatureCalculator<N, DataType>::get_feature_size(
                const std::string& feature_name
                ) const {
            std::map<std::string, size_t>::const_iterator it;
            it = feature_sizes_.find(feature_name);
            if (it == feature_sizes_.end()) {
                return 0;
            } else {
                return it->second;
            }
        }

        template<int N, typename DataType>
        size_t FeatureCalculator<N, DataType>::get_feature_size() const {
            size_t size = 0;
            for (
                 typename StringDataPairVectorType::const_iterator it = feature_scales_.begin();
                 it != feature_scales_.end();
                 it++
                 ) {
                size += get_feature_size(it->first);
            }
            return size;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate_gaussian_smoothing(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArrayView<N+1, DataType>& features,
                DataType feature_scale) const
        {
            vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
            vigra::gaussianSmoothMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(results),
                        feature_scale,
                        conv_options_);
            return 0;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate_laplacian_of_gaussian(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArrayView<N+1, DataType>& features,
                DataType feature_scale) const
        {
            vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
            vigra::laplacianOfGaussianMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(results),
                        feature_scale,
                        conv_options_);
            return 0;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate_gaussian_gradient_magnitude(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArrayView<N+1, DataType>& features,
                DataType feature_scale)
        {
            vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
            vigra::VectorNormFunctor<vigra::TinyVector<DataType, N> > norm;
        #ifndef USE_PARALLEL_FEATURES
            // reuse eigenvalue_temp
            if (eigenvalue_temp_.shape() != image.shape()) {
                eigenvalue_temp_.reshape(image.shape());
            }
            vigra::gaussianGradientMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(eigenvalue_temp_),
                        feature_scale,
                        conv_options_);
            vigra::transformMultiArray(
                        srcMultiArrayRange(eigenvalue_temp_),
                        destMultiArray(results),
                        norm);
        #else
            vigra::MultiArray<N, vigra::TinyVector<DataType, N> > temp(image.shape());
            vigra::gaussianGradientMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(temp),
                        feature_scale,
                        conv_options_);
            vigra::transformMultiArray(
                        srcMultiArrayRange(temp),
                        destMultiArray(results),
                        norm);
        #endif
            return 0;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate_difference_of_gaussians(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArrayView<N+1, DataType>& features,
                DataType feature_scale)
        {
            vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
            vigra::gaussianSmoothMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(results),
                        feature_scale,
                        conv_options_);
        #ifndef USE_PARALLEL_FEATURES
            if (feature_temp_.shape() != image.shape()) {
                feature_temp_.reshape(image.shape());
            }
            vigra::gaussianSmoothMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(feature_temp_),
                        feature_scale * 0.66,
                        conv_options_);
            results -= feature_temp_;
        #else
            vigra::MultiArray<N, DataType> temp(image.shape());
            vigra::gaussianSmoothMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(temp),
                        feature_scale * 0.66,
                        conv_options_);
            results -= temp;
        #endif
            return 0;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate_structure_tensor_eigenvalues(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArrayView<N+1, DataType>& features,
                DataType feature_scale)
        {
        #ifndef USE_PARALLEL_FEATURES
            if (tensor_temp_.shape() != image.shape()) {
                tensor_temp_.reshape(image.shape());
            }
            if (eigenvalue_temp_.shape() != image.shape()) {
                eigenvalue_temp_.reshape(image.shape());
            }
            vigra::structureTensorMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(tensor_temp_),
                        feature_scale,
                        feature_scale * 0.5,
                        conv_options_);
            vigra::tensorEigenvaluesMultiArray(
                        srcMultiArrayRange(tensor_temp_),
                        destMultiArray(eigenvalue_temp_));
            features = eigenvalue_temp_.expandElements(N);
        #else
            vigra::MultiArray<N, vigra::TinyVector<DataType, (N*(N+1))/2> > tensor(
                        image.shape());
            vigra::MultiArray<N, vigra::TinyVector<DataType, N> > eigenvalues(
                        image.shape());
            vigra::structureTensorMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(tensor),
                        feature_scale,
                        feature_scale * 0.5,
                        conv_options_);
            vigra::tensorEigenvaluesMultiArray(
                        srcMultiArrayRange(tensor),
                        destMultiArray(eigenvalues));
            features = eigenvalues.expandElements(N);
        #endif
            return 0;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate_hessian_of_gaussian_eigenvalues(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArrayView<N+1, DataType>& features,
                DataType feature_scale)
        {
        #ifndef USE_PARALLEL_FEATURES
            // using tensor as hessian here
            if (tensor_temp_.shape() != image.shape()) {
                tensor_temp_.reshape(image.shape());
            }
            if (eigenvalue_temp_.shape() != image.shape()) {
                eigenvalue_temp_.reshape(image.shape());
            }
            vigra::hessianOfGaussianMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(tensor_temp_),
                        feature_scale,
                        conv_options_);
            vigra::tensorEigenvaluesMultiArray(
                        srcMultiArrayRange(tensor_temp_),
                        destMultiArray(eigenvalue_temp_));
            features = eigenvalue_temp_.expandElements(N);
        #else
            vigra::MultiArray<N, vigra::TinyVector<DataType, (N*(N+1))/2> > hessian(
                        image.shape());
            vigra::MultiArray<N, vigra::TinyVector<DataType, N> > eigenvalues(
                        image.shape());
            vigra::hessianOfGaussianMultiArray(
                        srcMultiArrayRange(image),
                        destMultiArray(hessian),
                        feature_scale,
                        conv_options_);
            vigra::tensorEigenvaluesMultiArray(
                        srcMultiArrayRange(hessian),
                        destMultiArray(eigenvalues));
            features = eigenvalues.expandElements(N);
        #endif
            return 0;
        }

        template<int N, typename DataType>
        int FeatureCalculator<N, DataType>::calculate(
                const vigra::MultiArrayView<N, DataType>& image,
                vigra::MultiArray<N+1, DataType>& features)
        {
            std::cout << "\tcalculating " << get_feature_size() << " features" << std::endl;
            typedef typename vigra::MultiArrayShape<N+1>::type FeaturesShapeType;
            typedef typename vigra::MultiArrayShape<N>::type ImageShapeType;
            typedef typename vigra::MultiArrayView<N+1, DataType> FeaturesViewType;
            // initialize offset and size of the current features along the
            // feature vectors
            std::vector<size_t> offsets;
            FeaturesShapeType features_shape = append_to_shape<N>(
                        image.shape(),
                        get_feature_size());
            if (features.shape() != features_shape) {
                features.reshape(features_shape);
            }

            // store all offsets and keep scope of variable offset within the
            // for loop
            {
                size_t offset = 0;
                for (
                     typename StringDataPairVectorType::const_iterator it = feature_scales_.begin();
                     it != feature_scales_.end();
                     it++
                     ) {
                    offsets.push_back(offset);
                    offset += get_feature_size(it->first);
                }
            }

        #ifdef USE_PARALLEL_FEATURES
        #pragma omp parallel for
        #endif
            for(size_t i = 0; i < feature_scales_.size(); i++) {
                // get the offset and size of the current feature in the feature
                // arrays
                const size_t& offset = offsets[i];
                const size_t& size = get_feature_size(feature_scales_[i].first);
                // create the bounding box from box_min to box_max
                FeaturesShapeType box_min(0);
                box_min[N] = offset;
                FeaturesShapeType box_max = append_to_shape<N>(image.shape(), offset + size);
                // create a view to this bounding box
                FeaturesViewType features_view = features.subarray(box_min, box_max);
                // branch between the different features
                const std::string& feature_name = feature_scales_[i].first;
                const DataType& scale = feature_scales_[i].second;

                if (!feature_name.compare("GaussianSmoothing")) {
                    calculate_gaussian_smoothing(image, features_view, scale);
                } else if (!feature_name.compare("LaplacianOfGaussian")) {
                    calculate_laplacian_of_gaussian(image, features_view, scale);
                } else if (!feature_name.compare("GaussianGradientMagnitude")) {
                    calculate_gaussian_gradient_magnitude(image, features_view, scale);
                } else if (!feature_name.compare("DifferenceOfGaussians")) {
                    calculate_difference_of_gaussians(image, features_view, scale);
                } else if (!feature_name.compare("StructureTensorEigenvalues")) {
                    calculate_structure_tensor_eigenvalues(image, features_view, scale);
                } else if (!feature_name.compare("HessianOfGaussianEigenvalues")) {
                    calculate_hessian_of_gaussian_eigenvalues(image, features_view, scale);
                } else {
                    throw std::runtime_error("Invalid feature name used : " + feature_name);
                }
            }
            return 0;
        }
    }
}

#endif // _FEATURE_CALCULATOR_H_
