import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import fastfilters as ff
    WITH_FAST_FILTERS=True
    logger.info("Using fast filters")
except ImportError:
    WITH_FAST_FILTERS=False
    logger.warning("Not using fast filters")

class FeatureComputer(object):
    def __init__(self, featureWithSigmaList):
        self.featuresWithSigmas = featureWithSigmaList
        self.windowSize = 3.5

        # something like:
        # [["GaussianSmoothing", 1.0], ["GaussianSmoothing", 3.5], ["GaussianSmoothing", 10.0], ["LaplacianOfGaussian", 0.7], ["LaplacianOfGaussian", 1.6], ["LaplacianOfGaussian", 5.0], ["GaussianGradientMagnitude", 0.7], ["GaussianGradientMagnitude", 1.6], ["GaussianGradientMagnitude", 5.0], ["DifferenceOfGaussians", 0.7], ["DifferenceOfGaussians", 1.6], ["DifferenceOfGaussians", 5.0], ["StructureTensorEigenvalues", 1.0], ["StructureTensorEigenvalues", 3.5], ["StructureTensorEigenvalues", 10.0], ["HessianOfGaussianEigenvalues", 1.0], ["HessianOfGaussianEigenvalues", 3.5], ["HessianOfGaussianEigenvalues", 10.0]]

    def compute(self, volume):
        assert len(volume.shape) in [2, 3], "Can only compute features for 2 or 3 dimensional arrays!"
        windowSize = self.windowSize
        featureMaps = []
        for feature, sigma in self.featuresWithSigmas:
            if feature == "GaussianSmoothing":
                res = ff.gaussianSmoothing(volume, sigma, window_size=windowSize)
                res = np.expand_dims(res, axis=-1)
            elif feature == "LaplacianOfGaussian":
                res = ff.laplacianOfGaussian(volume, sigma, window_size=windowSize)
                res = np.expand_dims(res, axis=-1)
            elif feature == "GaussianGradientMagnitude":
                res = ff.gaussianGradientMagnitude(volume, sigma, window_size=windowSize)
                res = np.expand_dims(res, axis=-1)
            elif feature == "DifferenceOfGaussians":
                res = ff.gaussianSmoothing(volume, sigma, window_size=windowSize) - ff.gaussianSmoothing(volume, 0.66 * sigma, window_size=windowSize)
                res = np.expand_dims(res, axis=-1)
            elif feature == "StructureTensorEigenvalues":
                res = ff.structureTensorEigenvalues(volume, sigma * 0.5, sigma, window_size=windowSize)
            elif feature == "HessianOfGaussianEigenvalues":
                res = ff.hessianOfGaussianEigenvalues(volume, sigma, window_size=windowSize)
            else:
                raise ValueError("Invalid feature selected")
            featureMaps.append(res)

        return np.concatenate(featureMaps, axis=-1)

    def computeAndCrop(self, volume, blockWithHalo):
        squeezedVolume = volume.squeeze()
        dim = len(squeezedVolume.shape)
        # print("Got volume of shape {} and dim {}".format(squeezedVolume.shape, dim))
        # compute features of full block
        stackedFeatureMaps = self.compute(squeezedVolume)
        # print("Yielded features of shape {}".format(stackedFeatureMaps.shape))
        # make it 5D again
        stackedFeatureMaps = np.expand_dims(stackedFeatureMaps, axis=0)
        if dim == 2:
            stackedFeatureMaps = np.expand_dims(stackedFeatureMaps, axis=-2)

        # print("Transformed to shape: ", stackedFeatureMaps.shape)
        # cut away the halo
        start = blockWithHalo.innerBlockLocal.begin
        stop = blockWithHalo.innerBlockLocal.end
        # print("Returning crop from {} to {} and shape {}".format(start, stop, blockWithHalo.innerBlockLocal.shape))
        return stackedFeatureMaps[:,start[1]:stop[1], start[2]:stop[2], start[3]:stop[3], :]
