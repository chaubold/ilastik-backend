import logging
import json
import tempfile
import argparse
import sys
import numpy as np
import h5py
import vigra
from pprint import pprint
import requests

from utils.registry import Registry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure a ilastik as a service for pixel classification and thresholding',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--registry-ip', type=str, required=True,
                        help='IP of the registry service, running at port 6380')
    parser.add_argument('--dataprovider-ip', type=str, required=True,
                        help='IP:port of the dataprovider to use')
    parser.add_argument('--cache-ip', type=str, required=True,
                        help='IP:port of the caching redis server')
    parser.add_argument('--clear-logs', action='store_true', help='Clears the log field in the registry')

    parser.add_argument('--project', type=str, required=True, 
                        help='ilastik project with trained random forest')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('-c', '--channel', type=int, default=0, help='channel')
    parser.add_argument('-s', '--sigmas', type=float, action='append', default=None, 
                        help='smoothing sigmas, defaults to 1 in each spatial dimension')
    parser.add_argument('--blocksize', type=int, default=64, 
                        help='size of blocks in all 2 or 3 dimensions, used to blockify all processing')
    options = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    registry = Registry(options.registry_ip)

    # read configuration from project file
    with h5py.File(options.project, 'r') as ilp:
        # extract selected features from ilastik project file
        scales = ilp['FeatureSelections/Scales']
        featureNames = ilp['FeatureSelections/FeatureIds']
        selectionMatrix = ilp['FeatureSelections/SelectionMatrix'].value
        numPixelClassificationLabels = len(ilp['PixelClassification/LabelNames'].value)

        assert(selectionMatrix.shape[0] == featureNames.shape[0])
        assert(selectionMatrix.shape[1] == scales.shape[0])

        selectedFeatureScalePairs = []

        for f in range(featureNames.shape[0]):
            for s in range(scales.shape[0]):
                if selectionMatrix[f][s]:
                    selectedFeatureScalePairs.append((featureNames[f].decode(), scales[s]))

        # extract random forest and read it as a binary blob
        _, fname = tempfile.mkstemp(suffix='.h5')
        with h5py.File(fname, 'w') as rf:
            out_rf = rf.create_group('PixelClassification')
            ilp.copy('PixelClassification/ClassifierForests', out_rf)
        with open(fname, 'rb') as rf:
            rfBlob = rf.read()

    print("Found selected features:")
    pprint(selectedFeatureScalePairs)
    selectedFeatureScalePairs = json.dumps(selectedFeatureScalePairs)
    print("Sending them to {}".format(options.registry_ip))
    registry.set(registry.DATA_PROVIDER_IP, options.dataprovider_ip)
    registry.set(registry.CACHE_IP, options.cache_ip)

    # ilastik workflow configuration
    registry.set(registry.PC_FEATURES, selectedFeatureScalePairs)
    registry.set(registry.PC_RANDOM_FOREST, rfBlob)
    registry.set(registry.THRESHOLD_CHANNEL, options.channel)
    registry.set(registry.THRESHOLD_VALUE, options.threshold)
    
    if options.sigmas is not None:
        registry.set(registry.THRESHOLD_SIGMAS, options.sigmas)

    r = requests.get('http://{ip}/info/dim'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(options.dataprovider_ip))
    dim = int(r.text)
    assert dim in [2,3], "Individual frames must have dimension 2 or 3!"
    blockShape = [options.blocksize] * dim
    if dim == 2:
        blockShape.append(1)
    print("Using block shape {}".format(blockShape))
    registry.set(registry.BLOCKSIZE, '_'.join([str(b) for b in blockShape]))

    print("Done, checking contents:")
    registry.printContents()