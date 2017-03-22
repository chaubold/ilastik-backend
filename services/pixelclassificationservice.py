'''
The pixel classification service can take raw data and compute features and predict given a random forest
'''

import json
import tempfile
import argparse
import sys
import traceback
import numpy as np
import h5py
import vigra
from pprint import pprint
import redis
import requests

from flask import Flask, send_file, request
from flask_autodoc import Autodoc

# C++ module containing all the important methods
import pyilastikbackend as pib
from utils.servicehelper import returnDataInFormat
from utils.voxels_nddata_codec import VoxelsNddataCodec

# flask setup
app = Flask("pixelclassificationservice")
doc = Autodoc(app)

# global variable storing the backend instance and the redis client
pixelClassificationBackend = None
redisClient = None

# --------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------
def getBlockRawData(blockIdx):
    '''
    Get the raw data of a block
    '''
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"

    roi = pixelClassificationBackend.getRequiredRawRoiForFeatureComputationOfBlock(blockIdx)
    beginStr = '_'.join(map(str,roi.begin))
    endStr = '_'.join(map(str, roi.end))

    r = requests.get('http://{ip}/raw/raw/roi?extents_min={b}&extents_max={e}'.format(ip=options.dataprovider_ip, b=beginStr, e=endStr), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get raw data of block {b} from {ip}".format(b=blockIdx, ip=options.dataprovider_ip))
    shape = roi.shape
    codec = VoxelsNddataCodec(dtype)
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def readBlockFromCache(blockIdx):
    ''' if a redis client was initialized, try to find the block and its shape info and return the contents as numpy array '''
    if redisClient is not None:
        # read from cache
        cachedBlock = redisClient.get('prediction-{}-block'.format(blockIdx))
        cachedShape = redisClient.get('prediction-{}-shape'.format(blockIdx))
        if cachedBlock and cachedShape:
            try:
                cachedShape = cachedShape.decode().split('_')
                shape, dtype = list(map(int, cachedShape[:-1])), cachedShape[-1]
                print("Found block {} of shape {} and dtype {} in cache!".format(blockIdx, shape, dtype))
                return np.fromstring(cachedBlock, dtype=dtype).reshape(shape)
            except:
                print("ERROR when retrieving block from cache:")
                traceback.print_exc(file=sys.stdout)
    return None

def saveBlockToChache(blockIdx, blockData):
    ''' if a redis client was initialized, store the block there '''
    if redisClient is not None:
        # save to cache
        redisClient.set('prediction-{}-block'.format(blockIdx), blockData.tostring())
        shapeStr = '_'.join([str(d) for d in blockData.shape] + [str(blockData.dtype)])
        print(shapeStr)
        redisClient.set('prediction-{}-shape'.format(blockIdx), shapeStr)

def clearCache():
    ''' Remove all prediction blocks and their shapes from the cache '''
    if redisClient is not None:
        for k in redisClient.scan_iter(match='prediction-*'):
            redisClient.delete(k)

def processBlock(blockIdx):
    '''
    Main computational method for processing blocks
    '''
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"
    rawData = getBlockRawData(blockIdx)

    cachedBlock = readBlockFromCache(blockIdx)
    if cachedBlock is not None:
        return cachedBlock
    
    print("Input block min {} max {} dtype {} shape {}".format(rawData.min(), rawData.max(), rawData.dtype, rawData.shape))
    features = pixelClassificationBackend.computeFeaturesOfBlock(blockIdx, rawData)
    print("Feature block min {} max {} dtype {} shape {}".format(features.min(), features.max(), features.dtype, features.shape))
    predictions = pixelClassificationBackend.computePredictionsOfBlock(blockIdx, features)
    print("Prediction block min {} max {} dtype {} shape {}".format(predictions.min(), predictions.max(), predictions.dtype, predictions.shape))

    saveBlockToChache(blockIdx, predictions)
    
    return predictions

def createRoi(start, stop):
    ''' helper to create a 2D or 3D block '''
    if dim == 2:
        return pib.Block_2d(np.array(start), np.array(stop))
    elif dim == 3:
        return pib.Block_3d(np.array(start), np.array(stop))

# --------------------------------------------------------------
@app.route('/prediction/<format>/<int:blockIdx>')
@doc.doc()
def get_prediction(format, blockIdx):
    '''
    Get a predicted block in the specified format (raw / tiff / png / hdf5).
    '''
    data = processBlock(blockIdx)
    return returnDataInFormat(data, format)

# --------------------------------------------------------------
@app.route('/prediction/numclasses')
@doc.doc()
def get_prediction_num_classes():
    ''' Return the number of classes predicted by the currently loaded random forest '''
    return str(pixelClassificationBackend.numberOfClasses)

# --------------------------------------------------------------
@app.route('/doc')
def documentation():
    ''' serve an API documentation '''
    return doc.html(title='Pixel Classification API', author='Carsten Haubold')

# ----------------------------------------------------------------------------------------
# run server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a pixel classification service',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=8888, help='port on which to run service')
    # parser.add_argument('-d', '--dim', type=int, default=2, help='Whether the service should deal with 2D or 3D data.')
    # parser.add_argument('-t', '--dtype', type=str, default='uint8',
    #                     help='Which format the input data has, one of [uint8, uint16, float].')
    parser.add_argument('--project', type=str, required=True, 
                        help='ilastik project with trained random forest')
    parser.add_argument('--dataprovider-ip', type=str, required=True, 
                        help='ip and port of dataprovider')
    parser.add_argument('--blocksize', type=int, default=64, 
                        help='size of blocks in all 2 or 3 dimensions, used to blockify all processing')
    parser.add_argument('--use-caching', action='store_true', 
                        help='use caching for features and predictions, assumes a redis server to be running!')

    options = parser.parse_args()

    if options.use_caching:
        redisClient = redis.StrictRedis()
        # get rid of previously stored blocks
        # clearCache()

    # read dataset config from data provider service
    r = requests.get('http://{ip}/info/dtype'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query datatype from dataprovider at ip: {}".format(options.dataprovider_ip))
    dtype = r.text
    
    r = requests.get('http://{ip}/info/dim'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(options.dataprovider_ip))
    dim = int(r.text)
    blockShape = [options.blocksize] * dim

    r = requests.get('http://{ip}/info/shape'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query shape from dataprovider at ip: {}".format(options.dataprovider_ip))
    shape = list(map(int, r.text.split('_')))

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

    print("Found selected features:")
    pprint(selectedFeatureScalePairs)
    print("Found dataset of size {} and dimensionality {}".format(shape, dim))
    print("Using block shape {}".format(blockShape))

    # configure pixelClassificationBackent
    # TODO: write a factory method for the constructor!
    if dim == 2:
        if dtype == 'uint8':
            pixelClassificationBackend = pib.PixelClassification_2d_uint8()
        elif dtype == 'uint16':
            pixelClassificationBackend = pib.PixelClassification_2d_uint16()
        elif dtype == 'float32':
            pixelClassificationBackend = pib.PixelClassification_2d_float32()
        else:
            raise ValueError("Dataset has unsupported datatype {}".format(dtype))

        blocking = pib.Blocking_2d([0,0], shape, blockShape)
    elif dim == 3:
        if dtype == 'uint8':
            pixelClassificationBackend = pib.PixelClassification_3d_uint8()
        elif dtype == 'uint16':
            pixelClassificationBackend = pib.PixelClassification_3d_uint16()
        elif dtype == 'float32':
            pixelClassificationBackend = pib.PixelClassification_3d_float32()
        else:
            raise ValueError("Dataset has unsupported datatype {}".format(dtype))

        blocking = pib.Blocking_3d([0,0,0], shape, blockShape)
    else:
        raise ValueError("Wrong data dimensionality, must be 2 or 3, got {}".format(dim))

    print("Dataset consists of {} blocks".format(blocking.numberOfBlocks))

    pixelClassificationBackend.configureDatasetSize(blocking)
    pixelClassificationBackend.configureSelectedFeatures(selectedFeatureScalePairs)
    pixelClassificationBackend.loadRandomForest(options.project, 'PixelClassification/ClassifierForests/Forest', 4)

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)

