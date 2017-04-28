'''
The pixel classification service can take raw data and compute features and predict given a random forest
'''
import logging
import json
import tempfile
import argparse
import sys
import numpy as np
import time
import atexit
import h5py
import vigra
from pprint import pprint
import requests
from requests.adapters import HTTPAdapter

from flask import Flask, send_file, request
from flask_autodoc import Autodoc

# C++ module containing all the important methods
import pyilastikbackend as pib
from utils.servicehelper import returnDataInFormat, RedisCache, getOwnPublicIp
from utils.voxels_nddata_codec import VoxelsNddataCodec
from utils.queues import TaskQueueSubscription, FinishedQueuePublisher
from utils.registry import Registry
from utils.redisloghandler import RedisLogHandler

# flask setup
app = Flask("pixelclassificationservice")
doc = Autodoc(app)

# global variable storing the backend instance and the redis client
logger = logging.getLogger(__name__)
registry = None
dataprovider_ip = None
pixelClassificationBackend = None
cache = None
dtype = None
dim = None
shape = None
blocking = None
blockShape = None
finishedQueuePublisher = None
taskQueueSubscription = None
session = requests.Session() # to allow connection pooling

# --------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------
def getBlockRawData(blockIdx):
    '''
    Get the raw data of a block
    '''
    assert pixelClassificationBackend is not None, "Configuration not ready!"
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"

    roi = pixelClassificationBackend.getRequiredRawRoiForFeatureComputationOfBlock(blockIdx)
    beginStr = '_'.join([str(s) for s in roi.begin])
    endStr = '_'.join([str(s) for s in roi.end])

    r = session.get('http://{ip}/raw/raw/roi?extents_min={b}&extents_max={e}'.format(ip=dataprovider_ip, b=beginStr, e=endStr), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get raw data of block {b} from {ip}".format(b=blockIdx, ip=dataprovider_ip))
    roi_shape = roi.shape
    codec = VoxelsNddataCodec(dtype)
    rawData = codec.decode_to_ndarray(r.raw, roi_shape)

    return rawData

def processBlock(blockIdx):
    '''
    Main computational method for processing blocks
    '''
    assert pixelClassificationBackend is not None, "Configuration not ready!"
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"

    cachedBlock, isDummy = cache.readBlock(blockIdx)
    if not isDummy and cachedBlock is not None:
        return cachedBlock
    
    rawData = getBlockRawData(blockIdx)
    t0 = time.time()
    logger.info("Input block {} min {} max {} dtype {} shape {}".format(blockIdx, rawData.min(), rawData.max(), rawData.dtype, rawData.shape))
    features = pixelClassificationBackend.computeFeaturesOfBlock(blockIdx, rawData)
    logger.info("Feature block min {} max {} dtype {} shape {}".format(features.min(), features.max(), features.dtype, features.shape))
    predictions = pixelClassificationBackend.computePredictionsOfBlock(features)
    t1 = time.time()
    logger.info("Prediction block min {} max {} dtype {} shape {} took {}".format(predictions.min(), predictions.max(), predictions.dtype, predictions.shape, t1 - t0))

    cache.saveBlock(blockIdx, predictions)
    
    return predictions

def createRoi(start, stop):
    ''' helper to create a 5D block '''
    return pib.Block_5d(np.asarray(start), np.asarray(stop))

def loadRandomForest():
    '''
    Get the binary blob from the registry and load it as random forest into the pixel classification backend
    '''
    rfBlob = registry.get(registry.PC_RANDOM_FOREST)
    _, fname = tempfile.mkstemp(suffix='.h5')
    with open(fname, 'wb') as rf:
        rf.write(rfBlob)
    pixelClassificationBackend.loadRandomForest(fname, 'PixelClassification/ClassifierForests/Forest', 4)

def loadSelectedFeatures():
    '''
    Load features from registry and pass them to pixel classification backend
    '''
    selectedFeatureScalePairs = json.loads(registry.get(registry.PC_FEATURES))
    pixelClassificationBackend.configureSelectedFeatures(selectedFeatureScalePairs)

def configure():
    global dataprovider_ip
    global dtype
    global dim
    global shape
    global blocking
    global blockShape
    global pixelClassificationBackend

    dataprovider_ip = registry.get(registry.DATA_PROVIDER_IP)
    # allow 5 retries for requests to dataprovider ip:
    session.mount('http://{}'.format(dataprovider_ip), HTTPAdapter(max_retries=5))

    blocksize = registry.get(registry.BLOCKSIZE)
    # read dataset config from data provider service
    r = session.get('http://{ip}/info/dtype'.format(ip=dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query datatype from dataprovider at ip: {}".format(dataprovider_ip))
    dtype = r.text
    
    r = session.get('http://{ip}/info/dim'.format(ip=dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(dataprovider_ip))
    dim = int(r.text)
    assert dim in [2,3], "Individual frames must have dimension 2 or 3!"

    r = session.get('http://{ip}/info/shape'.format(ip=dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query shape from dataprovider at ip: {}".format(dataprovider_ip))
    shape = list(map(int, r.text.split('_')))
    assert len(shape) == 5, "Data Provider is not serving 5D data, cannot work with anything else."

    
    logger.info("Found dataset of size {} and dimensionality {}".format(shape, dim))

    # set up blocking
    blockShape = [1]*5
    blockShape[1:4] = [int(b) for b in blocksize.split('_')]
    logger.info("Using block shape {}".format(blockShape))
    blocking = pib.Blocking_5d([0]*5, shape, blockShape)
    logger.info("Dataset consists of {} blocks".format(blocking.numberOfBlocks))

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
    elif dim == 3:
        if dtype == 'uint8':
            pixelClassificationBackend = pib.PixelClassification_3d_uint8()
        elif dtype == 'uint16':
            pixelClassificationBackend = pib.PixelClassification_3d_uint16()
        elif dtype == 'float32':
            pixelClassificationBackend = pib.PixelClassification_3d_float32()
        else:
            raise ValueError("Dataset has unsupported datatype {}".format(dtype))
    else:
        raise ValueError("Wrong data dimensionality, must be 2 or 3, got {}".format(dim))

    pixelClassificationBackend.configureDatasetSize(blocking)
    loadSelectedFeatures()
    loadRandomForest()

def isConfigured():
    if any((v is None for v in [dataprovider_ip, dtype, dim, shape, 
                                blocking, blockShape, pixelClassificationBackend])):
        return False
    return True

@atexit.register
def shutdown():
    logger.info("Deregistering PC worker from registry")
    registry.remove(registry.PIXEL_CLASSIFICATION_WORKER_IPS, '{}:{}'.format(getOwnPublicIp(), options.port))

# --------------------------------------------------------------
# entry point for incoming messages via the task queue
# --------------------------------------------------------------
def processBlockCallback(blockIdx):
    assert isConfigured(), "Service must be configure by calling /setup first!"
    logger.debug("processing block {} request from task queue".format(blockIdx))
    try:
        processBlock(blockIdx)
        finishedQueuePublisher.finished(str(blockIdx))
    except:
        logger.exception("could not process block!")

# --------------------------------------------------------------
# REST Api
# --------------------------------------------------------------
@app.route('/prediction/<format>/<int:blockIdx>')
@doc.doc()
def get_prediction(format, blockIdx):
    '''
    Get a predicted block in the specified format (raw / tiff / png / hdf5).
    '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    data = processBlock(blockIdx)
    return returnDataInFormat(data, format)

# --------------------------------------------------------------
@app.route('/prediction/numclasses')
@doc.doc()
def get_prediction_num_classes():
    ''' Return the number of classes predicted by the currently loaded random forest '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str(pixelClassificationBackend.numberOfClasses)

# --------------------------------------------------------------
@app.route('/prediction/blockshape')
@doc.doc()
def get_prediction_blockshape():
    ''' Return the blockshape used for the dataset '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return '_'.join([str(s) for s in pixelClassificationBackend.blocking.blockShape])

# --------------------------------------------------------------
@app.route('/prediction/cachedblockids')
@doc.doc()
def get_prediction_list_cached_blocks():
    ''' Return a list of blockIds which are already in cache '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str(', '.join([str(b) for b in cache.listCachedBlocks()]))

# --------------------------------------------------------------
@app.route('/setup')
@doc.doc()
def setup():
    ''' set up pixel classification worker based on the values stored in the registry '''
    configure()
    return "Done!"

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
    parser.add_argument('--registry-ip', type=str, required=True,
                        help='IP of the registry service, running at port 6380')
    parser.add_argument('--clear-cache', action='store_true', 
                        help='clear the cache from all currently contained blocks!')
    parser.add_argument('--num-workers', type=int, default=1, 
                        help='number of worker threads')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Turn on verbose logging', default=False)

    options = parser.parse_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logHandler = RedisLogHandler(options.registry_ip, 6380, Registry.LOG, ip='{}@{}:{}'.format(app.name, getOwnPublicIp(), options.port))
    logHandler.setLevel(level=logging.DEBUG)
    logging.getLogger().addHandler(logHandler)

    # set up registry connection and query values
    registry = Registry(options.registry_ip)
    cache_ip = registry.get(registry.CACHE_IP)
    finishedQueuePublisher = FinishedQueuePublisher(host=options.registry_ip)

    cache = RedisCache(cache_ip)
    if options.clear_cache:
        # get rid of previously stored blocks
        cache.clear()

    # register this service in the registry
    registry.set(registry.PIXEL_CLASSIFICATION_WORKER_IPS, '{}:{}'.format(getOwnPublicIp(), options.port))

    logger.info("Starting {} worker threads".format(options.num_workers))
    for i in range(options.num_workers):
        taskQueueSubscription = TaskQueueSubscription(processBlockCallback, host=options.registry_ip)
        taskQueueSubscription.start()

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)#, processes=4)

