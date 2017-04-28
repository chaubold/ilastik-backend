'''
The pixel classification service can take raw data and compute features and predict given a random forest
'''
import argparse
import numpy as np
import atexit
import time
import threading
from pprint import pprint
import requests
from requests.adapters import HTTPAdapter
import concurrent.futures

from flask import Flask, request
from flask_autodoc import Autodoc

# C++ module containing all the important methods
import pyilastikbackend as pib
from utils.servicehelper import returnDataInFormat, RedisCache, getBlockRawData, getBlocksInRoi, combineBlocksToVolume, collectPredictionBlocksForRoi, createRoi, getOwnPublicIp
from utils.queues import FinishedQueueSubscription, TaskQueuePublisher, FinishedBlockCollectorThread
from utils.voxels_nddata_codec import VoxelsNddataCodec
from utils.registry import Registry
from utils.redisloghandler import RedisLogHandler

# logging
import logging
logger = logging.getLogger(__name__)

# flask setup
app = Flask("ilastikgateway")
doc = Autodoc(app)

# global variable storing the backend instance and the redis cache, etc.
registry = None
blocking = None
cache = None
rawDtype = None
dim = None
shape = None
numClasses = None
blocking = None
blockShape = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
finishedQueueSubscription = None
taskQueuePublisher = None
session = requests.Session() # to allow connection pooling
thresholding_ip = None
dataprovider_ip = None
pixelclassification_ip = None

# --------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------

def configure():
    global rawDtype
    global dim
    global shape
    global numClasses
    global blockShape
    global blocking
    global thresholding_ip
    global dataprovider_ip
    global pixelclassification_ip

    thresholding_ip = registry.get(registry.THRESHOLDING_IP)
    dataprovider_ip = registry.get(registry.DATA_PROVIDER_IP)
    pixelclassification_ips = registry.get(registry.PIXEL_CLASSIFICATION_WORKER_IPS)

    assert len(pixelclassification_ips) > 0, "No pixel classification service available, cannot process any data!"
    assert thresholding_ip is not None and thresholding_ip != '', "No thresholding service available, cannot process any data!"
    pixelclassification_ip = pixelclassification_ips[0]

    # call setup of the PC workers and thresholding
    for ip in pixelclassification_ips + [thresholding_ip]:
        print("Setting up service at ip:", ip)
        r = session.get('http://{ip}/setup'.format(ip=ip))
        if r.status_code != 200:
            raise RuntimeError("Could not setup service at ip: {}".format(ip))

    # allow 5 retries for requests to dataprovider and pixelclassification ip:
    session.mount('http://{}'.format(dataprovider_ip), HTTPAdapter(max_retries=5))
    session.mount('http://{}'.format(pixelclassification_ip), HTTPAdapter(max_retries=5))

    # read dataset config from data provider service
    r = session.get('http://{ip}/info/dtype'.format(ip=dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query datatype from dataprovider at ip: {}".format(dataprovider_ip))
    rawDtype = r.text
    
    r = session.get('http://{ip}/info/dim'.format(ip=dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(dataprovider_ip))
    dim = int(r.text)

    r = session.get('http://{ip}/info/shape'.format(ip=dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query shape from dataprovider at ip: {}".format(dataprovider_ip))
    shape = list(map(int, r.text.split('_')))
    assert len(shape) == 5, "Expected 5D data, but got shape {}".format(shape)

    r = session.get('http://{ip}/prediction/numclasses'.format(ip=pixelclassification_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query num classes from pixel classification at ip: {}".format(pixelclassification_ip))
    numClasses = int(r.text)

    r = session.get('http://{ip}/prediction/blockshape'.format(ip=pixelclassification_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query blockshape from pixel classification at ip: {}".format(pixelclassification_ip))
    blockShape = list(map(int, r.text.split('_')))
    assert len(shape) == 5, "Expected 5D blocks, but got block shape {}".format(blockShape)
    blocking = pib.Blocking_5d([0]*5, shape, blockShape)

    print("Found dataset of size {} and dimensionality {}".format(shape, dim))
    print("Using block shape {}".format(blockShape))
    print("Dataset consists of {} blocks".format(blocking.numberOfBlocks))

def isConfigured():
    if any((v is None for v in [rawDtype, dim, shape, numClasses, blockShape, blocking, 
                                thresholding_ip, dataprovider_ip, pixelclassification_ip])):
        return False
    return True

@atexit.register
def shutdown():
    logger.info("Deregistering Gateway from registry")
    registry.set(registry.GATEWAY_IP, '')

# --------------------------------------------------------------
# REST Api
# --------------------------------------------------------------
@app.route('/raw/<format>/roi')
@doc.doc()
def get_raw_roi(format):
    '''
    Get the raw data of a roi in the specified format (raw / tiff / png /hdf5 ).
    The roi is specified by appending "?extents_min=t_x_y_z_c&extents_max=t_x_y_z_c" to requested the URL.
    '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    assert all(a < b for a,b in zip(start, stop)), "End point must be greater than start point"
    assert len(start) == 5, "Expected 5D start coordinate"
    assert len(stop) == 5, "Expected 5D stop coordinate"
    roi = createRoi(start, stop)

    blocksToProcess = getBlocksInRoi(blocking, start, stop, dim)

    # Serial version:
    # blockData = [getBlockRawData(b) for b in blocksToProcess]

    # Using ThreadPoolExecutor:
    def partialGetBlockRawData(blockIdx):
        return getBlockRawData(blockIdx, rawDtype, blocking, dataprovider_ip, session)

    blockData = list(executor.map(partialGetBlockRawData, blocksToProcess))
    data = combineBlocksToVolume(blocksToProcess, blockData, blocking, roi)

    return returnDataInFormat(data, format)


# --------------------------------------------------------------
@app.route('/prediction/<format>/roi')
@doc.doc()
def get_prediction_roi(format):
    '''
    Get a predicted roi in the specified format (raw / tiff / png / hdf5).
    The roi is specified by appending "?extents_min=t_x_y_z_c&extents_max=t_x_y_z_c" to requested the URL.
    '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    # get blocks in ROI
    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    assert all(a < b for a,b in zip(start, stop)), "End point must be greater than start point"
    assert len(start) == 5, "Expected 5D start coordinate"
    assert len(stop) == 5, "Expected 5D stop coordinate"
    
    data = collectPredictionBlocksForRoi(blocking, start, stop, dim, cache, finishedQueueSubscription, taskQueuePublisher)

    return returnDataInFormat(data, format)

# --------------------------------------------------------------
@app.route('/labelimage/<format>/roi')
@doc.doc()
def get_labelimage_roi(format):
    '''
    Get the thresholded labelimage of exactly one time frame in the specified format (raw / tiff / png / hdf5).
    The roi must be specified by appending "?extents_min=t_x_y_z_c&extents_max=t_x_y_z_c" to requested the URL.
    '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    # get blocks in ROI
    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    assert all(a < b for a,b in zip(start, stop)), "End point must be greater than start point"
    assert len(start) == 5, "Expected 5D start coordinate"
    assert len(stop) == 5, "Expected 5D stop coordinate"
    assert stop[0] - start[0] == 1, "Can only serve single time frames"

    frame = start[0]
    r = session.get('http://{ip}/labelvolume/raw/{frame}'.format(ip=thresholding_ip, frame=frame), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get labelimage of frame {f} from {ip}".format(f=frame, ip=thresholding_ip))

    codec = VoxelsNddataCodec(np.uint32)
    frame_shape = [1] + shape[1:4] + [1]
    labelImage = codec.decode_to_ndarray(r.raw, frame_shape)

    return returnDataInFormat(labelImage[:, start[1]:stop[1], start[2]:stop[2], start[3]:stop[3], :], format)

# --------------------------------------------------------------
@app.route('/prediction/info/numclasses')
@doc.doc()
def get_prediction_num_classes():
    ''' Return the number of classes predicted by the currently loaded random forest '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str(numClasses)

# --------------------------------------------------------------
@app.route('/raw/info/dtype')
@doc.doc()
def info_dtype():
    ''' Return the datatype of the dataset as string '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return rawDtype

# --------------------------------------------------------------
@app.route('/raw/info/shape')
@doc.doc()
def info_shape():
    ''' Return the shape of the dataset as string with underscore delimiters '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return '_'.join(map(str, shape))

# --------------------------------------------------------------
@app.route('/raw/info/dim')
@doc.doc()
def info_dim():
    ''' Return the dimensionality of the data (2 or 3) '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str(dim)

# --------------------------------------------------------------
@app.route('/setup')
@doc.doc()
def setup():
    ''' set up gateway based on the values stored in the registry '''
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
    parser = argparse.ArgumentParser(description='Run an ilastik gateway',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=8080, help='port on which to run service')
    parser.add_argument('--registry-ip', type=str, required=True,
                        help='IP of the registry service, running at port 6380')
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
    cache = RedisCache(cache_ip)

    finishedQueueSubscription = FinishedQueueSubscription(host=options.registry_ip)
    finishedQueueSubscription.start()
    taskQueuePublisher = TaskQueuePublisher(host=options.registry_ip)

    # register the gateway in the registry
    registry.set(registry.GATEWAY_IP, '{}:{}'.format(getOwnPublicIp(), options.port))

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)

