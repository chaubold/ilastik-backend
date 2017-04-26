'''
The thresholding service takes prediction maps, presmoothes them if needed, and returns a segmentation.
Works with full frames only!
'''
import logging
import json
import tempfile
import argparse
import sys
import numpy as np
import h5py
import atexit
import vigra
from pprint import pprint
import requests
from requests.adapters import HTTPAdapter

from flask import Flask, send_file, request
from flask_autodoc import Autodoc

import pyilastikbackend as pib
from utils.servicehelper import returnDataInFormat, RedisCache, collectPredictionBlocksForRoi, getOwnPublicIp
from utils.voxels_nddata_codec import VoxelsNddataCodec
from utils.queues import FinishedQueueSubscription, TaskQueuePublisher, FinishedBlockCollectorThread
from utils.registry import Registry

# flask setup
app = Flask(__name__)
doc = Autodoc(app)
logger = logging.getLogger(__name__)

# global variables
cache = None
session = requests.Session() # to allow connection pooling
finishedQueueSubscription = None
taskQueuePublisher = None
threshold = None
channel = None
presmoothingSigmas = None
blockShape = None
blocking = None
rawDtype = None
dim = None
shape = None
numClasses = None

# --------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------

# --------------------------------------------------------------
# entry point for incoming messages via the task queue
# --------------------------------------------------------------
def processFrame(volume):
    assert channel is not None, "Channel must be specified before Thresholding can run!"
    assert threshold is not None, "threshold value must be specified before Thresholding can run!"
    assert presmoothingSigmas is not None, "presmoothingSigmas must be specified before Thresholding can run!"
    print("Got volume of shape {} and dtype {}".format(volume[..., channel].squeeze().shape, volume.dtype))
    print("Using presmoothing sigmas {}".format(presmoothingSigmas))
    smoothedChannel = vigra.filters.gaussianSmoothing(volume[..., channel].squeeze(), presmoothingSigmas)
    print("Smoothed channel has shape {} and dtype {}".format(smoothedChannel.shape, smoothedChannel.dtype))
    mask = smoothedChannel > threshold
    print("Mask has shape {} and dtype {}".format(mask.shape, mask.dtype))
    if len(mask.shape) == 2:
        labelImage = vigra.analysis.labelImageWithBackground(mask.astype(np.uint32))
    else:
        labelImage = vigra.analysis.labelVolumeWithBackground(mask.astype(np.uint32))

    # add back channel and time axes
    labelImage = np.expand_dims(labelImage, axis=0)
    labelImage = np.expand_dims(labelImage, axis=-1)
    
    return labelImage

def processFrameCallback(frame):
    logger.debug("processing block {} request from task queue".format(frame))
    # retrieve frame from cache?
    volume = np.zeros((10,10,10)) # dummy
    labelImage = processFrame(frame)
    # save result to cache
    # finishedQueuePublisher.finished(str(frame))

def configure():
    global channel
    global threshold
    global presmoothingSigmas
    global rawDtype
    global dim
    global shape
    global numClasses
    global blockShape
    global blocking

    dataprovider_ip = registry.get(registry.DATA_PROVIDER_IP)
    pixelclassification_ip = registry.get(registry.PIXEL_CLASSIFICATION_WORKER_IPS)[0]
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


    channel = int(registry.get(registry.THRESHOLD_CHANNEL))
    threshold = float(registry.get(registry.THRESHOLD_VALUE))
    try:
        presmoothingSigmas = [int(s) for s in registry.get(registry.THREDHOLD_SIGMAS).split('_')]
    except:
        presmoothingSigmas = []
    
    if len(presmoothingSigmas) == 0:
        presmoothingSigmas = [1.0] * dim
    else:
        assert len(presmoothingSigmas) == dim, "Number of smoothing sigmas must match the dimensionality of the data"

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
    if any((v is None for v in [channel, threshold, presmoothingSigmas, rawDtype, dim, 
                                shape, numClasses, blockShape, blocking])):
        return False
    return True

@atexit.register
def shutdown():
    logger.info("Deregistering Thresholding worker from registry")
    registry.set(registry.THRESHOLDING_IP, '')


# --------------------------------------------------------------
# REST Api
# --------------------------------------------------------------
@app.route('/labelvolume/<format>/<int:frame>', methods=['GET'])
@doc.doc()
def get_label_volume(format, frame):
    ''' Process the selected frame and return the label volume in the specified format '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    start = [frame, 0, 0, 0, 0]
    stop = list(shape)
    stop[0] = frame + 1

    volume = collectPredictionBlocksForRoi(blocking, start, stop, dim, cache, finishedQueueSubscription, taskQueuePublisher)
    labelImage = processFrame(volume)
    return returnDataInFormat(labelImage, format)

# --------------------------------------------------------------
@app.route('/threshold/get', methods=['GET'])
@doc.doc()
def get_threshold():
    ''' Return the threshold used to define foreground pixels from probabilities '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str(threshold)

@app.route('/threshold/set', methods=['POST'])
@doc.doc()
def set_threshold():
    ''' Set the threshold in a json object like {"value": 0.5} '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    message = request.get_json(force=True)

    try:
        global threshold
        threshold = message['value']
    except KeyError as e:
        return '{}: {}\n'.format(type(e), str(e)), 400 # return "Bad Request" because it was malformed
    return "Threshold set successfully"

# --------------------------------------------------------------
@app.route('/channel/get', methods=['GET'])
@doc.doc()
def get_channel():
    ''' Return the channel used to define foreground pixels from probabilities '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str(channel)

@app.route('/channel/set', methods=['POST'])
@doc.doc()
def set_channel():
    ''' Set the channel in a json object like {"value": 1} '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    message = request.get_json(force=True)

    try:
        global channel
        channel = message['value']
    except KeyError as e:
        return '{}: {}\n'.format(type(e), str(e)), 400 # return "Bad Request" because it was malformed
    return "Channel specified successfully"

# --------------------------------------------------------------
@app.route('/presmoothingsigmas/get', methods=['GET'])
@doc.doc()
def get_presmoothingsigmas():
    ''' Return the presmoothingsigmas that are applied before thresholding. May be "None" if no presmoothing is desired '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    return str('_'.join(presmoothingSigmas))

@app.route('/presmoothingsigmas/set', methods=['POST'])
@doc.doc()
def set_presmoothingsigmas():
    ''' 
    Set the presmoothingsigmas in a json object like {"value": [0.5, 1.0, 1.0] }. 
    May be {"value": null} if no presmoothing is desired.
    '''
    assert isConfigured(), "Service must be configure by calling /setup first!"
    message = request.get_json(force=True)

    try:
        global presmoothingSigmas
        presmoothingSigmas = message['value']
    except KeyError as e:
        return '{}: {}\n'.format(type(e), str(e)), 400 # return "Bad Request" because it was malformed
    return "Presmoothing sigmas set successfully"

# --------------------------------------------------------------
@app.route('/setup')
@doc.doc()
def setup():
    ''' set up thresholding service based on the values stored in the registry '''
    configure()
    return "Done!"

# --------------------------------------------------------------
@app.route('/doc')
def documentation():
    ''' serve an API documentation '''
    return doc.html(title='Thresholding API', author='Carsten Haubold')

# ----------------------------------------------------------------------------------------
# run server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a thresholding service',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=8889, help='port on which to run service')
    parser.add_argument('--registry-ip', type=str, required=True,
                        help='IP of the registry service, running at port 6380')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Turn on verbose logging', default=False)

    options = parser.parse_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # set up registry connection and query values
    registry = Registry(options.registry_ip)
    thresholding_ip = registry.get(registry.THRESHOLDING_IP)
    message_broker_ip = registry.get(registry.MESSAGE_BROKER_IP)
    cache_ip = registry.get(registry.CACHE_IP)
    cache = RedisCache(cache_ip)

    finishedQueueSubscription = FinishedQueueSubscription(host=message_broker_ip)
    finishedQueueSubscription.start()
    taskQueuePublisher = TaskQueuePublisher(host=message_broker_ip)

    # register this service in the registry
    registry.set(registry.THRESHOLDING_IP, '{}:{}'.format(getOwnPublicIp(), options.port))

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)#, processes=4)

