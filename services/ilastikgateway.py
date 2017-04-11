'''
The pixel classification service can take raw data and compute features and predict given a random forest
'''
import argparse
import numpy as np
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
from utils.servicehelper import returnDataInFormat, RedisCache, getBlockRawData, getBlocksInRoi, combineBlocksToVolume, collectPredictionBlocksForRoi, createRoi
from utils.queues import FinishedQueueSubscription, TaskQueuePublisher, FinishedBlockCollectorThread
from utils.voxels_nddata_codec import VoxelsNddataCodec

# logging
import logging
logger = logging.getLogger(__name__)

# flask setup
app = Flask("ilastikgateway")
doc = Autodoc(app)

# global variable storing the backend instance and the redis cache, etc.
blocking = None
cache = RedisCache()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
finishedQueueSubscription = FinishedQueueSubscription()
finishedQueueSubscription.start()
taskQueuePublisher = TaskQueuePublisher()
session = requests.Session() # to allow connection pooling

# --------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------

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
        return getBlockRawData(blockIdx, rawDtype, blocking, options.dataprovider_ip, session)

    blockData = list(executor.map(partialGetBlockRawData, blocksToProcess))
    data = combineBlocksToVolume(blocksToProcess, blockData, blocking, roi)

    return returnDataInFormat(data, format)


# --------------------------------------------------------------
@app.route('/prediction/<format>/roi')
@doc.doc()
def get_prediction_roi(format):
    '''
    Get a predicted roi in the specified format (raw / tiff / png / hdf5).
    The roi is specified by appending "?extents_min=x_y_z&extents_max=x_y_z" to requested the URL.
    '''

    # get blocks in ROI
    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    assert all(a < b for a,b in zip(start, stop)), "End point must be greater than start point"
    assert len(start) == 5, "Expected 5D start coordinate"
    assert len(stop) == 5, "Expected 5D stop coordinate"
    
    data = collectPredictionBlocksForRoi(blocking, start, stop, dim, cache, finishedQueueSubscription, taskQueuePublisher)

    return returnDataInFormat(data, format)

# --------------------------------------------------------------
@app.route('/prediction/info/numclasses')
@doc.doc()
def get_prediction_num_classes():
    ''' Return the number of classes predicted by the currently loaded random forest '''
    return str(numClasses)


# --------------------------------------------------------------
@app.route('/raw/info/dtype')
@doc.doc()
def info_dtype():
    ''' Return the datatype of the dataset as string '''
    return rawDtype

# --------------------------------------------------------------
@app.route('/raw/info/shape')
@doc.doc()
def info_shape():
    ''' Return the shape of the dataset as string with underscore delimiters '''
    return '_'.join(map(str, shape))

# --------------------------------------------------------------
@app.route('/raw/info/dim')
@doc.doc()
def info_dim():
    ''' Return the dimensionality of the data (2 or 3) '''
    return str(dim)

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
    parser.add_argument('--pixelclassification-ip', type=str, required=True, 
                        help='ip and port of pixelclassification service')
    parser.add_argument('--dataprovider-ip', type=str, required=True, 
                        help='ip and port of dataprovider')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Turn on verbose logging', default=False)

    options = parser.parse_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # allow 5 retries for requests to dataprovider and pixelclassification ip:
    session.mount('http://{}'.format(options.dataprovider_ip), HTTPAdapter(max_retries=5))
    session.mount('http://{}'.format(options.pixelclassification_ip), HTTPAdapter(max_retries=5))

    # read dataset config from data provider service
    r = session.get('http://{ip}/info/dtype'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query datatype from dataprovider at ip: {}".format(options.dataprovider_ip))
    rawDtype = r.text
    
    r = session.get('http://{ip}/info/dim'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(options.dataprovider_ip))
    dim = int(r.text)

    r = session.get('http://{ip}/info/shape'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query shape from dataprovider at ip: {}".format(options.dataprovider_ip))
    shape = list(map(int, r.text.split('_')))
    assert len(shape) == 5, "Expected 5D data, but got shape {}".format(shape)

    r = session.get('http://{ip}/prediction/numclasses'.format(ip=options.pixelclassification_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query num classes from pixel classification at ip: {}".format(options.pixelclassification_ip))
    numClasses = int(r.text)

    r = session.get('http://{ip}/prediction/blockshape'.format(ip=options.pixelclassification_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query blockshape from pixel classification at ip: {}".format(options.pixelclassification_ip))
    blockShape = list(map(int, r.text.split('_')))
    assert len(shape) == 5, "Expected 5D blocks, but got block shape {}".format(blockShape)
    blocking = pib.Blocking_5d([0]*5, shape, blockShape)

    print("Found dataset of size {} and dimensionality {}".format(shape, dim))
    print("Using block shape {}".format(blockShape))
    print("Dataset consists of {} blocks".format(blocking.numberOfBlocks))

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)

