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
from utils.servicehelper import returnDataInFormat, RedisCache
from utils.queues import FinishedQueueSubscription, TaskQueuePublisher
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
def getBlockRawData(blockIdx):
    '''
    Get the raw data of a block
    '''
    assert 0 <= blockIdx < blocking.numberOfBlocks, "Invalid blockIdx selected"

    roi = blocking.getBlock(blockIdx)
    beginStr = '_'.join(map(str,roi.begin))
    endStr = '_'.join(map(str, roi.end))

    r = session.get('http://{ip}/raw/raw/roi?extents_min={b}&extents_max={e}'.format(ip=options.dataprovider_ip, b=beginStr, e=endStr), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get raw data of block {b} from {ip}".format(b=blockIdx, ip=options.dataprovider_ip))
    shape = roi.shape
    codec = VoxelsNddataCodec(rawDtype)
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def getBlockPrediction(blockIdx):
    '''
    Get the prediction of a block
    '''
    assert 0 <= blockIdx < blocking.numberOfBlocks, "Invalid blockIdx selected"
    roi = blocking.getBlock(blockIdx)

    r = session.get('http://{ip}/prediction/raw/{b}'.format(ip=options.pixelclassification_ip, b=blockIdx), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get prediction of block {b} from {ip}".format(b=blockIdx, ip=options.pixelclassification_ip))
    
    shape = roi.shape
    shape[-1] = numClasses

    codec = VoxelsNddataCodec('float32')
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def getBlocksInRoi(start, stop):
    '''
    Compute the list of blocks that need to be processed to serve the requested ROI
    '''
    startIdx = blocking.getSurroundingBlockIndex(start)
    startBlock = blocking.getBlock(startIdx)

    stopIdx = blocking.getSurroundingBlockIndex(stop)
    stopBlock = blocking.getBlock(stopIdx)

    shape = stopBlock.end - startBlock.begin
    blocksPerDim = np.ceil(shape / blocking.blockShape)

    blockIds = []
    coord = np.zeros_like(start)
    for t in range(int(blocksPerDim[0])):
        coord[0] = startBlock.begin[0] + blocking.blockShape[0] * t
        for x in range(int(blocksPerDim[1])):
            coord[1] = startBlock.begin[1] + blocking.blockShape[1] * x
            for y in range(int(blocksPerDim[2])):
                coord[2] = startBlock.begin[2] + blocking.blockShape[2] * y
                if dim == 3:
                    for z in range(int(blocksPerDim[3])):
                        coord[3] = startBlock.begin[3] + blocking.blockShape[3] * z
                        blockIds.append(blocking.getSurroundingBlockIndex(coord))
                else:
                    blockIds.append(blocking.getSurroundingBlockIndex(coord))

    print("Range {}-{} is covered by blocks: {}".format(start, stop, blockIds))

    return blockIds

def combineBlocksToVolume(blockIds, blockContents, roi=None):
    '''
    Stitch blocks into one 5D numpy volume, which will have the size of the 
    smallest bounding box containing all specified blocks or the size of the provided roi (which should have .begin, .end, and .shape).
    The number of channels (last axis) will be adjusted to match the block contents
    '''
    assert len(blockIds) == len(blockContents), "Must provide the same number of block indices and contents"
    assert len(blockContents) > 0, "Cannot combine zero blocks"
    blocks = [blocking.getBlock(b) for b in blockIds]
    start = np.min([b.begin for b in blocks],axis=0)
    stop = np.max([b.end for b in blocks],axis=0)
    print("Got blocks covering {} to {}".format(start, stop))
    shape = stop - start

    if roi is not None:
        assert all(start <= roi.begin), "Provided blocks do not start at roi beginning"
        assert all(roi.end <= stop), "Provided blocks end at {} before roi end {}".format(stop, roi.end)

    numChannels = blockContents[0].shape[-1]
    shape[-1] = numChannels
    volume = np.zeros(shape, dtype=blockContents[0].dtype)

    print("Filling volume of shape {}".format(volume.shape))

    for block, data in zip(blocks, blockContents):
        blockStart = block.begin - start
        blockEnd = block.end - start
        # the last (channels) dim of start and end will be ignored, we just fill in all that we have
        volume[blockStart[0]:blockEnd[0], blockStart[1]:blockEnd[1], blockStart[2]:blockEnd[2], blockStart[3]:blockEnd[3], ...] = data

    if roi is not None:
        print("Cropping volume of shape {} to roi from {} to {}".format(volume.shape, roi.begin, roi.end))
        volume = volume[roi.begin[0]-start[0]:roi.end[0]-start[0], roi.begin[1]-start[1]:roi.end[1]-start[1], roi.begin[2]-start[2]:roi.end[2]-start[2], roi.begin[3]-start[3]:roi.end[3]-start[3], ...]

    return volume

def createRoi(start, stop):
    ''' helper to create a 5D or 3D block '''
    return pib.Block_5d(np.asarray(start), np.asarray(stop))

# --------------------------------------------------------------
class FinishedBlockCollectorThread(threading.Thread):
    '''
    Little helper thread that waits for all required blocks to be available.
    The thread finishes if all blocks have been found.
    '''
    def __init__(self, blocksToProcess):
        super(FinishedBlockCollectorThread, self).__init__()
        self._requiredBlocks = set(blocksToProcess)
        logger.debug("Waiting for blocks {}".format(self._requiredBlocks))
        self._requiredBlocksLock = threading.Lock()
        self.collectedBlocks = {} # dict with key=blockId, value=np.array of data
        self._availableBlocks = []
        self._availableBlocksLock = threading.Lock()

        # create a callback that appends found blocks to the availableBlocks list
        def finishedBlockCallback(blockId):
            isRequired = False
            with self._requiredBlocksLock:
                if blockId in self._requiredBlocks:
                    isRequired = True
                    self._requiredBlocks.remove(blockId)
            if isRequired:
                logger.debug("got finished message for required block {}".format(blockId))
                with self._availableBlocksLock:
                    self._availableBlocks.append(blockId)

        # on finished block messages, call the callback!
        self._callbackId = finishedQueueSubscription.registerCallback(finishedBlockCallback)

    def run(self):
        keepAlive = True
        while keepAlive:
            time.sleep(0.05)

            with self._availableBlocksLock:
                tempAvailableIds = self._availableBlocks[:] # copy!
                self._availableBlocks = []

            for b in tempAvailableIds:
                blockData, isDummy = cache.readBlock(b)
                assert not isDummy and blockData is not None, "Received Block finished message but it was not available in cache!"
                logger.debug("Fetching available block {}".format(b))
                self.collectedBlocks[b] = blockData

            # quit if all blocks are found
            with self._requiredBlocksLock:
                with self._availableBlocksLock:
                    if len(self._requiredBlocks) == 0 and len(self._availableBlocks) == 0:
                        keepAlive = False
        self._shutdown()
                    

    def _shutdown(self):
        logger.debug("Received all required blocks, shutting down FinishedBlockCollectorThread")
        ''' called from run() before the thread exits, stop retrieving block callbacks '''
        finishedQueueSubscription.removeCallback(self._callbackId)

    def removeBlockRequirements(self, blocks):
        ''' remove the given blocks from the list of blocks we are waiting for '''
        with self._requiredBlocksLock:
            for b in blocks:
                self._requiredBlocks.remove(b)
        logger.debug("Found blocks {} already, only {} remaining".format(blocks, self._requiredBlocks))

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

    blocksToProcess = getBlocksInRoi(start, stop)
    # blockData = [getBlockRawData(b) for b in blocksToProcess]
    blockData = list(executor.map(getBlockRawData, blocksToProcess))
    data = combineBlocksToVolume(blocksToProcess, blockData, roi)

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
    roi = createRoi(start, stop)
    blocksToProcess = getBlocksInRoi(start, stop)
    blockData = {}

    # start listening for finishing blocks right here 
    # because some might finish (blocksInFlight) between the time when we check the cache for this block and 
    # before we'd start listening after looping over all required blocks
    finishedBlockCollector = FinishedBlockCollectorThread(blocksToProcess)
    finishedBlockCollector.start()

    # try to get blocks from cache. If they are not there yet, emplace dummy block so others will not
    # request the computation again
    missingBlocks = []
    blocksInFlight = []
    for b in blocksToProcess:
        block, foundDummy = cache.readBlock(b, insertDummyIfNotFound=True)
        if foundDummy:
            blocksInFlight.append(b)
        else:
            if block is None:
                missingBlocks.append(b)
            else:
                blockData[b] = block

    # enqueue tasks to compute missing blocks
    finishedBlockCollector.removeBlockRequirements(list(blockData.keys()))
    for b in missingBlocks:
        taskQueuePublisher.enqueue(b)

    # wait for all missing blocks to be found
    finishedBlockCollector.join()
    logger.debug("All blocks needed by request were retrieved!")
    blockData.update(finishedBlockCollector.collectedBlocks)

    # blockData = list(executor.map(getBlockPrediction, blocksToProcess))
    # blockData = [getBlockPrediction(b) for b in blocksToProcess]

    blockDataList = [blockData[b] for b in blocksToProcess]
    data = combineBlocksToVolume(blocksToProcess, blockDataList, roi)

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

