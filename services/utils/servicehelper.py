import tempfile
import traceback
import numpy as np
from flask import Flask, send_file, request
from utils.voxels_nddata_codec import VoxelsNddataCodec
import redis
import logging
import pyilastikbackend as pib
from utils.queues import  FinishedBlockCollectorThread
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Block Cache
# --------------------------------------------------------------
class Cache(object):
    ''' Cache dummy that always returns None on a cache access '''
    def __init__(self):
        pass

    def readBlock(self, blockIdx, insertDummyIfNotFound=False):
        '''
        Try to find the block and its shape info and return the contents as numpy array.

        If insertDummyIfNotFound is True, it will check for the contents of the block or insert a dummy value atomically,
        so that the next requests will see that somebody asked for that block before.

        Returns a tuple (None or the block as numpy array, boolean flag specifying whether a dummy was found)
        '''
        return None, False

    def saveBlock(self, blockIdx, blockData):
        ''' store the block '''
        pass

    def listCachedBlocks(self):
        ''' return the ids of all cached blocks '''
        return []

    def clear(self):
        pass

class RedisCache(Cache):
    ''' 
    A redis cache running at the specified ip and port.
    '''
    _dummyBlock = 'dummy'

    def __init__(self, ip='0.0.0.0:6379', maxmemory='500mb'):
        super(RedisCache, self).__init__()
        host, port = ip.split(':')
        self._redisClient = redis.StrictRedis(host=host, port=port)

        # Configure cache to have a specific maximum memory usage and evict keys using a LRU policy.
        # See https://redis.io/topics/lru-cache for details
        self._redisClient.config_set('maxmemory', maxmemory)
        self._redisClient.config_set('maxmemory-policy', 'allkeys-lru')

    def readBlock(self, blockIdx, insertDummyIfNotFound=False):
        '''
        Try to find the block and its shape info and return the contents as numpy array.

        If insertDummyIfNotFound is True, it will check for the contents of the block or insert a dummy value (not yet! atomically),
        so that the next requests will see that somebody asked for that block before.

        Returns a tuple (None or the block as numpy array, boolean flag specifying whether a dummy was found)
        '''

        cachedBlock = self._redisClient.get('prediction-{}-block'.format(blockIdx))
        cachedShape = self._redisClient.get('prediction-{}-shape'.format(blockIdx))
        if cachedBlock and cachedShape:
            # if we found a dummy block
            if cachedShape.decode() == self._dummyBlock:
                return None, True
            try:
                cachedShape = cachedShape.decode().split('_')
                shape, dtype = list(map(int, cachedShape[:-1])), cachedShape[-1]
                logger.debug("Found block {} of shape {} and dtype {} in cache!".format(blockIdx, shape, dtype))
                return np.fromstring(cachedBlock, dtype=dtype).reshape(shape), False
            except:
                if insertDummyIfNotFound:
                    self._redisClient.set('prediction-{}-block'.format(blockIdx), '')
                    self._redisClient.set('prediction-{}-shape'.format(blockIdx), self._dummyBlock)
                logger.exception("ERROR when retrieving existing block from cache. Proceeding as if it was not found")

        return None, False

    def saveBlock(self, blockIdx, blockData):
        ''' store the block '''
    
        # save to cache
        self._redisClient.set('prediction-{}-block'.format(blockIdx), blockData.tostring())
        shapeStr = '_'.join([str(d) for d in blockData.shape] + [str(blockData.dtype)])
        logger.debug(shapeStr)
        self._redisClient.set('prediction-{}-shape'.format(blockIdx), shapeStr)

    def listCachedBlocks(self):
        ''' return the ids of all cached blocks '''
        return sorted([k.decode().split('-')[1] for k in self._redisClient.scan_iter(match='prediction-*-block')])

    def clear(self):
        ''' Remove all prediction blocks and their shapes from the cache '''
        for k in self._redisClient.scan_iter(match='prediction-*'):
            self._redisClient.delete(k)

# --------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------
def returnDataInFormat(data, format):
    '''
    Handle sending the requested data back in the specified format
    '''
    assert format in ['raw', 'tiff', 'png', 'hdf5'], "Invalid Format selected"
    if format == 'raw':
        stream = VoxelsNddataCodec(data.dtype).create_encoded_stream_from_ndarray(data)
        return send_file(stream, mimetype=VoxelsNddataCodec.VOLUME_MIMETYPE)
    elif format in ('tiff', 'png'):
        import vigra
        _, fname = tempfile.mkstemp(suffix='.'+format)
        vigra.impex.writeImage(data.squeeze(), fname, dtype='NBYTE')
        # TODO: delete file?
        return send_file(fname)
    elif format == 'hdf5':
        import h5py
        _, fname = tempfile.mkstemp(suffix='.'+format)
        with h5py.File(fname, 'w') as f:
            f.create_dataset('exported_data', data=data)
        # TODO: delete file?
        return send_file(fname)

def adjustVolumeAxisOrder(volume, inputAxes, outputAxes='txyzc'):
    """
    This method allows to convert a given `volume` (with given `inputAxes` ordering)
    into a different axis ordering, specified as `outputAxes` string (e.g. "xyzt").

    Allowed axes are `t`, `x`, `y`, `z`, `c`.

    The default format volumes are converted to is "txyzc", axes that are missing in the input
    volume are created with size 1.
    """
    assert(isinstance(volume, np.ndarray))
    assert(len(volume.shape) == len(inputAxes))
    assert(len(outputAxes) >= len(inputAxes))
    assert(not any(a not in 'txyzc' for a in outputAxes))
    assert(not any(a not in 'txyzc' for a in inputAxes))

    outVolume = volume

    # find present and missing axes
    positions = {}
    missingAxes = []
    for axis in outputAxes:
        try:
            positions[axis] = inputAxes.index(axis)
        except ValueError:
            missingAxes.append(axis)

    # insert missing axes at the end
    for m in missingAxes:
        outVolume = np.expand_dims(outVolume, axis=-1)
        positions[m] = outVolume.ndim - 1

    # transpose
    axesRemapping = [positions[a] for a in outputAxes]
    outVolume = np.transpose(outVolume, axes=axesRemapping)

    return outVolume

def adjustCoordinateAxesOrder(coord, inputAxes, outputAxes='txyzc', allowAxisDrop=True):
    """
    This method allows to convert a given `coord` inside a volume with the given `inputAxes`
    into a different axis ordering, specified as `outputAxes` string (e.g. "xyzt").

    Allowed axes are `t`, `x`, `y`, `z`, `c`.

    The default format volumes are converted to is "txyzc", axes that are missing in the input
    volume are created with size 1. Axes that are missing in the output order but are present in the input
    are only allowed to have values 0 or 1 because these are the values one would use to access this singleton axis.
    """
    assert(len(coord) == len(inputAxes))
    if not allowAxisDrop:
        assert len(outputAxes) >= len(inputAxes), "Cannot drop axes during conversion"
    else:
        for idx, a in enumerate(inputAxes):
            if a not in outputAxes:
                assert 0 <= coord[idx] <= 1, "Cannot drop axes that have a value different from 0 or 1"
    assert(not any(a not in 'txyzc' for a in outputAxes))
    assert(not any(a not in 'txyzc' for a in inputAxes))

    outCoord = list(coord)

    # find present and missing axes
    positions = {}
    missingAxes = []
    for axis in outputAxes:
        try:
            positions[axis] = inputAxes.index(axis)
        except ValueError:
            missingAxes.append(axis)

    # insert missing axes at the end
    for m in missingAxes:
        outCoord.append(1)
        positions[m] = len(outCoord) - 1

    # transpose
    axesRemapping = [positions[a] for a in outputAxes]
    outCoord = np.asarray(outCoord)[axesRemapping]

    return outCoord

def getBlockRawData(blockIdx, dtype, blocking, dataprovider_ip, session):
    '''
    Get the raw data of a block
    '''
    assert 0 <= blockIdx < blocking.numberOfBlocks, "Invalid blockIdx selected"

    roi = blocking.getBlock(blockIdx)
    beginStr = '_'.join(map(str,roi.begin))
    endStr = '_'.join(map(str, roi.end))

    r = session.get('http://{ip}/raw/raw/roi?extents_min={b}&extents_max={e}'.format(ip=dataprovider_ip, b=beginStr, e=endStr), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get raw data of block {b} from {ip}".format(b=blockIdx, ip=dataprovider_ip))
    shape = roi.shape
    codec = VoxelsNddataCodec(dtype)
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def getBlockPrediction(blockIdx, blocking, pixelclassification_ip, session):
    '''
    Get the prediction of a block
    '''
    assert 0 <= blockIdx < blocking.numberOfBlocks, "Invalid blockIdx selected"
    roi = blocking.getBlock(blockIdx)

    r = session.get('http://{ip}/prediction/numclasses'.format(ip=pixelclassification_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query num classes from pixel classification at ip: {}".format(pixelclassification_ip))
    numClasses = int(r.text)

    r = session.get('http://{ip}/prediction/raw/{b}'.format(ip=pixelclassification_ip, b=blockIdx), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get prediction of block {b} from {ip}".format(b=blockIdx, ip=pixelclassification_ip))
    
    shape = roi.shape
    shape[-1] = numClasses

    codec = VoxelsNddataCodec('float32')
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def getBlocksInRoi(blocking, start, stop, dim):
    '''
    Compute the list of blocks that need to be processed to serve the requested ROI
    '''
    startIdx = blocking.getSurroundingBlockIndex(start)
    startBlock = blocking.getBlock(startIdx)

    stop = np.asarray(stop) - 1 # stop is exclusive but the "surroundingBlockIndex" check works inclusive
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

    print("Range {}-{} is covered by blocks: {}".format(start, stop+1, blockIds))

    return blockIds

def combineBlocksToVolume(blockIds, blockContents, blocking, roi=None):
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

def collectPredictionBlocksForRoi(blocking, start, stop, dim, cache, finishedQueueSubscription, taskQueuePublisher):
    roi = createRoi(start, stop)
    blocksToProcess = getBlocksInRoi(blocking, start, stop, dim)
    blockData = {}

    # start listening for finishing blocks right here 
    # because some might finish (blocksInFlight) between the time when we check the cache for this block and 
    # before we'd start listening after looping over all required blocks
    finishedBlockCollector = FinishedBlockCollectorThread(blocksToProcess, finishedQueueSubscription, cache)
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

    blockDataList = [blockData[b] for b in blocksToProcess]
    return combineBlocksToVolume(blocksToProcess, blockDataList, blocking, roi)

def getOwnPublicIp():
    ''' helper method to extract the public IP address by which clients can access this very machine '''
    try:
        # AWS-style:
        import subprocess
        ip = subprocess.check_output(['wget','-qO-', 'http://instance-data/latest/meta-data/public-ipv4'])
    except subprocess.CalledProcessError:
        try:
            # Google Cloud:
            import subprocess
            ip = subprocess.check_output(['curl', 'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip', '-H', 'Metadata-Flavor: Google'])
        except subprocess.CalledProcessError:
            try:
                # in normal environments use hostname
                import socket
                ip = socket.gethostbyname(socket.gethostname())
            except:
                # nothing works (e.g. in eduroam), assume we're running local
                logger.warn("Was not able to determine own IP, assuming we run locally and using IP 0.0.0.0")
                return '0.0.0.0'
    return ip
