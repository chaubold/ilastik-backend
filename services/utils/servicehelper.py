import tempfile
import traceback
import numpy as np
from flask import Flask, send_file, request
from utils.voxels_nddata_codec import VoxelsNddataCodec
import redis
import logging
logger = logging.getLogger(__name__)

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
