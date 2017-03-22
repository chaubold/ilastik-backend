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
import concurrent.future

from flask import Flask, send_file, request
from flask_autodoc import Autodoc

# C++ module containing all the important methods
import pyilastikbackend as pib
from utils.servicehelper import returnDataInFormat
from utils.voxels_nddata_codec import VoxelsNddataCodec

# flask setup
app = Flask("ilastikgateway")
doc = Autodoc(app)

# global variable storing the backend instance and the redis client
blocking = None
executor = ThreadPoolExecutor(max_workers=40)

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

    r = requests.get('http://{ip}/raw/raw/roi?extents_min={b}&extents_max={e}'.format(ip=options.dataprovider_ip, b=beginStr, e=endStr), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get raw data of block {b} from {ip}".format(b=blockIdx, ip=options.dataprovider_ip))
    shape = roi.shape
    codec = VoxelsNddataCodec(rawDtype)
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def processBlock(blockIdx):
    '''
    Get the prediction of a block
    '''
    assert 0 <= blockIdx < blocking.numberOfBlocks, "Invalid blockIdx selected"
    roi = blocking.getBlock(blockIdx)
    r = requests.get('http://{ip}/prediction/raw/{b}'.format(ip=options.pixelclassification_ip, b=blockIdx), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get prediction of block {b} from {ip}".format(b=blockIdx, ip=options.pixelclassification_ip))
    shape = tuple(roi.shape) + (numClasses,)

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
    for x in range(int(blocksPerDim[0])):
        coord[0] = startBlock.begin[0] + blocking.blockShape[0] * x
        for y in range(int(blocksPerDim[1])):
            coord[1] = startBlock.begin[1] + blocking.blockShape[1] * y
            if dim == 3:
                for z in range(int(blocksPerDim[2])):
                    coord[2] = startBlock.begin[2] + blocking.blockShape[2] * z
                    blockIds.append(blocking.getSurroundingBlockIndex(coord))
            else:
                blockIds.append(blocking.getSurroundingBlockIndex(coord))

    print("Range {}-{} is covered by blocks: {}".format(start, stop, blockIds))

    return blockIds

def combineBlocksToVolume(blockIds, blockContents, roi=None):
    '''
    Stitch blocks into one numpy volume, which will have the size of the 
    smallest bounding box containing all specified blocks or the size of the provided roi (which should have .begin, .end, and .shape)
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

    additionalAxes = blockContents[0].shape[dim:]
    print("additional axes: {}".format(additionalAxes))
    if len(additionalAxes) > 0:
        volume = np.zeros(tuple(shape) + additionalAxes, dtype=blockContents[0].dtype)
    else:
        volume = np.zeros(shape, dtype=blockContents[0].dtype)

    print("Have volume of shape {}".format(volume.shape))

    for block, data in zip(blocks, blockContents):
        blockStart = block.begin - start
        blockEnd = block.end - start

        if dim == 2:
            print("Inserting data into a subarray of shape {} from a {} block".format(volume[blockStart[0]:blockEnd[0], blockStart[1]:blockEnd[1], ...].shape, data.shape))
            volume[blockStart[0]:blockEnd[0], blockStart[1]:blockEnd[1], ...] = data
        elif dim == 3:
            volume[blockStart[0]:blockEnd[0], blockStart[1]:blockEnd[1], blockStart[2]:blockEnd[2], ...] = data

    if roi is not None:
        print("Cropping volume of shape {} to roi from {} to {}".format(volume.shape, roi.begin, roi.end))
        if dim == 2:
            volume = volume[roi.begin[0]-start[0]:roi.end[0]-start[0], roi.begin[1]-start[1]:roi.end[1]-start[1], ...]
        elif dim == 3:
            volume = volume[roi.begin[0]-start[0]:roi.end[0]-start[0], roi.begin[1]-start[1]:roi.end[1]-start[1], roi.begin[2]-start[2]:roi.end[2]-start[2], ...]

    return volume

def createRoi(start, stop):
    ''' helper to create a 2D or 3D block '''
    if dim == 2:
        return pib.Block_2d(np.array(start), np.array(stop))
    elif dim == 3:
        return pib.Block_3d(np.array(start), np.array(stop))

# --------------------------------------------------------------
@app.route('/raw/<format>/roi')
@doc.doc()
def get_raw_roi(format):
    '''
    Get the raw data of a roi in the specified format (raw / tiff / png /hdf5 ).
    The roi is specified by appending "?extents_min=x_y_z&extents_max=x_y_z" to requested the URL.
    '''

    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    roi = createRoi(start, stop)

    blocksToProcess = getBlocksInRoi(start, stop)
    blockData = [getBlockRawData(b) for b in blocksToProcess]
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

    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    roi = createRoi(start, stop)

    blocksToProcess = getBlocksInRoi(start, stop)
    blockData = [processBlock(b) for b in blocksToProcess]
    data = combineBlocksToVolume(blocksToProcess, blockData, roi)

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
    parser.add_argument('--blocksize', type=int, default=64, 
                        help='size of blocks in all 2 or 3 dimensions, used to blockify all processing')
    
    options = parser.parse_args()

    # read dataset config from data provider service
    r = requests.get('http://{ip}/info/dtype'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query datatype from dataprovider at ip: {}".format(options.dataprovider_ip))
    rawDtype = r.text
    
    r = requests.get('http://{ip}/info/dim'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(options.dataprovider_ip))
    dim = int(r.text)
    blockShape = [options.blocksize] * dim

    r = requests.get('http://{ip}/info/shape'.format(ip=options.dataprovider_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query shape from dataprovider at ip: {}".format(options.dataprovider_ip))
    shape = list(map(int, r.text.split('_')))

    r = requests.get('http://{ip}/prediction/numclasses'.format(ip=options.pixelclassification_ip))
    if r.status_code != 200:
        raise RuntimeError("Could not query num classes from pixel classification at ip: {}".format(options.pixelclassification_ip))
    numClasses = int(r.text)

    print("Found dataset of size {} and dimensionality {}".format(shape, dim))
    print("Using block shape {}".format(blockShape))

    # configure pixelClassificationBackent
    # TODO: write a factory method for the constructor!
    if dim == 2:
        blocking = pib.Blocking_2d([0,0], shape, blockShape)
    elif dim == 3:
        blocking = pib.Blocking_3d([0,0,0], shape, blockShape)
    else:
        raise ValueError("Wrong data dimensionality, must be 2 or 3, got {}".format(dim))

    print("Dataset consists of {} blocks".format(blocking.numberOfBlocks))

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)

