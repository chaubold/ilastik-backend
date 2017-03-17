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
def getBlockRawData(blockIdx, withHalo=True):
    '''
    Get the raw data of a block
    '''
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"

    if withHalo:
        roi = pixelClassificationBackend.getRequiredRawRoiForFeatureComputationOfBlock(blockIdx)
    else:
        roi = pixelClassificationBackend.blocking.getBlock(blockIdx)
    beginStr = '_'.join(map(str,roi.begin))
    endStr = '_'.join(map(str, roi.end))

    r = requests.get('http://{ip}/raw/raw/roi?extents_min={b}&extents_max={e}'.format(ip=options.dataprovider_ip, b=beginStr, e=endStr), stream=True)
    if r.status_code != 200:
        raise RuntimeError("Could not get raw data of block {b} from {ip}".format(b=blockIdx, ip=options.dataprovider_ip))
    shape = roi.shape
    codec = VoxelsNddataCodec(dtype)
    rawData = codec.decode_to_ndarray(r.raw, shape)

    return rawData

def processBlock(blockIdx):
    '''
    Main computational method for processing blocks
    '''
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"
    rawData = getBlockRawData(blockIdx)

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


    print("Input block min {} max {} dtype {} shape {}".format(rawData.min(), rawData.max(), rawData.dtype, rawData.shape))
    features = pixelClassificationBackend.computeFeaturesOfBlock(blockIdx, rawData)
    print("Feature block min {} max {} dtype {} shape {}".format(features.min(), features.max(), features.dtype, features.shape))
    predictions = pixelClassificationBackend.computePredictionsOfBlock(blockIdx, features)
    print("Prediction block min {} max {} dtype {} shape {}".format(predictions.min(), predictions.max(), predictions.dtype, predictions.shape))

    if redisClient is not None:
        # save to cache
        redisClient.set('prediction-{}-block'.format(blockIdx), predictions.tostring())
        shapeStr = '_'.join([str(d) for d in predictions.shape] + [str(predictions.dtype)])
        print(shapeStr)
        redisClient.set('prediction-{}-shape'.format(blockIdx), shapeStr)
    
    return predictions

def getBlocksInRoi(start, stop):
    '''
    Compute the list of blocks that need to be processed to serve the requested ROI
    '''
    blk = pixelClassificationBackend.blocking
    
    startIdx = blk.getSurroundingBlockIndex(start)
    startBlock = blk.getBlock(startIdx)

    stopIdx = blk.getSurroundingBlockIndex(stop)
    stopBlock = blk.getBlock(stopIdx)

    shape = stopBlock.end - startBlock.begin
    blocksPerDim = np.ceil(shape / blk.blockShape)

    blockIds = []
    coord = np.zeros_like(start)
    for x in range(int(blocksPerDim[0])):
        coord[0] = start[0] + blk.blockShape[0] * x
        for y in range(int(blocksPerDim[1])):
            coord[1] = start[1] + blk.blockShape[1] * y
            if dim == 3:
                for z in range(int(blocksPerDim[2])):
                    coord[2] = start[2] + blk.blockShape[2] * z
                    blockIds.append(blk.getSurroundingBlockIndex(coord))
            else:
                blockIds.append(blk.getSurroundingBlockIndex(coord))

    print("Range {}-{} is covered by blocks: {}".format(start, stop, blockIds))

    return blockIds

def combineBlocksToVolume(blockIds, blockContents, roi=None):
    '''
    Stitch blocks into one numpy volume, which will have the size of the 
    smallest bounding box containing all specified blocks or the size of the provided roi (which should have .begin, .end, and .shape)
    '''
    assert len(blockIds) == len(blockContents), "Must provide the same number of block indices and contents"
    assert len(blockContents) > 0, "Cannot combine zero blocks"
    blocks = [pixelClassificationBackend.blocking.getBlock(b) for b in blockIds]
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
            volume = volume[roi.begin[0]-start[0]:roi.end[0]-stop[0], roi.begin[1]-start[1]:roi.end[1]-stop[1], ...]
        elif dim == 3:
            volume = volume[roi.begin[0]-start[0]:roi.end[0]-stop[0], roi.begin[1]-start[1]:roi.end[1]-stop[1], roi.begin[2]-start[2]:roi.end[2]-stop[2], ...]

    return volume

def createRoi(start, stop):
    ''' helper to create a 2D or 3D block '''
    if dim == 2:
        return pib.Block_2d(np.array(start), np.array(stop))
    elif dim == 3:
        return pib.Block_3d(np.array(start), np.array(stop))

# --------------------------------------------------------------
@app.route('/raw/<format>/<int:blockIdx>')
@doc.doc()
def get_raw(format, blockIdx):
    '''
    Get the raw data of a block in the specified format (raw / tiff / png /hdf5 ).
    '''

    data = getBlockRawData(blockIdx)
    return returnDataInFormat(data, format)

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
    blockData = [getBlockRawData(b, withHalo=False) for b in blocksToProcess]
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

    app.run(host='0.0.0.0', port=options.port, debug=False)