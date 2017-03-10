'''
The pixel classification service can take raw data and compute features and predict given a random forest
'''

import json
import tempfile
import argparse
import numpy as np
import h5py
import vigra
from pprint import pprint

from flask import Flask, send_file
from flask_autodoc import Autodoc

# C++ module containing all the important methods
import pyilastikbackend as pib
# streaming numpy voxel data format
from voxels_nddata_codec import VoxelsNddataCodec

# flask setup
app = Flask("pixelclassificationservice")
doc = Autodoc(app)

# global variable storing the backend instance
pixelClassificationBackend = None

# --------------------------------------------------------------
def processBlock(blockIdx):
    '''
    Main computational method for processing blocks
    '''
    assert 0 <= blockIdx < pixelClassificationBackend.blocking.numberOfBlocks, "Invalid blockIdx selected"

    roi = pixelClassificationBackend.getRequiredRawRoiForFeatureComputationOfBlock(blockIdx)

    with h5py.File(options.raw_data_file, 'r') as f:
        if dim == 2:
            rawData = f[options.raw_data_path][roi.begin[0]:roi.end[0], roi.begin[1]:roi.end[1]]
        elif dim == 3:
            rawData = f[options.raw_data_path][roi.begin[0]:roi.end[0], roi.begin[1]:roi.end[1], roi.begin[2]:roi.end[2]]

    features = pixelClassificationBackend.computeFeaturesOfBlock(blockIdx, rawData)
    predictions = pixelClassificationBackend.computePredictionsOfBlock(blockIdx, features)

    return predictions

# --------------------------------------------------------------
@app.route('/prediction/<format>/<int:blockIdx>')
@doc.doc()
def get_prediction(format, blockIdx):
    '''
    Get a predicted block in the specified format (raw / tiff / png).

    '''
    assert format in ['raw', 'tiff', 'png'], "Invalid Format selected"

    data = processBlock(blockIdx)

    if format == 'raw':
        data = np.asarray(data, order='C')
        stream = VoxelsNddataCodec(data.dtype).create_encoded_stream_from_ndarray(data)
        return send_file(stream, mimetype=VoxelsNddataCodec.VOLUME_MIMETYPE)
    elif format in ('tiff', 'png'):
        _, fname = tempfile.mkstemp(suffix='.'+format)
        vigra.impex.writeImage(data.squeeze(), fname, dtype='NBYTE')
        # TODO: delete file?
        return send_file(fname)
    else:
        raise RuntimeError("unsupported format: {}".format(format))

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
    parser.add_argument('--raw-data-file', type=str, required=True, 
                        help='hdf5 file containing the raw data to process')
    parser.add_argument('--raw-data-path', type=str, required=True, 
                        help='path inside raw data HDF5 file to the raw data volume')
    parser.add_argument('--blocksize', type=int, default=64, 
                        help='size of blocks in all 2 or 3 dimensions, used to blockify all processing')

    options = parser.parse_args()

    # read configuration from project file and raw data
    with h5py.File(options.project, 'r') as ilp:
        with h5py.File(options.raw_data_file, 'r') as raw:
            dtype = str(raw[options.raw_data_path].dtype)
            shape = raw[options.raw_data_path].shape
            dim = len(shape)
            blockShape = [64] * dim

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
        if dtype == 'uint16':
            pixelClassificationBackend = pib.PixelClassification_2d_uint16()
        if dtype == 'float32':
            pixelClassificationBackend = pib.PixelClassification_2d_float32()

        blocking = pib.Blocking_2d([0,0], shape, blockShape)
    elif dim == 3:
        if dtype == 'uint8':
            pixelClassificationBackend = pib.PixelClassification_3d_uint8()
        if dtype == 'uint16':
            pixelClassificationBackend = pib.PixelClassification_3d_uint16()
        if dtype == 'float32':
            pixelClassificationBackend = pib.PixelClassification_3d_float32()

        blocking = pib.Blocking_3d([0,0,0], shape, blockShape)
    else:
        raise ValueError("Wrong data dimensionality, must be 2 or 3, got {}".format(dim))

    print("Dataset consists of {} blocks".format(blocking.numberOfBlocks))

    pixelClassificationBackend.configureDatasetSize(blocking)
    pixelClassificationBackend.configureSelectedFeatures(selectedFeatureScalePairs)
    pixelClassificationBackend.loadRandomForest(options.project, 'PixelClassification/ClassifierForests/Forest', 4)

    app.run(host='0.0.0.0', port=options.port, debug=False)