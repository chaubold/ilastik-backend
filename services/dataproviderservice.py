'''
The raw data provider serves raw data from a 
'''

import json
import argparse
import sys
import traceback
import numpy as np
import h5py
from pprint import pprint

from flask import Flask, send_file, request
from flask_autodoc import Autodoc

# streaming numpy voxel data format
from utils.servicehelper import returnDataInFormat, adjustCoordinateAxesOrder, adjustVolumeAxisOrder

# flask setup
app = Flask("dataproviderservice")
doc = Autodoc(app)

# --------------------------------------------------------------
@app.route('/raw/<format>/roi')
@doc.doc()
def get_raw_roi(format):
    '''
    Get the raw data of a roi in the specified format (raw / tiff / png /hdf5 ).
    The roi is specified by appending "?extents_min=t_x_y_z_c&extents_max=t_x_y_z_c" to requested the URL.
    The extents_max field allows negative indices as in python list indexing.
    '''

    start = list(map(int, request.args['extents_min'].split('_')))
    stop = list(map(int, request.args['extents_max'].split('_')))
    assert len(start) == 5, "Start coordinate must have 5 dimensions"
    assert len(stop) == 5, "End coordinate must have 5 dimensions"
    
    # handle negative indices by adding the shape of the respective dimension
    stop = [shape[i]+x if x < 0 else x for i, x in enumerate(stop)]

    assert all(x >= 0 for x in start), "Cannot have negative start coordinates"
    assert all(a < b for a,b in zip(start, stop)), "End point must be greater than start point"
    assert all(x < s for x, s in zip(start, shape)), "ROI begin exceeds shape of dataset"
    assert all(x <= s for x, s in zip(stop, shape)), "ROI end exceeds shape of dataset"
    
    # transform indices back to input axes order
    start = adjustCoordinateAxesOrder(start, 'txyzc', inputAxes)
    stop = adjustCoordinateAxesOrder(stop, 'txyzc', inputAxes)

    with h5py.File(options.raw_data_file, 'r') as f:    
        if dim == 2:
            rawData = f[options.raw_data_path][start[0]:stop[0], start[1]:stop[1]]
        elif dim == 3:
            rawData = f[options.raw_data_path][start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]

        # always return in txyzc order
        rawData = adjustVolumeAxisOrder(rawData, inputAxes, 'txyzc')

    return returnDataInFormat(rawData, format)

# --------------------------------------------------------------
@app.route('/info/dtype')
@doc.doc()
def info_dtype():
    ''' Return the datatype of the dataset as string '''
    return dtype

# --------------------------------------------------------------
@app.route('/info/shape')
@doc.doc()
def info_shape():
    ''' Return the shape of the dataset as string with underscore delimiters '''
    return '_'.join(map(str, shape))

# --------------------------------------------------------------
@app.route('/info/dim')
@doc.doc()
def info_dim():
    ''' Return the dimensionality of the data (2 or 3), ignoring timeframes and channels '''
    return str(dim)

# --------------------------------------------------------------
@app.route('/doc')
def documentation():
    ''' serve an API documentation '''
    return doc.html(title='Data Provider API', author='Carsten Haubold')

# ----------------------------------------------------------------------------------------
# run server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a pixel classification service',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=9000, help='port on which to run service')
    parser.add_argument('--raw-data-file', type=str, required=True, 
                        help='hdf5 file containing the raw data to process')
    parser.add_argument('--raw-data-path', type=str, required=True, 
                        help='path inside raw data HDF5 file to the raw data volume')
    parser.add_argument('--raw-data-axes', type=str, default='txyzc',
                        help='Axes description of the dataset')
    
    options = parser.parse_args()
    inputAxes = options.raw_data_axes

    # read configuration from project file and raw data
    with h5py.File(options.raw_data_file, 'r') as raw:
        dtype = str(raw[options.raw_data_path].dtype)
        shape = adjustCoordinateAxesOrder(raw[options.raw_data_path].shape, inputAxes, 'txyzc', allowAxisDrop=False)

        dim = 3
        if shape[3] == 1:
            dim = 2

    app.run(host='0.0.0.0', port=options.port, debug=False, threaded=True)

