import tempfile
from flask import Flask, send_file, request
from utils.voxels_nddata_codec import VoxelsNddataCodec

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
