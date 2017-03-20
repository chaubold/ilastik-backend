import sys
import time
import httplib
import requests
import signal
import numpy as np

from PyQt4.QtCore import QObject, pyqtSignal, QString

from volumina.pixelpipeline.asyncabcs import RequestABC, SourceABC
from volumina.layer import GrayscaleLayer
from volumina.viewer import Viewer

from voxels_nddata_codec import VoxelsNddataCodec

class VoxelClientRequest(object):

    def __init__(self, hostname, layer, dtype, start, stop, dim, channels):
        '''
        start and stop are given as txyzc tuples.
        channels are the number of channels the dataset has, 0 if it doesn't have a channel axis (same amount of data as for =1)
        '''
        assert np.all(np.array(start) < np.array(stop)), "End values cannot be smaller or equal than start values!"
        print("Requesting roi from {} to {}".format(start, stop))
        self.hostname = hostname
        self.layer = layer
        self.dtype = dtype
        self.start = start[1:-1] # strip away time point for now, and skip channel
        self.stop = stop[1:-1] # strip away time point for now, and skip channel
        self.channelRange = (start[-1], stop[-1])
        self.dim = dim
        if dim == 2:
            # skip empty z axes for 2D images
            self.start = self.start[:-1]
            self.stop = self.stop[:-1]

        self.channels = channels

    def wait( self ):
        start_str = '_'.join(map(str, self.start))
        stop_str = '_'.join(map(str, self.stop))
        
        print("Retrieving roi from {} to {}".format(start_str, stop_str))

        r = requests.get("http://{hostname}/{layer}/raw/roi".format(**self.__dict__),
                         params={'extents_min': start_str, 'extents_max': stop_str},
                         stream=True)

        r.raise_for_status()
        
        # request block, which will either have exactly the specified (2D or 3D) shape,
        # or have one additional (channel) axis
        shape = np.array(self.stop) - self.start
        print("Expecting shape {} of dtype {}".format(shape, self.dtype))
        codec = VoxelsNddataCodec(self.dtype)
        arr = codec.decode_to_ndarray(r.raw, shape)

        # slice channels if any
        if len(arr.shape) > self.dim:
            assert 0 <= self.channelRange[0] < self.channels
            assert 0 <= self.channelRange[1] < self.channels
            arr = arr[...,self.channelRange[0]:self.channelRange[1]]
        else:
            assert self.channelRange == (0,1), "Cannot select channels from a plain image without channel dimension"
            arr = np.expand_dims(arr, axis=-1) # add channel dimension

        # add z axis again
        if self.dim == 2:
            arr = np.expand_dims(arr, axis=2)
        # add t axis
        arr = np.expand_dims(arr, axis=0)

        print("Returning block of shape {}".format(arr.shape))
        return arr

class VoxelClientSource(QObject):
    
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int)

    def __init__(self, data_hostname, pixelclass_hostname, layer="raw"):
        super(VoxelClientSource, self).__init__()
        self.data_hostname = data_hostname
        self.hostname = pixelclass_hostname
        self.layer = layer
        
        # read dataset config from data provider service
        r = requests.get('http://{ip}/info/dtype'.format(ip=data_hostname))
        if r.status_code != 200:
            raise RuntimeError("Could not query datatype from dataprovider at ip: {}".format(data_hostname))
        self._dtype = str(r.text)
        
        r = requests.get('http://{ip}/info/dim'.format(ip=data_hostname))
        if r.status_code != 200:
            raise RuntimeError("Could not query dimensionaliy from dataprovider at ip: {}".format(data_hostname))
        self.dim = int(r.text)

        r = requests.get('http://{ip}/info/shape'.format(ip=data_hostname))
        if r.status_code != 200:
            raise RuntimeError("Could not query shape from dataprovider at ip: {}".format(data_hostname))
        self.shape = list(map(int, r.text.split('_')))

        if self.layer == 'raw':
            self.numChannels = 1 # TODO: adjust for multi channel input!
        else:
            r = requests.get('http://{ip}/prediction/numclasses'.format(ip=self.hostname))
            r.raise_for_status()
            self.numChannels = int(r.text)

        # pad shape to 5 dims
        self.shape = [1] + self.shape # add time dimension at the front

        if self.dim == 2:
            self.shape += [1] # add z dimension if needed

        self.shape += [self.numChannels] # raw has one channel for now
        print("Voxel source for layer {} has shape {}".format(self.layer, self.shape))

    def numberOfChannels(self):
        return self.numChannels

    def dtype(self):
        return np.dtype(self._dtype).type

    def request( self, slicing ):
        # Convert slicing to (start, stop)
        start, stop = zip(*[(s.start, s.stop) for s in slicing])
        start = [0 if x is None else x for x in start]
        stop = [b if a is None else a for a,b in zip(stop, self.shape)]

        return VoxelClientRequest( self.hostname, self.layer, self._dtype, start, stop, self.dim, self.numChannels )

    def setDirty( self, slicing ):
        self.isDirty.emit(slicing)

    def __eq__( self, other ):
        return  self.hostname == other.hostname and self.data_hostname == other.data_hostname

    def __ne__( self, other ):
        return not (self == other)
    
    def clean_up(self):
        pass


if __name__ == "__main__":
    from PyQt4.QtGui import QApplication
    
    # DEBUG = False
    # if DEBUG:
    #     print "DEBUGGING with localhost:8000"
    #     sys.argv += ["localhost:8000"]
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixelclass-ip", type=str, help="ip:port of pixel classification service")
    parser.add_argument("--dataprovider-ip", type=str, help="ip:port of dataprovider service")
    args = parser.parse_args()

    app = QApplication([])

    viewer = Viewer()
    
    # Raw
    raw_source = VoxelClientSource(args.dataprovider_ip, args.pixelclass_ip, layer='raw')
    raw_layer = GrayscaleLayer(raw_source)
    raw_layer.numberOfChannels = raw_source.numberOfChannels()
    raw_layer.name = QString("Raw")

    viewer.dataShape = raw_source.shape
    viewer.layerstack.append(raw_layer)

    # Predictions
    # raw_source = VoxelClientSource(args.dataprovider_ip, args.pixelclass_ip, layer='prediction')
    # raw_layer = GrayscaleLayer(raw_source)
    # raw_layer.numberOfChannels = raw_source.numberOfChannels()
    # raw_layer.name = QString("Prediction")

    # viewer.dataShape = raw_source.shape
    # viewer.layerstack.append(raw_layer)

        
    # r = requests.get("http://{hostname}/api/list-datasets".format(hostname=hostname))
    # if r.status_code != httplib.OK:
    #     raise RuntimeError("Could not fetch dataset list: {}".format(r.status_code))
    
    # dset_names = r.json()
    # print dset_names
    # if len(dset_names) < 1:
    #     raise RuntimeError("No datasets listed.")
    
    # For now, just pick the first one.
    # dataset_name = dset_names[0]
    # r = requests.get("http://{hostname}/api/source-states/{dataset_name}".format(**locals()))
    # r.raise_for_status()
    # states = r.json()
    
    # datasources = {}
    # viewer = Viewer()
    # for state in states:
    #     assert state['axes'] == 'txyzc', \
    #         "For now, sources must adhere to volumina's wacky axis order. (Got {})".format(state['axes'])
    #     source = VoxelClientSource(args.hostname, dataset_name, state['name'], state)
    #     datasources[state['name']] = source

    #     layer = GrayscaleLayer(source)
    #     layer.numberOfChannels = state['shape'][-1]
    #     layer.name = QString(state['name'])

    #     viewer.dataShape = state['shape']
    #     viewer.layerstack.append(layer)
    
    # stopped = [False]
    # def state_monitor_thread():
    #     # TODO: This detects changes to the datasources we already have,
    #     #       but doesn't handle sources disappearing or appearing.
    #     while not stopped[0]:
    #         r = requests.get("http://{hostname}/api/source-states/{dataset_name}"
    #                          .format(hostname=hostname, dataset_name=dataset_name))
    #         r.raise_for_status()
    #         new_states = { state['name'] : state for state in r.json() }
    #         for name, source in datasources.items():
    #             source.update_state(new_states[name])
            
    #         time.sleep(0.1)

    # import threading
    # th = threading.Thread(target=state_monitor_thread)
    # th.daemon = True
    # th.start()

    # set CTRL+C behave as its default: quit!
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    viewer.show()
    viewer.raise_()
    app.exec_()