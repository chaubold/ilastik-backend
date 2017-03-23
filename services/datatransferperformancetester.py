import argparse
import requests
from flask import Flask, send_file, request
from utils.servicehelper import returnDataInFormat
from utils.voxels_nddata_codec import VoxelsNddataCodec
import time
import numpy as np
import math, random, sys
from operator import itemgetter
app = Flask("datatransferperformancetester")

shape = None
dtype = None
block = None

# --------------------------------------------------------------
class ProgressBar:
    def __init__(self, start=0, stop=100):
        self._state = 0
        self._start = start
        self._stop = stop

    def reset(self, val=0):
        self._state = val

    def show(self, increase=1):
        self._state += increase
        if self._state > self._stop:
            self._state = self._stop

        # show
        pos = float(self._state - self._start)/(self._stop - self._start)
        try:
            sys.stdout.write("\r[%-20s] %d%%" % ('='*int(20*pos), (100*pos)))

            if self._state == self._stop:
                sys.stdout.write('\n')
                sys.stdout.flush()
        except IOError:
            pass

# --------------------------------------------------------------
# Command line plotting taken from https://github.com/imh/hipsterplot/blob/master/hipsterplot.py

CHAR_LOOKUP_SYMBOLS = [(0, ' '), # Should be sorted
                       (1, '.'),
                       (2, ':'),
                       #(3, '!'),
                       (4, '|'),
                       #(8, '+'),
                       (float("inf"), '#')]

def charlookup(num_chars):
    """ Character for the given amount of elements in the bin """
    return next(ch for num, ch in CHAR_LOOKUP_SYMBOLS if num_chars <= num)


def bin_generator(data, bin_ends):
    """ Yields a list for each bin """
    max_idx_end = len(bin_ends) - 1
    iends = enumerate(bin_ends)

    idx_end, value_end = next(iends)
    bin_data = []
    for el in sorted(data):
        while el >= value_end and idx_end != max_idx_end:
            yield bin_data
            bin_data = []
            idx_end, value_end = next(iends)
        bin_data.append(el)

    # Finish
    for unused in iends:
        yield bin_data
        bin_data = []
    yield bin_data


def enumerated_reversed(seq):
    """ A version of reversed(enumerate(seq)) that actually works """
    return zip(range(len(seq) - 1, -1, -1), reversed(seq))


def plot(y_vals, x_vals=None, num_x_chars=70, num_y_chars=15):
    """
    Plots the values given by y_vals. The x_vals values are the y indexes, by
    default, unless explicitly given. Pairs (x, y) are matched by the x_vals
    and y_vals indexes, so these must have the same length.
    The num_x_chars and num_y_chars inputs are respectively the width and
    height for the output plot to be printed, given in characters.
    """
    if x_vals is None:
        x_vals = list(range(len(y_vals)))
    elif len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have the same length")

    ymin = min(y_vals)
    ymax = max(y_vals)
    xmin = min(x_vals)
    xmax = max(x_vals)

    xbinwidth = (xmax - xmin) / num_x_chars
    y_bin_width = (ymax - ymin) / num_y_chars

    x_bin_ends = [(xmin + (i+1) * xbinwidth, 0) for i in range(num_x_chars)]
    y_bin_ends = [ymin + (i+1) * y_bin_width for i in range(num_y_chars)]

    columns_pairs = bin_generator(zip(x_vals, y_vals), x_bin_ends)
    yloop = lambda *args: [charlookup(len(el)) for el in bin_generator(*args)]
    ygetter = lambda iterable: map(itemgetter(1), iterable)
    columns = (yloop(ygetter(pairs), y_bin_ends) for pairs in columns_pairs)
    rows = list(zip(*columns))

    for idx, row in enumerated_reversed(rows):
        y_bin_mid = y_bin_ends[idx] - y_bin_width * 0.5
        print("{:10.4f} {}".format(y_bin_mid, "".join(row)))

# --------------------------------------------------------------
@app.route('/shape')
def info_shape():
    ''' Return the shape of the dataset as string with underscore delimiters '''
    return '_'.join(map(str, shape))

# --------------------------------------------------------------
@app.route('/dtype')
def info_dtype():
    ''' Return the datatype used for the block'''
    return dtype

# --------------------------------------------------------------
@app.route('/block')
def block():
    ''' Return the shape of the dataset as string with underscore delimiters '''
    print("Serving block of shape {} and dtype {}".format(block.shape, block.dtype))
    return returnDataInFormat(block, 'raw')

# ----------------------------------------------------------------------------------------
# run server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an ilastik gateway',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=8080, help='port on which to run service')
    parser.add_argument('--server-ip', type=str, default=None,
                        help='IP of the server. If this is given this script runs the client side, otherwise the server')    
    parser.add_argument('-b', '--blocksize', type=int, default=64, 
                        help='size of blocks in all 2 or 3 dimensions, only needed on server side')
    parser.add_argument('-d', '--dims', type=int, default=2, 
                        help='number of dimensions for blocks')
    parser.add_argument('-t', '--dtype', type=str, default='uint8', 
                        help='datatype of the block')
    parser.add_argument('-n', '--num-iterations', type=int, default=100, 
                        help='How many packets to request (from the client)')
    parser.add_argument('--server-use-threads', action='store_true')
    parser.add_argument('--server-num-processes', type=int, default=1)

    
    options = parser.parse_args()

    if not options.server_ip:
        print("Running server")

        shape = [options.blocksize] * options.dims
        block = np.random.random(shape).astype(options.dtype)
        dtype = options.dtype

        app.run(host='0.0.0.0', port=options.port, debug=False, threaded=options.server_use_threads, processes=options.server_num_processes)

    else:
        print("Running client")
        # get which block size the server serves
        r = requests.get('http://{ip}/shape'.format(ip=options.server_ip))
        r.raise_for_status()
        shape = list(map(int, r.text.split('_')))

        r = requests.get('http://{ip}/dtype'.format(ip=options.server_ip))
        r.raise_for_status()
        dtype = r.text
        print("Using blocks of shape {} and dtype {} for speed test, {} bytes each...".format(shape, dtype, np.prod(shape) * np.dtype(dtype).type().nbytes))

        times = []
        pb = ProgressBar(0,options.num_iterations)
        pb.show(0)

        # get the block a fixed number of times
        for i in range(options.num_iterations):
            startTime = time.time()
            r = requests.get('http://{ip}/block'.format(ip=options.server_ip), stream=True)
            r.raise_for_status()
            codec = VoxelsNddataCodec(dtype)
            block = codec.decode_to_ndarray(r.raw, shape)
            endTime = time.time()
            times.append(endTime - startTime)
            pb.show()
        pb.show()

        print("Retrieving {} blocks of shape {} and dtype {} took:".format(options.num_iterations, shape, dtype))
        print("\ttotal:\t\t{} secs".format(sum(times)))
        print("\tmean:\t\t{} secs".format(np.mean(times)))
        print("\tvar:\t\t{} secs".format(np.var(times)))
        print("\tmin:\t\t{} secs".format(min(times)))
        print("\tmax:\t\t{} secs".format(max(times)))
        print("------------------------------------------------------------------------------------------------------------")
        plot(times)

    