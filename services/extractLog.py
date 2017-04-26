import logging
import json
import tempfile
import argparse
import sys
import numpy as np
import h5py
import vigra
from pprint import pprint
import requests

from utils.registry import Registry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the log of the ilastik cloud',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--registry-ip', type=str, required=True,
                        help='IP of the registry service, running at port 6380')
    parser.add_argument('--out', type=str, required=True,
                        help='Filename where to save the log')
    options = parser.parse_args()

    registry = Registry(options.registry_ip)
    registry.writeLogsToFile(options.out)
    print("Done")
