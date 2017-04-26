"""
The registry object
"""

import redis
import logging
logger = logging.getLogger(__name__)

class Registry(object):
    '''
    The service registry contains a list of IP adresses of running services, as well as some global configuration.

    This is an extremely simple implementation that uses a central Redis store for all data, 
    and no checks or whatsoever for possibly failing services.

    If that is required, we might want to switch to [etcd](https://github.com/coreos/etcd) or [Consul](https://www.consul.io/)

    TODO: implement a way to observe changes using pub/sub on specific keys (using https://redis.io/topics/notifications)
    '''

    DATA_PROVIDER_IP = "DATA_PROVIDER_IP"
    THRESHOLDING_IP = "THRESHOLDING_IP"
    GATEWAY_IP = "GATEWAY_IP"
    PIXEL_CLASSIFICATION_WORKER_IPS = "PIXEL_CLASSIFICATION_WORKER_IPS"
    CACHE_IP = "CACHE_IP"
    MESSAGE_BROKER_IP = "MESSAGE_BROKER_IP"
    PC_FEATURES = "PC_FEATURES"
    PC_RANDOM_FOREST = "PC_RANDOM_FOREST"
    THRESHOLD_VALUE = "THRESHOLD_VALUE"
    THRESHOLD_CHANNEL = "THRESHOLD_CHANNEL"
    THREDHOLD_SIGMAS = "THREDHOLD_SIGMAS"
    BLOCKSIZE = "BLOCKSIZE"

    allowedKeys = {
        DATA_PROVIDER_IP: "The IP:port at which the dataprovider is running. Usually outside of the ilastik machine 'network'",
        THRESHOLDING_IP: "IP:port address of the thresholding service",
        GATEWAY_IP: "IP:port of the ilastik gateway",
        PIXEL_CLASSIFICATION_WORKER_IPS: "List of IP:port addresses of pixelclassification workers",
        CACHE_IP: "IP:port address of the Redis instance used for caching. Must be running at the default port 6379.",
        MESSAGE_BROKER_IP: "IP:port of the rabbitMQ instance used for task queuing",
        PC_FEATURES: "Selected pixel classification features as JSON",
        PC_RANDOM_FOREST: "Binary blob of the HDF5 file containing the random forest for pixel classification",
        THRESHOLD_VALUE: "Thresholding value at which probability a pixel counts as foreground",
        THRESHOLD_CHANNEL: "Which channel of the probabilities to use for thresholding",
        THREDHOLD_SIGMAS: "String in x_y_z format of smoothing sigmas to apply on the probabilities before thresholding. Use zeros or negative values to disable smoothing.",
        BLOCKSIZE: "String in x_y_z format of the block size to use for the dataset"
    }

    def __init__(self, host, port=6380):
        self._redisClient = redis.StrictRedis(host=host, port=port)
        logger.info("Connected to Registry at {}:{}".format(host, port))
        self.printContents(logOnly=True)

    def get(self, key):
        '''
        Gets the value from the registry, checking that only valid keys are used.
        Lists are returned as python lists
        '''
        key = str(key).upper()
        if key not in self.allowedKeys.keys():
            raise ValueError("{} is no valid registry entry".format(key))

        if key == self.PIXEL_CLASSIFICATION_WORKER_IPS:
            # Warning: if the list size shrinks between the len and range call, this might throw an error?
            return [ip.decode() for ip in self._redisClient.lrange(key, 0, self._redisClient.llen(key))]
        elif key == self.PC_RANDOM_FOREST:
            return self._redisClient.get(key)
        else:
            v = self._redisClient.get(key)
            if v:
                return v.decode()
            else:
                return None

    def set(self, key, value):
        '''
        Sets the value in the registry, checking that only valid keys are used.
        Lists are automatically appended.
        '''
        key = str(key).upper()
        if key not in self.allowedKeys.keys():
            raise ValueError("{} is no valid registry entry".format(key))

        if key == self.PIXEL_CLASSIFICATION_WORKER_IPS:
            self._redisClient.rpush(key, value)
        else:
            return self._redisClient.set(key, value)

    def remove(self, key, value):
        '''
        removal is only allowed for pixel classification worker IPs
        '''
        key = str(key).upper()
        if key != self.PIXEL_CLASSIFICATION_WORKER_IPS:
            raise ValueError("can only remove values from worker IPs")

        value = str(value)
        if self._redisClient.lrem(key, 0, value) == 0:
            logger.warn("Tried to delete {} from {}, but is not present".format(value, key))
        else:
            logger.info("Removed {} from {}".format(value, key))

    def printContents(self, logOnly=False):
        ''' debug printing '''
        if logOnly:
            out = logger.info
        else:
            out = print

        for k in self.allowedKeys.keys():
            if k == self.PC_RANDOM_FOREST:
                out("{}=binary blob of length {} ({})".format(k, len(self.get(k)), self.allowedKeys[k]))
            else:    
                out("{}={} ({})".format(k, self.get(k), self.allowedKeys[k]))

