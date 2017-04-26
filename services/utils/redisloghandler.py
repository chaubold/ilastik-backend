import logging
import redis

class RedisLogHandler(logging.Handler):
    """
    A class which sends log messages to a redis instance
    """
    def __init__(self, host, port, key, ip=''):
        """
        Initialize the instance with the host, the request URL, and the method
        ("GET" or "POST")
        """
        logging.Handler.__init__(self)
        self._redisClient = redis.StrictRedis(host=host, port=port)
        self._key = key
        self._formatter = logging.Formatter('%(levelname)s:%(asctime)s:{}:%(module)s:%(message)s'.format(ip))

    def emit(self, record):
        """
        Emit a record by appending it to the list with the name `key` in the redis instance
        """
        try:
            self._redisClient.rpush(self._key, self._formatter.format(record))
        except:
            self.handleError(record)
