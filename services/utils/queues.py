import redis
import time
import logging
import threading
logger = logging.getLogger(__name__)

# using the registry Redis, and keyspace notifications to "listen" on channels

# --------------------------------------------------------------

class FinishedQueuePublisher(object):
    def __init__(self, name='finished-blocks', host='0.0.0.0', port=6380):
        '''
        A publisher for finished blocks
        '''
        self.host = host
        self.name = name
        self._redis = redis.StrictRedis(host=host, port=port)
    
    def finished(self, blockId):
        message = str(blockId)
        logger.info("Publishing in channel {}: {} ".format(self.name, message))
        self._redis.rpush(self.name, message)

class FinishedQueueSubscription(threading.Thread):
    def __init__(self, name='finished-blocks', host='0.0.0.0', port=6380):
        '''
        Start listening for messages on the finished queue (by subscribing to the publisher).
        There should be only one instance of this subscription per process, and whoever wants to be 
        notified by incoming messages should register a callback here.
        '''
        super(FinishedQueueSubscription, self).__init__()
        self.host = host
        self.name = name
        self.port = port
        self._redis = redis.StrictRedis(host=host, port=port)
        self._redis.config_set('notify-keyspace-events', 'Kl') # listen on keyspace events of lists!
        self._nextCallbackId = 0
        self._callbacks = {}
        self._callbacksLock = threading.Lock()

    def run(self):
        ''' wait for finished block replies and call callbacks '''
        def consumerCallback(body):
            logger.debug("Found finished message for block {}".format(body))
            # body is a bytes object containing an int as string
            with self._callbacksLock:
                for cb in self._callbacks.values():
                    cb(int(body.decode()))

        # start listening for messages on changes to the specified list
        p = self._redis.pubsub(ignore_subscribe_messages=True)
        p.subscribe('__keyspace@0__:{}'.format(self.name))

        currentListIndex = self._redis.llen(self.name)
        i = 0
        while True:
            m = p.get_message()
            listWasAppendedTo = m and 'push' in m['data'].decode()

            if listWasAppendedTo or i > 10:
                while self._redis.llen(self.name) > currentListIndex:
                    consumerCallback(self._redis.lindex(self.name, currentListIndex))
                    currentListIndex += 1
                i = 0
            else:
                i += 1
                time.sleep(0.01)


    def registerCallback(self, callback):
        ''' Register a callback function. Returns this callback's ID which is needed if the callback is to removed later '''
        with self._callbacksLock:
            cbId = self._nextCallbackId
            self._nextCallbackId += 1
            self._callbacks[cbId] = callback
            return cbId

    def removeCallback(self, cbId):
        ''' Remove a callback based on its ID '''
        with self._callbacksLock:
            if cbId in self._callbacks:
                del self._callbacks[cbId]

# --------------------------------------------------------------
class FinishedBlockCollectorThread(threading.Thread):
    '''
    Little helper thread that waits for all required blocks to be available.
    The thread finishes if all blocks have been found.
    '''
    def __init__(self, blocksToProcess, finishedQueueSubscription, cache):
        super(FinishedBlockCollectorThread, self).__init__()
        self._requiredBlocks = set(blocksToProcess)
        logger.debug("Waiting for blocks {}".format(self._requiredBlocks))
        self._requiredBlocksLock = threading.Lock()
        self.collectedBlocks = {} # dict with key=blockId, value=np.array of data
        self._availableBlocks = []
        self._availableBlocksLock = threading.Lock()
        self._finishedQueueSubscription = finishedQueueSubscription
        self._cache = cache

        # create a callback that appends found blocks to the availableBlocks list
        def finishedBlockCallback(blockId):
            isRequired = False
            with self._requiredBlocksLock:
                if blockId in self._requiredBlocks:
                    isRequired = True
                    self._requiredBlocks.remove(blockId)
            if isRequired:
                logger.debug("got finished message for required block {}".format(blockId))
                with self._availableBlocksLock:
                    self._availableBlocks.append(blockId)

        # on finished block messages, call the callback!
        self._callbackId = self._finishedQueueSubscription.registerCallback(finishedBlockCallback)

    def run(self):
        keepAlive = True
        while keepAlive:
            time.sleep(0.05)

            with self._availableBlocksLock:
                tempAvailableIds = self._availableBlocks[:] # copy!
                self._availableBlocks = []

            for b in tempAvailableIds:
                blockData, isDummy = self._cache.readBlock(b)
                assert not isDummy and blockData is not None, "Received Block finished message but it was not available in cache!"
                logger.debug("Fetching available block {}".format(b))
                self.collectedBlocks[b] = blockData

            # quit if all blocks are found
            with self._requiredBlocksLock:
                with self._availableBlocksLock:
                    if len(self._requiredBlocks) == 0 and len(self._availableBlocks) == 0:
                        keepAlive = False
        self._shutdown()
                    

    def _shutdown(self):
        logger.debug("Received all required blocks, shutting down FinishedBlockCollectorThread")
        ''' called from run() before the thread exits, stop retrieving block callbacks '''
        self._finishedQueueSubscription.removeCallback(self._callbackId)

    def removeBlockRequirements(self, blocks):
        ''' remove the given blocks from the list of blocks we are waiting for '''
        with self._requiredBlocksLock:
            for b in blocks:
                self._requiredBlocks.remove(b)
        logger.debug("Found blocks {} already, only {} remaining".format(blocks, self._requiredBlocks))

# --------------------------------------------------------------

class TaskQueuePublisher(object):
    def __init__(self, name='block-computation-tasks', host='0.0.0.0', port=6380):
        '''
        A publisher for finished blocks
        '''
        self.host = host
        self.name = name
        self._redis = redis.StrictRedis(host=host, port=port)

    def enqueue(self, blockId):
        logger.debug("Enqueueing task for block {}".format(blockId))
        self._redis.rpush(self.name, str(blockId))

class TaskQueueSubscription(threading.Thread):
    def __init__(self, callback, name='block-computation-tasks', host='0.0.0.0', port=6380):
        '''
        Start listening for messages on the finished queue (by subscribing to the publisher).
        There should be one instance per worker process or thread.
        '''
        super(TaskQueueSubscription, self).__init__()
        self.host = host
        self.name = name
        self.callback = callback
        self._redis = redis.StrictRedis(host=host, port=port)
        self._redis.config_set('notify-keyspace-events', 'Kl') # listen on keyspace events of lists!

    def run(self):
        def consumerCallback(body):
            logger.debug("TaskQueueSubscription: Got message {}".format(body))
            # body is a bytes object containing an int as string
            self.callback(int(body.decode()))

        # start listening for messages on changes to the specified list, 
        p = self._redis.pubsub(ignore_subscribe_messages=True)
        p.subscribe('__keyspace@0__:{}'.format(self.name))

        i = 10 # check for present tasks right at the beginning
        while True:
            m = p.get_message()
            listWasAppendedTo = m and 'push' in m['data'].decode()

            if listWasAppendedTo or i > 10:
                # if a task is found, pop it from the list so that nobody else gets it
                task = self._redis.lpop(self.name)
                if task:
                    consumerCallback(task)
                i = 0
            else:
                i += 1
                time.sleep(0.01)
