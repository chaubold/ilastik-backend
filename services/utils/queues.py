import pika
import time
import logging
import threading
logger = logging.getLogger(__name__)

# see the RabbitMQ tutorials for the different styles of pub/sub with fanout, or the work queues by prefetching only one message per worker.

# --------------------------------------------------------------

class FinishedQueuePublisher(object):
    def __init__(self, name='finished-blocks', host='0.0.0.0'):
        '''
        A publisher for finished blocks
        '''
        self.host = host
        self.name = name
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self._channel = self._connection.channel()
        try:
            self._channel.exchange_delete(exchange=self.name)
        except:
            logger.debug("Channel {} did not exist, so we couldn't remove it...".format(self.name))
        
        self._channel.exchange_declare(exchange=self.name, type='fanout')


    def finished(self, blockId):
        message = str(blockId)
        logger.info("Publishing in channel {}: {} ".format(self.name, message))
        self._channel.basic_publish(exchange=self.name, routing_key='', body=message)


class FinishedQueueSubscription(threading.Thread):
    def __init__(self, name='finished-blocks', host='0.0.0.0'):
        '''
        Start listening for messages on the finished queue (by subscribing to the publisher).
        There should be only one instance of this subscription per process, and whoever wants to be 
        notified by incoming messages should register a callback here.
        '''
        super(FinishedQueueSubscription, self).__init__()
        self.host = host
        self.name = name
        self._nextCallbackId = 0
        self._callbacks = {}
        self._callbacksLock = threading.Lock()

    def run(self):
        # start listening for messages on the specified queue
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        channel = connection.channel()
        channel.exchange_declare(exchange=self.name, type='fanout')
        result = channel.queue_declare(exclusive=True)
        queue_name = result.method.queue
        
        channel.queue_bind(exchange=self.name, queue=queue_name)

        def consumerCallback(channel, method, properties, body):
            logger.debug("Found finished message for block {}".format(body))
            # body is a bytes object containing an int as string
            with self._callbacksLock:
                for cb in self._callbacks.values():
                    cb(int(body.decode()))

        channel.basic_consume(consumerCallback, queue=queue_name, no_ack=True)
        channel.start_consuming()

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
    def __init__(self, name='block-computation-tasks', host='0.0.0.0'):
        '''
        A publisher for finished blocks
        '''
        self.host = host
        self.name = name
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=name) # add durable=True if we want to make queue durable so that messages can survive even if RabbitMQ dies.

    def enqueue(self, blockId):
        logger.debug("Enqueueing task for block {}".format(blockId))
        self._channel.basic_publish(exchange='', routing_key=self.name, body=str(blockId), 
            properties=pika.BasicProperties(delivery_mode=2)) # make message persistent


class TaskQueueSubscription(threading.Thread):
    def __init__(self, callback, name='block-computation-tasks', host='0.0.0.0'):
        '''
        Start listening for messages on the finished queue (by subscribing to the publisher).
        There should be one instance per worker process or thread.
        '''
        super(TaskQueueSubscription, self).__init__()
        self.host = host
        self.name = name
        self.callback = callback

    def run(self):
        # start listening for messages on the specified queue
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        channel = connection.channel()
        channel.queue_declare(queue=self.name)

        def consumerCallback(channel, method, properties, body):
            logger.debug("TaskQueueSubscription: Got message {}".format(body))
            # body is a bytes object containing an int as string
            self.callback(int(body.decode()))

            channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(consumerCallback, queue=self.name)
        channel.start_consuming()