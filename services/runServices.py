import subprocess
import argparse
import atexit
import requests
import time
from multiprocessing.dummy import Pool

from utils.registry import Registry
from utils.servicehelper import RedisCache

@atexit.register
def shutdown():
    def stopContainer(dns):
        try:
            subprocess.check_call(["ssh", "-oStrictHostKeyChecking=no", "-i", options.pem, "ubuntu@{}".format(dns), "sudo", "docker", "stop", "test"])
            subprocess.check_call(["ssh", "-oStrictHostKeyChecking=no", "-i", options.pem, "ubuntu@{}".format(dns), "sudo", "docker", "rm", "test"])
            print("docker container at {} shut down".format(dns))
        except:
            print("Couldn't shut down pixel worker {}".format(dns))
    
    hosts = options.pixelclass_dns + [options.thresholding_dns, options.gateway_dns]
    pool = Pool(processes=len(hosts))
    pool.map(stopContainer, hosts)

    try:
        # Download log:
        registry = Registry(options.registry_ip)
        registry.writeLogsToFile(options.logfile)
    except:
        print("Couldn't get log from registry")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up an ilastik cluster on AWS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--registry-ip', type=str, required=True,
                        help='IP of the registry to use')

    parser.add_argument('--pem', type=str, required=True, help='ssh key for the AWS machines')
    parser.add_argument('--thresholding-dns', type=str, required=True, help='DNS name of the thresholding service')
    parser.add_argument('--gateway-dns', type=str, required=True, help='DNS name of the gateway service')
    parser.add_argument('--pixelclass-dns', type=str, nargs='+', help='DNS name of the pixel class serviceS')
    parser.add_argument('--logfile', type=str, required=True, help='Filename where to save the log')
    parser.add_argument('--clear-cache', action='store_true', help='clear the cache from all currently contained blocks!')

    options = parser.parse_args()

    print("Cleaning logs and tasks in registry")
    registry = Registry(options.registry_ip)
    registry._redisClient.delete(registry.LOG)
    registry._redisClient.delete('block-computation-tasks')
    registry._redisClient.delete('finished-blocks')

    if options.clear_cache:
        # get rid of previously stored blocks
        print("Clearing the cache")
        cache_ip = registry.get(registry.CACHE_IP)
        cache = RedisCache(cache_ip)
        cache.clear()

    # start pixel classification
    commandlines = []
    for dns in options.pixelclass_dns:
        print("Starting PC worker ", dns)
        commandlines.append(["ssh", "-oStrictHostKeyChecking=no", "-i", options.pem, "ubuntu@{}".format(dns), "sudo", "docker", "run", "-d", "-p", "8888:8888", "--name", "test", "hcichaubold/ilastikbackend:0.3", "python", "pixelclassificationservice.py", "--registry-ip", options.registry_ip, "--verbose"])

    print("Starting thresholding worker ", options.thresholding_dns)
    commandlines.append(["ssh", "-oStrictHostKeyChecking=no", "-i", options.pem, "ubuntu@{}".format(options.thresholding_dns), "sudo", "docker", "run", "-d", "-p", "8889:8889", "--name", "test", "hcichaubold/ilastikbackend:0.3", "python", "thresholdingservice.py", "--registry-ip", options.registry_ip, "--verbose"])

    print("Starting thresholding worker ", options.gateway_dns)
    commandlines.append(["ssh", "-oStrictHostKeyChecking=no", "-i", options.pem, "ubuntu@{}".format(options.gateway_dns), "sudo", "docker", "run", "-d", "-p", "8080:8080", "--name", "test", "hcichaubold/ilastikbackend:0.3", "python", "ilastikgateway.py", "--registry-ip", options.registry_ip, "--verbose"])

    pool = Pool(processes=len(commandlines))
    pool.map(subprocess.check_call, commandlines)

    def checkConnection(ip, port, name):
        couldConnect = False
        while not couldConnect:
            try:
                r = requests.get('http://{}:{}/doc'.format(ip, port), timeout=1)
                if r.status_code != 200:
                    print('{}(@{}:{}) not reachable yet...'.format(name, ip, port))
                    time.sleep(5)
                else:
                    couldConnect = True
            except:
                print('{}(@{}:{}) not reachable yet...'.format(name, ip, port))
                time.sleep(5)

    checkConnection(options.gateway_dns, 8080, "gateway")
    checkConnection(options.thresholding_dns, 8889, "thresholdingservice")
    for pcIp in options.pixelclass_dns:
        checkConnection(pcIp, 8888, "pixelclassificationservice")

    r = requests.get('http://{}:8080/setup'.format(options.gateway_dns))
    if r.status_code != 200:
        print('Gateway could not be configured...')
        exit()

    print("All instances set up, gateway running at {}:8080".format(options.gateway_dns))

    print("Waiting for the user to press Ctrl+C before instances are shut down again\nPrinting log from all services\n\n\n\n\n")
    try:
        lastLogEntry = 0
        while True:
            time.sleep(0.1)
            if registry._redisClient.llen(registry.LOG) > lastLogEntry:
                logMessages = [m.decode() for m in registry._redisClient.lrange(registry.LOG, lastLogEntry, registry._redisClient.llen(registry.LOG))]
                for m in logMessages:
                    print(m)
                lastLogEntry += len(logMessages)
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping down instances")
