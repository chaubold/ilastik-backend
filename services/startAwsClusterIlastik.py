import boto3
import time
import argparse
import atexit
import botocore.exceptions
import requests

from utils.registry import Registry

@atexit.register
def shutdown():
    try:
        # to stop the instance: (remains available as configured machine, can be started again)
        ec2Client.stop_instances(InstanceIds=[redisInstance[0].id])
        # to terminate the instance: (will be gone after some time after shutdown)
        ec2Client.terminate_instances(InstanceIds=[redisInstance[0].id])
        print("Cache redis shut down")
    except:
        print("Couldn't shut down redis instance")

    try:
        # to stop the instance: (remains available as configured machine, can be started again)
        ec2Client.stop_instances(InstanceIds=[i.id for i in pcInstances])
        # to terminate the instance: (will be gone after some time after shutdown)
        ec2Client.terminate_instances(InstanceIds=[i.id for i in pcInstances])
        print("pc workers shut down")
    except:
        print("Couldn't shut down pixel class workers")

    try:
        # to stop the instance: (remains available as configured machine, can be started again)
        ec2Client.stop_instances(InstanceIds=[thresholdingInstance[0].id])
        # to terminate the instance: (will be gone after some time after shutdown)
        ec2Client.terminate_instances(InstanceIds=[thresholdingInstance[0].id])
        print("thresholding worker shut down")
    except:
        print("Couldn't shut down thresholding instance")

    try:
        # to stop the instance: (remains available as configured machine, can be started again)
        ec2Client.stop_instances(InstanceIds=[gatewayInstance[0].id])
        # to terminate the instance: (will be gone after some time after shutdown)
        ec2Client.terminate_instances(InstanceIds=[gatewayInstance[0].id])
        print("gateway shut down")
    except:
        print("Couldn't shut down gateway instance")

    try:
        # Download log:
        registry = Registry(registryIp)
        registry.writeLogsToFile(options.logfile)
    except:
        print("Couldn't get log from registry")

    try:
        # to stop the instance: (remains available as configured machine, can be started again)
        ec2Client.stop_instances(InstanceIds=[registryInstance[0].id])
        # to terminate the instance: (will be gone after some time after shutdown)
        ec2Client.terminate_instances(InstanceIds=[registryInstance[0].id])
        print("registry shut down")
    except:
        print("Couldn't get log and shut down registry instance")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up an ilastik cluster on AWS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--numPcWorkers', type=int, default=2,
                        help='Number of pixel classification workers to start')
    parser.add_argument('--logfile', type=str, required=True,
                        help='Filename where to save the log')
    parser.add_argument('--dataprovider-ip', type=str, required=True,
                        help='IP:port of the dataprovider to use')

    # Ilastik configuration
    parser.add_argument('--project', type=str, required=True, 
                        help='ilastik project with trained random forest')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('-c', '--channel', type=int, default=0, help='channel')
    parser.add_argument('-s', '--sigmas', type=float, action='append', default=None, 
                        help='smoothing sigmas, defaults to 1 in each spatial dimension')
    parser.add_argument('--blocksize', type=int, default=64, 
                        help='size of blocks in all 2 or 3 dimensions, used to blockify all processing')
    options = parser.parse_args()


    # --------------------------------------------------------------------------------
    boto3.setup_default_session(region_name='us-east-1')
    iam = boto3.resource('iam')
    ec2 = boto3.resource('ec2')
    iamClient = boto3.client('iam')
    ec2Client = boto3.client('ec2')

    # --------------------------------------------------------------------------------
    # delete all old stuff
    # try:
    #     print("deleting ilastikuser")
    #     iamClient.delete_user(UserName='ilastikuser')
    # except:
    #     print("Error when delete ilastikuser")

    # try:
    #     role = iamClient.get_role(RoleName='ilastikInstanceRole')
    #     iamClient.delete_policy(PolicyArn=role['Role']['Arn'])
    #     print("Deleted policy")
    # except:
    #     print("Error when deleting policy")

    # try:
    #     iamClient.delete_role(RoleName='ilastikInstanceRole')
    #     print("Deleted policy")
    # except:
    #     print("Error when deleting role")

    # try:
    #     ec2Client.delete_security_group(GroupName='ilastikFirewallSettingsSecurityGroup')
    #     print("Deleted security group")
    # except:
    #     print("error when Deleting security group")

    # try:
    #     iamClient.delete_instance_profile(InstanceProfileName='ilastikInstanceProfile')
    #     print("Deleted instance profile")
    # except:
    #     print("error when Deleting instance profile")


    # --------------------------------------------------------------------------------
    # new user
    print("Creating ilastikuser")
    try:
        iamClient.create_user(UserName='ilastikuser')
        # iamClient.create_access_key(UserName='ilastikuser')
    except:
        print("user already exists")

    # --------------------------------------------------------------------------------
    # add new policies and roles
    print("Creating policy")
    try:
        # policy that allows the current user to assign instance profiles to instances:
        doc='''{"Version": "2012-10-17",
                "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "iam:PassRole",
                        "iam:ListInstanceProfiles",
                        "ec2:*"
                    ],
                    "Resource": "*"
                }
                ]
        }'''

        policy = iamClient.create_policy(PolicyName='ilastikInstanceRole', PolicyDocument=doc)['Policy']
    except:
        print("policy already exists")
        policies = iamClient.list_policies()
        policy = [p for p in policies['Policies'] if p['PolicyName'] == 'ilastikInstanceRole'][0]

    # --------------------------------------------------------------------------------
    print("Adding policy to user")
    try:
        # give the policy to the user
        iu = iam.User('ilastikuser')
        iu.attach_policy(PolicyArn=policy['Arn'])

        roleDoc = '''{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "sts:AssumeRole",
                    "Principal": {"Service": "ec2.amazonaws.com" },
                    "Effect": "Allow"
                }
            ]
        }'''

        # create the role with full SQS and S3 access:
        iamClient.create_role(RoleName='ilastikInstanceRole', AssumeRolePolicyDocument=roleDoc)
        role = iam.Role('ilastikInstanceRole')
        # role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonSQSFullAccess')
        # role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')
    except:
        print("user already has those permissions")

    # --------------------------------------------------------------------------------
    print("Creating security group")
    try:
        # create a security group that allows SSH access (tcp port 22) from everywhere to our instances:
        ec2Client.create_security_group(GroupName='ilastikFirewallSettingsSecurityGroup', Description='Security Group for ilastik that allows SSH access for debugging')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=22, ToPort=22, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=8888, ToPort=8888, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=8889, ToPort=8889, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=8080, ToPort=8080, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=9000, ToPort=9000, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=6379, ToPort=6379, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=6380, ToPort=6380, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=4369, ToPort=4369, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=25672, ToPort=25672, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=5671, ToPort=5671, CidrIp='0.0.0.0/0')
        ec2Client.authorize_security_group_ingress(GroupName='ilastikFirewallSettingsSecurityGroup', IpProtocol='tcp', FromPort=5672, ToPort=5672, CidrIp='0.0.0.0/0')
    except:
        print("Security group already exists")

    try:
        iamClient.create_instance_profile(InstanceProfileName='ilastikInstanceProfile')
    except:
        print("Instance profile already exists")
        
    try:
        # create an instance profile from the role
        iamClient.add_role_to_instance_profile(InstanceProfileName='ilastikInstanceProfile', RoleName='ilastikInstanceRole')
    except:
        print("Could not add role to instance profile")

    # create instances
    # --------------------------------------------------------------------------------------------------------------------
    # Instance will be booted immediately!
    instanceCreated = False
    while not instanceCreated:
        try:
            registryInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                                    MinCount=1, # choose a larger number here if you want more than one instance!
                                                    MaxCount=1,
                                                    KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                                    InstanceType='t2.micro', # select the instance type
                                                    SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                                    IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                                    UserData="""#!/bin/bash
            sudo apt-get update --fix-missing
            sudo apt-get install -y docker.io
            sudo docker run -d -p 6380:6379 --name registry bitnami/redis:latest
            """)
            instanceCreated = True
        except botocore.exceptions.ClientError:
            print("Couldn't create instance yet, waiting 5 seconds to make sure all roles and profiles are available")
            time.sleep(5)

    # to figure out the dns name or IP:
    registryInstance[0].wait_until_running()
    registryInstance[0].load()
    registryIp = registryInstance[0].public_ip_address
    print("Registry running at IP: {}".format(registryIp))

    couldConnect = False
    while not couldConnect:
        try:
            registry = Registry(registryIp)
            dummy = registry.get(registry.DATA_PROVIDER_IP)
            print("Registry is available! Moving on...")
            couldConnect = True
        except:
            print("couldn't connect to registry redis yet, sleeping for 2 seconds")
            time.sleep(2)

    # --------------------------------------------------------------------------------------------------------------------
    instanceCreated = False
    while not instanceCreated:
        try:
            redisInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                            MinCount=1, # choose a larger number here if you want more than one instance!
                                            MaxCount=1,
                                            KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                            InstanceType='t2.micro', # select the instance type
                                            SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                            IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                            UserData="""#!/bin/bash
            sudo apt-get update --fix-missing
            sudo apt-get install -y docker.io
            sudo docker run -d -p 6379:6379 --name redis bitnami/redis:latest
            """)
            instanceCreated = True
        except botocore.exceptions.ClientError:
            print("Couldn't create instance yet, waiting 5 seconds to make sure all roles and profiles are available")
            time.sleep(5)

    # to figure out the dns name or IP:
    redisInstance[0].wait_until_running()
    redisInstance[0].load()
    cacheIp = redisInstance[0].public_ip_address
    print("Cache running at IP: {}".format(cacheIp))

    # --------------------------------------------------------------------------------------------------------------------
    # Configure cluster:
    import subprocess

    if options.sigmas:
        sigmaParams = ["--sigmas"] + [str(s) for s in options.sigmas]
    else:
        sigmaParams = []

    subprocess.check_call(["python", "configurePCWorkflow.py",
                             "--registry-ip", registryIp,
                             "--cache-ip", cacheIp+":6379",
                             "--dataprovider-ip", options.dataprovider_ip,
                             "--project", options.project,
                             "--threshold", str(options.threshold),
                             "--channel", str(options.channel),
                             "--blocksize", str(options.blocksize)] 
                          + sigmaParams)

    # --------------------------------------------------------------------------------------------------------------------
    instanceCreated = False
    while not instanceCreated:
        try:
            pcInstances = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                            MinCount=options.numPcWorkers, # choose a larger number here if you want more than one instance!
                                            MaxCount=options.numPcWorkers,
                                            KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                            InstanceType='t2.micro', # select the instance type
                                            SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                            IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                            UserData="""#!/bin/bash
            sudo apt-get update --fix-missing
            sudo apt-get install -y docker.io
            sudo docker run -d -p 8888:8888 --name test hcichaubold/ilastikbackend:0.3 python pixelclassificationservice.py --registry-ip {}
            """.format(registryIp))
            instanceCreated = True
        except botocore.exceptions.ClientError:
            print("Couldn't create instance yet, waiting 5 seconds to make sure all roles and profiles are available")
            time.sleep(5)

    for pi in pcInstances:
        pi.wait_until_running()
        pi.load()
        print("Pixel Classification Worker at IP {}".format(pi.public_ip_address))

    # --------------------------------------------------------------------------------------------------------------------
    instanceCreated = False
    while not instanceCreated:
        try:
            thresholdingInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                            MinCount=1, # choose a larger number here if you want more than one instance!
                                            MaxCount=1,
                                            KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                            InstanceType='t2.micro', # select the instance type
                                            SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                            IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                            UserData="""#!/bin/bash
            sudo apt-get update --fix-missing
            sudo apt-get install -y docker.io
            sudo docker run -d -p 8889:8889 --name test hcichaubold/ilastikbackend:0.3 python thresholdingservice.py --registry-ip {}
            """.format(registryIp))
            instanceCreated = True
        except botocore.exceptions.ClientError:
            print("Couldn't create instance yet, waiting 5 seconds to make sure all roles and profiles are available")
            time.sleep(5)

    # to figure out the dns name or IP:
    thresholdingInstance[0].wait_until_running()
    thresholdingInstance[0].load()
    thresholdingIp = thresholdingInstance[0].public_ip_address
    print("Thresholding running at IP: {}".format(thresholdingIp))

    # --------------------------------------------------------------------------------------------------------------------
    instanceCreated = False
    while not instanceCreated:
        try:
            gatewayInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                            MinCount=1, # choose a larger number here if you want more than one instance!
                                            MaxCount=1,
                                            KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                            InstanceType='t2.micro', # select the instance type
                                            SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                            IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                            UserData="""#!/bin/bash
            sudo apt-get update --fix-missing
            sudo apt-get install -y docker.io
            sudo docker run -d -p 8080:8080 --name test hcichaubold/ilastikbackend:0.3 python ilastikgateway.py --registry-ip {}
            """.format(registryIp))
            instanceCreated = True
        except botocore.exceptions.ClientError:
            print("Couldn't create instance yet, waiting 5 seconds to make sure all roles and profiles are available")
            time.sleep(5)

    # to figure out the dns name or IP:
    gatewayInstance[0].wait_until_running()
    gatewayInstance[0].load()
    gatewayIp = gatewayInstance[0].public_ip_address
    print("Gateway is started at {}:8080, trying to connect...".format(gatewayIp))
    couldConnect = False
    while not couldConnect:
        try:
            r = requests.get('http://{}:8080/doc'.format(gatewayIp), timeout=1)
            if r.status_code != 200:
                print('Gateway not reachable yet...')
                time.sleep(5)
            else:
                couldConnect = True
        except:
            print('Gateway not reachable yet...')
            time.sleep(5)

    print("All instances set up, gateway running at {}:8080".format(gatewayIp))

    # --------------------------------------------------------------------------------------------------------------------

    print("Waiting for the user to press Ctrl+C before instances are shut down again")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping down instances")

    # machines are shut down thanks to @atexit.register
