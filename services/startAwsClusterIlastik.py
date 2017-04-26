import boto3
import time

boto3.setup_default_session(region_name='us-east-1')
iam = boto3.resource('iam')
ec2 = boto3.resource('ec2')
iamClient = boto3.client('iam')
ec2Client = boto3.client('ec2')

# --------------------------------------------------------------------------------
# delete all old stuff
try:
  print("deleting ilastikuser")
  iamClient.delete_user(UserName='ilastikuser')
except:
  print("Error when delete ilastikuser")

try:
  role = iamClient.get_role(RoleName='ilastikInstanceRole')
  iamClient.delete_policy(PolicyArn=role['Role']['Arn'])
  print("Deleted policy")
except:
  print("Error when deleting policy")

try:
  iamClient.delete_role(RoleName='ilastikInstanceRole')
  print("Deleted policy")
except:
  print("Error when deleting role")

try:
  ec2Client.delete_security_group(GroupName='ilastikFirewallSettingsSecurityGroup')
  print("Deleted security group")
except:
  print("error when Deleting security group")

try:
  iamClient.delete_instance_profile(InstanceProfileName='ilastikInstanceProfile')
  print("Deleted instance profile")
except:
  print("error when Deleting instance profile")


# --------------------------------------------------------------------------------
# new user
print("Creating ilastikuser")
try:
  iamClient.create_user(UserName='ilastikuser')
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

  policy = iamClient.create_policy(PolicyName='ilastikInstanceRole', PolicyDocument=doc)
except:
  print("policy already exists")

# --------------------------------------------------------------------------------
print("Adding policy to user")
try:
  # give the policy to the user
  iamClient.create_access_key(UserName='ilastikuser')
  iu = iam.User('ilastikuser')
  iu.attach_policy(PolicyArn=policy['Policy']['Arn'])

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
  role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonSQSFullAccess')
  role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')
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

  # create an instance profile from the role
  instanceProfile = iamClient.create_instance_profile(InstanceProfileName='ilastikInstanceProfile')
  iamClient.add_role_to_instance_profile(InstanceProfileName='ilastikInstanceProfile', RoleName='ilastikInstanceRole')
except:
  print("Security group already exists")


print("Waiting 20 seconds to make sure all roles and profiles are available")
time.sleep(20)

# create instances
numWorkers = 4

# --------------------------------------------------------------------------------------------------------------------
# Instance will be booted immediately!
registryInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                 MinCount=1, # choose a larger number here if you want more than one instance!
                                 MaxCount=1,
                                 KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                 InstanceType='t2.micro', # select the instance type
                                 SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                 IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                 UserData="""#!/bin/bash
sudo apt-get update --fix-missing
sudo docker run -d -p 6380:6379 --name registry bitnami/redis:latest
""")


# to figure out the dns name or IP:
registryInstance[0].wait_until_running()
registryInstance[0].load()
registryIp = registryInstance[0].public_ip_address
print("Registry running at IP: {}".format(registryIp))

# --------------------------------------------------------------------------------------------------------------------
redisInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                 MinCount=1, # choose a larger number here if you want more than one instance!
                                 MaxCount=1,
                                 KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                 InstanceType='t2.micro', # select the instance type
                                 SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                 IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                 UserData="""#!/bin/bash
sudo apt-get update --fix-missing
sudo docker run -d -p 6379:6379 --name redis bitnami/redis:latest
""")

# to figure out the dns name or IP:
redisInstance[0].wait_until_running()
redisInstance[0].load()
cacheIp = redisInstance[0].public_ip_address
print("Cache running at IP: {}".format(cacheIp))

# --------------------------------------------------------------------------------------------------------------------
rabbitMqInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                 MinCount=1, # choose a larger number here if you want more than one instance!
                                 MaxCount=1,
                                 KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                 InstanceType='t2.micro', # select the instance type
                                 SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                 IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                 UserData="""#!/bin/bash
sudo apt-get update --fix-missing
sudo docker run -d -p 4369:4369 -p 25672:25672 -p 5671-5672:5671-5672 --name rabbitmq rabbitmq:3
""")

# to figure out the dns name or IP:
rabbitMqInstance[0].wait_until_running()
rabbitMqInstance[0].load()
messageBrokerIp = rabbitMqInstance[0].public_ip_address
print("Message Broker running at IP: {}".format(messageBrokerIp))

# --------------------------------------------------------------------------------------------------------------------
pcInstances = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                 MinCount=numWorkers, # choose a larger number here if you want more than one instance!
                                 MaxCount=numWorkers,
                                 KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                 InstanceType='t2.micro', # select the instance type
                                 SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                 IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                 UserData="""#!/bin/bash
sudo apt-get update --fix-missing
sudo docker run --name test hcichaubold/ilastikbackend:0.1 python pixelclassificationservice.py --registry-ip {}
""".format(registryIp))

for pi in pcInstances:
  pi.wait_until_running()
  pi.load()
  print("Pixel Classification Worker at IP {}".format(pi.public_ip_address))

# --------------------------------------------------------------------------------------------------------------------
thresholdingInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                 MinCount=1, # choose a larger number here if you want more than one instance!
                                 MaxCount=1,
                                 KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                 InstanceType='t2.micro', # select the instance type
                                 SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                 IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                 UserData="""#!/bin/bash
sudo apt-get update --fix-missing
sudo docker run --name test hcichaubold/ilastikbackend:0.1 python thresholdingservice.py --registry-ip {}
""".format(registryIp))

# to figure out the dns name or IP:
thresholdingInstance[0].wait_until_running()
thresholdingInstance[0].load()
thresholdingIp = thresholdingInstance[0].public_ip_address
print("Thresholding running at IP: {}".format(thresholdingIp))

# --------------------------------------------------------------------------------------------------------------------
gatewayInstance = ec2.create_instances(ImageId='ami-f4cc1de2', # specify a machine image
                                 MinCount=1, # choose a larger number here if you want more than one instance!
                                 MaxCount=1,
                                 KeyName='aws', # specify an SSH key pair that you have created in the AWS console to be able to SSH to the machine. Comment out if not needed.
                                 InstanceType='t2.micro', # select the instance type
                                 SecurityGroups=['ilastikFirewallSettingsSecurityGroup'],
                                 IamInstanceProfile={'Name':'ilastikInstanceProfile'},
                                 UserData="""#!/bin/bash
sudo apt-get update --fix-missing
sudo docker run --name test hcichaubold/ilastikbackend:0.1 python ilastikgateway.py --registry-ip {}
""".format(registryIp))

# to figure out the dns name or IP:
gatewayInstance[0].wait_until_running()
gatewayInstance[0].load()
gatewayIp = gatewayInstance[0].public_ip_address

print("All instances set up, gateway running at {}:8080".format(gatewayIp))

# --------------------------------------------------------------------------------------------------------------------

print("Waiting for the user to press Ctrl+C before instances are shot down again")
try:
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  print("Ctrl+C pressed. Stopping down instances")

# --------------------------------------------------------------------------------------------------------------------

# to stop the instance: (remains available as configured machine, can be started again)
ec2Client.stop_instances(InstanceIds=[registryInstance[0].id])
# to terminate the instance: (will be gone after some time after shutdown)
ec2Client.terminate_instances(InstanceIds=[registryInstance[0].id])

# to stop the instance: (remains available as configured machine, can be started again)
ec2Client.stop_instances(InstanceIds=[redisInstance[0].id])
# to terminate the instance: (will be gone after some time after shutdown)
ec2Client.terminate_instances(InstanceIds=[redisInstance[0].id])

# to stop the instance: (remains available as configured machine, can be started again)
ec2Client.stop_instances(InstanceIds=[rabbitMqInstance[0].id])
# to terminate the instance: (will be gone after some time after shutdown)
ec2Client.terminate_instances(InstanceIds=[rabbitMqInstance[0].id])

# to stop the instance: (remains available as configured machine, can be started again)
ec2Client.stop_instances(InstanceIds=[i.id for i in pcInstances])
# to terminate the instance: (will be gone after some time after shutdown)
ec2Client.terminate_instances(InstanceIds=[i.id for i in pcInstances])

# to stop the instance: (remains available as configured machine, can be started again)
ec2Client.stop_instances(InstanceIds=[thresholdingInstance[0].id])
# to terminate the instance: (will be gone after some time after shutdown)
ec2Client.terminate_instances(InstanceIds=[thresholdingInstance[0].id])

# to stop the instance: (remains available as configured machine, can be started again)
ec2Client.stop_instances(InstanceIds=[gatewayInstance[0].id])
# to terminate the instance: (will be gone after some time after shutdown)
ec2Client.terminate_instances(InstanceIds=[gatewayInstance[0].id])

