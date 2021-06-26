import subprocess
from time import sleep
import json

vmlist = ['Standard_B2s']*8 + ['Standard_B2ms']*8

HEADER = '\033[1m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def run(cmd, shell=True):
  data = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if 'ERROR' in data.stderr.decode():
    print(cmd)
    print(FAIL)
    print(data.stderr.decode())
    print(ENDC)
    exit()
  return data.stdout.decode()

#################

print(f'{HEADER}Create Azure VM{ENDC}')
for i, size in enumerate(vmlist):
  name = f'vm{i+1}'
  run(f'az vm create --resource-group vm1_group --name {name} --size {size} --image UbuntuLTS --ssh-key-values id_rsa.pub --admin-username ansible')

# #################

print(f'{HEADER}Wait for deployment (1 minute){ENDC}')
sleep(60)

#################

print(f'{HEADER}Open port 8081{ENDC}')
for i, size in enumerate(vmlist):
  name = f'vm{i+1}'
  run(f'az vm open-port --resource-group vm1_group --name {name} --port 8081')

#################

print(f'{HEADER}Install new kernel{ENDC}')
for i, size in enumerate(vmlist):
  name = f'vm{i+1}'
  cmd = f"az vm run-command invoke -g vm1_group -n {name} --command-id RunShellScript --scripts 'sudo apt install -y -f linux-image-4.15.0-1009-azure linux-tools-4.15.0-1009-azure linux-cloud-tools-4.15.0-1009-azure linux-headers-4.15.0-1009-azure linux-modules-4.15.0-1009-azure linux-modules-extra-4.15.0-1009-azure'"
  run(cmd)
