import numpy as np
import json
from framework.server.restClient import *
import subprocess
import requests
import logging
import os
import platform
from framework.node.Powermodels.PMXeon_X5570 import *
from framework.metrics.Disk import *
from framework.metrics.RAM import *
from framework.metrics.Bandwidth import *

class Datacenter():
    
    def __init__(self, hosts, env):
        self.num_hosts = len(hosts)
        self.hosts = hosts
        self.env = env if env == 'Vagrant' else 'Ansible'
        self.types = {'Power' : [1]}
        
    def generateHosts(self):
        hosts = []
        if self.env == 'Vagrant':
            if 'Windows' in platform.system():
                with open('framework/server/scripts/instructions_arch.json') as f:
                    arch_dict = json.load(f)
                instructions = arch_dict[platform.machine()]
            else:
                instructions = subprocess.run("bash -c framework/server/scripts/callIPS_instr.sh", shell=True,stdout=subprocess.PIPE)
                instructions  = int((instructions.stdout.decode()).splitlines()[0])
        for IP in self.hosts:
            payload = {"opcode": "hostDetails"+self.env}
            resp = requests.get("http://"+IP+":8081/host", data=json.dumps(payload))
            data = json.loads(resp.text)
            logging.error("Host details collected from: {}".format(IP))
            print(data)
            IPS = instructions/(float(data['clock']) * 1000) if self.env == 'Vagrant' else data['IPS']
            Power = PMXeon_X5570()
            Ram = RAM(data['Total_Memory'], data['Ram_read'], data['Ram_write'])
            Disk_ = Disk(data['Total_Disk'], data['Disk_read'], data['Disk_write'])
            Bw = Bandwidth(data['Bandwidth'], data['Bandwidth'])
            hosts.append((IP, IPS, Ram, Disk_, Bw, Power))
        return hosts
