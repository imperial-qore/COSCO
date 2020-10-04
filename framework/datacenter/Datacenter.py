import numpy as np
import uuid
import json
import pprint
import sqlite3
from framework.server.restClient import *
import subprocess
import requests
import logging
import os
import platform
from framework.node.Powermodels.PMXeon_X5570 import *

class Datacenter():
    
    def __init__(self, hosts):
        self.num_hosts = len(hosts)
        self.hosts = hosts
        self.types = {'Power' : [1]}
        
    def generateHosts(self):
        hosts = []
        if 'Windows' in platform.system():
            with open('framework/server/scripts/instructions_arch.json') as f:
                arch_dict = json.load(f)
            instructions = arch_dict[platform.machine()]
        else:
            instructions = subprocess.run("bash -c framework/server/scripts/callIPS_instr.sh", shell=True,stdout=subprocess.PIPE)
            instructions  = int((instructions.stdout.decode()).splitlines()[0])
        for IP in self.hosts:
            payload = {"opcode": "hostDetails"}
            resp = requests.get("http://"+IP+":8081/host", data=json.dumps(payload))
            data = json.loads(resp.text)
            logging.error("Host details collected from: {}".format(IP))
            IPS = instructions
            Power = PMXeon_X5570()
            hosts.append((IP,IPS, Ram, Disk_, Bw,Power))
        return hosts
