import numpy as np
import json
from framework.server.restClient import *
import subprocess
import requests
import logging
import os
import platform
from metrics.powermodels.PMRaspberryPi import *
from metrics.powermodels.PMB2s import *
from metrics.powermodels.PMB4ms import *
from metrics.powermodels.PMB8ms import *
from metrics.powermodels.PMXeon_X5570 import *
from metrics.Disk import *
from metrics.RAM import *
from metrics.Bandwidth import *
from utils.Utils import *

import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

class Datacenter():
    
    def __init__(self, hosts, env, env_type):
        self.num_hosts = len(hosts)
        self.hosts = hosts
        self.env = env
        self.env_type = env_type
        self.types = {'Power' : [1]}

    def parallelizedFunc(self, IP):
        payload = {"opcode": "hostDetails"+self.env_type}
        resp = requests.get("http://"+IP+":8081/request", data=json.dumps(payload))
        data = json.loads(resp.text)
        return data

    def generateHosts(self):
        print(color.HEADER+"Obtaining host information and generating hosts"+color.ENDC)
        hosts = []
        with open('framework/config/'+self.env+'_config.json', "r") as f:
            config = json.load(f)
        powermodels = [server["powermodel"] for server in config[self.env.lower()]['servers']]
        if self.env_type == 'Virtual':
            with open('framework/server/scripts/instructions_arch.json') as f:
                arch_dict = json.load(f)
            instructions = arch_dict[platform.machine()]
        outputHostsData = Parallel(n_jobs=num_cores)(delayed(self.parallelizedFunc)(i) for i in self.hosts)
        for i, data in enumerate(outputHostsData):
            IP = self.hosts[i]
            logging.error("Host details collected from: {}".format(IP))
            print(color.BOLD+IP+color.ENDC, data)
            IPS = (instructions * config[self.env.lower()]['servers'][i]['cpu'])/(float(data['clock']) * 1000000) if self.env_type == 'Virtual' else data['MIPS']
            Power = eval(powermodels[i]+"()")
            Ram = RAM(data['Total_Memory'], data['Ram_read'], data['Ram_write'])
            Disk_ = Disk(data['Total_Disk'], data['Disk_read'], data['Disk_write'])
            Bw = Bandwidth(data['Bandwidth'], data['Bandwidth'])
            hosts.append((IP, IPS, Ram, Disk_, Bw, Power))
        return hosts
