#!/usr/bin/env python

import dockerclient
import docker
import json
from string import Template
import os
import sys
import signal
import subprocess
import logging
import requests
import codes
import dockerclient
import psutil
import time


class RequestRouter():
    def __init__(self, config):
        self.containerClient = dockerclient.DockerClient(config["dockerurl"])
        self.hostIP = config["hostIP"]
        self.interface = config["interface"]
    
    def getAllContainers(self):
        rc,containerlist=self.containerClient.listContainers()
        return containerlist
 
    def getContainer(self, containerId):
        return self.containerClient.inspectContainer(containerId)
    
    def hostDetailsVagrant(self):
        rc = codes.SUCCESS
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        data = subprocess.run("./scripts/calIPS_clock_vagrant.sh", shell=True,stdout=subprocess.PIPE)
        data  = (data.stdout.decode()).splitlines()
        bw = ((subprocess.run("sudo ethtool "+self.interface+" | grep Speed",shell=True,stdout=subprocess.PIPE)).stdout.decode()).split()[1][0:4]
        payload ={"Total_Memory":int(float(memory.total/(1024*1024))),"Total_Disk":int(float(disk.total/(1024*1024))),"Bandwidth":int(bw),"clock":data[0],"Ram_read":int(float(data[3])),"Ram_write":int(float(data[4])),"Disk_read":int(float(data[1])),"Disk_write":int(float(data[2]))}
        data = json.dumps(payload)
        return rc,data

    def hostDetailsAnsible(self):
        rc = codes.SUCCESS
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        data = subprocess.run("./scripts/calIPS_ansible.sh", shell=True,stdout=subprocess.PIPE)
        data  = (data.stdout.decode()).splitlines()
        bw = ((subprocess.run("sudo ethtool "+self.interface+" | grep Speed",shell=True,stdout=subprocess.PIPE)).stdout.decode()).split()[1][0:4]
        payload ={"Total_Memory":int(float(memory.total/(1024*1024))),"Total_Disk":int(float(disk.total/(1024*1024))),"Bandwidth":int(bw),"IPS":data[0],"Ram_read":int(float(data[3])),"Ram_write":int(float(data[4])),"Disk_read":int(float(data[1])),"Disk_write":int(float(data[2]))}
        data = json.dumps(payload)
        return rc,data

    def gethostStat(self):
        rc = codes.SUCCESS 
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()[2]
        disk = psutil.disk_usage('/')
        disk_total = disk.total
        ts = time.time()
        payload ={"ip":self.hostIP,"time-stamp":ts,"cpu":cpu,"memory":memory,"disk":disk_total}
        data = json.dumps(payload)
        return rc,data

    def getContainersStat(self):
        Stats = {}
        rc = codes.SUCCESS 
        Stats['time-stamp'] = time.time()
        Stats['hostIP'] = self.hostIP
        Stat = []
        containers = self.getAllContainers()
        for container in containers:
            stats=self.containerClient.stats(container.id)
            stats=stats[1]
            cpu_count = len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
            cpu_percent = 0.0
            memory_percent = 0.0
            disk_percent = 0.0
            cpu_delta = float(stats["cpu_stats"]["cpu_usage"]["total_usage"]) - float(stats["precpu_stats"]["cpu_usage"]["total_usage"])
            system_delta = float(stats["cpu_stats"]["system_cpu_usage"]) - float(stats["precpu_stats"]["system_cpu_usage"])
            if system_delta > 0.0:
                cpu_percent = cpu_delta / system_delta * 100.0 * cpu_count
            cpu = cpu_percent
            time_stamp = time.time()
            if stats["memory_stats"]["limit"] != 0:
                memory_percent = float(stats["memory_stats"]["usage"]) / float(stats["memory_stats"]["limit"]) * 100.0
            payload ={"name":container.name,"time":time_stamp,"cpu":cpu_percent,"memory":memory_percent,"disk":disk_percent}
            Stat.append(payload)
        Stats["stats"] = Stat
        data = json.dumps(Stats)    
        return rc,data
        
    def _createContainer_restore(self, config):
        return self.containerClient._create(config)
    def _createContainer(self, config):
        return self.containerClient.create(config)
    def _startContainer(self, name):
        return self.containerClient.start(name)
    def _stopContainer(self, name):
        return self.containerClient.stop(name)
    def _deleteContainer(self, name):
        return self.containerClient.delete(name)
    def _checkpointContainer(self,name):
        return self.containerClient.checkpoint(name)
    def _migrateContainer(self, config):
        return self.containerClient.migrate(config)
    def _restoreContainer(self, config):
        return self.containerClient.restore(config)
   
   
    def handleHostOp(self,payload):
        opcode = payload["opcode"]
        if opcode == "hostDetailsVagrant":
            return self.hostDetailsVagrant()
        if opcode == "hostDetailsAnsible":
            return self.hostDetailsAnsible()
        if opcode == "hostStat":
            return self.gethostStat()

    def handleContainerOp(self, payload):
        opcode = payload["opcode"]
        name = payload["name"]
        if opcode == "create":
            return self._createContainer(payload)
        elif opcode == "start":
            return self._startContainer(name)
        elif opcode == "stop":
            return self._stopContainer(name)
        elif opcode == "delete":
            return self._deleteContainer(name)
        elif opcode == "checkpoint":
            return self.checkpoint(payload)
        elif opcode == "migrate":
            return self.migrate(payload)
        elif opcode == "restore":
            return self.restore(payload)
        elif opcode == "ContainerStat":
            return self.getContainersStat(payload)
        
        else:
            return codes.BAD_REQ

    def checkpoint(self,payload):
        container_name = payload["name"]
        checkpoint_name = payload["c_name"]
        container_image = payload["image"]
        checkpoint_tar="/tmp/"
        rc =  codes.SUCCESS
        data = "checkpoint successful"
        try:
            subprocess.call(["sudo","docker", "checkpoint","create",container_name,checkpoint_name])
        except ValueError:
            data= "Checkpoint not created"
            rc = codes.ERROR
        return rc,data

    def migrate(self, payload):
        targetIP = payload["targetIP"]
        container_name = payload["name"]
        checkpoint_name = payload["c_name"]
        container_image = payload["image"]
        checkpoint_tar="/tmp/"
        rc =  codes.SUCCESS
        data = "Migration successful"
        try:
            #cid = subprocess.Popen("docker inspect -f '{{.Id}}' "+container_name, shell=True, stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]
            cid = subprocess.Popen("docker inspect -f '{{.Id}}' "+container_name, shell=True,stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
            subprocess.call(["echo","/var/lib/docker/containers/"+str(cid)+"/checkpoints/ "+checkpoint_name+"/"])
            cmd = "sudo tar -zcf /tmp/"+container_name+"."+checkpoint_name+".tgz -C /var/lib/docker/containers/"+cid+"/"+"checkpoints/ "+checkpoint_name+"/"
            subprocess.call(["echo",cmd])
            subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE,universal_newlines=True)
           # subprocess.call(["sudo","tar","-zcf","/tmp/"+container_name+"."+checkpoint_name+".tgz","-C","/var/lib/docker/containers/"+cid+"/checkpoints/ "+checkpoint_name+"/"])
            subprocess.call(["scp", "-i", "~/agent/id_rsa","/tmp/"+container_name+"."+checkpoint_name+".tgz", "ubuntu@"+targetIP+":/tmp/"])
            subprocess.call(["sudo","rm","-rf","/tmp/"+container_name+"."+checkpoint_name+".tgz"])
        except ValueError:
            data = "Migrate checkpoint to "+targetIP+" not successful"
            rc = codes.ERROR
        return rc,data

    def restore(self, payload):
        container_name = payload["name"]
        checkpoint_name = payload["c_name"]
        container_image = payload["image"]
        rc = codes.SUCCESS
        data = "Container "+container_name+"restored successfully"
       
        try:
         
            cmd = "docker create --name "+container_name+" "+container_image
           
            cid = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
            
            cmd = "sudo tar -zxf /tmp/"+container_name+"."+checkpoint_name+".tgz -C /var/lib/docker/containers/"+str(cid)+"/"+"checkpoints/"
          
            subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE,universal_newlines=True)
          
        except ValueError:
            data = "restore not successful"
            rc = codes.ERROR    
        try:    
            subprocess.call(["docker", "start","--checkpoint",checkpoint_name, container_name])
        except ValueError:
            data = "Container"+container_name+"cannot be started"
            rc = codes.ERROR
        return rc,data

