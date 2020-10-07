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
import re

class RequestRouter():
    def __init__(self, config):
        self.containerClient = dockerclient.DockerClient(config["dockerurl"])
        self.hostIP = config["hostIP"]
        self.interface = config["interface"]
    
    def hostDetailsVagrant(self):
        rc = codes.SUCCESS
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        data = subprocess.run("./scripts/calIPS_clock_vagrant.sh", shell=True,stdout=subprocess.PIPE)
        data  = (data.stdout.decode()).splitlines()
        bw = ((subprocess.run("sudo ethtool "+self.interface+" | grep Speed",shell=True,stdout=subprocess.PIPE)).stdout.decode()).split()[1][0:4]
        payload ={"Total_Memory":int(float(memory.total/(1024*1024))),"Total_Disk":int(float(disk.total/(1024*1024))),"Bandwidth":int(bw),"clock":data[0],"Ram_read":int(float(data[3])),"Ram_write":int(float(data[4])),"Disk_read":int(float(data[1])),"Disk_write":int(float(data[2]))}
        data = json.dumps(payload)
        return rc, data

    def hostDetailsAnsible(self):
        rc = codes.SUCCESS
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        data = subprocess.run("./scripts/calIPS_ansible.sh", shell=True,stdout=subprocess.PIPE)
        data  = (data.stdout.decode()).splitlines()
        bw = ((subprocess.run("sudo ethtool "+self.interface+" | grep Speed",shell=True,stdout=subprocess.PIPE)).stdout.decode()).split()[1][0:4]
        payload ={"Total_Memory":int(float(memory.total/(1024*1024))),"Total_Disk":int(float(disk.total/(1024*1024))),"Bandwidth":int(bw),"IPS":data[0],"Ram_read":int(float(data[3])),"Ram_write":int(float(data[4])),"Disk_read":int(float(data[1])),"Disk_write":int(float(data[2]))}
        data = json.dumps(payload)
        return rc, data

    def gethostStat(self):
        rc = codes.SUCCESS 
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()[2]
        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024 * 1024)
        ts = time.time()
        payload = {"ip": self.hostIP, "time-stamp":ts, "cpu":cpu, "memory":memory, "disk":disk_total}
        data = json.dumps(payload)
        return rc, data

    def getContainersStat(self):
        Stats = {}
        rc = codes.SUCCESS 
        Stats['time-stamp'] = time.time()
        Stats['hostIP'] = self.hostIP
        Stat = []
        containers = self.containerClient.dclient1.containers(all=True)
        for container in containers:
            c_id = container['Id']
            c_name = container['Names'][0].replace('/', '')
            _, stats = self.containerClient.stats(c_id)
            inspect_data = self.containerClient.dclient1.inspect_container(c_id)['State']
            running = inspect_data['Running']
            finished_at = inspect_data['FinishedAt']
            error = inspect_data['Error']
            cpu_percent = 0.0
            memory_percent = 0.0
            disk_percent = 0.0
            disk_size = 0.0
            time_stamp = time.time()
            bw_up = stats['networks']['eth0']['tx_bytes'] / (1024 * 1024) if running else 0.0
            bw_down = stats['networks']['eth0']['rx_bytes'] / (1024 * 1024) if running else 0.0
            # Reference: https://github.com/moby/moby/blob/eb131c5383db8cac633919f82abad86c99bffbe5/cli/command/container/stats_helpers.go#L175
            if running:
                cpu_delta = float(stats["cpu_stats"]["cpu_usage"]["total_usage"]) - float(stats["precpu_stats"]["cpu_usage"]["total_usage"])
                system_delta = float(stats["cpu_stats"]["system_cpu_usage"]) - float(stats["precpu_stats"]["system_cpu_usage"])
                if system_delta > 0.0:
                    cpu_percent = cpu_delta / system_delta * 100.0
                if stats["memory_stats"]["limit"] != 0:
                    memory_percent = float(stats["memory_stats"]["usage"]) / float(stats["memory_stats"]["limit"]) * 100.0
                disk_stat = subprocess.Popen("docker ps --size -a", shell=True,stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
                for line in disk_stat.splitlines():
                    if c_name in line:
                        disk_stat = re.search('(virtual (.+?)MB)', line).group(1)
                        break
                disk_size = float(disk_stat.split(' ')[1][:-2])
            payload = {
                        "name": c_name,
                        "time": time_stamp,
                        "cpu": cpu_percent,
                        "memory": memory_percent,
                        "disk": disk_size, 
                        "bw_up": bw_up, 
                        "bw_down": bw_down,
                        "running": running,
                        "finished_at": finished_at,
                        "error": error
                    }
            Stat.append(payload)
        Stats["stats"] = Stat
        data = json.dumps(Stats)    
        return rc, data

    def handleRequestOp(self, payload):
        opcode = payload["opcode"]
        if opcode == "create":
            return self.containerClient.create(payload)
        elif opcode == "start":
            return self.containerClient.start(payload["name"])
        elif opcode == "stop":
            return self.containerClient.stop(payload["name"])
        elif opcode == "delete":
            return self.containerClient.delete(payload["name"])
        elif opcode == "checkpoint":
            return self.checkpoint(payload)
        elif opcode == "migrate":
            return self.migrate(payload)
        elif opcode == "restore":
            return self.restore(payload)
        elif opcode == "ContainerStat":
            return self.getContainersStat()
        elif opcode == "hostDetailsVagrant":
            return self.hostDetailsVagrant()
        elif opcode == "hostDetailsAnsible":
            return self.hostDetailsAnsible()
        elif opcode == "hostStat":
            return self.gethostStat()
        else:
            return codes.BAD_REQ, ""

    def checkpoint(self,payload):
        container_name = payload["name"]
        checkpoint_name = payload["c_name"]
        rc, data = codes.SUCCESS, "Checkpoint successful"
        try:
            subprocess.call(["sudo", "docker", "checkpoint", "create", container_name, checkpoint_name])
        except ValueError:
            rc, data = codes.ERROR, "Checkpoint not created"
        return rc, json.dumps({'message': data})

    def migrate(self, payload):
        targetIP = payload["targetIP"]
        container_name = payload["name"]
        checkpoint_name = payload["c_name"]
        uname = payload["uname"]
        rc, data =  codes.SUCCESS, "Migration successful"
        try:
            cid = subprocess.Popen("docker inspect -f '{{.Id}}' "+container_name, shell=True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
            cmd = "sudo tar -zcf /tmp/"+container_name+"."+checkpoint_name+".tgz -C /var/lib/docker/containers/"+cid+"/"+"checkpoints/ "+checkpoint_name+"/"
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
            self.containerClient.delete(container_name)
            subprocess.call(["scp", "-i", "~/agent/id_rsa","/tmp/"+container_name+"."+checkpoint_name+".tgz", uname+"@"+targetIP+":/tmp/"])
            subprocess.call(["sudo","rm","-rf","/tmp/"+container_name+"."+checkpoint_name+".tgz"])
        except ValueError:
            data = "Migrate checkpoint to "+targetIP+" not successful"
            rc = codes.ERROR
        return rc, json.dumps({'message': data})

    def restore(self, payload):
        container_name = payload["name"]
        checkpoint_name = payload["c_name"]
        container_image = payload["image"]
        rc, data = codes.SUCCESS,"Container "+container_name+" restored successfully"
        try:
            cmd = "docker create --name "+container_name+" "+container_image
            cid = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
            cmd = "sudo tar -zxf /tmp/"+container_name+"."+checkpoint_name+".tgz -C /var/lib/docker/containers/"+str(cid)+"/"+"checkpoints/"
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,universal_newlines=True)
            subprocess.call(["sudo","rm","-rf","/tmp/"+container_name+"."+checkpoint_name+".tgz"])
        except ValueError:
            rc, data = codes.ERROR, "restore not successful"
        output = subprocess.run("docker start --checkpoint "+checkpoint_name+" "+container_name, shell=True,stderr=subprocess.PIPE)
        output = output.stderr.decode()
        if 'Error' in output:
            rc, data = codes.ERROR, output.split(":")[1]
        return rc, json.dumps({'message': data})

