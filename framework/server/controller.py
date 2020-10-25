#
# Controller: Handles all operations between framework and cluster

import json
import os
import sys
from string import Template
import requests
import pdb
from framework.server.common.codes import *
import logging
import framework.server.restClient as rclient
from time import time
from datetime import datetime

class RequestHandler():
    def __init__(self, database, env):
        self.db = database
        self.env = env

    def basic_call(self, json_body, opcode, hostIP):
        start = time()
        payload = {
            "opcode": opcode,
            "image": json_body["fields"]["image"],
            "host_ip": hostIP,
            "name": json_body["fields"]["name"]
        }
        rc = rclient.HandleRequest(payload, hostIP, self.env)
        self.env.logger.debug(payload)
        self.env.logger.debug("Response from "+opcode+"d container", rc)
        return rc, time() - start

    def create(self, json_body, hostIP):
        return self.basic_call(json_body, "create", hostIP)
        
    def destroy(self, json_body, hostIP):
        return self.basic_call(json_body, "delete", hostIP)

    def gethostStat(self, hostIP):
        message = "Host stats collected successfully"
        data = rclient.HandleRequest({"opcode": "hostStat"}, hostIP, self.env)
        datapoint =  {
                    "measurement": "hostStat",
                    "tags": {
                                "host_ip": data["ip"]
                            },
                    "fields":
                            {
                                "cpu": data["cpu"],
                                "memory": data["memory"],
                                "disk": data["disk"]
                            },
                    "time": datetime.utcnow().isoformat(sep='T'),
                } if 'server_error' not in data else {}
        self.env.logger.debug(datapoint)
        if 'server_error' not in data: self.db.insert([datapoint])
        return datapoint, message
               
    def getContainerStat(self, hostIP):
        message = "Container stats collected successfully"
        data = rclient.HandleRequest({"opcode": "ContainerStat"}, hostIP, self.env)
        datapoints = []
        if 'server_error' not in data:
            for container_dict in data['stats']:
                datapoints.append({
                        "measurement": "ContainerStat",
                        "tags": {
                                    "host_ip": data["hostIP"],
                                    "container_name": container_dict['name']
                                },
                        "fields": container_dict,
                        "time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
                    })
        self.env.logger.debug(datapoints)
        if 'server_error' not in data: self.db.insert(datapoints)
        return datapoints, message
            
    def checkpoint(self, ccid, cid, cur_host_ip):
        # print("Checkpoint started")
        start = time()
        payload = {
                "opcode": "checkpoint",
                "c_name": str(ccid)+"_"+str(cid),
                "name": str(ccid)+"_"+str(cid)
        } 
        rc = rclient.HandleRequest(payload, cur_host_ip, self.env)
        self.env.logger.debug("checkpoint completed, response is container:"+str(ccid)+"_"+str(cid)+", host:"+cur_host_ip)
        self.env.logger.debug(payload)
        return rc, time() - start
    
    def migrate(self, ccid, cid, cur_host_ip, tar_host_ip):
        # print("Migration started")
        start = time()
        payload = {
                "opcode": "migrate",
                "uname": 'vagrant' if self.env.environment == 'Vagrant' else 'ansible',
                "c_name": str(ccid)+"_"+str(cid),
                "name": str(ccid)+"_"+str(cid),
                "targetIP": tar_host_ip
        }
        rc = rclient.HandleRequest(payload, cur_host_ip, self.env)
        self.env.logger.debug("Migrated from "+cur_host_ip+" to "+tar_host_ip+" for container: "+str(ccid)+"_"+str(cid))
        self.env.logger.debug(payload)
        return rc, time() - start
    
    def restore(self, ccid, cid, image, tar_host_ip):
        start = time()
        name = str(ccid)+"_"+str(cid)
        payload = {
            "opcode": 'restore',
            "c_name": name,
            "name": name,
            "image": image
        }
        rc = rclient.HandleRequest(payload, tar_host_ip, self.env)
        self.env.logger.debug("Restore container "+str(ccid)+"_"+str(cid)+" at "+tar_host_ip)
        self.env.logger.debug(payload)
        return rc, time() - start