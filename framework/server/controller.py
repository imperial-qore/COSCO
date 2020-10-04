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
import time
import framework.server.restClient as rclient



class RequestHandler():
    def __init__(self, database):
        self.db=database


    def Create(self,json_body):
        host_id = json_body["fields"]["Host_id"]
        payload = {}
        query="SELECT * FROM host WHERE host_id="+"'"+str(host_id)+"'"+";"
        print(query)
        result=self.db.select(query)
        host_ip = list(result)[0][0]["host_ip"]
        image = json_body["fields"]["image"]
        cpu_shares = 2
        memory = '521m'
        payload["opcode"] = 'create'
        payload["image"] = image
        payload["host_ip"] = host_ip
        payload["cpu_shares"] = cpu_shares
        payload["memory"] = memory
        payload["name"] = json_body["fields"]["name"]
        
        rc = rclient.HandleContainer(payload,host_ip)
        print("Resposne from created coatient",rc)
        

    def destroy(self,json_body):
        host_id = json_body["fields"]["Host_id"]
        payload = {}
        query="SELECT * FROM host WHERE host_id="+"'"+str(host_id)+"'"+";"
        result=self.db.select(query)
        host_ip = list(result)[0][0]["host_ip"]
        payload["opcode"] = 'delete'
        payload["host_ip"] = host_ip
        payload["name"] = json_body["fields"]["name"]
        
        rc = rclient.HandleContainer(payload,host_ip)
        return rc

    def gethostStat(self):
        mes = "Host stas collected successfully"
        payload = dict()
        payload["opcode"]="hostStat"
  
        query="SELECT * FROM host;"
        host=self.db.select(query)
        hostList = list(host)[0][0]["host_ip"]
             
        if(hostList):
            for host in hostList:
                resp = resp = rclient.HandleHost(payload,host)
                data = json.dumps(resp.json())
               # print(data)
                datapoint =  {
                            "measurement":"hostStat",
                            "tags": {
                                        "host_ip":data["ip"],
                                    },
                            "time": data["time-stamp"],
                            "fields":
                                    {
                                        "cpu":data["cpu"],
                                        "memory":data["memory"],
                                        "disk":data["disk_total"]
                                    }
                        }

                result=self.db.insert(datapoint)
        else:
      
            mes = "No host to collect the container stats"
        return mes
               
    def getContainerStat(self):
        query="SELECT * FROM host;"
        host=self.db.select(query)
        hostList = list(host)
        print(hostList)
        payload = dict()
        payload['opcode']='ContainerStat'
       
        datapoints={}
        if hostList :
            for host in hostList:
                resp = rclient.HostDetails(payload,host)
                print(resp)
                for i in range(len(resp)):
                    temp_data = resp[i]
                    data_point = {} 
                    data_point["measurement"] = ContainerStat
                    data_point["tags"]["name"] = temp_data["name"]
                    data_point["fields"]["memory"] = temp_data["memory"]
                    data_point["fields"]["disk"] = temp_data["disk"]
                    data_point["fields"]["cpu"]= temp_data["cpu"]
                    data_point["time"] = temp_data["time-stamp"]
                    print(data_point)
                    datapoints.append(data_point)
            result = self.db.insert(datapoints)
        else:
        
            mes = "No host to collect the container stats"
        return mes
            

    def getHostDetails(host):
        
        HostDetails = rclient.HostDetails(host_ip)
        
        return HostDetails

    def checkpoint(self,creation_id,host_id):
        print("checkpoint started")
        startTime=time.time()
        payload = {}
             
        query="SELECT * FROM host WHERE host_id="+"'"+str(host_id)+"'"+";"
        print(query) 
        host=self.db.select(query)
        host_ip = list(host)[0][0]["host_ip"]

        query="SELECT * FROM CreatedContainers WHERE creation_id="+"'"+str(creation_id)+"'"+";"
        container=self.db.select(query)

        payload["opcode"] = 'checkpoint'
        payload["host_ip"] = host_ip
        payload["c_name"] = list(container)[0][0]["name"]
        payload["name"] = list(container)[0][0]["name"]
        payload["image"] = list(container)[0][0]["image"]
 
        rc = rclient.HandleContainer(payload,host_ip)
        print("checkpoint completed","response is",rc,"container:",list(container)[0][0]["name"],"host:",host_ip)
        print("Time taken for checkpoint",time.time()-startTime)
        return rc
    
    def migrate(self, cur_host,tar_host,creation_id):
        print("Migration started at time")
        startTime=time.time()
        payload = {}
        payload["opcode"] = 'checkpoint'
        
        # Get Container details
        query="SELECT * FROM CreatedContainers WHERE creation_id="+"'"+str(creation_id)+"'"+";"
        container = self.db.select(query)
       
        payload["c_name"] = list(container)[0][0]["name"]
        payload["name"] = list(container)[0][0]["name"]
        payload["image"] = list(container)[0][0]["image"]
        
        # Get current  host details
        currenthost = "SELECT * FROM host WHERE host_id="+"'"+str(cur_host)+"'"+";"
        current_host=self.db.select(currenthost)
        source_host_ip = list(current_host)[0][0]["host_ip"]
        payload["host_ip"] = source_host_ip
        payload["source_ip"] = source_host_ip
        # Get target host details
        targethost = "SELECT * FROM host WHERE host_id="+"'"+str(tar_host)+"'"+";"
        target_host=self.db.select(targethost)
        target_host_ip = list(target_host)[0][0]["host_ip"]
        payload["targetIP"] = target_host_ip
        
        rc = rclient.HandleContainer(payload,source_host_ip)
        print("Migrated from",source_host_ip,"to",target_host_ip,"for container:",list(container)[0][0]["name"],rc)
        print("Time taken for migration",time.time()-startTime)
        return rc
    
    def restore(self,json_body):

        host_id = json_body["fields"]["Host_id"]
        print("Restoring process started in host:",host_id)
        payload = {}
        query="SELECT * FROM host WHERE host_id="+"'"+str(host_id)+"'"+";"
        result = self.db.select(query)
        host_ip = list(result)[0][0]["host_ip"]
        image = json_body["fields"]["image"]
        cpu_shares = 2
        memory = '521m'
        payload["opcode"] = 'create'
        payload["image"] = image
        payload["host_ip"] = host_ip
        payload["cpu_shares"] = cpu_shares
        payload["memory"] = memory
        payload["name"] = json_body["fields"]["name"]
        rc = rclient.HandleContainer(payload,host_ip)
        print("Restore process completed:",rc)
        return rc