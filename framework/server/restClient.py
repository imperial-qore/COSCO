import requests
import json
import sys
from string import Template

import framework.server.common.codes as codes

OPERATE_CONTAINER = Template("http://$HOST:$PORT/container")
OPERATE_HOST = Template("http://$HOST:$PORT/host")


def HandleContainer(payload,host):
    resp = ""
    port = 8081
    clientUrl = OPERATE_CONTAINER.substitute(HOST = host, PORT = port)
    
    try:
        resp = requests.post(clientUrl, data=json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        resp = codes.FAILED
    return resp

def HostDetails(payload,host):
    resp = ""
    port = 8081
    clientUrl = OPERATE_HOST.substitute(HOST = host, PORT = port)
    print(clientUrl)
        #payload = {"opcode": "create", "params": containerCfg}
    rc = codes.SUCCESS  # TODO: This is not used
    try:
        resp = requests.post(clientUrl, data=json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        rc = codes.FAILED
    return resp


