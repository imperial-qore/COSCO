import requests
import json
from string import Template
from enum import Enum
import framework.server.common.codes as codes

class InvalidResponse(Enum):
	text = 'invalid'

REQUEST_TEMPLATE = Template("http://$HOST:$PORT/request")
CONTAINER_PORT = 8081

def HandleRequest(payload, host, framework):
    resp = ""
    clientUrl = REQUEST_TEMPLATE.substitute(HOST = host, PORT = CONTAINER_PORT)
    print(json.dumps(payload), clientUrl)
    try:
        resp = requests.get(clientUrl, data=json.dumps(payload), timeout = 420)
    except Exception as e:
        resp = InvalidResponse()
        resp.text = e + ' for payload = ' + json.dumps(payload)
    framework.logger.debug("Response received by server from agent "+host+" : "+resp.text)
    return json.loads(resp.text)
