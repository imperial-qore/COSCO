import requests
import json
from string import Template

import framework.server.common.codes as codes

REQUEST_TEMPLATE = Template("http://$HOST:$PORT/request")
CONTAINER_PORT = 8081

def HandleRequest(payload, host, framework):
    resp = ""
    clientUrl = REQUEST_TEMPLATE.substitute(HOST = host, PORT = CONTAINER_PORT)
    try:
        resp = requests.get(clientUrl, data=json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        resp = codes.FAILED
    framework.logger.debug("Response received by server from agent "+host+" : "+resp.text)
    return json.loads(resp.text)
