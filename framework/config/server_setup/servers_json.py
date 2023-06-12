import json

def get_ips_from_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    servers = data["vlan"]["servers"]
    ips = [server["ip"] for server in servers]
    return ips

if __name__ == "__main__":
    ips = get_ips_from_json('/home/erfan/Project/COSCO/framework/config/VLAN_config.json')
    for ip in ips:
        print(ip)