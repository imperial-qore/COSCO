import os
import logging
import json
import re
from subprocess import call, Popen, PIPE
from getpass import getpass
from utils.Utils import *

def setup(cfg):
    # For ansible setup
    host = []
    num_hosts = len(cfg["servers"])
    for i in range(num_hosts):
            logging.debug("Creating enviornment with configuration file as  :{}".format(cfg))
            typeID = i%2 # np.random.randint(0,3) # i%3 #
            flavor = instance_type[typeID]
            vm = {}
            vm["flavor"]=flavor
            vm["name"]="SimpeE-Worker-"+str(i)
            host.append(vm)
            cfg["hosts"]=host    
    cfg = json.dumps(cfg)   

    call(["ansible-playbook","playbooks/client.yml","-e",cfg])

# VLAN setup functions

def run_cmd_pwd(cmd, password):
    os.system("bash -c \"echo "+password+" | sudo -S "+cmd+" &> /dev/null\"")

def run_cmd(cmd):
    os.system("bash -c \""+cmd+"\"")

def setupVLANEnvironment(cfg, mode):
    with open(cfg, "r") as f:
        config = json.load(f)
    HOST_IPS = [server['ip'] for server in config['vlan']['servers']]
    if mode in [0, 1]:
        MAIN_DIR = os.getcwd().replace('\\', '/').replace('C:', '/mnt/c')
        password = getpass(color.BOLD+'Please enter linux password:'+color.ENDC)
        run_cmd_pwd("rm /etc/ansible/hosts", password)
        run_cmd_pwd("cp framework/install_scripts/ssh_keys/id_rsa ~/id_rsa", password)
        run_cmd_pwd("cp framework/install_scripts/ssh_keys/id_rsa.pub ~/id_rsa.pub", password)
        with open("framework/config/hosts", "w") as f:
            f.write("[agents]\n")
            for ip in HOST_IPS:
                f.write(ip+" ansible_ssh_private_key_file=~/id_rsa ansible_ssh_user=ansible\n")
        run_cmd_pwd("cp framework/config/hosts /etc/ansible/hosts", password)
        run_cmd_pwd("cp framework/config/ansible.cfg /etc/ansible/ansible.cfg", password)
        run_cmd("ansible-playbook framework/config/VLAN_ansible.yml")
    uname = "ansible"
    for ip in HOST_IPS:
        res = call(["ssh", "-o", "StrictHostKeyChecking=no", "-i", "framework/install_scripts/ssh_keys/id_rsa", uname+"@"+ip, "~/agent/scripts/delete.sh"], shell=True, stdout=PIPE, stderr=PIPE)  
        res = call(["ssh", "-o", "StrictHostKeyChecking=no", "-i", "framework/install_scripts/ssh_keys/id_rsa", "-t", uname+"@"+ip, "sudo service docker restart"], shell=True, stdout=PIPE, stderr=PIPE)  
    return HOST_IPS

def destroyVLANEnvironment(cfg, mode):
    return

# Vagrant setup functions

def setupVagrantEnvironment(cfg, mode):
    with open(cfg, "r") as f:
        config = json.load(f)
    if mode in [0, 1]:
        MAIN_DIR = os.getcwd().replace('\\', '/')
        os.chdir(r"framework/config/") 
        with open('Vagrantfile', 'r') as file:
            data = file.read()
        custom_list = "servers=[\n"
        host_ip = []
        for i, datapoint in enumerate(config['vagrant']['servers']):
            custom_list += "\t{\n\t\t:hostname => 'vm"+str(i+1)+"',\n\t\t:ip => '192.168.0."+str(i+2)+"',\n\t\t:box => '"
            custom_list += config['vagrant']['box']
            custom_list += "',\n\t\t:ram => "+str(datapoint['ram'])+",\n\t\t:cpu => "+str(datapoint['cpu'])+",\n\t\t:disk => '"+str(datapoint['disk'])+"GB'\n\t}"
            host_ip.append("192.168.0."+str(i+2))
            if i != len(config['vagrant']['servers']) - 1:
                custom_list += ","
            custom_list += "\n"
        custom_list += "]\n\n"
        data = re.sub(r"servers=\[((.|\n)*)agent_path=", custom_list+"\nagent_path=", data)
        data = re.sub(r"agent_path=((.|\n)*)Vagrant", "agent_path='"+MAIN_DIR+"/framework/agent'\n\nVagrant", data)
        with open('Vagrantfile', 'w') as file:
            file.write(data)
        call(["vagrant", "up", "--parallel"])
        os.chdir("../../") 
        return host_ip
    HOST_IPS = ['192.168.0.'+str(i+2) for i in range(len(config['vagrant']['servers']))]
    uname = "vagrant"
    for ip in HOST_IPS:
        res = call(["ssh", "-o", "StrictHostKeyChecking=no", "-i", "framework/install_scripts/ssh_keys/id_rsa", uname+"@"+ip, "~/agent/scripts/delete.sh"], shell=True, stdout=PIPE, stderr=PIPE)  
    return HOST_IPS

def destroyVagrantEnvironment(cfg, mode):
    if mode in [0, 3]:
        call(["vagrant", "destroy -f"])
