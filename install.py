from sys import version
from utils.ColorUtils import *

if float(version[0:3]) < 3.7:
	print(color.FAIL+'Python 3.7 or above required!'+color.ENDC)
	exit()

from os import mkdir, remove, startfile, path, system, environ

system('pip install -r requirements.txt')

import wget
from zipfile import ZipFile
import platform
from shutil import copyfile
import sys
import subprocess
from getpass import getpass

# Install Ansible
password = getpass(color.BOLD+'Please enter linux password:'+color.ENDC)

def run_cmd_pwd(cmd):
	system("bash -c \"echo "+password+" | sudo -S "+cmd+" -f\"")

def run_cmd(cmd):
	system("bash -c \""+cmd+"\"")

# Installing InfluxDB
if 'Windows' in platform.system():
	install_dir = 'InfluxDB'
	influxdb_install_path = 'C:/Program Files/' + install_dir
	if not path.exists(influxdb_install_path+'/influxdb-1.8.3-1/influxd.exe'):
		print(color.HEADER+'Installing InfluxDB'+color.ENDC)
		url = 'https://dl.influxdata.com/influxdb/releases/influxdb-1.8.3_windows_amd64.zip'
		mkdir(influxdb_install_path)
		filename = wget.download(url)
		zf = ZipFile(filename, 'r')
		zf.extractall(influxdb_install_path)
		zf.close()
		for folder in ['meta', 'data', 'wal']:
			mkdir(influxdb_install_path+'/'+folder)
		copyfile('framework/install_scripts/influxdb.conf', influxdb_install_path+'/influxdb-1.8.3-1/influxdb.conf')
		remove(filename)
		print('InfluxDB service runs as a separate front-end window. Please minimize this window.')
		startfile(influxdb_install_path+'/influxdb-1.8.3-1/influxd.exe')
elif 'Linux' in platform.system():
	run_cmd_pwd('apt install influxdb')

# Installing Vagrant
if 'Windows' in platform.system():
	trial = subprocess.run("where vagrant.exe", shell=True,stdout=subprocess.PIPE)
	stdout = trial.stdout.decode()
	if 'Cloud not find' in stdout:
		if sys.maxsize > 2**32: # 64-bit system
			link = 'https://releases.hashicorp.com/vagrant/2.2.10/vagrant_2.2.10_x86_64.msi'
		else: # 32-bit system
			link = 'https://releases.hashicorp.com/vagrant/2.2.10/vagrant_2.2.10_i686.msi'
		filename = wget.download(link)
		print('\n'+color.HEADER+'Please follow the prompts for installing Vagrant'+color.ENDC)
		startfile(filename)
		remove(filename)
elif 'Linux' in platform.system():
	run_cmd_pwd('apt install vagrant')

# Install VirtualBox
if 'Windows' in platform.system():
	trial = subprocess.run("where virtualbox.exe", shell=True,stdout=subprocess.PIPE)
	stdout = trial.stdout.decode()
	if 'Cloud not find' in stdout:
		link = 'https://download.virtualbox.org/virtualbox/6.0.24/VirtualBox-6.0.24-139119-Win.exe'
		filename = wget.download(link)
		print('\n'+color.HEADER+'Please follow the prompts for installing VirtualBox'+color.ENDC)
		startfile(filename)
		remove(filename)
elif 'Linux' in platform.system():
	run_cmd_pwd('apt install virtualbox')

# Install WSL if Windows
if 'Windows' in platform.system():
	trial = subprocess.run("where bash.exe", shell=True,stdout=subprocess.PIPE)
	stdout = trial.stdout.decode()
	if 'Cloud not find' in stdout:
		system("powershell.exe -ExecutionPolicy Bypass 'wsl-ubuntu-powershell-master/0_enable_wsl.ps1'")

# Copy SSH keys
ssh_dir = 'C:'+environ['homepath']+'\\.ssh' if 'Windows' in platform.system() else '~/.ssh'
if not path.exists(ssh_dir):
	mkdir(ssh_dir)
for filename in ['id_rsa', 'id_rsa.pub']:
	copy('framework/install_scripts/ssh_keys/'+filename, ssh_dir)

if not path.exists('framework/agent/DockerImages/'):
	mkdir('framework/agent/DockerImages/')

run_cmd_pwd("apt install ansible")

# Build docker images
print(color.HEADER+'Building Docker Images'+color.ENDC)
run_cmd("framework/install_scripts/install_docker.sh")

print(color.GREEN+"All packages installed."+color.ENDC)
