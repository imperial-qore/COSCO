from sys import version
from utils.ColorUtils import *

if float(version[0:3]) < 3.6:
	print(color.FAIL+'Python 3.6 or above required!'+color.ENDC)
	exit()

import platform
from os import mkdir, makedirs, remove, path, system, environ, getcwd
if 'Windows' in platform.system():
	from os import startfile

system('python3 -m pip install -r requirements.txt')
system('python3 -m pip install -U scikit-learn')
system('python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html')

import wget
from zipfile import ZipFile
from shutil import copy
import sys
import os
import subprocess
from getpass import getpass

# Install Ansible
password = getpass(color.BOLD+'Please enter linux password:'+color.ENDC)

def run_cmd_pwd(cmd, password):
    os.system("bash -c \"echo "+password+" | sudo -S "+cmd+" &> /dev/null\"")

def run_cmd(cmd):
	system("bash -c \""+cmd+"\"")

# Install WSL if Windows
if 'Windows' in platform.system():
	trial = subprocess.run("where bash.exe", shell=True, stderr=subprocess.PIPE)
	stdout = trial.stderr.decode()
	if 'Could not find' in stdout:
		system("powershell.exe -ExecutionPolicy Bypass 'framework/install_scripts/install_wsl.ps1'")

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
	run_cmd_pwd('apt install influxdb', password)

# Installing Vagrant
if 'Windows' in platform.system():
	trial = subprocess.run("where vagrant.exe", shell=True, stderr=subprocess.PIPE)
	stdout = trial.stderr.decode()
	if 'Could not find' in stdout:
		if sys.maxsize > 2**32: # 64-bit system
			link = 'https://releases.hashicorp.com/vagrant/2.2.10/vagrant_2.2.10_x86_64.msi'
		else: # 32-bit system
			link = 'https://releases.hashicorp.com/vagrant/2.2.10/vagrant_2.2.10_i686.msi'
		filename = wget.download(link)
		print('\n'+color.HEADER+'Please follow the prompts for installing Vagrant'+color.ENDC)
		subprocess.call([getcwd()+'/'+filename], shell=True)
		remove(filename)
	set_disk = subprocess.run("setx VAGRANT_EXPERIMENTAL \"disks\"", shell=True, stderr=subprocess.PIPE)
elif 'Linux' in platform.system():
	run_cmd_pwd('apt install vagrant', password)
	set_disk = subprocess.run("export VAGRANT_EXPERIMENTAL=disks", shell=True, stderr=subprocess.PIPE)

# Install VirtualBox
if 'Windows' in platform.system():
	trial = subprocess.run("where virtualbox.exe", shell=True, stderr=subprocess.PIPE)
	stdout = trial.stderr.decode()
	if 'Could not find' in stdout:
		link = 'https://download.virtualbox.org/virtualbox/6.0.24/VirtualBox-6.0.24-139119-Win.exe'
		filename = wget.download(link)
		print('\n'+color.HEADER+'Please follow the prompts for installing VirtualBox'+color.ENDC)
		subprocess.call([getcwd()+'/'+filename], shell=True)
		remove(filename)
elif 'Linux' in platform.system():
	run_cmd_pwd('apt install virtualbox', password)

# Copy SSH keys
ssh_dir = 'C:'+environ['homepath']+'\\.ssh' if 'Windows' in platform.system() else environ['HOME']+'/.ssh'
if not path.exists(ssh_dir):
	makedirs(ssh_dir)
for filename in ['id_rsa', 'id_rsa.pub']:
	copy('framework/install_scripts/ssh_keys/'+filename, ssh_dir)

run_cmd_pwd("apt install ansible", password)
run_cmd_pwd("apt install dos2unix", password)
run_cmd_pwd("sudo chmod 400 framework/install_scripts/ssh_keys/id_rsa", password)

print(color.GREEN+"All packages installed."+color.ENDC)
