#!/bin/bash

sudo apt-get update
sudo apt-get -y install apt-transport-https ca-certificates curl criu software-properties-common python3-pip virtualenv python3-setuptools linux-tools-generic linux-tools-4.15.0-72-generic sysbench ioping
python3 -m pip install flask-restful inotify Flask psutil docker
sudo chmod +x $HOME/agent/agent.py

# Install Docker

sudo mkdir /etc/docker
sudo cp $HOME/agent/scripts/daemon.json /etc/docker
sudo apt-get -y install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
sudo groupadd docker 
sudo usermod -aG docker $USER
newgrp docker

# Setup Flask 

sudo cp ~/agent/scripts/flask.conf /etc/init.d/
sudo cp ~/agent/scripts/flask.service /lib/systemd/system/flask.service
sudo service flask start
sudo chmod +x ~/agent/scripts/delete.sh

# Load Docker images

sudo docker pull shreshthtuli/yolo
sudo docker pull shreshthtuli/pocketsphinx
sudo docker pull shreshthtuli/aeneas
