#!/bin/bash

sudo apt-get update
sudo apt-get install -f apt-transport-https ca-certificates curl criu software-properties-common python3-pip virtualenv python3-setuptools linux-tools-generic linux-tools-4.15.0-72-generic sysbench
pip install flask-restful inotify Flask psutil
sudo chmod +x $HOME/agent/agent.py
sudo mkdir /etc/docker
sudo cp $HOME/agent/daemon.json /etc/docker
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo groupadd docker 
sudo usermod -aG docker $USER
newgrp docker

