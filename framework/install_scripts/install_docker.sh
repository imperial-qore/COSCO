# Installing docker

sudo mkdir /etc/docker
sudo cp framework/agent/scripts/daemon.json /etc/docker
sudo apt-get -y install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
sudo groupadd docker 
sudo usermod -aG docker $USER
newgrp docker

# Building Docker images

docker build -t yolo framework/workload/DockerImages/yolo/
docker build -t pocketsphinx framework/workload/DockerImages/PocketSphinx/
docker build -t aeneas framework/workload/DockerImages/Aeneas/

# Create tar files of built images

docker save -o framework/agent/DockerImages/yolo.tar.gz yolo
docker save -o framework/agent/DockerImages/pocketsphinx.tar.gz pocketsphinx
docker save -o framework/agent/DockerImages/aeneas.tar.gz aeneas