curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# install prerequisites
sudo apt-get -y install python3-venv python3-pip
export PATH=$PATH:~/.local/bin

pip3 install opera ansible