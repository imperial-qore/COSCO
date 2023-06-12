sudo vim /etc/netplan/50-cloud-init.yaml

		addresses:
                            - 8.8.8.8

sudo netplan apply

sudo mkdir /etc/systemd/system/docker.service.d
sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf

[Service]
Environment="HTTP_PROXY=http://[username]:[password]@[proxy_address]:3832"
Environment="HTTPS_PROXY=http://[username]:[password`]@[proxy_address]:3832"

sudo systemctl daemon-reload
sudo systemctl restart docker

docker pull shreshthtuli/yolo
docker pull shreshthtuli/pocketsphinx
docker pull shreshthtuli/aeneas
