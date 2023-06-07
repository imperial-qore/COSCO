#!/bin/bash

password=123ansible

# Call the Python script and store the output (IP addresses)
ips=$(python servers_json.py)

# Loop over each IP address
for ip in $ips
do
    echo "Starting process for server with IP: $ip"

    # Connect to the server as 'ubuntu', create the 'ansible' user with sudo access
    ssh -o StrictHostKeyChecking=no ubuntu@$ip "
        sudo useradd -m ansible 
        echo 'ansible:${password}' | sudo chpasswd
        sudo bash -c 'echo \"ansible ALL=(ALL:ALL) ALL\" >> /etc/sudoers'
    "

    echo "User creation completed for server with IP: $ip"

    # Test the connection with new 'ansible' user
    sshpass -p $password ssh -o StrictHostKeyChecking=no -q ansible@$ip exit

    if [ $? -eq 0 ]; then
        echo "Successfully connected to server with IP: $ip using 'ansible' user"
    else
        echo "Failed to connect to server with IP: $ip using 'ansible' user"
    fi

    echo "Completed process for server with IP: $ip"
done
