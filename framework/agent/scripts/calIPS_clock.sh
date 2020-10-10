#! /bin/bash
cpus=$(nproc)
sudo perf stat -o output.txt  -e instructions -e task-clock  sysbench --num-threads=$cpus --test=cpu run  &> /dev/null
clock=$(cat output.txt | grep 'seconds time' |awk '{print $1;}'| while read spo; do echo $spo; done)
echo $clock

ioping -S64M -L -s4k -c 10 . > disk_read.txt
disk_read=$(cat disk_read.txt | grep 'completed')
echo $disk_read

ioping -S64M -L -s4k -W -c 10 . > disk_write.txt
disk_write=$(cat disk_write.txt | grep 'completed')
echo $disk_write

mkdir -p /tmp/ram
sudo mount -t tmpfs -o size=512M tmpfs /tmp/ram/
cd /tmp/ram/
ioping -S64M -L -s4k -c 10 . > ram_read.txt
ram_read=$(cat ram_read.txt | grep 'completed')
echo $ram_read

ioping -S64M -L -s4k -W -c 10 . > ram_write.txt
ram_write=$(cat ram_write.txt | grep 'completed')
echo $ram_write