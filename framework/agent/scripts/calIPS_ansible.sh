#! /bin/bash
sudo perf stat -o output.txt  -e instructions -e task-clock  sysbench --test=cpu run  &> /dev/null
clock=$(cat output.txt | grep task-clock |awk '{print $1;}'| while read spo; do echo $spo; done)
instructions=$(cat output.txt | grep instructions |awk '{print $1;}'| while read spo; do echo $spo; done)
echo ($instructions/($clock * 1000))

ioping -S64M -L -s4k -W -c 10 . > disk_read.txt
disk_read=$(cat disk_read.txt | grep 'completed' |awk '{print $10;}'| while read spo; do echo $spo; done)
disk_read=$(echo "$disk_read" | awk '{print ($1)*2048}')
echo $disk_read

ioping -A -D -s16k -c 10 . > disk_write.txt
disk_write=$(cat disk_write.txt | grep 'completed' |awk '{print $10;}'| while read spo; do echo $spo; done)
disk_write=$(echo "$disk_write" | awk '{print ($1)*2048}')
echo $disk_write

mkdir -p /tmp/ram
sudo mount -t tmpfs -o size=512M tmpfs /tmp/ram/
cd /tmp/ram/
ioping -S64M -L -s4k -W -c 10 . > ~/agent/ram_read.txt
ram_read=$(cat  ~/agent/ram_read.txt | grep 'completed' |awk '{print $10;}'| while read spo; do echo $spo; done)
ram_read=$(echo "$ram_read" | awk '{print ($1)*1073.74}')
echo $ram_read | perl -ne 'printf "%d\n", $_;'


ioping -A -s16k -c 10 . > ~/agent/ram_write.txt
ram_write=$(cat  ~/agent/ram_write.txt | grep 'completed' |awk '{print $10;}'| while read spo; do echo $spo; done)
ram_write=$(echo "$ram_write" | awk '{print ($1)*1073.74}')
echo $ram_write | perl -ne 'printf "%d\n", $_;'