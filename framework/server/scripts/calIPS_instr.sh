#! /bin/bash

sudo perf stat -o output.txt  -e instructions -e task-clock  sysbench --test=cpu run  &> /dev/null
instructions=$(cat output.txt | grep instructions |awk '{print $1;}'| while read spo; do echo $spo; done)
echo $instructions