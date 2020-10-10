[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/COSCO/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCOSCO&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/SimpleFogSim/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas)

# COSCO Framework
Coupled Simulation and Container Orchestration Framework for integrated Edge and Cloud Computing Environments

## Installation
- sudo python3 install.py
- python3 install.py (run in windows powershell (as admin))

## To be done by the user:
* Set environment type based on VMs/PMs of Datacenter object
* Power model required to be implemented in 'framework/node/Powermodels' or set from existing
* Config.json needs to be updated  --> based on environment
* Update based on arch --> 'framework/server/scripts/instructions_arch.json'
* Update SLA requirement based on application requirements --> 'framework/workload/\*.py'
