[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/COSCO/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCOSCO&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/SimpleFogSim/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas)

# COSCO Framework

COSCO is an AI based coupled-simulation and container orchestration framework for integrated Edge, Fog and Cloud Computing Environments.  It's a simple python based software solution, where academics or industrialists  can develop, simulate, test and deploy their scheduling policies. 

<img src="https://github.com/imperial-qore/COSCO/blob/master/wiki/COSCO.jpg" width="900" align="middle">


## Advantages of COSCO
1. Hassle free development of AI based scheduling algorithms in integrated edge, fog and cloud infrastructures.
2. Provides seamless integration of scheduling policies with simulated back-end for enhanced decision making.
3. Supports container migration physical deployments (not supported by other frameworks) using CRIU utility.
4. Multiple deployment support as per needs of the developers. (Vagrant VM testbed, VLAN Fog environment, Cloud based deployment using Azure/AWS/OpenStack)
5. Equipped with a smart real-time graph generation of utilization metrics using InfluxDB and Grafana.
6. Real time metrics monitoring, logging and consolidated graph generation using custom Stats logger.

The basic architecture of COSCO has two main packages: <br>
**Simulator:** It's a discrete event simulator and runs in a standalone system. <br>
**Framework:** It’s a kind of tool to test the scheduling algorithms in a physical(real time) fog/cloud environment with real world applications.

## Novel Scheduling Algorithms
We present two novel algorithms in this work: GOBI and GOBI*. GOBI uses a neural network as a surrogate model and gradient based optimization using backpropagation of gradients to input. With advances like cosine annealing and momentum allow us to converge to an optima quickly. Moreover, GOBI* leverages a coupled simulation engine like a digital-twin to further improve the surrogate accuracy and subsequently the scheduling decisions. Experiments conducted using real-world data on fog applications using the GOBI and GOBI* methods, show a significant improvement in terms of energy consumption, response time, Service Level Objective and scheduling time by up to 15, 40, 4, and 82 percent respectively when compared to the state-of-the-art algorithms.

## Supplementary video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/RZOWTj0rfBQ/0.jpg)](https://www.youtube.com/watch?v=RZOWTj0rfBQ)
  
## Quick Start Guide
To run the COSCO framework, install required packages using
```bash
pip3 install -r requirements.txt
```
To run the code with the required scheduler, modify line 104 of `main.py` to one of the several options including LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI.
```python
scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))
```

To run the simulator, use the following command
```bash
python3 main.py
```

## Wiki
Access the [wiki](https://github.com/imperial-qore/COSCO/wiki) for detailed installation instructions, implementing a custom scheduler and replication of results. All execution traces and training data is available at [Zenodo](https://zenodo.org/record/4897944) under CC License.

## Arxiv preprint
https://arxiv.org/abs/2104.14392.

## Cite this work
```
@article{tuli2021cosco,
  title={COSCO: Container Orchestration using Co-Simulation and Gradient Based Optimization for Fog Computing Environments},
  author={Tuli, Shreshth and Poojara, Shivananda and Srirama, Satish and Casale, Giuliano and Jennings, Nicholas R},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  year={2021}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
