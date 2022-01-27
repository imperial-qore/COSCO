<h1 align="center">COSCO Framework</h1>
<div align="center">
  <a href="https://github.com/imperial-qore/COSCO/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCOSCO&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
   <a href="https://github.com/imperial-qore/COSCO/actions">
    <img src="https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg" alt="Actions Status">
  </a>
  </a>
   <a href="https://doi.org/10.5281/zenodo.4897944">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4897944.svg" alt="Zenodo">
  </a>
 <br>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo" alt="Docker pulls yolo">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx" alt="Docker pulls pocketsphinx">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas" alt="Docker pulls aeneas">
  </a>
 <br>
   <a href="https://gitpod.io/#https://github.com/imperial-qore/COSCO/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>
</div>

COSCO is an AI based coupled-simulation and container orchestration framework for integrated Edge, Fog and Cloud Computing Environments. It's a simple python based software solution, where academics or industrialists can develop, simulate, test and deploy their scheduling policies. Further, this repo presents a novel gradient-based optimization strategy using deep neural networks as surrogate functions and co-simulations to facilitate decision making.

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

Supported workloads: (Simulator) [Bitbrains](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains) and [Azure2017/2019](https://github.com/Azure/AzurePublicDataset); (Framework) [DeFog](https://github.com/qub-blesson/DeFog) and [AIoTBench](https://www.benchcouncil.org/aibench/aiotbench/index.html).

Our main COSCO work uses the Bitbrains and DeFog workloads. An extended work, MCDS (see `workflow` branch), accepted in IEEE TPDS uses scientific workflows. Check [paper](https://arxiv.org/abs/2112.07269) and [code](https://github.com/imperial-qore/COSCO/tree/workflow).

## Novel Scheduling Algorithms
We present two novel algorithms in this work: GOBI and GOBI*. GOBI uses a neural network as a surrogate model and gradient based optimization using backpropagation of gradients to input. With advances like cosine annealing and momentum allow us to converge to an optima quickly. Moreover, GOBI* leverages a coupled simulation engine like a digital-twin to further improve the surrogate accuracy and subsequently the scheduling decisions. Experiments conducted using real-world data on fog applications using the GOBI and GOBI* methods, show a significant improvement in terms of energy consumption, response time, Service Level Objective and scheduling time by up to 15, 40, 4, and 82 percent respectively when compared to the state-of-the-art algorithms.

## Supplementary video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/RZOWTj0rfBQ/0.jpg)](https://www.youtube.com/watch?v=RZOWTj0rfBQ)

A detailed course on using the COSCO framework for deep learning based scheduling (deep surrogate optimization and co-simulation) in fog environments is available as a [youtube playlist](https://www.youtube.com/playlist?list=PLN_nzHzuaOBQijEwy2Fy8c09-dWYVe4XO).
 
## Quick Start Guide
To run the COSCO framework, install required packages using
```bash
python3 install.py
```
To run the code with the required scheduler, modify line 106 of `main.py` to one of the several options including LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI.
```python
scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))
```

To run the simulator, use the following command
```bash
python3 main.py
```

## Gitpod
You can directly run tests on the results using a Gitpod Workspace without needing to install anything on your local machine. Click "Open in Gitpod" below and test the code by running `python3 main.py`.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/imperial-qore/COSCO/)

## Wiki
Access the [wiki](https://github.com/imperial-qore/COSCO/wiki) for detailed installation instructions, implementing a custom scheduler and replication of results. All execution traces and training data is available at [Zenodo](https://zenodo.org/record/4897944) under CC License.

## Links
| Items | Contents | 
| --- | --- |
| **Paper** | https://ieeexplore.ieee.org/document/9448450 (with the "Code Reviewed Badge") |
| **Pre-print** | https://arxiv.org/pdf/2104.14392.pdf |
| **Documentation** | https://github.com/imperial-qore/COSCO/wiki |
| **Video** | https://youtu.be/RZOWTj0rfBQ |
| **Tutorial** | https://www.youtube.com/playlist?list=PLN_nzHzuaOBQijEwy2Fy8c09-dWYVe4XO | 
| **Extensions** | QoS aware scheduling ([TPDS'22](https://arxiv.org/abs/2112.07269), [code](https://github.com/imperial-qore/COSCO/tree/workflow)), Energy aware sustainable computing ([JSS'21](https://www.sciencedirect.com/science/article/abs/pii/S0164121221002211), [code](https://github.com/imperial-qore/COSCO/tree/ggcn)), EdgeAI ([SIGMETRICS'21](https://arxiv.org/abs/2110.04841), [NeurIPS'21](https://arxiv.org/abs/2110.02912)) and fault-tolerance ([Infocom'22](https://arxiv.org/abs/2112.02292), [code](https://github.com/imperial-qore/PreGAN)) |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship, H2020-825040 (RADON) |

## Cite this work
Our work is published in IEEE TPDS journal. Cite using the following bibtex entry.
```bibtex
@article{tuli2022tranad,
  title={TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  journal={arXiv preprint arXiv:2201.07284},
  year={2022}
}
```
## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
