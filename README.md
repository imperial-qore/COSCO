[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/COSCO/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCOSCO&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/SimpleFogSim/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas)

# Supplementary for th GOSH paper

## Novel Scheduling Algorithms
We present two novel algorithms in this work: GOSH and GOSH*. GOSH uses heteroscedastic deep surrogate model with uncertainty based exploration and higher-order optimization to quickly converge to near optimal scheduling decisions. GOSH* is the co-simulator based extension of GOSH.


## Quick Start Guide
To run the COSCO framework, install required packages using
```bash
python3 install.py
```
To run the code with the required scheduler, modify line 104 of `main.py` to one of the several options including LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI.
```python
scheduler = GOSHScheduler('energy_latency_'+str(HOSTS))
```

To run the simulator, use the following command
```bash
python3 main.py
```

## Wiki
Access the [wiki](https://github.com/imperial-qore/COSCO/wiki) for detailed installation instructions, implementing a custom scheduler and replication of results. All execution traces and training data is available at [Zenodo](https://zenodo.org/record/4897944) under CC License.

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
