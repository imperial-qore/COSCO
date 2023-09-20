

"""
"""

from .Scheduler import *
from .TRL.train import load_model
from .TRL.src.models import TransformerScheduler, EncoderScheduler

import pickle
import torch
from os import path, mkdir
from copy import deepcopy
from random import sample, randint
import numpy as np

class TRLScheduler(Scheduler):
    def __init__(self, data_type, training=True):
        self.data_type = data_type
        self.hosts = int(data_type.split('_')[-1])
        self.encoder_max_length = int(2.5*self.hosts) + 3
        self.decoder_max_length = int(1.5*self.hosts) + 2
        self.probs_len = int(1.5*self.hosts**2) + 1
        self.model = EncoderScheduler(self.encoder_max_length, 
                                          self.decoder_max_length, 5, 50*5, 
                                          self.prob_len, self.hosts)
        
        self.model = load_model(self.model)
        
    def run_transformer(self):
        SOD = [[1]*1]
        EOD = [[2]*1]
        PAD = [0]*1        
        #TODO add more info
        contInfo = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]#, c.getRAM()[0], c.getDisk()[0], c.createAt, c.startAt
        contInfo = np.array(self.padding(contInfo, int(1.5*self.hosts), PAD, 
                                         pad_left=True))
        hostInfo = np.array([(host.getCPU()) for host in self.env.hostlist])# , host.getRAMAvailable()[0], host.getDiskAvailable(), 0, 0
        mainInfo = np.append(np.append(np.append(SOD, contInfo, 0), EOD, 0), 
                             np.append(hostInfo, EOD, 0), 0)
        
        allocateInfo = [[1]*10]; prev_alloc = {}; PAD1 = [0]*10; step = 0
        for c in self.env.containerlist:
            if c: hId = c.getHostID(); prev_alloc[c.id] = hId
            if c and hId != -1: 
                step += 1
                host = c.getHost()
                allocateInfo.append((c.getBaseIPS(), c.getRAM()[0], c.getDisk()[0], \
                                      c.createAt, c.startAt, host.getIPSAvailable(), \
                                          host.getRAMAvailable()[0],  host.getDiskAvailable(), \
                                              0, 0))
        allocateInfo = np.array(self.padding(allocateInfo, int(1.5*self.hosts)+1, 
                                             PAD1, pad_left=False))
        mainInfo = torch.tensor(mainInfo, dtype=torch.float32, 
                                requires_grad=True).unsqueeze(0)
        allocateInfo = torch.tensor(allocateInfo, dtype=torch.float32, 
                                    requires_grad=True).unsqueeze(0)
        decisions, actions, log_probs, encoder_inputs, steps, steps, decoder_inputs = \
            self.model.generateSteps(self.env, mainInfo, allocateInfo, step)
        
        return decisions, actions, log_probs, mainInfo, encoder_inputs, steps, decoder_inputs 
        
    def padding(self, sequence, final_length, padding_token, pad_left = True):
        pads = [padding_token] * (final_length - len(sequence))
        if pad_left: return np.appen(pads, sequence , 0)  
        else: return np.appen(sequence, pads, 0)   
    
    def selection(self):
        return []

    def placement(self, containerIDs):
        decision, _, _, _, _ = self.run_GOBI()
        return decision