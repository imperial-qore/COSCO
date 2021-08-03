import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase
import dgl.function as fn
from .constants import *

class HeteroRGCNLayer(nn.Module):
    # The GCN layer which has been borrowed from DGL. Contains a link to the source module.
    # Source = https://docs.dgl.ai/en/0.4.x/tutorials/hetero/1_basics.html
    def __init__(self, in_size, out_size, etypes, activation):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name : nn.Linear(in_size, out_size) for name in etypes})
        self.activation = activation

    def forward(self, G, features):
        funcs = {}
        for etype in G.etypes:
            Wh = self.weight[etype](features)
            G.nodes['object'].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return self.activation(G.nodes['object'].data['h'])

class GatedRGCNLayer(nn.Module):
    # The Gated GCN layer which has been borrowed from DGL. Contains a link to the source module.
    # Source = https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatedgraphconv.html#GatedGraphConv
    def __init__(self, in_size, out_size, activation):
        super(GatedRGCNLayer, self).__init__()
        self.weight = nn.Linear(in_size, out_size)
        self.reduce = nn.Linear(in_size, out_size)
        self.activation = activation
        self.gru = LayerNormGRUCell(out_size, out_size, bias=True)

    def forward(self, G, features):
        funcs = {}; feat = self.activation(self.reduce(features))
        for _ in range(N_TIMESEPS):
            Wh = self.weight(features)
            G.ndata['Wh'] = Wh
            G.update_all(fn.copy_u('Wh', 'm'), fn.mean('m', 'h'))
            feat = self.gru(G.ndata['h'], feat)
        return self.activation(feat)

class GatedHeteroRGCNLayer(nn.Module):
    # The Gated GCN layer which has been borrowed from DGL. Contains a link to the source module.
    # Source = https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatedgraphconv.html#GatedGraphConv
    def __init__(self, in_size, out_size, etypes, activation):
        super(GatedHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name : nn.Linear(in_size, out_size) for name in etypes})
        self.reduce = nn.Linear(in_size, out_size)
        self.activation = activation
        self.gru = LayerNormGRUCell(out_size, out_size, bias=True)

    def forward(self, G, features):
        funcs = {}; feat = self.activation(self.reduce(features))
        for _ in range(N_TIMESEPS):
            for etype in G.etypes:
                Wh = self.weight[etype](features)
                G.nodes['object'].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
            G.multi_update_all(funcs, 'sum')
            feat = self.gru(G.nodes['object'].data['h'], feat)
        return self.activation(feat)

class LayerNormGRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

        self.ln_resetgate = nn.LayerNorm(hidden_size)
        self.ln_inputgate = nn.LayerNorm(hidden_size)
        self.ln_newgate = nn.LayerNorm(hidden_size)
        self.ln = {
            'resetgate': self.ln_resetgate,
            'inputgate': self.ln_inputgate,
            'newgate': self.ln_newgate,
        }

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)
        return self._LayerNormGRUCell(
            input, hx,
            self.weight_ih, self.weight_hh, self.ln,
            self.bias_ih, self.bias_hh,
        )

    def _LayerNormGRUCell(self, input, hidden, w_ih, w_hh, ln, b_ih=None, b_hh=None):
    	
	    gi = F.linear(input, w_ih, b_ih)
	    gh = F.linear(hidden, w_hh, b_hh)
	    i_r, i_i, i_n = gi.chunk(3, 1)
	    h_r, h_i, h_n = gh.chunk(3, 1)

	    # use layernorm here
	    resetgate = torch.sigmoid(ln['resetgate'](i_r + h_r))
	    inputgate = torch.sigmoid(ln['inputgate'](i_i + h_i))
	    newgate = torch.tanh(ln['newgate'](i_n + resetgate * h_n))
	    hy = newgate + inputgate * (hidden - newgate)

	    return hy

class fc_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn):
        super(fc_block, self).__init__()

        block = nn.Sequential()
        block.add_module('linear', nn.Linear(in_channels, out_channels))
        if norm:
            block.add_module('batchnorm', nn.BatchNorm1d(out_channels))
        if activation_fn is not None:
            block.add_module('activation', activation_fn())

        self.block = block

    def forward(self, x):
        return self.block(x)