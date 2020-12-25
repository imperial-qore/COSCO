import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch
import torch.nn.functional as F
import math
import random

class NPNLinear(nn.Module):
    def positive_s(self, x, use_sigmoid = 0):
        if use_sigmoid == 0:
            y = torch.log(torch.exp(x) + 1)
        else:
            y = F.sigmoid(x)
        return y

    def positive_s_inv(self, x, use_sigmoid = 0):
        if use_sigmoid == 0:
            y = torch.log(torch.exp(x) - 1)
        else:
            y = - torch.log(1 / x - 1)
        return y

    def __init__(self, in_channels, out_channels, dual_input = True, init_type = 0):
        # init_type 0: normal, 1: mixture of delta distr'
        super(NPNLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dual_input = dual_input

        self.W_m = nn.Parameter(2 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * (torch.rand(in_channels, out_channels) - 0.5))
        if init_type == 0:
            W_s_init = 0.01 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * torch.rand(in_channels, out_channels)
        else:
            bern = torch.bernoulli(torch.ones(in_channels, out_channels) * 0.5)
            W_s_init = bern * math.exp(-2) + (1 - bern) * math.exp(-14)
            print(W_s_init[:4,:4])
        self.W_s_ = nn.Parameter(self.positive_s_inv(W_s_init, 0))

        self.bias_m = nn.Parameter(torch.zeros(out_channels))
        if init_type == 0:
            self.bias_s_ = nn.Parameter(torch.ones(out_channels) * (-10))
        else:
            bern = torch.bernoulli(torch.ones(out_channels) * 0.5)
            bias_s_init = bern * math.exp(-2) + (1 - bern) * math.exp(-14)
            self.bias_s_ = nn.Parameter(self.positive_s_inv(bias_s_init, 0))

    def forward(self, x):
        if self.dual_input:
            x_m, x_s = x
        else:
            x_m = x
            x_s = x.clone()
            x_s = 0 * x_s

        o_m = torch.mm(x_m, self.W_m)
        o_m = o_m + self.bias_m.expand_as(o_m)

        #W_s = torch.log(torch.exp(self.W_s_) + 1)
        #bias_s = torch.log(torch.exp(self.bias_s_) + 1)
        W_s = self.positive_s(self.W_s_, 0)
        bias_s = self.positive_s(self.bias_s_, 0)

        o_s = torch.mm(x_s, W_s) + torch.mm(x_s, self.W_m * self.W_m) + torch.mm(x_m * x_m, W_s)
        o_s = o_s + bias_s.expand_as(o_s)

        #print('bingo om os')
        #print(o_m.data)
        #print(o_s.data)

        return o_m, o_s

class NPNRelu(nn.Module):
    def __init__(self):
        super(NPNRelu, self).__init__()
        self.scale = math.sqrt(8/math.pi) 

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = F.sigmoid(self.scale * o_m * (o_s ** (-0.5))) * o_m + torch.sqrt(o_s) / math.sqrt(2 * math.pi) * torch.exp(-o_m ** 2 / (2 * o_s))
        a_s = F.sigmoid(self.scale * o_m * (o_s ** (-0.5))) * (o_m ** 2 + o_s) + o_m * torch.sqrt(o_s) / math.sqrt(2 * math.pi) * torch.exp(-o_m ** 2 / (2 * o_s)) - a_m ** 2  # mbr
        return a_m, a_s

class NPNSigmoid(nn.Module):
    def __init__(self):
        super(NPNSigmoid, self).__init__()
        self.xi_sq = math.pi / 8
        self.alpha = 4 - 2 * math.sqrt(2)
        self.beta = - math.log(math.sqrt(2) + 1)

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = F.sigmoid(o_m / (1 + self.xi_sq * o_s) ** 0.5)
        a_s = F.sigmoid(self.alpha * (o_m + self.beta) / (1 + self.xi_sq * self.alpha ** 2 * o_s) ** 0.5) - a_m ** 2
        return a_m, a_s

class NPNDropout(nn.Module):
    def __init__(self, rate):
        super(NPNDropout, self).__init__()
        self.dropout = nn.Dropout2d(p = rate)
    def forward(self, x):
        assert(len(x) == 2)
        if self.training:
            self.dropout.train()
        else:
            self.dropout.eval()
        x_m, x_s = x
        x_m = x_m.unsqueeze(2)
        x_s = x_s.unsqueeze(2)
        x_com = torch.cat((x_m, x_s), dim = 2)
        x_com = x_com.unsqueeze(3)
        x_com = self.dropout(x_com)
        y_m = x_com[:,:,0,0]
        y_s = x_com[:,:,1,0]
        return y_m, y_s

def NPNBCELoss(pred_m, pred_s, label):
    loss = -torch.sum((torch.log(pred_m + 1e-10) * label + torch.log(1 - pred_m + 1e-10) * (1 - label))/ (pred_s + 1e-10) - torch.log(pred_s+ 1e-10))
    return loss

def KL_BG(pred_m, pred_s, label):
    loss = 0.5 * torch.sum((1 - label) * (pred_m ** 2 / pred_s + torch.log(torch.clamp(math.pi * 2 * pred_s, min=1e-6))) + label * ((pred_m - 1) ** 2 / pred_s + torch.log(torch.clamp(math.pi * 2 * pred_s, min=1e-6)))) / pred_m.size()[0] # min = 1e-6
    return loss

def L2_loss(pred, label):
    loss = torch.sum((pred - label) ** 2)
    return loss

def KL_loss(pred, label):
    assert(len(pred) == 2)
    pred_m, pred_s = pred
    # print(((pred_m - label) ** 2) / (pred_s) , torch.log(pred_s))
    loss = 0.5 * torch.sum(10 * ((pred_m - label) ** 2) / (pred_s) + torch.log(pred_s)) # may need epsilon
    return loss

def multi_logistic_loss(pred, label):
    assert(len(label.size()) == 1)
    print('bingo type\n', label.data.type())
    print('bingo label\n', pred[:, label])
    log_prob = torch.sum(torch.log(1 - pred)) + torch.sum(log(pred[:, label.data]) - log(1 - pred[:, label.data]))
    return -log_prob

def RMSE(pred, label):
    loss = torch.mean(torch.sum((pred - label) ** 2, 1), 0) ** 0.5
    return loss
