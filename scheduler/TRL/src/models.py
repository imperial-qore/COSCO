"""
"""
import torch
import math
import numpy as np
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from typing import Optional, List
from copy import deepcopy
from torch.distributions.categorical import Categorical

class PositionalEncoding(nn.Module):
    """
    """
    def __init__(self, embed_dimension, max_length, device):
        super().__init__()
        self.embed_dimension = embed_dimension
        self.max_length = max_length
        self.positionalEncoding = self.build_positional_encoding()
        self.device = device
        
    def build_positional_encoding (self):
        positional_encoding = np.zeros((self.max_length, self.embed_dimension))
        for pos in range(self.max_length):
            for i in range(0, self.embed_dimension, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.embed_dimension)))
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dimension)))
        return torch.from_numpy(positional_encoding).float()
    
    def forward (self, x):
        return x + self.positionalEncoding[:x.size(1), :].to(self.device)

class TransformerScheduler (nn.Module):
    def __init__(
        self,
        encoder_max_length: int,
        decoder_max_length: int,
        input_dim: int,
        output_dim: int,
        prob_len: int,
        host_num: int,
        device: torch.device = torch.device("cpu"),
        name = 'transformer',
    ):
        super().__init__()
        self.name = name
        self.device = device
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prob_len = prob_len
        self.host_num = host_num
        
        self.en_embed = nn.Linear(self.input_dim, self.output_dim, device=self.device)
        self.de_embed = nn.Linear(2*self.input_dim, self.output_dim, device=self.device)

        self.en_position_encode = PositionalEncoding(self.output_dim, 
                                                     self.encoder_max_length,
                                                     self.device)
        self.de_position_encode = PositionalEncoding(self.output_dim, 
                                                     self.decoder_max_length,
                                                     self.device)
        
        encoder_layers = TransformerEncoderLayer(
            d_model= self.output_dim,
            nhead= 1,
            dim_feedforward= 16,
            dropout= .1,
            batch_first= True,
        )
        self.encoder = TransformerEncoder(
            encoder_layers, 6
        )
        decoder_layers = TransformerDecoderLayer(
            d_model= self.output_dim,
            nhead= 1,
            dim_feedforward= 16,
            dropout= .1,
            batch_first= True,
        )
        self.decoder = TransformerDecoder(
            decoder_layers, 8
        )
        
        
        self.outer = nn.Linear(self.output_dim, self.prob_len, device=self.device)
            
        
        self.softmax = nn.Softmax(dim=-1)
    
    def generateSteps (
        self,
        encoder_in: torch.tensor,
        decoder_in: torch.tensor, 
        step: int,     
    ):
        tgt_mask = torch.tril(torch.ones(1*decoder_in.size(0),
                                         self.decoder_max_length,
                                         self.decoder_max_length) == 1) 
        tgt_mask = tgt_mask.float() 
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(self.device)
        
        decoder_padding_mask = None
        actions = np.zeros((0,1), dtype=int)
        log_probs = np.zeros((0,1), dtype=int)
        decoder_inputs=[]; decisions=[]
        for s in range(0, self.decoder_max_length-1):
            inner_step = self.decoder_max_length-1 if s+step+1 > self.decoder_max_length-1 \
                else s+step+1
            next_generate = self.forward(encoder_in, decoder_in, tgt_mask,
                                         decoder_padding_mask, inner_step)
            next_dist = Categorical(next_generate)
            next_decision = next_dist.sample()
            next_prob = next_dist.log_prob(next_decision).unsqueeze(0).cpu().detach().numpy()
            decoder_inputs.append(decoder_in)
            log_probs = np.append(log_probs, next_prob, 0)
            actions = np.append(actions, next_decision, 0)
            if next_decision == self.prob_len-1:
                #TODO
                break
            #TODO check allocated last beacuase of getting the change 
            cont_dec = int(next_decision / self.host_num) 
            host_dec = int(next_decision % self.host_num)
            decisions.append((cont_dec, host_dec))
            if s+step+1 > self.decoder_max_length-1:
                decoder_in=torch.cat([decoder_in[:,1:], torch.cat([encoder_in[:,cont_dec+1],
                                                                   encoder_in[:,-(self.host_num-host_dec+1)]], 1).unsqueeze(0)],1)
            else:
                decoder_in[:, inner_step]=torch.cat([encoder_in[:,cont_dec+1], 
                                                     encoder_in[:,-(self.host_num-host_dec+1)]], 1)
            
        return decisions, actions, log_probs, torch.cat(decoder_inputs, 0)
            
    def forward (
        self,
        encoder_in: torch.tensor,
        decoder_in: torch.tensor,
        tgt_mask: torch.tensor,
        decoder_padding_mask: Optional[torch.tensor] = None,
        step: Optional[int] = None,
    ):
        
        encoder_embedding = self.en_embed(encoder_in)* math.sqrt(self.output_dim)
        decoder_embedding = self.de_embed(decoder_in)* math.sqrt(self.output_dim)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        encod = self.encoder(self.en_position_encode(encoder_embedding))

        transformer_out = self.decoder(self.de_position_encode(decoder_embedding), 
                                       encod, tgt_mask=tgt_mask, 
                                       tgt_key_padding_mask=decoder_padding_mask)
        
        pos = torch.cat([step.unsqueeze(0)]*self.output_dim,0).T.unsqueeze(1).to(self.device)
        out = transformer_out.gather(1,pos).squeeze(1)
        
        return self.softmax(self.outer(out))
    

class EncoderScheduler (nn.Module):
    def __init__(
        self,
        encoder_max_length: int,
        decoder_max_length: int,
        input_dim: int,
        output_dim: int,
        prob_len: int,
        host_num: int,
        device: torch.device = torch.device("cpu"),
        name = 'EncoderScheduler',
    ):
        super().__init__()
        self.name = name
        self.device = device
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prob_len = prob_len
        self.host_num = host_num
        
        self.en_embed = nn.Linear(self.config.input_dim, self.config.output_dim).to(self.device)
        
        
        self.en_position_encode = PositionalEncoding(self.output_dim, 
                                                     self.encoder_max_length,
                                                     self.device)
        
        encoder_layers = TransformerEncoderLayer(
            d_model= self.output_dim,
            nhead= 1,
            dim_feedforward= 16,
            dropout= .1,
            batch_first= True,
        )
        self.encoder = TransformerEncoder(
            encoder_layers, 6
        )
        
        self.flatten = nn.Flatten().to(self.device)
        
        input_dim = (self.encoder_max_length) * self.output_dim
        self.outer = nn.Linear(self.output_dim, self.prob_len, device=self.device)
    
        self.softmax = nn.Softmax(dim=1).to(self.device)
    
    def generateSteps (
        self,
        env,
        encoder_in: torch.tensor,
        decoder_in: torch.tensor, 
        step: int,     
    ):
        pad=torch.tensor([[0]*encoder_in.size(2)]).unsqueeze(0)
        actions = []# np.zeros((0,1), dtype=int)
        log_probs = []#np.zeros((0,1), dtype=int)
        encoder_inputs=[]; decisions=[]; #reward_rate = []
        for s in range(0, self.decoder_max_length-1):
            next_generate = self.forward(encoder_in)
            next_dist = Categorical(next_generate)
            next_decision = next_dist.sample()
            next_prob = next_dist.log_prob(next_decision).unsqueeze(0).cpu().detach().numpy()
            log_probs = np.append(next_prob)#log_probs = np.append(log_probs, next_prob, 0)
            actions.append(next_decision)#actions = np.append(actions, next_decision, 0)
            encoder_inputs.append(encoder_in)
            if next_decision == self.prob_len-1:
                #TODO
                #reward_rate.append(0)
                break
            #TODO check allocated last beacuase of getting the change 
            cont_dec = int(next_decision / self.host_num) 
            host_dec = int(next_decision % self.host_num)
            decisions.append((cont_dec, host_dec))
            
            if env.getPlacementPossible(cont_dec, host_dec): 
                #reward_rate.append(1)
                encoder_in = torch.cat([pad, torch.cat([encoder_in[:,0:cont_dec], 
                                                        encoder_in[:,cont_dec+1:]],1)], 1)
                host = env.getHostByID(host_dec)
                ips = int(host.getApparentIPS() + env.getContainerByID(cont_dec).getApparentIPS())
                encoder_in[:,-(self.host_num-host_dec+1)] = 100 * (ips / host.ipsCap)
            #else : reward_rate.append(0)
            
            
        return decisions, actions, log_probs, encoder_inputs, None, None
            
    def forward (
        self,
        encoder_in: torch.tensor,
    ):
        
        encoder_embedding = self.en_embed(encoder_in)* math.sqrt(self.output_dim)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        encod = self.encoder(self.en_position_encode(encoder_embedding))
        flat = self.flatten(encod)
        
        return self.softmax(self.outer(flat))
     

class CriticNetwork1 (nn.Module):
    def __init__ (self, encoder_max_length, encoder_input_dim, 
                  decoder_max_length, decoder_input_dim, device,
                  hidden_dims: List = [512, 256, 128, 128, 64, 16],
                  name = 'mlp_cretic'):
        super().__init__()
        
        self.name = name
        self.flatten = nn.Flatten()
        self.device =device
        
        
        modules = []
        input_dim = (encoder_input_dim*encoder_max_length)+(decoder_input_dim*decoder_max_length)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_eo = nn.Sequential(*modules).to(device)
        
        
    
    def forward(self, external, internal):
        input_tensor = torch.cat([external.flatten(start_dim=1),internal.flatten(start_dim=1)],1)
        return self.critic_eo(input_tensor.to(self.device))
    
class CriticNetwork2 (nn.Module):
    def __init__ (self, encoder_max_length, encoder_input_dim, 
                  decoder_max_length, decoder_input_dim, device,
                  hidden_dims: List = [256, 128, 128, 64, 16],
                  name = 'mlp_cretic'):
        super().__init__()
        self.name = name
        self.device = device
        self.flatten = nn.Flatten().to(device)
        modules = []
        input_dim = encoder_max_length*encoder_input_dim,
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic = nn.Sequential(*modules).to(device)
    
    def forward(self, external, *args):
        return self.critic(self.flatten(external.to(self.device)))
    