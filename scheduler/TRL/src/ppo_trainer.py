
"""

"""
import torch
import numpy as np
from torch.optim import Adam

from .models import CriticNetwork2
from .reply_buffer import PPOReplayBuffer
from typing import List, Optional
from torch.distributions.categorical import Categorical


class PPOTRainer ():
    def __init__(
        self,
        actor_model: torch.nn.Module,
        env,
        ppo_epochs: int =5,
        gamma: float =.99,
        gae_lambda: float =.97,
        cliprange: float=.2,
    ): 
        self.actor_model = actor_model
        self.critic_model = CriticNetwork2(self.actor_model.encoder_max_length, 
                                           self.actor_model.input_dim, 
                                           self.actor_model.device)
        #TODO critic model load part
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=1e-7)
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr= 1e-7)
        self.buffer = PPOReplayBuffer()
        self.env = env
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cliprange = cliprange

    
    def save_mid_step (
        self, 
        observation: List[torch.Tensor],
        decision: List,
        filtered_decision: List,
        rewards,
        actions: List,
        log_probs: List,
        steps: Optional[List[torch.Tensor]] = None,
        internalObservations: Optional[List[torch.Tensor]] = None,
    ):
        '''length = len(decision)
        for index in range(length):
            if decision[index] not in filtered_decision:
                observation.pop(index)
                decision.pop(index)
                actions.pop(index)
                log_probs.pop(index)
                try:
                    steps.pop(index)
                    internalObservations.pop(index)
                except: pass'''
        vals = self.critic_model(torch.cat(observation, 0)).squeeze().tolist()
        
        print('valss', vals)
        self.buffer.save_mid_memory(observation, decision, rewards, actions, log_probs,
                                    vals)
    
    def save_final_step (
        self, 
        rewards
    ):
        self.buffer.save_final_memory(rewards)
    
    def generate_batch (
        self, 
        n_states: int,
        batch_size: int,
    ):
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        return batches
    
    def train_minibatch(
        self, 
        n_state: int,
        train_batch_size: int = 8,
    ):
        
        memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memortStp, memoryIObs \
            = self.buffer.get_memory(n_state)
        
        for _ in range(self.ppo_epochs):
            batches = self.generate_batch(n_state, train_batch_size)

            advantage = np.zeros(n_state, dtype=np.float32)
            for t in range(n_state-1):
                discount = 1
                a_t = 0
                for k in range(t, n_state-1):
                    a_t += discount*(memoryRwd[k] + self.gamma*memoryVal[k+1]*\
                                     (1) - memoryVal[k])                
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            
            for batch in batches:
                batchObs = torch.cat(memoryObs[batch].tolist(), 0).to(self.actor_model.device)
                batchActs = torch.tensor(memoryAct[batch], dtype=torch.float).to(self.actor_model.device)
                batchProbs = torch.tensor(memoryPrb[batch], dtype=torch.float).to(self.actor_model.device)
                batchVals = torch.tensor(memoryVal[batch], dtype=torch.float).to(self.actor_model.device)
                batchAdvs = torch.tensor(advantage[batch], dtype=torch.float).to(self.actor_model.device)
                
                batchObs.requires_grad = True
                #batchActs.requires_grad = True
                batchProbs.requires_grad = True
                batchVals.requires_grad = True
                batchAdvs.requires_grad = True
            
                generate = self.actor_model(batchObs)
                act_dist = Categorical(generate)
                
                new_log_probs = act_dist.log_prob(batchActs.squeeze().to(self.actor_model.device))
                newVal = self.critic_model(batchObs).squeeze()
                
                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = batchAdvs * prob_ratio

                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.cliprange,
                            1+self.cliprange)*batchAdvs
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = batchAdvs + batchVals
                
                critic_loss = (returns-newVal)**2
                
                critic_loss = torch.mean(critic_loss)

                total_loss = actor_loss + 0.5*critic_loss
                #total_loss.requires_grad=True
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()


        return self.memory.erase(n_state)
         
        

    