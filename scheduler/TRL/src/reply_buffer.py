
"""

"""
import torch
from typing import Optional

class PPOReplayBuffer ():
    def __init__ (self, link_number):
        self.link_number = link_number
        self.reset_normal_memory()
        self.reset_extra_memory()
        
    
    def reset_normal_memory (self) :
        self.normal_observation = [] 
        self.normal_internalObservation = []
        self.normal_action = [] 
        self.normal_prob = [] 
        self.normal_val = []
        self.normal_reward = []
        self.normal_steps = []
        self.normal_done = []
        
    def save_normal_step (
        self, 
        observation: torch.tensor,
        actions: torch.tensor,
        probs: torch.tensor,
        values: torch.tensor, 
        rewards: torch.tensor,
        done: bool,
        steps: Optional[torch.tensor] = None,
        internalObservations: Optional[torch.tensor] = None,

    ):
        self.normal_observation.append(torch.cat([observation.unsqueeze(1)]*self.link_number, 1).cpu())
        self.normal_action.append(torch.cat(actions,1).cpu())
        self.normal_prob.append(torch.cat(probs,1).cpu())
        self.normal_val.append(torch.cat(values,1).cpu())
        self.normal_reward.append(torch.cat(rewards,1).cpu())
        
        if done == True:
            self.normal_done.append(torch.tensor([False]*(self.link_number-1)+[True]).unsqueeze(0))
        else:
            self.normal_done.append(torch.tensor([done]*self.link_number).unsqueeze(0))
            
        if not isinstance(steps, type(None)):
            self.normal_steps.append(steps.unsqueeze(0).cpu())

        if not isinstance(internalObservations, type(None)):
            self.normal_internalObservation.append(torch.cat(internalObservations,0).unsqueeze(0).cpu())

    def get_normal_memory (self):
        try: return torch.cat(self.normal_observation, 0), \
                torch.cat(self.normal_action, 0), \
                torch.cat(self.normal_prob, 0), \
                torch.cat(self.normal_val, 0), \
                torch.cat(self.normal_reward, 0), \
                torch.cat(self.normal_done, 0), \
                torch.cat(self.normal_steps, 0), \
                torch.cat(self.normal_internalObservation, 0)
        except: return torch.cat(self.normal_observation, 0), \
                torch.cat(self.normal_action, 0), \
                torch.cat(self.normal_prob, 0), \
                torch.cat(self.normal_val, 0), \
                torch.cat(self.normal_reward, 0), \
                torch.cat(self.normal_done, 0)