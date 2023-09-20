
"""

"""
import torch
from typing import Optional, List
import numpy as np

class PPOReplayBuffer ():
    def __init__ (self, link_number):
        #self.link_number = link_number
        self.reset_final_memory()
        self.reset_mid_memory()
        
    
    def reset_final_memory (self) :
        self.final_observation = [] 
        self.final_internalObservation = []
        self.final_decision = [] 
        self.final_action = [] 
        self.final_prob = [] 
        self.final_val = []
        self.final_reward = []
        self.final_steps = []
        
    def reset_mid_memory (self) :
        self.mid_observation = [] 
        self.mid_internalObservation = []
        self.mid_decision = [] 
        self.mid_action = [] 
        self.mid_prob = [] 
        self.mid_val = []
        self.mid_steps = []
    
    def save_mid_step (
        self, 
        observation: List[torch.Tensor],
        decision: List,
        actions: List,
        probs: List,
        values: List, 
        steps: Optional[List[torch.Tensor]] = None,
        internalObservations: Optional[List[torch.Tensor]] = None,
    ):
        self.mid_observation += observation
        self.mid_decision += decision
        self.mid_action += actions
        self.mid_prob += probs
        self.mid_val += values
        
        if not isinstance(steps, type(None)):
            self.mid_steps += steps
            
        if not isinstance(internalObservations, type(None)):
            self.mid_internalObservation += internalObservations

    def save_final_memory (
        self,
        rewards:dict
    ):
        for i in rewards:
            condition = (np.array(self.mid_decision)[:,0] == i[0]) &  (np.array(self.mid_decision)[:,1] == i[1])
            index = np.where(condition)[0][0]
            self.final_observation.append(self.mid_observation.pop(index))
            self.final_decision.append(self.mid_decision.pop(index)) 
            self.final_action.append(self.mid_action.pop(index))
            self.final_prob.append(self.mid_prob.pop(index))
            self.final_val.append(self.mid_val.pop(index))
            self.final_reward.append(rewards[i])
            
            try:
                self.final_internalObservation.append(self.mid_internalObservation.pop(index))
                self.final_steps.append(self.mid_steps.pop(index))
        
    def get_memory (self):
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