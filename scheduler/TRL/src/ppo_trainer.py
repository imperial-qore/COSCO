
"""

"""
import torch
from torch.optim import Adam

from .models import CriticNetwork
from .reply_buffer import PPOReplayBuffer

class PPOTRainer ():
    def __init__(
        self,
        actor_model: torch.nn.Module,
        env,
    ):
        self.actor_model = actor_model
        self.critic_model = 
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=1e-7)
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr= 1e-7)
        self.buffer = PPOReplayBuffer
        self.env = env

    
    def save_mid_step (
        self, 
        observation: List[torch.Tensor],
        decision: List,
        filtered_decision: List,
        actions: List,
        log_probs: List,
        steps: Optional[List[torch.Tensor]] = None,
        internalObservations: Optional[List[torch.Tensor]] = None,
    ):
        for index in range(len(decision)):
            if decision[index] not in filtered_decision:
                observation.pop(index)
                decision.pop(index)
                actions.pop(index)
                log_probs.pop(index)
                try:
                    steps.pop(index)
                    internalObservations.pop(index)
                except: pass
        self.critic_model
        
        buffer.save_mid_memory()#TODO save
    
    def save_final_step (
        self, 
        rewards
    ):
        buffer.save_final_memory(rewards)
    
    def train_minibatch(self):
        pass

    def reward_function(self):
        pass

    def