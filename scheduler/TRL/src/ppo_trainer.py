
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

        return decision
        
    def train_minibatch(self):
        pass

    def reward_function(self):
        pass

    def