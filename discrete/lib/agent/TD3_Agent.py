'''
Created on Feb 6, 2024

@author: diegobenalcazar
'''

import abc
from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from discrete.lib.agent.AC_Agent import AC_Agent, AC_TargetAgent
from discrete.lib.agent.AC_easy_target_agent import AC_EasyTargetAgent
from discrete.lib.agent.feature_extractor import FeatureExtractor

class CriticNetwork(nn.Module):
    def __init__(
        self, 
        beta,            # learning rate
        input_dims,      # dimension of flattened observation
        fc1_dims, 
        fc2_dims, 
        n_actions, 
        name, 
        chkpt_dir='tmp/ddpg', 
        device='cpu'
    ):
        super(CriticNetwork, self).__init__()

        self.flattener = nn.Flatten()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        ############
        # CRITIC 1 #
        ############
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Action → same dimension as fc2_dims
        self.action_1_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        ############
        # CRITIC 2 #
        ############
        self.fc3 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc4 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action_2_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q2 = nn.Linear(self.fc2_dims, 1)

        # Move everything to device
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        """
        TD3 uses "two critics". We'll compute Q1 and Q2 for the same (state, action).
        """
        # Flatten the state first
        x = self.flattener(state)

        ########################
        # Critic 1 forward pass
        ########################
        c1 = F.relu(self.fc1(x))               # shape: [batch_size, fc1_dims]
        c1 = self.fc2(c1)                      # shape: [batch_size, fc2_dims]
        a1 = self.action_1_value(action)       # shape: [batch_size, fc2_dims]
        c1 = F.relu(c1 + a1)                   # combine state and action
        q1 = self.q1(c1)                       # shape: [batch_size, 1]

        ########################
        # Critic 2 forward pass
        ########################
        c2 = F.relu(self.fc3(x))
        c2 = self.fc4(c2)
        a2 = self.action_2_value(action)
        c2 = F.relu(c2 + a2)
        q2 = self.q2(c2)

        # Return as 1D (squeezed) for convenience
        return q1.squeeze(-1), q2.squeeze(-1)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))

        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, fc1_dims = 800, fc2_dims = 600, 
                chkpt_dir='tmp/ddpg', device='cpu'):
        super(ActorNetwork, self).__init__()

        # print(f"actor input dims: {input_dims}")
        self.flattener = nn.Flatten() #added
        # print(f"actor input dims: {input_dims}")
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'ddpg')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1=nn.Linear(self.input_dims, self.fc1_dims)
        
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # f1 = 0.003
        #torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        # f2 = 0.003
        #torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        #f3 = 0.003
        # print(f"\nfc2dims: {fc2_dims}, n_actions: {n_actions}\n")

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        #torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        #torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        # self.optimizer = torch.optim.Adam(self.parameters(), lr =alpha)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.device = device
        self.to(self.device)
        
    def forward(self, state):
        # print(f"actor state input: {state}")
        # print(f"actor state input shape: {state.shape}")
        
        x = self.flattener(state)
        # print(f"after flatten shape: {x.shape}")
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        a = self.tanh(self.mu(x))
        #x = x.squeeze()

        return a
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class TD3_Agent(AC_Agent):
    """
    Represents a TD3 agent that utilizes the integrated into the other tools chosen for the replay buffer and parallelization
    """
    
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.name = "TD3 Agent"
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.noise_clip = 0.5
        self.policy_noise_stddev = 0.2
        self.max_action = 1.0
        if os.getenv("NOISE_CLIP") is not None:
            self.noise_clip = float(os.getenv("NOISE_CLIP"))
            self.policy_noise_stddev = float(os.getenv("NOISE_STDDEV"))

        # Update Critic every (self.d) timesteps
        self.d = 2 # update timesteps
        if os.getenv("POLICY_FREQ") is not None:
            self.d = int(os.getenv("POLICY_FREQ"))

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

        self.flattener = nn.Flatten()
        
        self.actor = ActorNetwork(alpha=0.005, input_dims = np.prod(input_shape), n_actions=self.num_actions, name = 'Actor', device=self.device)
        self.actor.to(self.device)

        self.critic = CriticNetwork(beta=0.005, input_dims = np.prod(input_shape), fc1_dims= 800,fc2_dims=600, n_actions=self.num_actions, name='Critic', device=self.device)
        self.critic.to(self.device)

    def noise(self, action):
        return (torch.randn_like(action) * self.policy_noise_stddev).clamp(-self.noise_clip, self.noise_clip)

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "AC_Agent":
        return cls(input_shape, num_actions)
    
    # def choose_action(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.tensor:

    #     self.actor.eval()
    #     mu = self.actor.forward(observation).to(self.actor.device)

    #     # noise = torch.tensor(self.noise(),dtype=torch.float).to(self.actor.device)
    #     noise = self.noise(mu)
    #     mu_prime = (mu + noise).clamp(-self.max_action, self.max_action)

    #     self.actor.train()
    #     action = mu_prime.cpu().detach().numpy()

    #     return action
    
    def choose_action(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.tensor:

        
        mu = self.actor.forward(observation).to(self.actor.device)
        action = mu.cpu().detach().numpy()

        return action

    def calc_q_values_single(self, observation: torch.Tensor, automaton_state: int) -> torch.Tensor:
        """
        Calculate the q values for a single sample
        Default implementation just calls calc_q_values_batch
        """
        return self.calc_q_values_batch(observation.unsqueeze(0), torch.as_tensor([automaton_state], dtype=torch.long,
                                                                                  device=observation.device)).view((-1,))

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self(observation)
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        return self.calc_q_values_batch(observation, automaton_state).amax(dim=-1)

    def create_target_agent(self, tau=1) -> "AC_TargetAgent":
        return AC_EasyTargetAgent(self, TD3_Agent(self.input_shape, self.num_actions), tau=tau).to(self.device)
    
        