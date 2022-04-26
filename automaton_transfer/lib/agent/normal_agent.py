from typing import Tuple

import torch
from torch import nn as nn

from automaton_transfer.lib.agent.agent import Agent, TargetAgent
from automaton_transfer.lib.agent.easy_target_agent import EasyTargetAgent
from automaton_transfer.lib.agent.feature_extractor import FeatureExtractor


class DuelingQNetworkAgent(Agent):
    """A basic dueling q-agent"""

    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.input_shape = input_shape
        self.feature_extractor = FeatureExtractor(input_shape=input_shape)
        self.half_feat_extractor_output_size = self.feature_extractor.output_size // 2
        self.val_branch = nn.Linear(self.half_feat_extractor_output_size, 1)
        self.adv_branch = nn.Linear(self.half_feat_extractor_output_size, num_actions)
        self.flattener = nn.Flatten()
        self.num_actions = num_actions

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "Agent":
        return cls(input_shape, num_actions)

    def forward(self, obs):
        features = self.feature_extractor(obs)
        val_stream, adv_stream = torch.split(features, self.half_feat_extractor_output_size, dim=1)

        # Val and adv are because of dueling Q-network architecture, not actor critic
        val_stream = self.flattener(val_stream)
        adv_stream = self.flattener(adv_stream)

        val = self.val_branch(val_stream)
        adv = self.adv_branch(adv_stream)

        mean_adv = adv.mean(dim=1)

        q_vals = val + (adv - mean_adv.unsqueeze(1))  # Unsqueeze necessary so that it broadcasts correctly
        return q_vals

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self(observation)

    def create_target_agent(self) -> "TargetAgent":
        return EasyTargetAgent(self, DuelingQNetworkAgent(self.input_shape, self.num_actions)).to(
            self.adv_branch.weight.device)
