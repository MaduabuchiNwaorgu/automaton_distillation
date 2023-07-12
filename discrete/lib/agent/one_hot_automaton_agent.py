from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from automaton_transfer.lib.agent.agent import Agent, TargetAgent
from automaton_transfer.lib.agent.easy_target_agent import EasyTargetAgent
from automaton_transfer.lib.agent.feature_extractor import FeatureExtractor


def plane_one_hot(n, plane_shape, num_classes, dtype: torch.dtype):
    """
    n is a longtensor of shape (x), returns shape (x, num_classes, plane_shape)
    With a single plane for each n set to 1, and the rest to 0
    """
    ret = torch.zeros((len(n), num_classes, *plane_shape), dtype=dtype, device=n.device)
    ret[range(len(ret)), n] = 1
    return ret


# TODO: Would multi-hot encodings of a NFA state be equivalent, or more efficient?

class OneHotAutomatonAfterFeatureExtractorAgent(Agent):
    """
    Keep the feature extractor as with the base agent,
    and add in a one-hot encoding of the automaton state as input to the linear layers
    """

    def __init__(self, input_shape: Tuple, num_aut_states: int, num_actions: int):
        super().__init__()

        self.input_shape = input_shape
        self.feature_extractor = FeatureExtractor(input_shape=input_shape)
        self.half_feat_extractor_output_size = self.feature_extractor.output_size // 2
        self.val_branch = nn.Linear(self.half_feat_extractor_output_size + num_aut_states, 1)
        self.adv_branch = nn.Linear(self.half_feat_extractor_output_size + num_aut_states, num_actions)
        self.flattener = nn.Flatten()
        self.num_actions = num_actions
        self.num_aut_states = num_aut_states

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "Agent":
        return cls(input_shape, num_automaton_states, num_actions)

    def forward(self, obs, aut_states):
        features = self.feature_extractor(obs)
        val_stream, adv_stream = torch.split(features, self.half_feat_extractor_output_size, dim=1)

        # Val and adv are because of dueling Q-network architecture, not actor critic
        val_stream = self.flattener(val_stream)
        adv_stream = self.flattener(adv_stream)

        oh_aut_states = F.one_hot(aut_states, num_classes=self.num_aut_states)

        val_stream = torch.cat((val_stream, oh_aut_states), dim=1)
        adv_stream = torch.cat((adv_stream, oh_aut_states), dim=1)

        val = self.val_branch(val_stream)
        adv = self.adv_branch(adv_stream)

        mean_adv = adv.mean(dim=1)

        q_vals = val + (adv - mean_adv.unsqueeze(1))  # Unsqueeze necessary so that it broadcasts correctly
        return q_vals

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self(observation, automaton_states)

    def create_target_agent(self) -> "TargetAgent":
        return EasyTargetAgent(self, OneHotAutomatonAfterFeatureExtractorAgent(self.input_shape,
                                                                               self.num_aut_states,
                                                                               self.num_actions)).to(
            self.adv_branch.weight.device)


class OneHotAutomatonBeforeFeatExtractorAgent(Agent):
    """
    Add a one-hot encoding of the automaton state as input to the feature extractor,
    using a plane for each potential automaton state
    """

    def __init__(self, input_shape: Tuple, num_aut_states: int, num_actions: int):
        super().__init__()

        self.input_shape = input_shape
        input_chan, *plane_shape = input_shape
        self.plane_shape = plane_shape
        inner_input_shape = (input_chan + num_aut_states, *plane_shape)

        self.feature_extractor = FeatureExtractor(input_shape=inner_input_shape)
        self.half_feat_extractor_output_size = self.feature_extractor.output_size // 2
        self.val_branch = nn.Linear(self.half_feat_extractor_output_size, 1)
        self.adv_branch = nn.Linear(self.half_feat_extractor_output_size, num_actions)
        self.flattener = nn.Flatten()
        self.num_actions = num_actions
        self.num_aut_states = num_aut_states

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "Agent":
        return cls(input_shape, num_automaton_states, num_actions)

    def forward(self, obs, aut_states):
        one_hot_planes = plane_one_hot(aut_states, plane_shape=self.plane_shape, dtype=torch.float,
                                       num_classes=self.num_aut_states)
        full_obs = torch.cat((one_hot_planes, obs), dim=1)

        features = self.feature_extractor(full_obs)
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
        return self(observation, automaton_states)

    def create_target_agent(self) -> "TargetAgent":
        return EasyTargetAgent(self, OneHotAutomatonBeforeFeatExtractorAgent(self.input_shape,
                                                                             self.num_aut_states,
                                                                             self.num_actions)).to(
            self.adv_branch.weight.device)
