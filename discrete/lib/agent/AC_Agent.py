'''
Created on Feb 6, 2024

@author: diegobenalcazar
'''
import abc
from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn


class AC_Agent(ABC, nn.Module):
    """
    Represents a Q-learning agent that may choose to incorporate the current automaton state into its q-values
    """

    @classmethod
    @abc.abstractmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "AC_Agent":
        pass

    def calc_q_values_single(self, observation: torch.Tensor, automaton_state: int) -> torch.Tensor:
        """
        Calculate the q values for a single sample
        Default implementation just calls calc_q_values_batch
        """
        return self.calc_q_values_batch(observation.unsqueeze(0), torch.as_tensor([automaton_state], dtype=torch.long,
                                                                                  device=observation.device)).view(
            (-1,))

    @abc.abstractmethod
    def calc_q_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        """Automaton_state should be a LongTensor"""
        pass
    
    # @abc.abstractmethod
    # def choose_action(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
    #     """Automaton_state should be a LongTensor"""
    #     pass
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        return self.calc_q_values_batch(observation, automaton_state).amax(dim=-1)

    @abc.abstractmethod
    def create_target_agent(self) -> "AC_TargetAgent":
        """Clones the agent and its weights. Updates to the agent don't affect the target agent, but
        the target agent can "pull" updates from the source agent"""
        pass

class AC_TargetAgent(AC_Agent, ABC):
    """An abstraction to make double-Q learning easier to implement"""

    @abc.abstractmethod
    def update_weights(self):
        """Update the weights from the parent agent"""
        pass