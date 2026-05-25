'''
Created on Feb 6, 2024

@author: diegobenalcazar
'''
import torch
import copy

from discrete.lib.agent.AC_Agent import AC_Agent, AC_TargetAgent
from discrete.lib.agent.feature_extractor import FeatureExtractor


class AC_EasyTargetAgent(AC_TargetAgent):
    """
    Target agent that delegates to a copy of the source agent, and uses the state_dict when updating itself
    """

    def __init__(self, source: AC_Agent, initial_copy: AC_Agent, tau: float):
        super().__init__()
        self.source = source
        self.target = initial_copy
        self.tau = tau
        self.update_weights()

    @classmethod
    def create_agent(cls, input_shape: FeatureExtractor, num_automaton_states: int, num_actions: int) -> "AC_Agent":
        raise NotImplementedError("Can't construct target agent directly")

    def resize_num_aut_states(self, num_automaton_states: int):
        raise NotImplementedError("Shouldn't resize target agents directly; resize the source and update weights")

    def calc_q_values_single(self, observation: torch.Tensor, automaton_state: int) -> torch.Tensor:
        return self.target.calc_q_values_single(observation, automaton_state)

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self.target.calc_q_values_batch(observation, automaton_states)
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self.target.calc_v_values_batch(observation, automaton_states)
    
    def create_target_agent(self) -> "AC_TargetAgent":
        raise NotImplementedError("Shouldn't create targets of target agents")

    def update_weights(self):
        source_copy = copy.deepcopy(self.source.state_dict())
        target_copy = copy.deepcopy(self.target.state_dict())

        for param_tensor in source_copy:
            
            source_copy[param_tensor] = self.tau * source_copy[param_tensor]  + (1 - self.tau) * target_copy[param_tensor]

        self.target.load_state_dict(source_copy)
