import torch

from automaton_transfer.lib.agent.agent import Agent, TargetAgent
from automaton_transfer.lib.agent.feature_extractor import FeatureExtractor


class EasyTargetAgent(TargetAgent):
    """
    Target agent that delegates to a copy of the source agent, and uses the state_dict when updating itself
    """

    def __init__(self, source: Agent, initial_copy: Agent):
        super().__init__()
        self.source = source
        self.target = initial_copy
        self.update_weights()

    @classmethod
    def create_agent(cls, input_shape: FeatureExtractor, num_automaton_states: int, num_actions: int) -> "Agent":
        raise NotImplementedError("Can't construct target agent directly")

    def resize_num_aut_states(self, num_automaton_states: int):
        raise NotImplementedError("Shouldn't resize target agents directly; resize the source and update weights")

    def calc_q_values_single(self, observation: torch.Tensor, automaton_state: int) -> torch.Tensor:
        return self.target.calc_q_values_single(observation, automaton_state)

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self.target.calc_q_values_batch(observation, automaton_states)
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        return self.target.calc_v_values_batch(observation, automaton_states)
    
    def create_target_agent(self) -> "TargetAgent":
        raise NotImplementedError("Shouldn't create targets of target agents")

    def update_weights(self):
        # noinspection PyTypeChecker
        self.target.load_state_dict(self.source.state_dict())
