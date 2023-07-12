import abc
from abc import abstractmethod

import torch


class Automaton(abc.ABC):
    """Specifically, a DFA"""
    @property
    @abstractmethod
    def default_state(self) -> int:
        """
        :return: The starting state of the automaton
        """
        pass

    @property
    @abstractmethod
    def num_states(self) -> int:
        pass

    @property
    @abstractmethod
    def num_aps(self) -> int:
        pass

    @abstractmethod
    def step_batch(self, current_states: torch.tensor, aps_after_current: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def step_single(self, current_state: int, ap: int) -> int:
        """
        Given the automaton state and alphabet letter, get the next state
        (or -1 if the automaton is incomplete and we attempted a non-existent transition)
        """
        pass

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass
