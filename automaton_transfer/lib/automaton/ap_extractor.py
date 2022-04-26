import abc
from abc import ABC
from typing import Dict, List

import torch


class APExtractor(ABC):
    """
    At its core, an AP extractor is a function from a state to an integer.
    Realistically, it's an alphabet extractor, not an AP extractor
    """

    @abc.abstractmethod
    def num_transitions(self):
        """If there are 4 APs, this function should return 16- one for each combination of possible APs
        For compatibility with systems based on DFA minimization, which may not use APs at all"""
        pass

    @abc.abstractmethod
    def extract_aps_batch(self, observations: torch.Tensor, infos: List[Dict]) -> torch.LongTensor:
        """:returns a list of numbers between 0 and num_transitions - 1"""

    def extract_ap_single(self, observation: torch.Tensor, info: Dict) -> int:
        return int(self.extract_aps_batch(observation.unsqueeze(0), [info]).detach())

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass


