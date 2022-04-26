from typing import List, Dict

import torch

from automaton_transfer.lib.automaton.ap_extractor import APExtractor


class ProductAPExtractor(APExtractor):
    def __init__(self, ape1: APExtractor, ape2: APExtractor):
        self.ape1 = ape1
        self.ape2 = ape2

    def num_transitions(self):
        return self.ape1.num_transitions() * self.ape2.num_transitions()

    def extract_aps_batch(self, observations: torch.Tensor, infos: List[Dict]) -> torch.LongTensor:
        aps1 = self.ape1.extract_aps_batch(observations, infos)
        aps2 = self.ape2.extract_aps_batch(observations, infos)

        return (aps1 * self.ape2.num_transitions()) + aps2

    def state_dict(self):
        return {
            "inner1": self.ape1.state_dict(),
            "inner2": self.ape2.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.ape1.load_state_dict(state_dict["inner1"])
        self.ape2.load_state_dict(state_dict["inner2"])
