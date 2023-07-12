from typing import NamedTuple, Callable, List, Dict

import torch

from automaton_transfer.lib.automaton.ap_extractor import APExtractor


class AP(NamedTuple):
    name: str
    func: Callable[[Dict], bool]


class MineEnvApExtractor(APExtractor):
    def __init__(self, ap_funcs: List[Callable[[Dict], bool]], device: torch.device):
        self.ap_funcs = ap_funcs
        self.device = device

    def extract_aps_batch(self, observations: torch.Tensor, infos: List[Dict]) -> torch.LongTensor:
        # Last bit is for the typechecker
        return torch.as_tensor([self.extract_ap_single(observations[i], infos[i]) for i in range(len(infos))],
                               dtype=torch.long, device=self.device).long()

    def extract_ap_single(self, observation: torch.Tensor, info: Dict) -> int:
        total_idx = 0
        for idx, ap_func in enumerate(self.ap_funcs):
            if ap_func(info):
                total_idx += 2 ** idx  # Based on the ordering of the powerset in ltl_automaton

        return total_idx

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def num_transitions(self):
        return 2 ** len(self.ap_funcs)

