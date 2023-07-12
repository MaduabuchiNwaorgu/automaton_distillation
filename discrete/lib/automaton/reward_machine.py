import abc
from abc import ABC

import json
import os
import numpy as np
import torch

from automaton_transfer.lib.automaton.automaton import Automaton
from automaton_transfer.lib.config import Configuration

class RewardMachine(Automaton):
    def __init__(self, automaton: Automaton, reward_adj_list: np.ndarray, terminal_states: np.ndarray, name: str, device: torch.device, gamma: float = 0.99):
        self.reward_mat = torch.as_tensor(reward_adj_list, dtype=torch.float, device=device)
        self.inner_automaton = automaton
        self.device = device
        self.gamma = gamma
        self.terminal_states = torch.as_tensor(terminal_states, dtype=torch.float, device=device)
        
        self.value_iter()
        
        to_save = {
            "reward_mat": self.reward_mat.tolist(),
            "terminal_states": self.terminal_states.tolist(),
            "aut_num_q": torch.ones_like(self.q).tolist(),
            "aut_total_q": self.q.tolist(),
            "aut_num_v": torch.ones_like(self.v).tolist(),
            "aut_total_v": self.v.tolist()
        }

        if not os.path.exists("automaton_q"):
            os.mkdir("automaton_q")

        with open(f"automaton_q/{name}.json", "w") as f:
            json.dump(to_save, f)
    
    @staticmethod
    def from_json(config: Configuration, device: torch.device):
        with open(f"automaton_q/{config.run_name}.json", "r") as f:
            teacher_aut_info = json.load(f)
        
        return RewardMachine(config.automaton, teacher_aut_info["reward_mat"], teacher_aut_info["terminal_states"], config.run_name, device)
    
    def value_iter(self):
        self.q = torch.zeros_like(self.reward_mat)
        converged = torch.as_tensor(False, dtype=torch.bool, device=self.device)
        
        while not converged:
            converged |= True
            # print(self.q)
            
            for state in torch.where(1 - self.terminal_states)[0]:
                states = torch.ones(self.num_aps, dtype=torch.long, device=self.device) * state
                actions = torch.arange(self.num_aps, dtype=torch.long, device=self.device)
                
                new_states = self.step_batch(states, actions).long()
                
                new_q = self.reward_mat[state, actions] + self.gamma * self.q[new_states].amax(axis=1)
                
                converged &= torch.all(torch.abs(self.q[state, actions] - new_q) < 1e-10)
                
                self.q[state, actions] = new_q
                        
        self.v = self.q.amax(axis=1)
    
    @property
    def default_state(self) -> int:
        return self.inner_automaton.default_state

    @property
    def num_states(self) -> int:
        return self.inner_automaton.num_states

    @property
    def num_aps(self) -> int:
        return self.inner_automaton.num_aps

    def step_batch(self, current_states: torch.tensor, aps_after_current: torch.tensor) -> torch.tensor:
        return self.inner_automaton.step_batch(current_states, aps_after_current)

    def step_single(self, current_state: int, ap: int) -> int:
        return self.inner_automaton.step_single(current_state, ap)

    def state_dict(self):
        return self.inner_automaton.state_dict()

    def load_state_dict(self, state_dict):
        self.inner_automaton.load_state_dict(state_dict)