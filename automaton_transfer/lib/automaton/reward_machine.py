import abc
from abc import ABC

import json
import os
import numpy as np
import torch

from automaton_transfer.lib.automaton.automaton import Automaton
from automaton_transfer.lib.config import Configuration

class RewardMachine:
    def __init__(self, automaton: Automaton, reward_adj_list: np.ndarray, terminal_states: np.ndarray, name: str, device: torch.device, gamma: float = 0.99):
        self.reward_mat = torch.as_tensor(reward_adj_list, dtype=torch.float, device=device)
        self.aut = automaton
        self.device = device
        self.gamma = gamma
        self.terminal_states = terminal_states
        
        self.value_iter()
        
        to_save = {
            "reward_mat": reward_adj_list.tolist(),
            "terminal_states": terminal_states.tolist(),
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
        with open(f"automaton_q/{config.name}.json", "r") as f:
            teacher_aut_info = json.load(f)
        
        return RewardMachine(config.automaton, teacher_aut_info["reward_mat"], teacher_aut_info["terminal_states"], config.name, device)
    
    def value_iter(self):
        self.q = torch.zeros_like(self.reward_mat)
        converged = torch.as_tensor(False, dtype=torch.bool, device=self.device)
        
        while not converged:
            converged |= True
            # print(self.q)
            
            for state in torch.where(1 - self.terminal_states)[0]:
                states = torch.ones(self.aut.num_aps, dtype=torch.long, device=self.device) * state
                actions = torch.arange(self.aut.num_aps, dtype=torch.long, device=self.device)
                
                new_states = self.aut.step_batch(states, actions).long()
                
                new_q = self.reward_mat[state, actions] + self.gamma * self.q[new_states].amax(axis=1)
                
                converged &= torch.all(torch.abs(self.q[state, actions] - new_q) < 1e-10)
                
                self.q[state, actions] = new_q
                        
        self.v = self.q.amax(axis=1)