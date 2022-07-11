import abc
from abc import ABC

import json
import os
import numpy as np
import torch

from automaton_transfer.lib.automaton.automaton import Automaton

class RewardMachine:
    def __init__(self, automaton: Automaton, reward_adj_list: np.ndarray, name: str, device: torch.device):
        self.r = torch.as_tensor(reward_adj_list, dtype=torch.float, device=device)
        self.device = device
        
        self.value_iter(automaton)
        
        to_save = {
            "aut_num_q": torch.ones_like(self.q).tolist(),
            "aut_total_q": self.q.tolist(),
            "aut_num_v": torch.ones_like(self.v).tolist(),
            "aut_total_v": self.v.tolist()
        }

        if not os.path.exists("automaton_q"):
            os.mkdir("automaton_q")

        with open(f"automaton_q/{name}.json", "w") as f:
            json.dump(to_save, f)
    
    def value_iter(self, automaton):
        self.q = torch.zeros_like(self.r)
        converged = torch.as_tensor(False, dtype=torch.bool, device=self.device)
        
        while not converged:
            converged |= True
            print(self.q)
            
            for state in range(automaton.num_states):
                states = torch.ones(automaton.num_aps, dtype=torch.long, device=self.device) * state
                actions = torch.arange(automaton.num_aps, dtype=torch.long, device=self.device)
                
                new_states = automaton.step_batch(states, actions).long()
                
                new_q = self.r[state, actions] + self.q[new_states].amax(axis=1)
                
                converged &= torch.all(torch.abs(self.q[state, actions] - new_q) < 1e-10)
                
                self.q[state, actions] = new_q
                        
        self.v = self.q.amax(axis=1)