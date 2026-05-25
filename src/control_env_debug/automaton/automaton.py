import os
import json
import numpy as np
import gym
from flloat.parser.ltlf import LTLfParser
from pythomata.impl.symbolic import SymbolicDFA

AUT_CACHE_NAME = "automaton_cache.json"

class LTLfAutomaton:
    def __init__(self, adj_matrix, initial_state, terminal_states):
        self.adj_matrix = np.array(adj_matrix, dtype=int)
        self.u0 = initial_state
        self.T = set(terminal_states)

    @property
    def num_states(self):
        return self.adj_matrix.shape[0]

    def step(self, u_current, ap_id):
        if u_current >= self.num_states or ap_id >= self.adj_matrix.shape[1]:
            return u_current
        return self.adj_matrix[u_current, ap_id]

    def reset(self):
        return self.u0

    @staticmethod
    def from_ltlf(ltlf, propositions):
        prop_names = sorted(propositions.keys())
        cache_key = ltlf + repr(prop_names)

        # Load from cache if available
        if os.path.exists(AUT_CACHE_NAME):
            with open(AUT_CACHE_NAME, 'r') as f:
                try:
                    cache = json.load(f)
                    if cache_key in cache:
                        print(f"Loading automaton for '{ltlf}' from cache.")
                        data = cache[cache_key]
                        return LTLfAutomaton(data['adj'], data['u0'], data['T'])
                except json.JSONDecodeError:
                    cache = {}
        else:
            cache = {}

        print(f"Compiling LTLf formula: '{ltlf}'...")
        parser = LTLfParser()
        formula = parser(ltlf)
        dfa: SymbolicDFA = formula.to_automaton().determinize()

        state_map = {state: i for i, state in enumerate(dfa.states)}
        num_states = len(state_map)
        num_aps = len(prop_names)
        adj_matrix = -np.ones((num_states, 2**num_aps), dtype=int)

        for symbolic_state, u_id in state_map.items():
            for ap_id in range(2**num_aps):
                symbol = {}
                for i, prop_name in enumerate(prop_names):
                    symbol[prop_name] = bool((ap_id >> i) & 1)
                next_symbolic_state = dfa.get_successor(symbolic_state, symbol)
                if next_symbolic_state is not None:
                    adj_matrix[u_id, ap_id] = state_map[next_symbolic_state]

        initial_state = state_map[dfa.initial_state]
        terminal_states = {state_map[s] for s in dfa.accepting_states}

        cache[cache_key] = {
            "adj": adj_matrix.tolist(),
            "u0": initial_state,
            "T": list(terminal_states)
        }
        with open(AUT_CACHE_NAME, 'w') as f:
            json.dump(cache, f, indent=4)

        print("Automaton built and saved to cache.")
        return LTLfAutomaton(adj_matrix, initial_state, terminal_states)

def get_ap_id(events_str, prop_names):
    ap_id = 0
    for i, prop in enumerate(prop_names):
        if prop in events_str:
            ap_id += (1 << i)
    return ap_id
