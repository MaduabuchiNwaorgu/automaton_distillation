import json
from json import JSONDecodeError
from os.path import exists
from typing import List, Dict
import platform

import numpy as np
import torch
import re
# import spot
from flloat.parser.ltlf import LTLfParser
from ltlf2dfa.parser.ltlf import LTLfParser as ltlf2dfaParser
from pythomata.impl.symbolic import SymbolicDFA

from automaton_transfer.lib.automaton.automaton import Automaton

# It can be slow to compile LTLf into an automaton, so we keep the results of this on disk
AUT_CACHE_NAME = "aut_cache.json"

# spot.setup()


def get_aut_json_key(ltlf: str, ap_names: List[str]):
    """A human-readable "hash" for the automaton and aps. Should be unique for non-pathological cases."""
    return ltlf + repr(ap_names)


def load_aut_from_cache(ltlf: str, ap_names: List[str], device: torch.device):
    """If the automaton corresponding to ltlf and ap_names exists in the cache, return it. Otherwise, return None."""
    aut_key = get_aut_json_key(ltlf=ltlf, ap_names=ap_names)

    try:
        with open(AUT_CACHE_NAME, "r") as f:
            j: Dict = json.load(f)
            sd = j.get(aut_key, None)
            if not sd:
                return None

            np.asarray(sd["adj_list"], dtype=np.int)
            return LTLAutomaton(np.asarray(sd["adj_list"], dtype=np.int), sd["init_state"], device)
    except (JSONDecodeError, FileNotFoundError):
        return None


def save_aut_to_cache(ltlf: str, ap_names: List[str], adj_matrix: np.ndarray, init_state: int):
    """Save the given automaton, overwriting it in the cache if it exists already"""
    if not exists(AUT_CACHE_NAME):
        j = {}
    else:
        with open(AUT_CACHE_NAME, "r") as f:
            j = json.load(f)

    j[get_aut_json_key(ltlf, ap_names)] = {
        "adj_list": adj_matrix.tolist(),
        "init_state": init_state
    }

    with open(AUT_CACHE_NAME, "w") as f:
        json.dump(j, f)


class LTLAutomaton(Automaton):
    """
    Stateless representation of a deterministic automaton
    """

    @property
    def default_state(self) -> int:
        return self.initial_state

    @property
    def num_states(self) -> int:
        return len(self.adj_list)

    @property
    def num_aps(self) -> int:
        return len(self.adj_list[0])

    def step_batch(self, current_states: torch.tensor, aps_after_current: torch.tensor) -> torch.tensor:
        return self.adj_mat[current_states, aps_after_current]

    def step_single(self, current_state: int, ap: int) -> int:
        return self.adj_list[current_state][ap]

    def state_dict(self):
        return {
            "adj_mat": self.adj_mat,
            "init": self.initial_state
        }

    def load_state_dict(self, state_dict):
        self.initial_state = state_dict["init"]
        self.adj_mat = torch.as_tensor(state_dict["adj_mat"], device=self.device)
        self.adj_list = self.adj_mat.tolist()

    def __init__(self, adj_list: np.ndarray, default_state: int, device: torch.device):
        self.adj_mat = torch.as_tensor(adj_list, device=device)
        self.adj_list = self.adj_mat.tolist()
        self.initial_state = default_state
        self.device = device

    @staticmethod
    def from_ltlf(ltlf: str, ap_names: List[str], device: torch.device):
        """
        Construct an automaton graph from a DFA
        :param ltlf: The ltlf formula
        :param ap_names: An ordered list of names for the atomic propositions
        """
        cached_automaton = load_aut_from_cache(ltlf=ltlf, ap_names=ap_names, device=device)
        if cached_automaton:
            return cached_automaton

        # Parse to DFA
        if platform.system() == 'Linux':
            # parser = ltlf2dfaParser()
            # parsed = parser(ltlf)
            # mona_dfa = parsed.to_dfa()
            # dot_dfa = spot.translate(ltlf, 'deterministic').to_str('dot')
            # dfa: SymbolicDFA = from_dot_spot(dot_dfa)
            ltl_parser = LTLfParser()
            parsed_formula = ltl_parser(ltlf)
            dfa: SymbolicDFA = parsed_formula.to_automaton().determinize()
            pass
        else:
            ltl_parser = LTLfParser()
            parsed_formula = ltl_parser(ltlf)
            dfa: SymbolicDFA = parsed_formula.to_automaton().determinize()

        print("Done with DFA conversion")

        # Convert from SymbolicDFA to an adjacency list with an integer alphabet
        adj_matrix = -np.ones((len(dfa.states), 2 ** len(ap_names)), dtype=np.int)
        iter = 0
        for state in dfa.states:
            iter += 1
            for ap_num in range(2 ** len(ap_names)):
                ap_combination = []
                num_remaining = ap_num

                # Powerset of ap_names in a more predictable order than more_itertools.powerset
                for name in ap_names:
                    if num_remaining % 2 == 1:
                        ap_combination.append(name)
                    num_remaining //= 2

                sym = {ap: True for ap in ap_combination}

                trans_to = dfa.get_successor(state, sym)
                adj_matrix[state, ap_num] = trans_to

        save_aut_to_cache(ltlf=ltlf, ap_names=ap_names, adj_matrix=adj_matrix, init_state=dfa.initial_state)

        return LTLAutomaton(adj_matrix, dfa.initial_state, device)


def from_dot(dfa):
    new_automaton = SymbolicDFA()
    states = set()
    outgoing = {}
    lines = dfa.split('\n')
    start = False
    for line in lines:
        if line[0] == '}':
            break
        if start is False:
            if 'init ->' in line:
                new_automaton.create_state()
                states.add(int(line[-2]) - 1)
                outgoing[int(line[-2]) - 1] = 0
                start = True
        else:
            temp = line[:line.index('[')].split(' ')
            initial = int(temp[1]) - 1
            receive = int(temp[3]) - 1
            label = line[line.index('"') + 1:-3]
            if initial not in states:
                states.add(initial)
                outgoing[initial] = 0
                new_automaton.create_state()
            if receive not in states:
                states.add(receive)
                new_automaton.create_state()
                outgoing[receive] = 0
            outgoing[initial] += 1
            new_automaton.add_transition((initial, label, receive))
    for key in outgoing.keys():
        if outgoing[key] == 1:
            new_automaton.set_accepting_state(key, True)
    new_automaton.set_initial_state(0)
    return new_automaton.determinize()


def from_dot_spot(dfa):
    new_automaton = SymbolicDFA()
    initial_state = -1
    current_state = 0
    states = {0}
    lines = dfa.split('\n')
    for line in lines:
        if re.match('\s+I -> \d+', line):
            initial_state = int(line.split(' ')[-1])
        if line == '}':
            break
        if re.match('\s+\d+ \[label=.+\]', line):
            if int(line[2]) in states:
                continue
            new_automaton.create_state()
            current_state += 1
            states.add(current_state)
        elif re.match('\s+\d+ -> \d+ \[label=.+\]', line):
            temp = line[:line.index('[')].split(' ')
            initial = int(temp[2])
            receive = int(temp[4])
            while receive not in states:
                new_automaton.create_state()
                current_state += 1
                states.add(current_state)
            label = line[line.index('<') + 1:-2]
            if 'amp;' in label:
                label = label.replace('amp;', '')
            if '!' in label:
                label = label.replace('!', '~')
            new_automaton.add_transition((initial, label, receive))
    new_automaton.set_initial_state(initial_state)
    return new_automaton
