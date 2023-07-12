import abc
from abc import ABC

import torch

from automaton_transfer.lib.automaton.automaton import Automaton


class TargetAutomaton(Automaton, ABC):
    """
    Represents an automaton where each state has an associated V-value and each transition has an associated Q-value
    """

    @abc.abstractmethod
    def target_q_values(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        """Get the Q-value for a given automaton state and transition"""
        pass

    @abc.abstractmethod
    def target_q_weights(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        """How much should the Q-value for a state and transition be weighted versus the "real" target Q value"""
        pass
    
    @abc.abstractmethod
    def target_reward_shaping(self, aut_states: torch.Tensor, aut_states_after_current: torch.Tensor) -> torch.Tensor:
        """"""
        pass

    @abc.abstractmethod
    def update_training_observed_count(self, aut_states: torch.Tensor, aps: torch.Tensor):
        """
        The automaton can keep track of how many times a state and transition have been seen during training.
        Note that this is called when the state is sampled from the replay buffer, not when it is seen in the actual env
        """
        pass

class AbstractTargetAutomatonWrapper(TargetAutomaton, ABC):
    """
    Wraps an inner "real" automaton, and delegates all of the non-target-automaton-specific functionality
    to this inner automaton
    """

    def __init__(self, inner_automaton: Automaton):
        self.inner_automaton = inner_automaton

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


class AnnealTargetAutomaton(AbstractTargetAutomatonWrapper, ABC):
    """
    Anneal between the given automaton weights and the real target Q values according to how many times an
    automaton transition was seen during training.
    This is probably the most useful class to extend in this file
    """

    def __init__(self, inner_automaton: Automaton, teacher_aut_info, min_source_q_count: int,
                 device: torch.device):
        """
        :param source_q_total: This is divided by source_q_count to obtain the q values of a transition
        :param source_q_count: How many times was a transition seen in the teacher instance
        :param min_source_q_count: If the transition wasn't seen at least this many times in the teacher instance,
                                    weight will be zero. Must be >= 1
        """
        super().__init__(inner_automaton)
        self.min_source_q_count = min_source_q_count
        self.source_q_total = torch.as_tensor(teacher_aut_info["aut_total_q"], dtype=torch.float, device=device)
        self.source_q_count = torch.as_tensor(teacher_aut_info["aut_num_q"], dtype=torch.int, device=device)
        self._cached_source_q_values = torch.zeros_like(self.source_q_total)

        self.target_q_count = torch.zeros_like(self.source_q_count)
        self.device = device

        self.recalc_cache()

    def target_q_values(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        return self._cached_source_q_values[aut_states, aps]

    def recalc_cache(self):
        self._cached_source_q_values = torch.where(self.source_q_count != 0,
                                                   self.source_q_total / self.source_q_count,
                                                   torch.as_tensor(0.0, dtype=torch.float, device=self.device))
    
    @abc.abstractmethod
    def calc_q_weights(self, source_q_count, target_q_count, iter_num):
        """
        How should the given automaton transition be weighted?
        This function can return junk where source_q_count < min_source_q_count, as long as it doesn't error
        :param source_q_count: How many times this transition was seen in the teacher
        :param target_q_count: How many times it was seen in the student
        :param iter_num: How many iterations of training have been taken in the student
        """
        pass

    def target_q_weights(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        return torch.where(self.source_q_count[aut_states, aps] >= self.min_source_q_count,
                           self.calc_q_weights(source_q_count=self.source_q_count[aut_states, aps],
                                               target_q_count=self.target_q_count[aut_states, aps], iter_num=iter_num),
                           torch.as_tensor(0.0, dtype=torch.float, device=self.device))

    def update_training_observed_count(self, aut_states: torch.Tensor, aps: torch.Tensor):
        # aut_states: [0, 1, 0, 1]
        # aps: [0, 0, 0, 1]
        # output: [[0, 0], [1, 0], [0, 0], [1, 1]]
        aut_state_ap = torch.stack((aut_states, aps), dim=1)

        # indices: [[0, 0], [1, 0], [1, 1]
        # counts: [2, 1, 1]
        indices, counts = torch.unique(aut_state_ap, return_counts=True, dim=0)
        self.target_q_count[indices[:, 0], indices[:, 1]] += counts

    def state_dict(self):
        return {
            "source_q_total": self.source_q_total,
            "source_q_count": self.source_q_count,
            "target_q_count": self.target_q_count,
            "inner_aut_sd": self.inner_automaton.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.inner_automaton.load_state_dict(state_dict["inner_aut_sd"])
        self.source_q_total = state_dict["source_q_total"]
        self.source_q_count = state_dict["source_q_count"]
        self.target_q_count = state_dict["target_q_count"]

        self.recalc_cache()


class ExponentialAnnealTargetAutomaton(AnnealTargetAutomaton):
    """
    Decrease the importance of the source values exponentially as more target transitions are observed
    """

    def __init__(self, inner_automaton: Automaton, teacher_aut_info, min_source_q_count: int,
                 device: torch.device,
                 exponent_base: float):
        super().__init__(inner_automaton=inner_automaton,
                         teacher_aut_info=teacher_aut_info,
                         min_source_q_count=min_source_q_count,
                         device=device)

        self.exponent_base = exponent_base

    def calc_q_weights(self, source_q_count, target_q_count, iter_num):
        return torch.pow(self.exponent_base, target_q_count)
    
    def target_reward_shaping(self, aut_states, aut_states_after_current):
        return torch.zeros_like(aut_states)

class RewardShapingTargetAutomaton(AnnealTargetAutomaton):
    def __init__(self, inner_automaton: Automaton, teacher_aut_info, min_source_q_count: int,
                device: torch.device, gamma: float):
        super().__init__(inner_automaton=inner_automaton,
                         teacher_aut_info=teacher_aut_info,
                         min_source_q_count=min_source_q_count,
                         device=device)
        
        source_v_total = torch.as_tensor(teacher_aut_info["aut_total_v"], dtype=torch.float, device=device)
        source_v_count = torch.as_tensor(teacher_aut_info["aut_num_v"], dtype=torch.float, device=device)
        
        self.v = source_v_total / (source_v_count + 1e-20)
        self.gamma = gamma
    
    def calc_q_weights(self, source_q_count, target_q_count, iter_num):
        return torch.as_tensor(0.0, dtype=torch.float, device=self.device)
    
    def target_reward_shaping(self, aut_states, aut_states_after_current):
        return self.v[aut_states.long()] - self.gamma * self.v[aut_states_after_current.long()]