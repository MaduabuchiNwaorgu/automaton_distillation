import torch

from automaton_transfer.lib.automaton.target_automaton import TargetAutomaton


class ProductTargetAutomaton(TargetAutomaton):
    """
    An automaton produced by taking the product of two sub-automata
    Must be used with ProductAPExtractor
    Steps both sub-automata in sync with each other, and takes the average of their q-values and weights
    Note that transition counts are stored separately for each sub-automaton.
    If transition 1 in automaton 1 is taken twice, with transition 1 in automaton 2 and transition 2 in automaton 2,
    the weight is such that 1-1 was stepped twice, and both 2-1 and 2-2 have each been stepped once
    """

    def __init__(self, automaton_1: TargetAutomaton, automaton_2: TargetAutomaton):
        self.automaton_1 = automaton_1
        self.automaton_2 = automaton_2

    def inner_to_main_state(self, state1, state2):
        return (state1 * self.automaton_2.num_states) + state2

    def main_to_inner_states(self, state):
        return (state // self.automaton_2.num_states), (state % self.automaton_2.num_states)

    def main_to_inner_aps(self, ap):
        return (ap // self.automaton_2.num_aps), (ap % self.automaton_2.num_aps)

    @property
    def default_state(self) -> int:
        return self.inner_to_main_state(self.automaton_1.default_state, self.automaton_2.default_state)

    @property
    def num_states(self) -> int:
        return self.automaton_1.num_states * self.automaton_2.num_states

    @property
    def num_aps(self) -> int:
        return self.automaton_1.num_aps * self.automaton_2.num_aps

    def step_batch(self, current_states: torch.tensor, aps_after_current: torch.tensor) -> torch.tensor:
        states1, states2 = self.main_to_inner_states(current_states)
        aps1, aps2 = self.main_to_inner_aps(aps_after_current)

        next_states1 = self.automaton_1.step_batch(states1, aps1)
        next_states2 = self.automaton_2.step_batch(states2, aps2)
        return self.inner_to_main_state(next_states1, next_states2)

    def step_single(self, current_state: int, ap: int) -> int:
        state1, state2 = self.main_to_inner_states(current_state)
        ap1, ap2 = self.main_to_inner_aps(ap)

        next_state1 = self.automaton_1.step_single(state1, ap1)
        next_state2 = self.automaton_2.step_single(state2, ap2)
        return self.inner_to_main_state(next_state1, next_state2)

    def state_dict(self):
        return {
            "inner1": self.automaton_1.state_dict(),
            "inner2": self.automaton_2.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.automaton_1.load_state_dict("inner1")
        self.automaton_2.load_state_dict("inner2")

    def target_q_values(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        states1, states2 = self.main_to_inner_states(aut_states)
        aps1, aps2 = self.main_to_inner_aps(aps)
        weights1 = self.automaton_1.target_q_weights(states1, aps1, iter_num)
        weights2 = self.automaton_2.target_q_weights(states2, aps2, iter_num)

        return ((self.automaton_1.target_q_values(states1, aps1, iter_num) * weights1)
                + (self.target_q_values(states2, aps2, iter_num) * weights2)) / (weights1 + weights2)

    def target_q_weights(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        states1, states2 = self.main_to_inner_states(aut_states)
        aps1, aps2 = self.main_to_inner_aps(aps)

        return (self.automaton_1.target_q_weights(states1, aps1, iter_num)
                + self.automaton_2.target_q_weights(states2, aps2, iter_num)) / 2

    def update_training_observed_count(self, aut_states: torch.Tensor, aps: torch.Tensor):
        states1, states2 = self.main_to_inner_states(aut_states)
        aps1, aps2 = self.main_to_inner_aps(aps)

        self.automaton_1.update_training_observed_count(states1, aps1)
        self.automaton_2.update_training_observed_count(states2, aps2)


class ProductTargetAutomatonIntersectionWeights(ProductTargetAutomaton):
    """
    If transition 1 in automaton 1 is taken twice, with transition 1 in automaton 2 and transition 2 in automaton 2,
    this is counted as two separate transitions altogether- the fact that transition 1-1 was taken twice is not relevant
    """

    def __init__(self, automaton1, automaton2, device: torch.device):
        super().__init__(automaton1, automaton2)
        self.target_q_count = torch.zeros((self.num_states, self.num_aps), dtype=torch.int32, device=device)

    def update_training_observed_count(self, aut_states: torch.Tensor, aps: torch.Tensor):
        super().update_training_observed_count(aut_states, aps)

        # aut_states: [0, 1, 0, 1]
        # aps: [0, 0, 0, 1]
        # output: [[0, 0], [1, 0], [0, 0], [1, 1]]
        aut_state_ap = torch.stack((aut_states, aps), dim=1)

        # indices: [[0, 0], [1, 0], [1, 1]
        # counts: [2, 1, 1]
        indices, counts = torch.unique(aut_state_ap, return_counts=True, dim=0)
        self.target_q_count[indices[:, 0], indices[:, 1]] += counts

    def target_q_weights(self, aut_states: torch.Tensor, aps: torch.Tensor, iter_num: int) -> torch.Tensor:
        """Honestly not quite sure what would be a good function here"""
        raise NotImplemented

    def state_dict(self):
        return {
            'super': super().state_dict(),
            'target_q_count': self.target_q_count
        }

    def load_state_dict(self, state_dict):
        self.target_q_count = state_dict['target_q_count']
        super().load_state_dict(state_dict['super'])
