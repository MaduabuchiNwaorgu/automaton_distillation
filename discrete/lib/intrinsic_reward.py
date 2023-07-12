import abc
from typing import Generic, TypeVar, List

import torch

# Note that intrinsic reward wasn't actually used, but there was no need to remove this for now

T = TypeVar("T")


class IntrinsicRewardBatchCalculator(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def create_state(self, num_in_batch: int) -> T:
        """
        :return: The state of an intrinsic reward calculator for num_in_batch separate instances
        """
        pass

    @abc.abstractmethod
    def calc_intr_rewards_batch(self, calculator_state: T, current_states, actions, next_states, ext_rewards, dones,
                                current_aut_states, aps, next_aut_states) -> torch.Tensor:
        """This will mutate calculator_state. If done is passed in, calculator state should automatically reset"""
        pass


class IntrinsicRewardCalculatorBatchWrapper(Generic[T], IntrinsicRewardBatchCalculator[List[T]]):
    """Wrap a single intrinsic-reward calculator to work with batches"""

    def __init__(self, inner_calc: "IntrinsicRewardCalculator[T]", device: torch.device):
        self.inner_calc = inner_calc
        self.device = device

    def create_state(self, num_in_batch: int) -> List[T]:
        return [self.inner_calc.create_state() for _ in range(num_in_batch)]

    def calc_intr_rewards_batch(self, batch_calculator_state: List[T], *other_args) -> torch.Tensor:
        inner_rewards = []
        for i, calc_state in enumerate(batch_calculator_state):
            other_args_indexed = tuple(arg[i] for arg in other_args)
            inner_rewards.append(self.inner_calc.calc_intr_reward(calc_state, *other_args_indexed))

        return torch.as_tensor(inner_rewards, device=self.device)


class IntrinsicRewardCalculator(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def create_state(self) -> T:
        pass

    @abc.abstractmethod
    def calc_intr_reward(self, calculator_state: T, current_state, action, next_state, ext_reward, done,
                         current_aut_state, ap, next_aut_state) -> float:
        """This will mutate calculator_state. If done is passed in, calculator state should automatically reset"""
        pass


class DummyIntrinsicRewardCalculator(IntrinsicRewardCalculator[None]):
    def create_state(self) -> None:
        return None

    def calc_intr_reward(self, calculator_state: None, current_state, action, next_state, ext_reward, done,
                         current_aut_state, ap, next_aut_state) -> float:
        return 0.0
