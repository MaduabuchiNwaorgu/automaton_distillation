import abc
from abc import ABC, ABCMeta
from typing import NamedTuple, List, Any, Tuple, Iterator, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class TraceStep(NamedTuple):
    """A single step in a sequence"""
    state: np.ndarray
    action: np.ndarray
    ext_reward: float
    done: bool
    # Note that fields below this point are calculable from the fields above here, given the AP extractor, automaton, and trace history
    starting_aut_state: int
    ap: int
    intr_reward: float


class RolloutSample(NamedTuple):
    """
    A series of steps, but batched for efficiency.
    These aren't necessarily sequential; random sampling can produce this.
    All of these have the same length
    """
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    aut_states: torch.Tensor
    aps: torch.Tensor
    next_aut_states: torch.Tensor


class RolloutBuffer(ABC, metaclass=ABCMeta):
    @classmethod
    @abc.abstractmethod
    def create_empty(cls, capacity: int, input_shape: Tuple, state_dtype: torch.dtype,
                     device: torch.device):
        pass

    @abc.abstractmethod
    def add_episode(self, trace_steps: List[TraceStep], last_state: torch.Tensor, last_aut_state: int):
        """
        Add an episode to the replay buffer
        :param trace_steps A trace of the episode, excluding the terminal state
        :param last_state The terminal state, seen when done=true
        :param last_aut_state The last automaton state, based on transitioning the automaton given the APs extracted from the last state
        """
        pass

    @abc.abstractmethod
    def sample(self, batch_size: int, num_aut_states: Optional[int] = None, priority_scale: float = 0.0) -> Tuple[
        RolloutSample, torch.Tensor, torch.Tensor]:
        """Don't include samples that correspond to aut states above num_aut_states"""
        pass

    @abc.abstractmethod
    def get_list_of_all_traces(self) -> List[List[int]]:
        """Specifically, the indices of the traces"""
        pass

    @abc.abstractmethod
    def num_filled_approx(self) -> int:
        """Any reasonable ballpark estimate"""
        pass

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abc.abstractmethod
    def set_priorities(self, indices: torch.Tensor, errors: torch.Tensor):
        """For priority experience replay"""
        pass

    @abc.abstractmethod
    def reset_all_priorities(self):
        """Call this on each new generation"""
        pass

    @abc.abstractmethod
    def iterate_episode_indices(self) -> Iterator[torch.Tensor]:
        pass

    @abc.abstractmethod
    def get_rollout_sample_from_indices(self, indices: torch.Tensor) -> RolloutSample:
        pass


class CircularRolloutBuffer(RolloutBuffer):
    """Inspired by the Deepsynth rollout buffer, but adapted a bit to specific needs
    (for example, only samples from whole episodes)
    Uses vectorized ops as much as possible"""

    @classmethod
    def create_empty(cls, capacity: int, input_shape: Tuple, state_dtype: torch.dtype,
                     device: torch.device):
        return cls(capacity, input_shape, state_dtype, device)

    def __init__(self, capacity=1000000, input_shape=(84, 84),
                 state_dtype: torch.dtype = torch.uint8, device: torch.device = "cuda:0"):
        """
        Arguments:
            capacity: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            device: Where to store everything
        """
        self.capacity = capacity
        self.input_shape = input_shape
        self.device = device
        self.state_dtype = state_dtype

        self.write_head = 0  # INVARIANT: Always 0 <= write_head < capacity
        self.num_filled = 0  # All spaces that are filled, including meaningless buffer

        self.actions = torch.zeros((self.capacity,), dtype=torch.long, device=device)
        self.intr_rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self.ext_rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self.dfa_states = torch.zeros((self.capacity,), dtype=torch.long, device=device)
        self.aps = torch.zeros((self.capacity,), dtype=torch.long, device=device)
        self.frames = torch.zeros((self.capacity, *self.input_shape), dtype=state_dtype, device=device)
        self.terminal_flags = torch.zeros((self.capacity,), dtype=torch.bool, device=device)
        self.valid_samples = torch.zeros((self.capacity,), dtype=torch.bool, device=device)
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float, device=device)

        # Useful so that we don't have parts of episodes: the whole thing gets zeroed out when necessary
        self.next_episode_start = -torch.ones((self.capacity,), dtype=torch.long, device=device)

    def write_to_and_move_head_batch(self, mapping: List[Tuple[torch.Tensor, torch.Tensor]]):
        # See add_episode for example of usage

        len_to_write = mapping[0][1].shape[0]  # The first src tensor's first dimension

        for dest_tensor, src_tensor in mapping:
            if self.write_head + len_to_write <= self.capacity:
                # Everything fits before we need to roll over to the beginning of the tape
                dest_tensor[self.write_head: self.write_head + len_to_write] = src_tensor
            else:
                # Need to split the write in two sections
                write_len_before_split = self.capacity - self.write_head
                write_len_after_split = len_to_write - write_len_before_split

                dest_tensor[self.write_head:] = src_tensor[:write_len_before_split]
                dest_tensor[:write_len_after_split] = src_tensor[write_len_before_split:]

        self.write_head = (self.write_head + len_to_write) % self.capacity

    def write_to_and_move_head(self, mapping: List[Tuple[torch.Tensor, Any]]):
        for dest_tensor, src_tensor in mapping:
            dest_tensor[self.write_head] = src_tensor

        self.write_head = (self.write_head + 1) % self.capacity

    def clear_between(self, start_idx, end_idx):
        """
        Zero out between two indices. Does not update num_filled. ASSUMPTION end_idx >= start_idx
        :param end_idx: Exclusive
        """
        self.actions[start_idx:end_idx] = 0
        self.intr_rewards[start_idx:end_idx] = 0
        self.ext_rewards[start_idx:end_idx] = 0
        self.dfa_states[start_idx:end_idx] = 0
        self.aps[start_idx:end_idx] = 0
        self.frames[start_idx:end_idx] = 0
        self.terminal_flags[start_idx:end_idx] = 0
        self.valid_samples[start_idx:end_idx] = False
        self.next_episode_start[start_idx:end_idx] = -1
        self.priorities[start_idx:end_idx] = 0

    def ensure_space_and_zero_out(self, space_required: int):
        """
        Zero out enough so that there are space_required steps available in front of the write_head,
        plus to the end of an episode if we end up zeroing out to the middle of an episode
        """
        free_spaces_in_front_of_write_head = self.capacity - self.num_filled
        if free_spaces_in_front_of_write_head >= space_required:
            return

        next_entry_allowed_to_be_filled = (self.write_head + space_required) % self.capacity
        last_entry_that_needs_to_be_erased = (next_entry_allowed_to_be_filled - 1) % self.capacity
        erase_head = (self.write_head + free_spaces_in_front_of_write_head) % self.capacity

        # We don't want to leave partial episodes hanging around, so we also zero out the rest of the episode,
        # if it turns out we would erase until the middle of an episode
        # This idx is exclusive
        clear_between_erase_head_and_this_idx = int(self.next_episode_start[last_entry_that_needs_to_be_erased])

        # The if condition at the beginning of the function should prevent this from happening
        assert clear_between_erase_head_and_this_idx != -1, "Trying to zero out non-existing episode"

        if erase_head <= clear_between_erase_head_and_this_idx:  # No need to loop around
            self.clear_between(erase_head, clear_between_erase_head_and_this_idx)
        else:  # Loop around to the start
            self.clear_between(erase_head, self.capacity)
            self.clear_between(0, clear_between_erase_head_and_this_idx)

        numel_erased = (clear_between_erase_head_and_this_idx - erase_head) % self.capacity
        self.num_filled -= numel_erased

    def add_episode(self, trace_steps: List[TraceStep], last_state: torch.Tensor, last_aut_state: int):
        # Plus one because of last_state
        total_buffer_space_required = len(trace_steps) + 1
        next_episode_start = (self.write_head + total_buffer_space_required) % self.capacity
        self.ensure_space_and_zero_out(total_buffer_space_required)

        max_priority = max(float(self.priorities.max()), 1.0)

        for t_step in trace_steps:
            self.write_to_and_move_head([
                (self.actions, t_step.action),
                (self.intr_rewards, t_step.intr_reward),
                (self.ext_rewards, t_step.ext_reward),
                (self.dfa_states, t_step.starting_aut_state),
                (self.aps, t_step.ap),
                (self.frames, t_step.state),
                (self.terminal_flags, t_step.done),
                (self.valid_samples, True),
                (self.next_episode_start, next_episode_start),
                (self.priorities, max_priority)
            ])

        self.write_to_and_move_head([
            (self.frames, last_state),
            (self.dfa_states, last_aut_state),
            (self.next_episode_start, next_episode_start)
        ])

        self.num_filled += total_buffer_space_required

    def get_states_and_next_states(self, indices) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the states and next states, taking history into account (read the comments in the function body)
        :param indices: A tensor of indices
        :return: Current states and next states
        """

        """
        [[index0],
         ...
         [index255]]
        """
        indices = indices.unsqueeze(1)

        """
        always produces
        [[0, 1]]
        """
        history_range = torch.arange(0, 2, device=self.device).unsqueeze(0)

        """
        [[index0  , index0 + 1  ],
         ...
         [index255, index255 + 1]
        """
        history_indices = (history_range + indices).long() % self.capacity

        all_states = self.frames[history_indices].float()

        current_states = all_states[:, 0]
        next_states = all_states[:, 1]

        return current_states, next_states

    def get_rollout_sample_from_indices(self, indices: torch.Tensor) -> RolloutSample:
        current_states, next_states = self.get_states_and_next_states(indices)
        aut_states = self.dfa_states[indices]
        next_indices = (indices + 1) % self.capacity
        next_aut_states = self.dfa_states[next_indices]

        rewards = self.intr_rewards[indices] + self.ext_rewards[indices]

        return RolloutSample(
            states=current_states,
            actions=self.actions[indices],
            rewards=rewards,
            next_states=next_states,
            dones=self.terminal_flags[indices],
            aut_states=aut_states,
            aps=self.aps[indices],
            next_aut_states=next_aut_states
        )

    def sample(self, batch_size: int, num_aut_states: Optional[int] = None, priority_scale=0.0, reward_machine=None) -> Tuple[
        RolloutSample, torch.Tensor, torch.Tensor]:
        """
        Sample from the rollout buffer
        """
        if num_aut_states is None:
            valid = self.valid_samples
        else:
            good_this_aut_state = self.dfa_states < num_aut_states
            good_next_aut_state = torch.roll(good_this_aut_state, -1, 0)

            valid = self.valid_samples & good_this_aut_state & good_next_aut_state

        scaled_priorities = torch.pow(self.priorities, priority_scale)

        sample_probs = valid.float() * scaled_priorities
        sample_probs = sample_probs / sample_probs.sum()

        indices = torch.multinomial(input=sample_probs, num_samples=batch_size, replacement=True)

        importance = 1 / sample_probs[indices]
        importance = importance / importance.max()

        rollout_sample = self.get_rollout_sample_from_indices(indices)
        
        if reward_machine:
            aut_states = torch.randint_like(rollout_sample.aut_states, low=0, high=num_aut_states).long()
            next_aut_states = reward_machine.step_batch(aut_states, rollout_sample.aps).long()
            rewards = reward_machine.reward_mat[rollout_sample.aut_states, rollout_sample.aps]
            dones = reward_machine.terminal_states[rollout_sample.next_aut_states].bool()
            
            rollout_sample = RolloutSample(
                states=rollout_sample.states,
                actions=rollout_sample.actions,
                rewards=rewards,
                next_states=rollout_sample.next_states,
                dones=dones,
                aut_states=aut_states,
                aps=rollout_sample.aps,
                next_aut_states=next_aut_states
            )
                
        
        return rollout_sample, indices, importance
    
    def sample_crm(self, batch_size: int, num_aut_states: Optional[int] = None, priority_scale=0.0) -> Tuple[
        RolloutSample, torch.Tensor, torch.Tensor]:
        
        if num_aut_states is None:
            valid = self.valid_samples
        else:
            good_this_aut_state = self.dfa_states < num_aut_states
            good_next_aut_state = torch.roll(good_this_aut_state, -1, 0)

            valid = self.valid_samples & good_this_aut_state & good_next_aut_state

        scaled_priorities = torch.pow(self.priorities, priority_scale)

        sample_probs = valid.float() * scaled_priorities
        sample_probs = sample_probs / sample_probs.sum()

        indices = torch.multinomial(input=sample_probs, num_samples=batch_size, replacement=True)

        importance = 1 / sample_probs[indices]
        importance = importance / importance.max()

        # return self.get_rollout_sample_from_indices(indices), indices, importance

    def iterate_episode_indices(self) -> Iterator[torch.Tensor]:
        """
        Iterate through all episodes. Please don't do anything that would affect write_head while iterating
        :return: The indices of each frame that is part of an episode
        """
        num_empty_slots = self.capacity - self.num_filled
        first_episode_start = (self.write_head + num_empty_slots) % self.capacity
        # What is the write head position if this was an infinite, rather than circular buffer
        write_head_premodulo = first_episode_start + self.num_filled

        assert write_head_premodulo == self.write_head or write_head_premodulo - self.capacity == self.write_head, "Did math wrong"

        iterator_head_premodulo = first_episode_start
        while iterator_head_premodulo < write_head_premodulo:  # Allowing us to potentially loop over to start once
            # Including buffer space before and after an episode
            this_episode_start = iterator_head_premodulo % self.capacity
            this_episode_len_incl_buffer = int(
                self.next_episode_start[this_episode_start] - this_episode_start) % self.capacity

            this_episode_real_len = this_episode_len_incl_buffer - 1  # Because of the state after

            indices = torch.arange(this_episode_start, this_episode_start + this_episode_real_len,
                                   device=self.device) % self.capacity

            yield indices

            iterator_head_premodulo += this_episode_len_incl_buffer

    def get_list_of_all_traces(self) -> List[List[int]]:
        return list(self.aps[indices].tolist() for indices in self.iterate_episode_indices())

    def num_filled_approx(self) -> int:
        return self.num_filled

    def set_priorities(self, indices, errors, offset=0.1):
        self.priorities[indices] = errors.abs() + offset

    def reset_all_priorities(self):
        self.priorities = self.valid_samples.float()

    def state_dict(self):
        return {
            "write_head": self.write_head,
            "num_filled": self.num_filled,
            "actions": self.actions,
            "intr_rewards": self.intr_rewards,
            "ext_rewards": self.ext_rewards,
            "dfa_states": self.dfa_states,
            "aps": self.aps,
            "frames": self.frames,
            "terminal_flags": self.terminal_flags,
            "valid_samples": self.valid_samples,
            "next_episode_start": self.next_episode_start,
            "priorities": self.priorities
        }

    def load_state_dict(self, state_dict):
        self.write_head = state_dict["write_head"]
        self.num_filled = state_dict["num_filled"]
        self.actions = state_dict["actions"]
        self.intr_rewards = state_dict["intr_rewards"]
        self.ext_rewards = state_dict["ext_rewards"]
        self.dfa_states = state_dict["dfa_states"]
        self.aps = state_dict["aps"]
        self.frames = state_dict["frames"]
        self.terminal_flags = state_dict["terminal_flags"]
        self.valid_samples = state_dict["valid_samples"]
        self.next_episode_start = state_dict["next_episode_start"]
        self.priorities = state_dict["priorities"]


class VecRolloutBufferHelper:
    """
    When using a VecEnv during training, it is a bit of a pain to actually manage all the half-completed states.
    Use VecRolloutBufferHelper to abstract that away and automatically add to the rollout buffer when an episode is done
    """

    def __init__(self, num_vec_envs: int, buffer: RolloutBuffer, logger: SummaryWriter,
                 no_done_on_out_of_time: bool):
        self.num_vec_envs = num_vec_envs
        self.in_progress_traces: List[List[TraceStep]] = [[] for _ in range(num_vec_envs)]
        self.buffer = buffer
        self.logger = logger
        self.no_done_on_out_of_time = no_done_on_out_of_time

    def add_vec_experiences(self,
                            current_states,
                            actions_after_current,
                            ext_rewards_after_current,
                            intr_rewards_after_current,
                            dones_after_current,
                            states_after_current,
                            current_aut_states,
                            aut_states_after_current,
                            aps_after_current,
                            infos,
                            global_step: int):
        len_of_dones = 0
        rew_of_dones = 0
        num_dones = 0

        for i in range(len(self.in_progress_traces)):
            this_trace: List[TraceStep] = self.in_progress_traces[i]

            this_trace.append(
                TraceStep(state=current_states[i],
                          action=actions_after_current[i],
                          ext_reward=ext_rewards_after_current[i],
                          intr_reward=intr_rewards_after_current[i],
                          done=dones_after_current[i],
                          starting_aut_state=current_aut_states[i],
                          ap=aps_after_current[i]
                          )
            )

            if dones_after_current[i]:
                len_of_dones += len(this_trace)
                rew_of_dones += sum(trace_step.ext_reward + trace_step.intr_reward for trace_step in this_trace)
                num_dones += 1

                # See comment in config about no_done_on_out_of_time
                if self.no_done_on_out_of_time and infos[i].get("TimeLimit.truncated", False):
                    this_trace[-1] = this_trace[-1]._replace(done=(~this_trace[-1].done))

                self.buffer.add_episode(this_trace, last_state=states_after_current[i],
                                        last_aut_state=aut_states_after_current[i])
                self.in_progress_traces[i] = []

        if num_dones > 0:
            self.logger.add_scalar("experience_generation/episode_len", len_of_dones / num_dones,
                                   global_step=global_step)
            self.logger.add_scalar("experience_generation/episode_rew", rew_of_dones / num_dones,
                                   global_step=global_step)