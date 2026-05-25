import torch
import math
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from discrete.lib.agent.agent import Agent, TargetAgent
from discrete.lib.automaton.ap_extractor import APExtractor
from discrete.lib.automaton.automaton import Automaton
from discrete.lib.automaton.target_automaton import TargetAutomaton
from discrete.lib.automaton.reward_machine import RewardMachine
from discrete.lib.checkpoint import save_checkpoint, Checkpoint,checkpoint_exists, load_checkpoint
from discrete.lib.config import Configuration
from discrete.lib.create_training_state import create_training_state
from discrete.lib.env.util import make_vec_env, make_env
from discrete.lib.intrinsic_reward import IntrinsicRewardCalculatorBatchWrapper
from discrete.lib.rollout_buffer import VecRolloutBufferHelper, RolloutBuffer, CircularRolloutBuffer
from discrete.lib.updater import Updater

from discrete.lib.training import take_eps_greedy_action_from_q_values, vec_env_distinct_episodes, reset_done_aut_states, TraceHelper


class History:
    def __init__(self, ap_extractor: APExtractor,
                 automaton: Automaton,
                 expert_policy: Agent,
                 config: Configuration,
                 ):

        self.env = make_vec_env(config.env_config, config.num_parallel_envs)
        self.policy = expert_policy
        self.automaton = automaton
        self.ap_extractor = ap_extractor
        self.config = config

        # Current states
        self.current_states = torch.as_tensor(self.env.reset(), device=config.device)
        self.current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                               device=config.device, dtype=torch.long)
        self.trace_helper = TraceHelper(config.num_parallel_envs)

        self.collected_states = []
        
    def collect_expert_samples(self, num_steps: int):
        """
        Step the environment `num_steps` times using the teacher’s policy,
        and store all transitions in the rollout buffer.
        """
       
        for _ in range(num_steps):
            # with torch.no_grad():
            q_values = self.policy.calc_q_values_batch(
                torch.as_tensor(self.current_states, device=self.config.device, dtype=torch.float),
                                             self.current_aut_states
            )
            actions = take_eps_greedy_action_from_q_values(q_values, self.config.epsilon)

            # Step env
            obs, rewards, dones, infos = self.env.step(actions)
            obs = torch.as_tensor(obs, device=self.config.device)
            rewards = torch.as_tensor(rewards, device=self.config.device)
            dones = torch.as_tensor(dones, device=self.config.device)

            # Distinct episodes logic
            states_after_current, next_states = vec_env_distinct_episodes(obs, infos)

            # AP extraction
            aps_after_current = self.ap_extractor.extract_aps_batch(states_after_current, infos)
            self.trace_helper.add_aps(aps_after_current)
            aut_states_after_current = self.automaton.step_batch(self.current_aut_states, aps_after_current)
            self.trace_helper.finalize_step(dones)

            next_aut_states = reset_done_aut_states(aut_states_after_current, dones, self.automaton)
            self.current_states = next_states
            self.current_aut_states = next_aut_states

            self.collected_states.append(obs)
            # Update current states
            self.current_states = next_states

    def get_minibatches(self, batch_size: int):
        """
        Yield mini-batches of states for your representation learning model.
        """
        data_size = len(self.collected_states)
        if data_size == 0:
            return
        
        # Shuffle the states if you want random samples
        random.shuffle(self.collected_states)
        
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            batch = self.collected_states[start_idx:end_idx]
            # Stack them into a tensor of shape (batch_size, *obs_shape)
            yield torch.stack(batch, dim=0)
