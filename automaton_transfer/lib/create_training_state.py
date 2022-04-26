from typing import Tuple

import torch

from automaton_transfer.lib.agent.agent import Agent
from automaton_transfer.lib.automaton.ap_extractor import APExtractor
from automaton_transfer.lib.automaton.automaton import Automaton
from automaton_transfer.lib.checkpoint import checkpoint_exists, load_checkpoint
from automaton_transfer.lib.config import Configuration
from automaton_transfer.lib.env.util import make_env
from automaton_transfer.lib.rollout_buffer import RolloutBuffer


def create_training_state(config: Configuration) -> Tuple[Agent, RolloutBuffer, APExtractor, Automaton, int]:
    """
    Loads training state from a checkpoint, or creates a default training state if no checkpoint exists
    """
    sample_env = make_env(config.env_config)
    agent = config.agent_cls.create_agent(sample_env.observation_space.shape, config.automaton.num_states,
                                          sample_env.action_space.n).to(config.device)
    rollout_buffer = config.rollout_buffer_config.rollout_buffer_cls.create_empty(
        capacity=config.rollout_buffer_config.capacity,
        input_shape=sample_env.observation_space.shape,
        state_dtype=getattr(torch, sample_env.observation_space.dtype.name),  # Convert np dtype to torch dtype
        device=config.device
    )
    ap_extractor = config.ap_extractor
    automaton = config.automaton

    start_iter = 0

    if checkpoint_exists(config):
        print("Loading from checkpoint")
        checkpoint = load_checkpoint(config)
        start_iter = checkpoint.iter_num + 1
        ap_extractor.load_state_dict(checkpoint.ap_extractor_state)
        automaton.load_state_dict(checkpoint.automaton_state)
        rollout_buffer.load_state_dict(checkpoint.rollout_buffer_state)
        agent.load_state_dict(checkpoint.agent_state)
    else:
        print("NOT Loading from checkpoint")

    return agent, rollout_buffer, ap_extractor, automaton, start_iter
