from typing import NamedTuple, Type, Dict, Optional

import gym
import torch

import automaton_transfer.lib.agent.agent as ag
import automaton_transfer.lib.automaton.ap_extractor as ape
import automaton_transfer.lib.automaton.automaton as aut
import automaton_transfer.lib.intrinsic_reward as intrew
import automaton_transfer.lib.rollout_buffer as rb


class EnvConfig(NamedTuple):
    """All the information necessary to construct an environment"""
    env_name: str  # OpenAI gym environment
    kwargs: Dict
    wrapper_cls: Optional[Type[gym.Wrapper]] = None
    wrapper_kwargs: Dict = {}


class RolloutBufferConfig(NamedTuple):
    rollout_buffer_cls: Type["rb.RolloutBuffer"]
    capacity: int
    min_size_before_training: int
    priority_scale: float


class Configuration(NamedTuple):
    """
    Everything needed to run an environment.
    I don't want to have a separate config for runtime args vs environment config, etc.
    If command-line args are necessary, just parse them in the `run` package and put them in the configuration class
    """
    env_config: EnvConfig
    num_parallel_envs: int  # Really, how many cpu cores to use
    rollout_buffer_config: RolloutBufferConfig
    agent_cls: "Type[ag.Agent]"  # How to construct an agent
    automaton: "aut.Automaton"  # A pre-constructed automaton
    epsilon: float  # For epsilon-greedy action selection
    agent_train_batch_size: int  # When training the agent
    target_agent_update_every_steps: int  # Max number of steps before updating the DDQN target
    max_training_steps: int
    checkpoint_every_steps: int
    gamma: float  # Discount factor
    intrinsic_reward_calculator: "intrew.IntrinsicRewardCalculator"
    distill: bool # Whether or not to perform online policy distillation
    temperature: float # Temperature for policy distillation
    # When training, "done" implies that Q_max(next_state)=0. However, if the episode
    # terminated due to running out of time, this assumption may not make sense.
    # Set no_done_on_out_of_time=True to not have a "done" when the episode runs out of time
    no_done_on_out_of_time: bool
    ap_extractor: "ape.APExtractor"
    device: torch.device
    run_name: str
