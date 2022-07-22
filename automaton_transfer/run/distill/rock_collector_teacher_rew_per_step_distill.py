import os

import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.main import run_distill
from automaton_transfer.run.env.rock_collector import rock_collector_rew_per_step_env_config
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.rock_collector import rock_collector_aps, rock_collector_ltlf

device = torch.device("cuda:0")
config = teacher_config_v1(rock_collector_rew_per_step_env_config, "rock_collector_teacher_rew_per_step", device)

if __name__ == '__main__':
    run_distill(config)
