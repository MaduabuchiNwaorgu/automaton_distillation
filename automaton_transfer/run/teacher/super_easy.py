import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.super_easy import super_easy_env_config
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(super_easy_env_config, "super_easy", device)

if __name__ == '__main__':
    run_training(config)
