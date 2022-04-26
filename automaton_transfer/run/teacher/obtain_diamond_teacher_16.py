import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.obtain_diamond_16 import diamond_basic_env_config
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(diamond_basic_env_config, "obtain_diamond_teacher_16", device, max_training_steps=int(1e7))

if __name__ == '__main__':
    run_training(config)
