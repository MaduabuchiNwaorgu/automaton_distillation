import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.obtain_diamond_10 import diamond_basic_env_config, obtain_diamond_aps, \
    obtain_diamond_ltlf
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(diamond_basic_env_config, "obtain_diamond_teacher_10", device, max_training_steps=int(1e7),
                           ltlf=obtain_diamond_ltlf, aps=obtain_diamond_aps)

if __name__ == '__main__':
    run_training(config)
