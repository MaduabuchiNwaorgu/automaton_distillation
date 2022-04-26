import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.even_easier import even_easier_env_config
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(even_easier_env_config, "even_easier_2", device)

if __name__ == '__main__':
    run_training(config)
