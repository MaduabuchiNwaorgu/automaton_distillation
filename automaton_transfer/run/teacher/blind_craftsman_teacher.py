import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_env_config
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf

device = torch.device("cuda:0")
config = teacher_config_v1(blind_craftsman_env_config, "blind_craftsman_teacher", device)

if __name__ == '__main__':
    run_training(config)
