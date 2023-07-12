import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.blind_craftsman_7 import blind_craftsman_env_config_7
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf


device = torch.device("cuda:0")
config = teacher_config_v1(blind_craftsman_env_config_7, "blind_craftsman_teacher_7", device, aps=blind_craftsman_aps,
                           ltlf=blind_craftsman_ltlf)

if __name__ == '__main__':
    run_training(config)
