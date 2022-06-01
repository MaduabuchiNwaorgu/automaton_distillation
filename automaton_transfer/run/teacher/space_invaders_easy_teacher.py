import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.space_invaders_easy import space_invaders_config, space_invaders_aps, space_invaders_ltlf
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(space_invaders_config, "space_invaders_easy_teacher",
                           device, max_training_steps=int(5e6), aps=space_invaders_aps, ltlf=space_invaders_ltlf)

if __name__ == '__main__':
    run_training(config)
