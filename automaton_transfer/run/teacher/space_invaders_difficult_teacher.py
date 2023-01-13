import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.space_invaders_difficult import space_invaders_config
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(space_invaders_config, "space_invaders_difficult_teacher",
                           device, max_training_steps=int(5e6))

if __name__ == '__main__':
    run_training(config)
