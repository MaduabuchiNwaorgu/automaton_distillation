import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.dungeon_quest_7 import dungeon_quest_env_config_7
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf


device = torch.device("cuda:0")
config = teacher_config_v1(dungeon_quest_env_config_7, "dungeon_quest_teacher_7", device, aps=dungeon_quest_aps,
                           ltlf=dungeon_quest_ltlf)

if __name__ == '__main__':
    run_training(config)
