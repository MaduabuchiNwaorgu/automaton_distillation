import os

import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.main import run_distill
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf

device = torch.device("cuda:0")
config = teacher_config_v1(dungeon_quest_rew_per_step_env_config, "dungeon_quest_teacher_rew_per_step", device)

if __name__ == '__main__':
    run_distill(config)
