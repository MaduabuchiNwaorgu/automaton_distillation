import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.door_quest import door_quest_env_config
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.door_quest import door_quest_aps, door_quest_ltlf

device = torch.device("cuda:0")
config = teacher_config_v1(dungeon_quest_env_config, "door_quest_teacher", device, 
                            agent_cls=OneHotAutomatonAfterFeatureExtractorAgent)

if __name__ == '__main__':
    run_training(config)
