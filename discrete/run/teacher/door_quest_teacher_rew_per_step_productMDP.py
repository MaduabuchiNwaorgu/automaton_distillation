import torch

from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.door_quest import door_quest_rew_per_step_env_config
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.door_quest import door_quest_aps, door_quest_ltlf
from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent

device = torch.device("cuda:0")
config = teacher_config_v1(door_quest_rew_per_step_env_config, "door_quest_teacher_rew_per_step_productMDP", device, 
                            agent_cls=OneHotAutomatonAfterFeatureExtractorAgent, aps=door_quest_aps, ltlf=door_quest_ltlf)

if __name__ == '__main__':
    run_training(config)
