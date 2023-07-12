import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.labyrinth_7 import labyrinth_rew_per_step_env_config_7
from automaton_transfer.run.env.labyrinth import labyrinth_aps, labyrinth_ltlf
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(labyrinth_rew_per_step_env_config_7, "labyrinth_teacher_rew_per_step_7_productMDP",
                           device, agent_cls=OneHotAutomatonAfterFeatureExtractorAgent, aps=labyrinth_aps,
                           ltlf=labyrinth_ltlf)

if __name__ == '__main__':
    run_training(config)
