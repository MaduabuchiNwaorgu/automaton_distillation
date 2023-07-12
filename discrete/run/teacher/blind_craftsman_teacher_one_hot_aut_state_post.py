import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_rew_per_step_env_config
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config, "blind_craftsman_teacher_one_hot_aut_state_post",
                           device,
                           agent_cls=OneHotAutomatonAfterFeatureExtractorAgent)

if __name__ == '__main__':
    run_training(config)
