import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.main import run_distill
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_rew_per_step_env_config
from automaton_transfer.run.utils import teacher_config_v1
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf

device = torch.device("cuda:0")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config, "blind_craftsman_teacher_rew_per_step", device, 
                            aps=blind_craftsman_aps, ltlf=blind_craftsman_ltlf)

if __name__ == '__main__':
    run_distill(config)
