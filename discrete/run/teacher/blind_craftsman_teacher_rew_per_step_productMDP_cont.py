import torch

from discrete.lib.main import run_training
from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.run.env.blind_craftsman import blind_craftsman_rew_per_step_env_config_cont
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.run.utils import teacher_config_v1

from discrete.lib.agent.TD3_Agent import TD3_Agent

# from discrete.run.env.blind_craftsman_7 import blind_craftsman_rew_per_step_env_config_7_cont


device = torch.device("cuda")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config_cont, "blind_craftsman_teacher_rew_per_step_productMDP_cont", device,
                           agent_cls=TD3_Agent, aps=blind_craftsman_aps, 
                           ltlf=blind_craftsman_ltlf)

if __name__ == '__main__':
    run_training(config)
