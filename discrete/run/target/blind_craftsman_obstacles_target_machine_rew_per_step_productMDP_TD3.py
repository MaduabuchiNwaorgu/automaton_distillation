import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles_cont
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.run.utils import student_config_v1
from discrete.lib.agent.TD3_Agent import TD3_Agent


device = torch.device("cuda:0")
config = student_config_v1(
    env_config=blind_craftsman_rew_per_step_env_config_7_obstacles_cont,
    teacher_run_name="blind_craftsman_machine_rew_per_step",
    student_run_name="blind_craftsman_obstacles_target_machine_rew_per_step_productMDP_TD3",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    agent_cls=TD3_Agent,
    aps=blind_craftsman_aps,
    ltlf=blind_craftsman_ltlf
)

if __name__ == '__main__':
    run_training(config)
