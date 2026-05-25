import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.gold_mine_7 import gold_mine_rew_per_step_env_config_7_cont
from discrete.run.env.gold_mine_7 import gold_mine_automaton, gold_mine_ap_extractor
from discrete.run.utils import student_config_reward_machine
from discrete.lib.agent.TD3_Agent import TD3_Agent


device = torch.device("cuda:0")
config = student_config_reward_machine(
    env_config=gold_mine_rew_per_step_env_config_7_cont,
    teacher_run_name="gold_mine_machine_rew_per_step",
    student_run_name="gold_mine_target_machine_rew_per_step_CRM_TD3",
    device=device,
    agent_cls=TD3_Agent,
    max_training_steps=int(1e6),
    automaton=gold_mine_automaton,
    next_flag=True
)

# Add automaton to config
config = config._replace(automaton=gold_mine_automaton, ap_extractor=gold_mine_ap_extractor)

if __name__ == '__main__':
    run_training(config)
