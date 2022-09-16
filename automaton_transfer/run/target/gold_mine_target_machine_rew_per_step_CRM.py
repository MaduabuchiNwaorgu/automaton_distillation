import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.automaton.target_automaton import RewardShapingTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.gold_mine import gold_mine_rew_per_step_env_config
from automaton_transfer.run.env.gold_mine import gold_mine_automaton
from automaton_transfer.run.utils import student_config_reward_machine

device = torch.device("cuda:0")
config = student_config_reward_machine(
    env_config=gold_mine_rew_per_step_env_config,
    teacher_run_name="gold_mine_machine_rew_per_step",
    student_run_name="gold_mine_target_machine_rew_per_step_CRM",
    agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
    device=device,
    automaton=gold_mine_automaton
)

if __name__ == '__main__':
    run_training(config)
