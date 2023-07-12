import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.automaton.reward_machine import RewardMachine
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=dungeon_quest_rew_per_step_env_config,
    teacher_run_name="dungeon_quest_machine_rew_per_step",
    student_run_name="dungeon_quest_target_machine_q_rew_per_step_CRM_productMDP",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    reward_machine=True,
    agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
    aps=dungeon_quest_aps,
    ltlf=dungeon_quest_ltlf
)

if __name__ == '__main__':
    run_training(config)
