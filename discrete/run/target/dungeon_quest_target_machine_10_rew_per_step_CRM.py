import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.target_automaton import RewardShapingTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import student_config_reward_machine

device = torch.device("cuda:0")
config = student_config_reward_machine(
    env_config=dungeon_quest_rew_per_step_env_config,
    teacher_run_name="dungeon_quest_machine_rew_per_step",
    student_run_name="dungeon_quest_target_machine_10_rew_per_step_CRM",
    agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
    device=device,
    aps=dungeon_quest_aps,
    ltlf=dungeon_quest_ltlf,
    max_training_steps=int(2e6),
    checkpoint_name = "dungeon_quest_target_machine_7_rew_per_step_CRM"
)

if __name__ == '__main__':
    run_training(config)