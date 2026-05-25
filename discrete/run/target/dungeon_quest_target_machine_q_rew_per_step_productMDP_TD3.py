#Static Student

import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7, dungeon_quest_rew_per_step_env_config_7_cont
from discrete.run.env.dungeon_quest_7 import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import student_config_v1
from discrete.lib.agent.TD3_Agent import TD3_Agent


device = torch.device("cuda:0")
config = student_config_v1(
    env_config=dungeon_quest_rew_per_step_env_config_7_cont,
    teacher_run_name="dungeon_quest_machine_rew_per_step",
    student_run_name="dungeon_quest_target_machine_q_rew_per_step_productMDP_TD3",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    agent_cls=TD3_Agent,
    aps=dungeon_quest_aps,
    ltlf=dungeon_quest_ltlf,
    max_training_steps=int(1e6)
)

if __name__ == '__main__':
    run_training(config)
