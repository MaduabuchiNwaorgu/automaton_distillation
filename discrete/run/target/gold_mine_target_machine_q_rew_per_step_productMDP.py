import torch

from automaton_transfer.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.gold_mine import gold_mine_rew_per_step_env_config
from automaton_transfer.run.env.gold_mine import gold_mine_automaton, gold_mine_ap_extractor
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=gold_mine_rew_per_step_env_config,
    teacher_run_name="gold_mine_machine_rew_per_step",
    student_run_name="gold_mine_target_machine_q_rew_per_step_productMDP",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    agent_cls=OneHotAutomatonAfterFeatureExtractorAgent
)

# Add automaton to config
config = config._replace(automaton=gold_mine_automaton, ap_extractor=gold_mine_ap_extractor)

if __name__ == '__main__':
    run_training(config)
