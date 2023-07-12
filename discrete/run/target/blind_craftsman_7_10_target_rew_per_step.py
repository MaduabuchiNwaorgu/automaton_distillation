import torch

from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_rew_per_step_env_config
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=blind_craftsman_rew_per_step_env_config,
    teacher_run_name="blind_craftsman_teacher_rew_per_step_7",
    student_run_name="blind_craftsman_7_10_target_rew_per_step",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    }
)

if __name__ == '__main__':
    run_training(config)
