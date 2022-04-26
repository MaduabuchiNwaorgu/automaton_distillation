import torch

from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.obtain_diamond_16 import diamond_basic_env_config, obtain_diamond_aps, \
    obtain_diamond_ltlf
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=diamond_basic_env_config,
    teacher_run_name="obtain_diamond_teacher_10",
    student_run_name="obtain_diamond_10_16_target",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    aps=obtain_diamond_aps,
    ltlf=obtain_diamond_ltlf,
    max_training_steps=int(1e7)
)
if __name__ == '__main__':
    run_training(config)
