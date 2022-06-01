import torch

from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.space_invaders_difficult import space_invaders_config, \
    space_invaders_aps, space_invaders_ltlf
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=space_invaders_config,
    teacher_run_name="space_invaders_easy_teacher",
    student_run_name="space_invaders_easy_to_difficult_target",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.9999
    },
    aps=space_invaders_aps,
    ltlf=space_invaders_ltlf,
    max_training_steps=int(2e6)
)
if __name__ == '__main__':
    run_training(config)
