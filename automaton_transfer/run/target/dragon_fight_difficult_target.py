import torch

from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.dragon_fight_difficult import dragon_fight_config
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=dragon_fight_config,
    teacher_run_name="dragon_fight_difficult_teacher",
    student_run_name="dragon_fight_difficult_target",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.001
    },
    max_training_steps=int(2e6)
)

if __name__ == '__main__':
    run_training(config)
