import torch

from automaton_transfer.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.labyrinth import labyrinth_rew_per_step_env_config
from automaton_transfer.run.env.labyrinth import labyrinth_aps, labyrinth_ltlf
from automaton_transfer.run.utils import student_config_v1

device = torch.device("cuda:0")
config = student_config_v1(
    env_config=labyrinth_rew_per_step_env_config,
    teacher_run_name="labyrinth_teacher_rew_per_step_7_productMDP",
    student_run_name="labyrinth_7_10_target_rew_per_step_productMDP",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    aps=labyrinth_aps,
    ltlf=labyrinth_ltlf
)

if __name__ == '__main__':
    run_training(config)
