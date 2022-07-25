import torch

from automaton_transfer.lib.automaton.target_automaton import RewardShapingTargetAutomaton
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.rock_collector import rock_collector_rew_per_step_env_config
from automaton_transfer.run.env.rock_collector import rock_collector_aps, rock_collector_ltlf
from automaton_transfer.run.utils import student_config_reward_machine

device = torch.device("cuda:0")
config = student_config_reward_machine(
    env_config=rock_collector_rew_per_step_env_config,
    teacher_run_name="rock_collector_machine_rew_per_step",
    student_run_name="rock_collector_target_machine_rew_per_step_CRM",
    device=device,
    aps=rock_collector_aps,
    ltlf=rock_collector_ltlf
)

if __name__ == '__main__':
    run_training(config)
