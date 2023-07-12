import torch

from automaton_transfer.lib.agent.dqn_per_automaton_state_agent import DQNPerAutomatonStateAgent
from automaton_transfer.lib.main import run_training
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_rew_per_step_env_config
from automaton_transfer.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config, "blind_craftsman_dqn_per_aut_state", device,
                           agent_cls=DQNPerAutomatonStateAgent)

if __name__ == '__main__':
    run_training(config)
