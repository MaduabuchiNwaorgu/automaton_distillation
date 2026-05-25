import torch

from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles
from discrete.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config_7_obstacles, "blind_craftsman_obstacles_teacher_rew_per_step", device)

if __name__ == '__main__':
    run_training(config)
