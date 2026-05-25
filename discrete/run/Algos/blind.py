import torch
import argparse

from discrete.lib.main import run_policy_distillation
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cuda:0")
config = teacher_config_v1(
    dungeon_quest_rew_per_step_env_config,
    run_name="dungeon_quest_teacher_rew_per_step",
    # student_run_name="blind_craftsman_target_machine_7_rew_per_step_CRM",
    device = device,
    max_training_steps=int(2e6))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iterations", type=int, default=1000000)
    parser.add_argument("--student_batch_size", type=int, default=512)
    parser.add_argument("--loss_metric", type=str, default="kl_divergence")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--test_train_interval", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=2000)
    # parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    run_policy_distillation(config, run_name=config.run_name, args=args)


