print("entered training run...")

import torch
import time
import argparse

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles_cont, blind_craftsman_aps, blind_craftsman_ltlf
from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.lib.agent.TD3_Agent import TD3_Agent
from discrete.run.utils import student_config_reward_machine

from discrete.lib.agent.AC_Agent import AC_Agent
from discrete.run.env.dungeon_quest_7 import dungeon_quest_config_7, dungeon_quest_rew_per_step_env_config_7, dungeon_quest_rew_per_step_env_config_7_cont, dungeon_quest_aps, dungeon_quest_ltlf

print("imported all dependencies, checking for cuda")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to handle command line arguments for ALR, CLR, and Gamma.")
    
    # Add arguments
    parser.add_argument('--alr', type=float, default=0.001, help='Actor Learning Rate')
    parser.add_argument('--clr', type=float, default=0.001, help='Critic Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor (Gamma)')
    parser.add_argument('--batch-size', type=int, default=128, help='Buffer Batch Size')
    parser.add_argument('--tau', type=float, default=0.005, help='Target Transfer Tau')
    parser.add_argument('--total-steps', type=int, default=int(2e6), help='Buffer Batch Size')
    parser.add_argument('--path-to-out', type=str, default="", help='Path to place plots')

    
    # Parse arguments from command line
    args = parser.parse_args()
    
    # Assign parsed values to variables
    alr = args.alr
    clr = args.clr
    gamma = args.gamma
    batch_size = args.batch_size
    tau = args.tau
    max_training_steps = int(args.total_steps)
    
    config = student_config_reward_machine(
        env_config=blind_craftsman_rew_per_step_env_config_7_obstacles_cont,
        teacher_run_name="blind_craftsman_machine_rew_per_step",
        student_run_name="blind_craftsman_7_obstacles_target_rew_per_step_TD3_CPREP",
        agent_cls=TD3_Agent,
        device=device,
        aps=blind_craftsman_aps,
        ltlf=blind_craftsman_ltlf,
        next_flag=True)


    print("\n\n============================================")
    print(f"Training Teacher / Independent DDPG Agent")
    print(f"Max Training Steps: {max_training_steps}")
    print(f"LTLF: {blind_craftsman_ltlf}")
    print(f"this is the tau {tau}")
    # print(f"Hyperparameters: {}")
    print("============================================\n\n")
    start_time = time.time()
    run_training(config)
    print(f"Total elapsed time: {time.time() - start_time}")

