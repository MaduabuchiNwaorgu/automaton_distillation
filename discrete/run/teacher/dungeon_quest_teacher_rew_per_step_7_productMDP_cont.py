import torch
import argparse

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7_cont, dungeon_quest_aps, dungeon_quest_ltlf
# from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import teacher_config_v1

from discrete.lib.agent.TD3_Agent import TD3_Agent

device = torch.device("cuda:0")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")
    # assert False

max_training_steps=int(2000000)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to handle command line arguments for ALR, CLR, and Gamma.")
    
    # Add arguments
    parser.add_argument('--alr', type=float, default=0.001, help='Actor Learning Rate')
    parser.add_argument('--clr', type=float, default=0.001, help='Critic Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor (Gamma)')
    parser.add_argument('--batch-size', type=int, default=64, help='Buffer Batch Size')
    parser.add_argument('--tau', type=float, default=0.005, help='Target Transfer Tau')
    parser.add_argument('--total-steps', type=int, default=int(1e6), help='Buffer Batch Size')
    parser.add_argument('--path-to-out', type=str, default=None, help='Path to place plots')

    
    # Parse arguments from command line
    args = parser.parse_args()
    
    # Assign parsed values to variables
    alr = args.alr
    clr = args.clr
    gamma = args.gamma
    batch_size = args.batch_size
    tau = args.tau
    max_training_steps = int(args.total_steps)
    path_to_out = args.path_to_out

    config = teacher_config_v1(dungeon_quest_rew_per_step_env_config_7_cont, 
                           "dungeon_quest_teacher_rew_per_step_7_productMDP_TD3",
                           device, 
                           agent_cls=TD3_Agent, 
                           aps=dungeon_quest_aps,
                           ltlf=dungeon_quest_ltlf, 
                           max_training_steps=max_training_steps
                           )
    
    print("\n\n============================================")
    print(f"Training Teacher / Independent TD3 Agent")
    print(f"Max Training Steps: {max_training_steps}")
    print(f"LTLF: {dungeon_quest_ltlf}")
    print("============================================\n\n")

    run_training(config)
