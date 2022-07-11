import torch

from automaton_transfer.run.utils import construct_ap_extractor_automaton
from automaton_transfer.run.reward_machine.reward_machine import RewardMachine
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf

device = torch.device("cuda:0")

automaton, _ = construct_ap_extractor_automaton(dungeon_quest_aps, dungeon_quest_ltlf, device)
print([ap.name for ap in dungeon_quest_aps])
print(automaton.adj_mat)

# -0.1 per step
reward_adj_list = -0.1 * torch.ones_like(automaton.adj_mat)

# success = +99
reward_adj_list[automaton.adj_mat == 1] += 99

# key/sword/shield/dragon = +1
for i in range(automaton.num_states):
    reward_adj_list[i, automaton.adj_mat[i] != i] += 1

# no reward in terminal states
reward_adj_list[1, :] = 0
reward_adj_list[7, :] = 0

rm = RewardMachine(automaton, reward_adj_list, "dungeon_quest_machine_rew_per_step", device)