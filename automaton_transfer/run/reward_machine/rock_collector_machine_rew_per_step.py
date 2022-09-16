import torch

from automaton_transfer.run.utils import construct_ap_extractor_automaton
from automaton_transfer.lib.automaton.reward_machine import RewardMachine
from automaton_transfer.run.env.rock_collector import rock_collector_aps, rock_collector_ltlf

device = torch.device("cuda:0")

automaton, _ = construct_ap_extractor_automaton(rock_collector_aps, rock_collector_ltlf, device)
print([ap.name for ap in rock_collector_aps])
print(automaton.adj_mat)

# -0.1 per step
reward_adj_list = -0.1 * torch.ones_like(automaton.adj_mat)

# success = +100
reward_adj_list[automaton.adj_mat == 1] += 99

# key/sword/shield/dragon = +1
for i in range(automaton.num_states):
    reward_adj_list[i, automaton.adj_mat[i] != i] += 1

# no reward in terminal states
terminal_states = torch.as_tensor([0,0,0,0,1,0,0,0,0], dtype=torch.float, device=device)

rm = RewardMachine(automaton, reward_adj_list, terminal_states, "rock_collector_machine_rew_per_step", device)