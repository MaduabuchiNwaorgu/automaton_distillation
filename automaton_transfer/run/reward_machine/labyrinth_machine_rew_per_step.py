import torch

from automaton_transfer.run.utils import construct_ap_extractor_automaton
from automaton_transfer.lib.automaton.reward_machine import RewardMachine
from automaton_transfer.run.env.labyrinth import labyrinth_aps, labyrinth_ltlf

device = torch.device("cuda:0")

automaton, _ = construct_ap_extractor_automaton(labyrinth_aps, labyrinth_ltlf, device)
print([ap.name for ap in labyrinth_aps])
print(automaton.adj_mat)

# -0.1 per step
reward_adj_list = -0.1 * torch.ones_like(automaton.adj_mat)

# success = +100
reward_adj_list[automaton.adj_mat == 1] += 100

# no reward in terminal states
# terminal_states = torch.as_tensor([0,1,0,0,1,0,0,0,0,0], dtype=torch.float, device=device)

rm = RewardMachine(automaton, reward_adj_list, terminal_states, "labyrinth_machine_rew_per_step", device)