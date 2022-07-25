import torch

from automaton_transfer.run.utils import construct_ap_extractor_automaton
from automaton_transfer.lib.automaton.reward_machine import RewardMachine
from automaton_transfer.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf

device = torch.device("cuda:0")

automaton, _ = construct_ap_extractor_automaton(blind_craftsman_aps, blind_craftsman_ltlf, device)
print([ap.name for ap in blind_craftsman_aps])
print(automaton.adj_mat)

reward_adj_list = torch.zeros_like(automaton.adj_mat)

# success = +100
reward_adj_list[automaton.adj_mat == 2] += 100

# no reward in terminal states
terminal_states = torch.as_tensor([0,0,1,0], dtype=torch.float32, device=device)

rm = RewardMachine(automaton, reward_adj_list, terminal_states, "blind_craftsman_machine", device)