import torch

from automaton_transfer.run.utils import construct_ap_extractor_automaton
from automaton_transfer.lib.automaton.reward_machine import RewardMachine
from automaton_transfer.run.env.gold_mine import gold_mine_automaton, n

device = torch.device("cuda:0")

# -0.1 per step
reward_mat = -0.1 * torch.ones_like(gold_mine_automaton.adj_mat)

# success = +100
reward_mat[gold_mine_automaton.adj_mat == n+2] += 100

# gold = +n
reward_mat[0, 1<<n] += n

# silver = +1
ind = torch.arange(n, device=device)
reward_mat[ind, 1<<ind] += 1

# no reward in terminal states
terminal_states = torch.as_tensor([0]*(n+2) + [1,1], dtype=torch.float, device=device)

rm = RewardMachine(gold_mine_automaton, reward_mat, terminal_states, "gold_mine_machine_rew_per_step", device)