import torch

from automaton_transfer.run.utils import construct_ap_extractor_automaton
from automaton_transfer.run.reward_machine.reward_machine import RewardMachine
from automaton_transfer.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf

device = torch.device("cuda:0")

automaton, _ = construct_ap_extractor_automaton(dungeon_quest_aps, dungeon_quest_ltlf, device)
print([ap.name for ap in dungeon_quest_aps])
print(automaton.adj_mat)