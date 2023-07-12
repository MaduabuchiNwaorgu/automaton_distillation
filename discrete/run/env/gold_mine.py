from collections import Counter

import torch

from automaton_transfer.lib.automaton.ltl_automaton import LTLAutomaton
from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP, MineEnvApExtractor
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

gold_mine_config = MineWorldConfig(
    shape=(10, 10),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="at_home_done", consumable=False, grid_letter="H", inventory_modifier=Counter(),
                inventory_requirements=Counter(silver=10), reward=+100, terminal=True
            ),
            fixed_placements=[(0, 0)]
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="gold", consumable=True, grid_letter="g", inventory_modifier=Counter(gold=+1, nogold=-1),
                inventory_requirements=Counter(nosilver=10), reward=+10
            ),
            fixed_placements=[],
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="Gold", consumable=True, grid_letter="G", inventory_modifier=Counter(gold=+1), 
                inventory_requirements=Counter(gold=1, copper=30), reward=+100, terminal=True
            ),
            fixed_placements=[],
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="silver", consumable=True, grid_letter="S", inventory_modifier=Counter(silver=+1, nosilver=-1),
                inventory_requirements=Counter(nogold=1), reward=+1
            ),
            fixed_placements=[],
            random_placements=10
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="copper", consumable=True, grid_letter="C", inventory_modifier=Counter(copper=+1),
                inventory_requirements=Counter(gold=1)
            ),
            fixed_placements=[],
            random_placements=30
        ),
    ],
    inventory=[
        InventoryItemConfig(name="gold", default_quantity=0, capacity=2),
        InventoryItemConfig(name="silver", default_quantity=0, capacity=10),
        InventoryItemConfig(name="copper", default_quantity=0, capacity=50),
        InventoryItemConfig(name="nogold", default_quantity=1, capacity=1),
        InventoryItemConfig(name="nosilver", default_quantity=10, capacity=10),
    ]
)

gold_mine_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": gold_mine_config}
)

gold_mine_exp_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": gold_mine_config},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

gold_mine_rew_per_step_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": gold_mine_config},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

gold_mine_aps = [
    AP(name="s1", func=MineInventoryAP(inventory_item="silver", quantity=1)),
    AP(name="s2", func=MineInventoryAP(inventory_item="silver", quantity=2)),
    AP(name="s3", func=MineInventoryAP(inventory_item="silver", quantity=3)),
    AP(name="s4", func=MineInventoryAP(inventory_item="silver", quantity=4)),
    AP(name="s5", func=MineInventoryAP(inventory_item="silver", quantity=5)),
    AP(name="s6", func=MineInventoryAP(inventory_item="silver", quantity=6)),
    AP(name="s7", func=MineInventoryAP(inventory_item="silver", quantity=7)),
    AP(name="s8", func=MineInventoryAP(inventory_item="silver", quantity=8)),
    AP(name="s9", func=MineInventoryAP(inventory_item="silver", quantity=9)),
    AP(name="s10", func=MineInventoryAP(inventory_item="silver", quantity=10)),
    AP(name="g1", func=MineInventoryAP(inventory_item="gold", quantity=1)),
    AP(name="g2", func=MineInventoryAP(inventory_item="gold", quantity=2)),
    AP(name="at_home", func=MineLocationAP(location=(0, 0))),
]

gold_mine_ltlf = """F(at_home | g2) & (F(g1) -> !F(s1)) & (F(s1) -> !F(g1)) & (g1 R !g2) & 
                    (s1 R !s2) & (s2 R !s3) & (s3 R !s4) & (s4 R !s5) & (s5 R !s6) & 
                    (s6 R !s7) & (s7 R !s8) & (s8 R !s9) & (s9 R !s10) & (s10 R !at_home)"""

# Construct automaton manually
# APs: [0,n-1] = silver, [n,n+1] = gold, n+2 = home
# states: [0,n] = silver, n+1 = gold, n+2 = accept, n+3 = fail
device = torch.device('cuda:0')
n = 10
adj_matrix = torch.ones((n+4, 1<<(n+3)), dtype=torch.long, device=device) * (n+3)

ind = torch.arange(n, device=device)

adj_matrix[0, 0] = 0 # state = have nothing, ap = get nothing -> stay in state
adj_matrix[ind+1, 1<<ind] = ind+1 # state = have 3 silver, ap = get 3rd silver -> stay in state
adj_matrix[ind, 1<<ind] = ind+1 # state = have 2 silver, ap = get 3rd silver -> move to new state
adj_matrix[n, (1<<n-1) + (1<<n+2)] = n+2 # state = have n silver, ap = silver n + home -> accept
adj_matrix[n+2, (1<<n-1)] = n # state = have n silver + home, ap = leave home -> back to n

adj_matrix[0, 1<<n] = n+1 # state = have no silver, ap = get gold -> move to new state
adj_matrix[n+1, 1<<n] = n+1 # state = have 1 gold, ap = gold 1 -> stay in state
adj_matrix[n+1, 1<<n+1] = n+2 # state = have 1 gold, ap = gold 2 -> accept

adj_matrix = adj_matrix.cpu().numpy()

gold_mine_automaton = LTLAutomaton(adj_matrix, 0, device)
gold_mine_ap_extractor = MineEnvApExtractor(ap_funcs=[ap.func for ap in gold_mine_aps], device=device)