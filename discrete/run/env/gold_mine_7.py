from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit
from automaton_transfer.run.env.gold_mine import gold_mine_aps, gold_mine_ltlf, gold_mine_automaton, gold_mine_ap_extractor

gold_mine_config_7 = MineWorldConfig(
    shape=(7, 7),
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

gold_mine_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": gold_mine_config_7}
)

gold_mine_exp_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": gold_mine_config_7},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

gold_mine_rew_per_step_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": gold_mine_config_7},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

