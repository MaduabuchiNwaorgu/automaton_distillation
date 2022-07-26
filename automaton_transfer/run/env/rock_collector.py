from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

rock_collector_config = MineWorldConfig(
    shape=(10, 10),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="small_rock", consumable=True, grid_letter="o", inventory_modifier=Counter(srock=+1), reward=+1
            ),
            fixed_placements=[(0, 0)]
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="medium_rock", consumable=True, grid_letter="0", inventory_modifier=Counter(mrock=+1), reward=+1
            ),
            # fixed_placements=[(0, 0)]
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="big_rock", consumable=True, grid_letter="O", inventory_modifier=Counter(not_brock=-1),
                reward=+1
            ),
            # fixed_placements=[(2, 2)]
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="hill", consumable=False, grid_letter="^", inventory_modifier=Counter(),
                movement_requirements=Counter(not_brock=1)
            ),
            # fixed_placements=[(0, 1), (0, 4), (0, 6), (0, 7), (1, 0), (1, 5),
            # (1, 7), (1, 8), (2, 3), (2, 9), (3, 1), (3, 3), (3, 4), (4, 4),
            # (4, 5), (4, 6), (4, 7), (4, 8), (5, 0), (5, 1), (5, 2), (5, 8),
            # (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (7, 6), (7, 8),
            # (7, 9), (8, 0), (8, 1), (8, 5), (8, 6), (8, 8), (8, 9), (9, 1),
            # (9, 5), (9, 6), (9, 8), (9, 9)]
            fixed_placements=[(0, 1), (1, 0)],
            random_placements=30
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="at_home_done", consumable=False, grid_letter="H", inventory_modifier=Counter(),
                inventory_requirements=Counter(srock=1, mrock=1, not_brock=0), reward=+100, terminal=True
            ),
            fixed_placements=[(2, 0)]
        )
    ],
    inventory=[
        InventoryItemConfig(name="srock", default_quantity=0, capacity=1),
        InventoryItemConfig(name="mrock", default_quantity=0, capacity=1),
        InventoryItemConfig(name="not_brock", default_quantity=1, capacity=1)
    ]
)

rock_collector_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": rock_collector_config}
)

rock_collector_exp_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": rock_collector_config},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

rock_collector_rew_per_step_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": rock_collector_config},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

rock_collector_aps = [
    AP(name="at_home", func=MineLocationAP(location=(2, 0))),
    AP(name="brock", func=MineInventoryAP(inventory_item="not_brock", quantity=0)),
    AP(name="mrock", func=MineInventoryAP(inventory_item="mrock", quantity=1)),
    AP(name="srock", func=MineInventoryAP(inventory_item="srock", quantity=1))
]

# rock_collector_ltlf = "F(at_home) & (!at_home U brock) & (!at_home U mrock) & (!at_home U srock)"
rock_collector_ltlf = "F(brock) & F(mrock) & F(srock) & G((brock & mrock & srock) -> F(at_home))"
