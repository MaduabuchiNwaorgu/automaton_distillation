from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

labyrinth_config = MineWorldConfig(
    shape=(10, 10),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name='wall', consumable=False, inventory_modifier=Counter(), wall=True, grid_letter='|'
            ),
            fixed_placements=[[0, 1], [0, 5], [0, 9], [1, 1], [1, 3], [1, 5], [1, 7], [1, 9],
            [2, 1], [2, 3], [2, 5], [2, 7], [2, 9], [3, 1], [3, 3], [3, 5], [3, 7], [3, 9],
            [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [5, 1], [5, 3], [5, 5], [5, 7], [5, 9],
            [6, 1], [6, 3], [6, 5], [6, 7], [6, 9], [7, 1], [7, 3], [7, 5], [7, 7], [7, 9],
            [8, 1], [8, 3], [8, 5], [8, 7], [8, 9], [9, 3], [9, 7]]
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="red_key", consumable=True, grid_letter="r", inventory_modifier=Counter(rkey=+1)
            ),
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="blue_key", consumable=True, grid_letter="b", inventory_modifier=Counter(rkey=-1, bkey=+1, key=+1)
            ),
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="purple_key", consumable=True, grid_letter="p", inventory_modifier=Counter(pkey=+1, key=+1)
            ),
            fixed_placements=[(9, 9)]
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="at_home_done", consumable=False, grid_letter="H", inventory_modifier=Counter(),
                inventory_requirements=Counter(key=+1), reward=+100, terminal=True
            ),
            fixed_placements=[(0, 0)]
        )
    ],
    inventory=[
        InventoryItemConfig(name="rkey", default_quantity=0, capacity=1),
        InventoryItemConfig(name="bkey", default_quantity=0, capacity=1),
        InventoryItemConfig(name="pkey", default_quantity=0, capacity=1),
        InventoryItemConfig(name="key", default_quantity=0, capacity=1),
    ]
)

labyrinth_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": labyrinth_config}
)

labyrinth_exp_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": labyrinth_config},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

labyrinth_rew_per_step_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": labyrinth_config},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

labyrinth_aps = [
    AP(name="at_home", func=MineLocationAP(location=(0, 0))),
    AP(name="rkey", func=MineInventoryAP(inventory_item="rkey", quantity=1)),
    AP(name="bkey", func=MineInventoryAP(inventory_item="bkey", quantity=1)),
    AP(name="pkey", func=MineInventoryAP(inventory_item="pkey", quantity=1))
]

labyrinth_ltlf = "F(home) & (rkey R !bkey) & ((pkey | bkey) R !at_home)"
