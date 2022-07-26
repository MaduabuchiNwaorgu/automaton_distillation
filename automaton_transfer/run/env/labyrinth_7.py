from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

labyrinth_config_7 = MineWorldConfig(
    shape=(7, 7),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name='wall', consumable=False, inventory_modifier=Counter(), wall=True, grid_letter='|'
            ),
            fixed_placements=[[0, 1], [0, 5], [1, 1], [1, 3], [1, 5], [2, 1], [2, 3], [2, 5],
            [3, 1], [3, 3], [3, 5], [4, 1], [4, 3], [4, 5], [5, 1], [5, 3], [5, 5], [6, 3]]
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
            fixed_placements=[(0, 6)]
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

labyrinth_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": labyrinth_config_7}
)

labyrinth_exp_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": labyrinth_config_7},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

labyrinth_rew_per_step_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": labyrinth_config_7},
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
