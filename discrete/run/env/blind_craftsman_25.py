from collections import Counter

from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

blind_craftsman_config_25 = MineWorldConfig(
    shape=(25, 25),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="wood", consumable=True, grid_letter="W", inventory_modifier=Counter(wood=+1), reward=+1
            ),
            fixed_placements=[],
            random_placements=3
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="factory", consumable=False, grid_letter="F", inventory_modifier=Counter(wood=-1, tool=+1),
                reward=+1
            ),
            fixed_placements=[],
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="at_home_done", consumable=False, grid_letter="H", inventory_modifier=Counter(),
                inventory_requirements=Counter(tool=3), reward=+100, terminal=True
            ),
            fixed_placements=[(0, 0)]
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="river", consumable=False, grid_letter="R", inventory_modifier=Counter(), wall=True
            ),
            fixed_placements=[
                (12, 24),
                (12, 23),
                (12, 22),
                (12, 21),
                (12, 20),
                (12, 19),
                (12, 18),
                (12, 17),
                (12, 16),
                (12, 12),
                (12, 11),
                (12, 10),
                (13, 10),
                (14, 10),
                (15, 10),
                (16, 10),
                (17, 10),
                (18, 10),
                (19, 10),
                (20, 10),
                (21, 10),
                (22, 10),
                (23, 10),
                (24, 10)
            ]
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="boulder", consumable=False, grid_letter="B", inventory_modifier=Counter(), wall=True
            )
        )
    ],
    inventory=[
        InventoryItemConfig(name="wood", default_quantity=0, capacity=2),
        InventoryItemConfig(name="tool", default_quantity=0, capacity=3)
    ]
)

blind_craftsman_25_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": blind_craftsman_config_25}
)

blind_craftsman_25_exp_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": blind_craftsman_config_25},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

blind_craftsman_25_rew_per_step_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": blind_craftsman_config_25},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)
