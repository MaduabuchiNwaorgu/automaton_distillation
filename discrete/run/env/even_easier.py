from collections import Counter

from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep

even_easier_config = MineWorldConfig(
    shape=(10, 10),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="a", consumable=True, grid_letter="A", inventory_modifier=Counter(), reward=+100,
                terminal=True
            ),
            fixed_placements=[(0, 0)],
        ),
    ],
    inventory=[
    ]
)

even_easier_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": even_easier_config},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -1})
