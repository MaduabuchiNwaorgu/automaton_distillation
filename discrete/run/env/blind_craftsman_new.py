from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

blind_craftsman_config_new = MineWorldConfig(
    shape=(10, 10),
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
                action_name="stone", consumable=False, grid_letter="S", inventory_modifier=Counter(),
            ),
            fixed_placements=[],
            random_placements=3
        )    
    ],
    inventory=[
        InventoryItemConfig(name="wood", default_quantity=0, capacity=2),
        InventoryItemConfig(name="stone", default_quantity=0, capacity=2),
        InventoryItemConfig(name="tool", default_quantity=0, capacity=3)
    ]
)

blind_craftsman_env_config_new = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": blind_craftsman_config_new}
)

blind_craftsman_exp_env_config_new = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": blind_craftsman_config_new},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

blind_craftsman_rew_per_step_env_config_new = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": blind_craftsman_config_new},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

blind_craftsman_aps_new = [
    AP(name="at_home", func=MineLocationAP(location=(0, 0))),
    AP(name="tool_3", func=MineInventoryAP(inventory_item="tool", quantity=3)),
    AP(name="factory", func=MineInfoAutAP(ap_name="factory")),
    AP(name="wood", func=MineInfoAutAP(ap_name="wood")),
    AP(name="stone", func=MineInfoAutAP(ap_name="stone"))
]

blind_craftsman_ltlf_new = "G(wood & stone -> F(factory)) & F(tool_3 & at_home)"
