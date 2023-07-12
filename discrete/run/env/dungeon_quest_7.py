from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

dungeon_quest_config_7 = MineWorldConfig(
    shape=(7, 7),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="key", consumable=True, grid_letter="K", inventory_modifier=Counter(key=+1), reward=+1
            ),
            fixed_placements=[],
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="shield", consumable=True, grid_letter="S", inventory_modifier=Counter(shield=+1),
                reward=+1
            ),
            fixed_placements=[],
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="chest", consumable=False, grid_letter="C", inventory_modifier=Counter(key=-1, sword=+1),
                inventory_requirements=Counter(key=1), reward=+1
            ),
            fixed_placements=[],
            random_placements=1
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="dragon", consumable=True, grid_letter="D", inventory_modifier=Counter(),
                inventory_requirements=Counter(sword=1, shield=1), reward=+100, terminal=True
            ),
            fixed_placements=[],
            random_placements=1
        )
    ],
    inventory=[
        InventoryItemConfig(name="key", default_quantity=0, capacity=1),
        InventoryItemConfig(name="sword", default_quantity=0, capacity=1),
        InventoryItemConfig(name="shield", default_quantity=0, capacity=1)
    ]
)

dungeon_quest_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": dungeon_quest_config_7}
)

dungeon_quest_exp_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": dungeon_quest_config_7},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

dungeon_quest_rew_per_step_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": dungeon_quest_config_7},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

dungeon_quest_aps = [
    AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
    AP(name="sword", func=MineInventoryAP(inventory_item="sword", quantity=1)),
    AP(name="shield", func=MineInventoryAP(inventory_item="shield", quantity=1)),
    AP(name="dragon", func=MineInfoAutAP(ap_name="dragon"))
]

dungeon_quest_ltlf = "F(dragon) & (!sword U key) & (!dragon U sword) & (!dragon U shield)"
