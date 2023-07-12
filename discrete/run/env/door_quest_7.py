from collections import Counter

from automaton_transfer.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, \
    MineWorldTileType
from automaton_transfer.lib.env.rew_every_step import RewEveryStep
from automaton_transfer.lib.env.time_limit import TimeLimit

door_quest_7_config = MineWorldConfig(
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
                action_name="door", consumable=False, grid_letter="D", inventory_modifier=Counter(),
                inventory_requirements=Counter(key=1), reward=+100, terminal=True
            ),
            fixed_placements=[],
            random_placements=1
        )
    ],
    inventory=[
        InventoryItemConfig(name="key", default_quantity=0, capacity=1)
    ]
)

door_quest_7_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": door_quest_7_config}
)

door_quest_7_exp_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": door_quest_7_config},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

door_quest_7_rew_per_step_env_config = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": door_quest_7_config},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

door_quest_7_aps = [
    AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
    AP(name="door", func=MineInfoAutAP(ap_name="door"))
]

door_quest_7_ltlf = "F(door) & (!door U key)"
