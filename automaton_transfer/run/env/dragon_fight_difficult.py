from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP

difficult_config = ({'shape': (15, 15), 'crystals': 8, 'dragon_health': 5, 'player_health': 3, 'timesteps': 500})

dragon_fight_config = EnvConfig(
    env_name='DragonFightGridworld-v0',
    kwargs={'config': difficult_config})
