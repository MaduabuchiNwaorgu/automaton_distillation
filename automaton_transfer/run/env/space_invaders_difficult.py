from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP

# from automaton_transfer.lib.automaton

difficult_config = {'enemies': (10, 6), 'bunkers': 2, 'shape': (30, 40), 'alien_shape': (1, 2),
                    'player_shape': (2, 2), 'bunker_shape': (4, 4)}

space_invaders_config = EnvConfig(
    env_name='SpaceInvadersGridworld-v0',
    kwargs={'config': difficult_config})
