from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.automaton.dragon_fight_aps import InPathOfDragon, InPathOfBreath, CrystalsRemain, \
    CrystalDestroyed, DragonDamaged

difficult_config = ({'shape': (15, 15), 'crystals': 8, 'dragon_health': 5, 'player_health': 3, 'timesteps': 500})

dragon_fight_config = EnvConfig(
    env_name='DragonFightGridworld-v0',
    kwargs={'config': difficult_config})

dragon_fight_aps = [
    AP('in_path_of_dragon', InPathOfDragon()),
    AP('in_path_of_breath', InPathOfBreath()),
    AP('crystals_remain', CrystalsRemain()),
    AP('crystal_destroyed', CrystalDestroyed()),
    AP('dragon_damaged', DragonDamaged())
]

dragon_fight_ltlf = 'G(in_path_of_dragon -> F !in_path_of_dragon) & G(in_path_of_breath -> F !in_path_of_breath) & G(' \
                    'crystals_remain -> F crystal_destroyed) & G(!crystals_remain -> F dragon_damaged)'
