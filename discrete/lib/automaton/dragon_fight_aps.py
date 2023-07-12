class InPathOfDragon:
    """Checks if the player is in the path of the dragon"""

    def __init__(self):
        pass

    def __call__(self, info):
        dragon = info['dragon']
        player = info['player']
        if dragon.rushing or dragon.perching or dragon.pos != dragon.ring[dragon.path_ind]:
            # Dragon is currently not in its regular roaming path
            for pos in dragon.path:
                if pos[0] <= player.pos[0] < pos[0] + dragon.size[0] \
                        and pos[1] <= player.pos[1] < pos[1] + dragon.size[1]:
                    return True

            return False
        else:
            for pos in dragon.ring:
                if pos[0] <= player.pos[0] < pos[0] + dragon.size[0] \
                        and pos[1] <= player.pos[1] < pos[1] + dragon.size[1]:
                    return True

            return False


class InPathOfBreath:
    """Checks if the player is in the path of the breath (includes when it's landed)"""
    def __init__(self):
        pass

    def __call__(self, info):
        dragon_breath = info['dragon_breath']
        player = info['player']
        for breath in dragon_breath:
            if player.pos in breath.path:
                return True

        return False


class CrystalsRemain:
    """Check if there are any crystals remaining"""
    def __init__(self):
        pass

    def __call__(self, info):
        return len(info['crystals'].crystals) > 0


class CrystalDestroyed:
    """Check if a crystal was destroyed in the last timestep"""
    def __init__(self):
        self.num_crystals = None

    def __call__(self, info):
        crystals = info['crystals']
        if self.num_crystals is None:
            self.num_crystals = len(crystals.crystals)
            return False
        else:
            crystal_destroyed = (self.num_crystals == len(crystals.crystals))
            self.num_crystals = len(crystals.crystals)
            return crystal_destroyed


class DragonDamaged:
    """Check if the dragon was damaged in the last timestep"""
    def __init__(self):
        self.dragon_hp = None

    def __call__(self, info):
        dragon = info['dragon']
        if self.dragon_hp is None:
            self.dragon_hp = dragon.health
            return False
        else:
            dragon_damaged = (dragon.health == self.dragon_hp)
            self.dragon_hp = dragon.health
            return dragon_damaged
