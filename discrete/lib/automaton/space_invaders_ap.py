class InPathBullet:
    """Checks if the agent is in the path of a bullet"""
    def __init__(self):
        pass

    def __call__(self, info):
        bullets = info['alien_proj']
        player = info['player']
        for bullet in bullets:
            if player.pos[0] <= bullet.pos[0] < player.pos[0] + player.size[0]:
                return True

        return False


class BehindCover:
    """Checks if the agents hitbox is completely covered by bunkers"""
    def __init__(self):
        pass

    def __call__(self, info):
        player = info['player']
        bunkers = info['bunkers']
        player_pos = [player.pos[0] + i for i in range(player.size)]

        for point in bunkers:
            if point[0] in player_pos:
                player_pos.remove(point[0])

        if len(player_pos) == 0:
            return True
        return False


class EnemyRemaining:
    """Checks if there are any enemies remaining"""
    def __init__(self):
        pass

    def __call__(self, info):
        aliens = info['aliens']
        return len(aliens) > 0


class EnemyDestroyed:
    """Checks if an enemy was destroyed in the last timestep"""
    def __init__(self):
        self.num_aliens = None

    def __call__(self, info):
        aliens = info['aliens']
        if self.num_aliens is None:
            self.num_aliens = len(aliens)
        else:
            destroyed = (len(aliens) == self.num_aliens)
            self.num_aliens = len(aliens)
            return destroyed
