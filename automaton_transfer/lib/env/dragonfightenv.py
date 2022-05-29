from typing import Tuple, Dict
import os
import numpy as np
from collections import deque
from gym import spaces

from automaton_transfer.lib.env.saveloadenv import SaveLoadEnv


def path(pos1, pos2):
    mvmt = (pos2[0] - pos1[0], pos2[1] - pos1[1])
    N = max(abs(mvmt[0]), abs(mvmt[1]))
    if N == 0:
        return [pos1]
    pth = []
    for i in range(N + 1):
        step = i / N
        pth.append((round(pos1[0] + step * (pos2[0] - pos1[0])), round(pos1[1] + step * (pos2[1] - pos1[1]))))
    return pth


def distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def generate_ring(radius, center):
    start = (center[0] - radius, center[1] - radius)
    ring = [(start[0] + i, start[1]) for i in range(radius * 2)]
    ring = ring + [(start[0] + (radius * 2 - 1), start[1] + i) for i in range(1, radius * 2)]
    ring = ring + [(start[0] + (radius * 2 - 1) - i, start[1] + (radius * 2 - 1)) for i in range(1, radius * 2)]
    ring = ring + [(start[0], start[1] + (radius * 2 - 1) - i) for i in range(1, radius * 2 - 1)]
    return ring


class Dragon:
    def __init__(self, shape, health, ring, size=(2, 2)):
        self.size = size
        self.shape = shape
        self.health = 2
        self.ring = ring
        self.ind = np.random.choice(len(ring))
        self.pos = ring[self.ind]
        self.path = []
        self.path_ind = 0
        self.perched = False
        self.perch_duration = 10
        self.rushing = False
        self.perching = False
        self.cooldown = 10

    def roam(self):
        if self.pos != self.ring[self.ind]:
            self.path_ind = max(self.path_ind - 1, 0)
            self.pos = self.path[self.path_ind]
        else:
            self.ind = (self.ind + 1) % len(self.ring)
            self.pos = self.ring[self.ind]

    def rush(self):
        self.path_ind = min(self.path_ind + 1, len(self.path) - 1)
        if self.path_ind == len(self.path) - 1:
            self.cooldown = 10
            self.rushing = False
        self.pos = self.path[self.path_ind]

    def breath(self, player_pos):
        self.cooldown = 10
        return Breath(path(self.pos, player_pos))

    def perch(self):
        if not self.perched:
            self.path_ind = min(self.path_ind + 1, len(self.path) - 1)
            self.pos = self.path[self.path_ind]
            if self.path_ind == len(self.path) - 1:
                self.perched = True
        else:
            self.perch_duration -= 1
            if self.perch_duration == 0:
                self.perched = False
                self.perching = False
                self.perch_duration = 10
                self.cooldown = 10

    def hit(self, crystals):
        if len(crystals.crystals) == 0:
            self.health -= 1

    def step(self, player):
        self.cooldown = max(self.cooldown - 1, 0)
        if self.rushing:
            self.rush()
        elif self.perching:
            self.perch()
        elif self.cooldown > 0:
            self.roam()
        else:
            if np.random.uniform(0, 1, 1) > 0.5:
                # Chance to attack
                if np.random.uniform(0, 1, 1) > 0.5:
                    # If it decides to rush
                    self.path_ind = 0
                    self.path = path(self.pos, player.pos)
                    self.rushing = True
                    self.rush()
                else:
                    # Otherwise, it'll do the breath attack
                    return self.breath(player.pos)
            elif np.random.uniform(0, 1, 1) > 0.9:
                self.path_ind = 0
                self.perching = True
                self.path = path(self.pos, (self.shape[0] // 2 - self.size[0] // 2,
                                            self.shape[1] // 2 - self.size[1] // 2))
                self.perch()
            else:
                self.roam()
        return None

    def positions(self):
        return [(self.pos[0] + i, self.pos[1] + j) for i in range(self.size[0]) for j in range(self.size[1])]

    def path(self, pos):
        path = None



class Breath:
    def __init__(self, _path):
        self.path = _path
        self.path_ind = 0
        self.pos = self.path[0]
        self.inplace = (self.pos == self.path[-1])
        self.lifespan = 8

    def step(self):
        if not self.inplace:
            self.path_ind = min(self.path_ind + 1, len(self.path) - 1)
            self.pos = self.path[self.path_ind]
            self.inplace = (self.pos == self.path[-1])
        else:
            self.lifespan -= 1


class Player:
    def __init__(self, shape, health):
        self.shape = shape
        self.pos = (0, 0)
        self.melee = True
        self.max_ranged_cooldown = 3
        self.ranged_cooldown = 0
        self.health = health
        self.invuln = False
        self.iframe = 0

    def attack(self, crystals, dragon):
        self.ranged_cooldown = max(self.ranged_cooldown - 1, 0)
        if len(crystals.crystals) > 0:
            closest_crystal, dist = crystals.closest_crystal(self.pos)
        dragon_dist = distance(self.pos, dragon.pos)
        if len(crystals.crystals) == 0 or dragon_dist < dist:
            target = dragon
            target_dist = dragon_dist
        else:
            target = closest_crystal
            target_dist = dist
        if self.melee:
            if target_dist == 1:
                if isinstance(target, Dragon):
                    target.hit(crystals)
                else:
                    crystals.remove(target)
        else:
            if self.ranged_cooldown == 0:
                if target_dist == 0:
                    probability = 1
                else:
                    probability = min(1, 2 / target_dist)
                if np.random.uniform(0, 1, 1) < probability:
                    self.ranged_cooldown = self.max_ranged_cooldown
                    if isinstance(target, Dragon):
                        target.hit(crystals)
                    else:
                        crystals.remove(target)

    def switch_weapon(self):
        self.melee = not self.melee

    def move(self, action, crystals, dragon):
        """

        :param action: Actions: 0 Up, 1 Right, 2 Down, 3 Left, 4 Noop
        :param crystals:
        :param dragon:
        :return:
        """
        if action == 0:
            new_pos = (max(self.pos[0] - 1, 0), self.pos[1])
        elif action == 1:
            new_pos = (self.pos[0], min(self.pos[1] + 1, self.shape[1] - 1))
        elif action == 2:
            new_pos = (min(self.pos[0] + 1, self.shape[0] - 1), self.pos[1])
        elif action == 3:
            new_pos = (self.pos[0], max(self.pos[1] - 1, 0))
        else:
            new_pos = self.pos
        if new_pos in crystals:
            return
        elif new_pos in dragon.positions():
            self.hit()
            self.pos = new_pos
        else:
            self.pos = new_pos

    def build(self, crystals):
        if len(crystals.crystals) > 0:
            closest_crystal, _ = crystals.closest_crystal(self.pos)
            if distance(self.pos, closest_crystal.pos) == 1:
                crystals.build(closest_crystal)

    def step(self, action, crystals, dragon):
        """
        :param action: Actions: 0 Up, 1 Right, 2 Down, 3 Left, 4 Noop, 5 Attack, 6 Switch Weapon, 7 Build
        :param crystals
        :param dragon
        """
        if action <= 4:
            self.move(action, crystals, dragon)
        if action == 5:
            self.attack(crystals, dragon)
        if action == 6:
            self.switch_weapon()
        if action == 7:
            self.build(crystals)

    def hit(self):
        if self.invuln:
            self.iframe -= 1
            if self.iframe == 0:
                self.invuln = False
        else:
            self.iframe = 5
            self.invuln = True
            self.health -= 1


class Crystals:
    def __init__(self, num, ring):
        start = np.random.choice(len(ring))
        inc = len(ring) // num
        self.built = [2 for _ in range(num)]
        self.crystals = [Crystal(ring[(start + i * inc) % len(ring)]) for i in range(num)]

    def closest_crystal(self, pos):
        dist = [(crystal, self.distance(pos, crystal)) for crystal in self.crystals]
        return min(dist, key=lambda x: x[1])

    def distance(self, pos, crystal):
        return distance(pos, crystal.pos) + self.built[self.crystals.index(crystal)]

    def remove(self, crystal):
        ind = self.crystals.index(crystal)
        self.crystals.remove(crystal)
        self.built.pop(ind)

    def __contains__(self, item):
        for crystal in self.crystals:
            if item == crystal.pos:
                return True
        return False

    def build(self, crystal):
        ind = self.crystals.index(crystal)
        self.built[ind] = max(0, self.built[ind] - 1)


class Crystal:
    def __init__(self, pos):
        self.pos = pos


class DragonFight(SaveLoadEnv):

    def save_state(self):
        pass

    def load_state(self, state):
        pass

    def step(self, action):
        self.time += 1
        proj = self.dragon.step(self.player)
        if isinstance(proj, Breath):
            self.dragon_breath.append(proj)
        num_crystals = len(self.crystals.crystals)
        initial_hp = self.player.health
        self.player.step(action, self.crystals, self.dragon)
        expired_breath = []
        for breath in self.dragon_breath:
            breath.step()
            if breath.inplace and breath.pos == self.player.pos:
                self.player.hit()
                expired_breath.append(breath)
            if breath.lifespan == 0:
                expired_breath.append(breath)
        for breath in expired_breath:
            if breath in self.dragon_breath:
                self.dragon_breath.remove(breath)
        num_destroyed = num_crystals - len(self.crystals.crystals)
        hp_lost = initial_hp - self.player.health
        self.obs.append(self.get_state())
        player_health = [[self.player.health for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        dragon_health = [[self.dragon.health for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        weapon = [[1 if self.player.melee else 0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        built = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        for i in range(len(self.crystals.crystals)):
            crystal = self.crystals.crystals[i]
            built[crystal.pos[0]][crystal.pos[1]] = self.crystals.built[i]
        obs = list(self.obs)
        obs.append(player_health)
        obs.append(dragon_health)
        obs.append(weapon)
        obs.append(built)
        done = self.player.health == 0 or self.dragon.health == 0 or self.time == self.max_time
        reward = num_destroyed * 5 + hp_lost * (-50 / self.player_health)
        if self.player.health == 0:
            reward -= 50
        if self.dragon.health == 0:
            reward += 50
        return np.array(obs, dtype=np.uint8), reward, done, {}

    def reset(self):
        self.time = 0
        self.crystals = Crystals(self.crystal_count, self.inner_ring)
        self.dragon = Dragon(self.shape, self.dragon_health, self.outer_ring)
        self.player = Player(self.shape, self.player_health)
        self.dragon_breath = []
        for _ in range(self.obs.maxlen - 1):
            self.obs.append([[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])])
        self.obs.append(self.get_state())
        player_health = [[self.player.health for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        dragon_health = [[self.dragon.health for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        weapon = [[1 if self.player.melee else 0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        built = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        for i in range(len(self.crystals.crystals)):
            crystal = self.crystals.crystals[i]
            built[crystal.pos[0]][crystal.pos[1]] = self.crystals.built[i]
        obs = list(self.obs)
        obs.append(player_health)
        obs.append(dragon_health)
        obs.append(weapon)
        obs.append(built)
        return np.array(obs, dtype=np.uint8)

    def render(self, mode="human"):
        state = self.get_state()
        for row in state:
            print(row)

    def get_state(self):
        obs = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        # There will be logic checking that the player isn't in the same position as the crystals, but the dragon can be
        obs[self.player.pos[0]][self.player.pos[1]] = 1
        for crystal in self.crystals.crystals:
            obs[crystal.pos[0]][crystal.pos[1]] = 3
        for pos in self.dragon.positions():
            if pos[0] >= self.shape[0] or pos[1] >= self.shape[1]:
                continue
            obs[pos[0]][pos[1]] = 2
        for breath in self.dragon_breath:
            if breath.inplace:
                obs[breath.pos[0]][breath.pos[1]] = 5
            else:
                obs[breath.pos[0]][breath.pos[1]] = 4
        return obs

    def __init__(self, config: Dict):
        self.time = 0
        self.dragon = None
        self.crystals = None
        self.player = None
        self.dragon_breath = []
        self.shape = config['shape']
        self.crystal_count = config['crystals']
        self.dragon_health = config['dragon_health']
        self.player_health = config['player_health']
        self.max_time = config['timesteps']
        self.observation_space = spaces.Box(0, 5, shape=(8, self.shape[1], self.shape[0]), dtype=np.uint8)
        self.action_space = spaces.Discrete(8)
        self.obs = deque(maxlen=4)
        inner_rad = int(0.5 * self.shape[1]) // 2
        outer_rad = int(0.7 * self.shape[0]) // 2
        self.inner_ring = generate_ring(inner_rad, (self.shape[0] // 2, self.shape[1] // 2))
        self.outer_ring = generate_ring(outer_rad, (self.shape[0] // 2, self.shape[1] // 2))


if __name__ == '__main__':
    # Just used for testing/debugging code
    env = DragonFight({'shape': (10, 10), 'crystals': 6, 'dragon_health': 5, 'player_health': 3, 'timesteps': 250})
    for _ in range(50):
        env.reset()
        print(_)
        for _ in range(500):
            # env.render()
            # _, _, done, _ = env.step(int(input('Actions: 0 Up, 1 Right, 2 Down, 3 Left, 4 Noop, 5 Attack, 6 Switch '
            #                                    'Weapon, 7 Build')))
            # print(f'Dragon Health: {env.dragon.health}, Player Health: {env.player.health}')
            _, _, done, _ = env.step(np.random.choice(range(0, 8)))
            if done:
                break
