import abc
import functools
import random
import time
from collections import Counter
from random import Random
from typing import Tuple, Dict
import os
import numpy as np
from gym import spaces
from math import ceil

from automaton_transfer.lib.env.saveloadenv import SaveLoadEnv


class Alien:
    def __init__(self, pos, shape, size=(1, 2)):
        self.shape = shape
        self.pos = pos
        self.size = size
        self.gun_pos = (pos[0], pos[1] + self.size[1])

    def step(self, dir_x, dir_y):
        if dir_x < 0:
            self.pos = (max(self.pos[0] + dir_x, int(self.shape[0] * 0.05)), self.pos[1] + dir_y)
        else:
            self.pos = (min(self.pos[0] + dir_x, int(self.shape[0] * 0.95)), self.pos[1] + dir_y)
        self.gun_pos = (self.pos[0], self.pos[1] + 5)
        if self.pos[0] <= int(self.shape[0] * 0.07) or self.pos[0] >= int(self.shape[0] * 0.93):
            return True
        else:
            return False

    def fire(self):
        return AlienBullet(self.gun_pos)

    def distance(self, pos):
        return abs(self.gun_pos[0] - pos[0]) + abs(self.gun_pos[1] - pos[1])

    def contains(self, pos):
        return self.pos[0] <= pos[0] < self.pos[0] + self.size[0] and self.pos[1] <= pos[1] < self.pos[1] + self.size[1]

    def positions(self):
        return [(self.pos[0] + i, self.pos[1] + j) for i in range(self.size[0]) for j in range(self.size[1])]


class AlienBullet:
    def __init__(self, pos):
        self.pos = pos
        self.in_flash = True

    def step(self):
        self.pos = (self.pos[0], self.pos[1] + 1)


class Player:
    def __init__(self, shape, size=(2, 2)):
        self.pos = (0, shape[1] - size[1])
        self.board_shape = shape
        self.size = size
        self.lives = 3
        self.invulnerable = False
        self.invuln_frame = 0

    def contains(self, pos):
        return self.pos[0] <= pos[0] < self.pos[0] + self.size[0] and self.pos[1] <= pos[1] < self.pos[1] + self.size[1]

    def hit(self):
        self.lives -= 1
        self.invulnerable = True
        self.invuln_frame = 8

    def step(self, action):
        """
        :param action: Move or Fire, 0: Left, 1: Right, 2: Fire
        :return: PlayerBullet if fire
        """
        if self.invulnerable:
            self.invuln_frame -= 1
            if self.invuln_frame == 0:
                self.invulnerable = False

        if action == 0:
            self.pos = (max(self.pos[0] - 2, 0), self.pos[1])
            return None
        elif action == 1:
            self.pos = (min(self.pos[0] + 2, self.board_shape[0]), self.pos[1])
            return None
        elif action == 2:
            return self.fire()

    def fire(self):
        return PlayerBullet((self.pos[0] + 1, self.pos[1] - -1))

    def positions(self):
        return [(self.pos[0] + i, self.pos[1] + j) for i in range(self.size[0]) for j in range(self.size[1])]


class PlayerBullet:
    def __init__(self, pos):
        self.pos = pos
        self.in_flash = True

    def step(self):
        self.pos = (self.pos[0], self.pos[1] - 1)


class SpaceInvaders(SaveLoadEnv):

    def load_state(self, state):
        pass

    def save_state(self):
        pass

    def handle_enemy_collisions(self):
        collided_proj = []
        collided_enemy = []
        out_of_bounds = []
        for proj in self.player_projectiles:
            if proj.pos[0] <= 0:
                out_of_bounds.append(proj)
            for enemy in self.enemies:
                if enemy.contains(proj.pos):
                    collided_proj.append(proj)
                    collided_enemy.append(enemy)
                    break
        for proj in collided_proj:
            if proj in self.player_projectiles:
                self.player_projectiles.remove(proj)
        for enemy in collided_enemy:
            if enemy in self.enemies:
                self.enemies.remove(enemy)
        for proj in out_of_bounds:
            if proj in self.player_projectiles:
                self.player_projectiles.remove(proj)
        return len(collided_enemy)

    def handle_bunker_collisions(self):
        enemy_bullet_heads = {}
        player_bullet_heads = {}
        for proj in self.enemy_projectiles:
            enemy_bullet_heads[proj.pos] = proj
        for proj in self.player_projectiles:
            player_bullet_heads[proj.pos] = proj

        destroyed_bunker = []
        for pos in self.bunkers:
            for enemy in self.enemies:
                if enemy.contains(pos):
                    destroyed_bunker.append(pos)
            if pos in enemy_bullet_heads.keys():
                destroyed_bunker.append(pos)
                if (pos[0], pos[1] + 1) in self.bunkers:
                    destroyed_bunker.append((pos[0], pos[1] + 1))
                if enemy_bullet_heads[pos] in self.enemy_projectiles:
                    self.enemy_projectiles.remove(enemy_bullet_heads[pos])
            elif pos in player_bullet_heads:
                destroyed_bunker.append(pos)
                if (pos[0], pos[1] + 1) in self.bunkers:
                    destroyed_bunker.append((pos[0], pos[1] + 1))
                if player_bullet_heads[pos] in self.player_projectiles:
                    self.player_projectiles.remove(player_bullet_heads[pos])
        for val in destroyed_bunker:
            if val in self.bunkers:
                self.bunkers.remove(val)

    def handle_player_collisions(self):
        hit_proj = None
        out_of_bounds = []
        for proj in self.enemy_projectiles:
            if proj.pos[0] == 83:
                out_of_bounds.append(proj)
            if self.player.contains(proj.pos):
                hit_proj = proj

        if hit_proj is not None and not self.player.invulnerable:
            if hit_proj in self.enemy_projectiles:
                self.enemy_projectiles.remove(hit_proj)
            self.player.hit()
            return True

        for proj in out_of_bounds:
            if proj in self.enemy_projectiles:
                self.enemy_projectiles.remove(proj)

    def step_aliens(self):
        against_wall = False
        for enemy in self.enemies:
            if not against_wall:
                against_wall = enemy.step(self.dir_x, self.dir_y)
            else:
                enemy.step(self.dir_x, self.dir_y)
        if against_wall:
            self.dir_x *= -1
            self.dir_y = 1
        elif self.dir_y != 0:
            self.dir_y = 0

    def aliens_fire(self):
        front_aliens = {}
        for enemy in self.enemies:
            if enemy.gun_pos[0] not in front_aliens.keys():
                front_aliens[enemy.gun_pos[0]] = enemy
            elif front_aliens[enemy.gun_pos[0]].gun_pos[1] < enemy.gun_pos[1]:
                front_aliens[enemy.gun_pos[0]] = enemy
        front_aliens = list(front_aliens.values())
        distances = np.array([front_alien.distance(self.player.pos) for front_alien in front_aliens])
        distances = 3 / (distances + np.ones(shape=distances.shape))
        distances = np.nan_to_num(distances)
        probs = np.abs(distances) / np.sum(distances)
        firing_alien = np.random.choice(front_aliens, p=probs)
        self.enemy_projectiles.append(firing_alien.fire())

    def step_proj(self):
        for proj in self.enemy_projectiles:
            proj.step()
        for proj in self.player_projectiles:
            proj.step()

    def frequency(self):
        max_enemies = self.num_enemies[0] * self.num_enemies[1]
        if len(self.enemies) > max_enemies // 2:
            return 5
        elif len(self.enemies) > max_enemies // 4:
            return 2
        elif len(self.enemies) > 1:
            self.dir_x = 2 * np.sign(self.dir_x)
            return 2
        else:
            self.dir_x = 3 * np.sign(self.dir_x)
            return 1

    def step(self, action):
        self.timestep += 1
        self.step_proj()
        if self.timestep % self.frequency() == 0:
            self.step_aliens()
        if self.timestep % 2 == 0:
            self.aliens_fire()
        player_bullet = self.player.step(action)
        if player_bullet is not None and self.time_before_fire == 0:
            self.player_projectiles.append(player_bullet)
            self.time_before_fire = 5
        elif self.time_before_fire > 0:
            self.time_before_fire -= 1
        num_destroyed = self.handle_enemy_collisions()
        self.num_destroyed += num_destroyed
        player_hit = self.handle_player_collisions()
        self.handle_bunker_collisions()
        obs = [self.prev_obs, self.get_state()]
        self.prev_obs = obs[1]
        reward = 0
        reward += 10 / (self.num_enemies[0] * self.num_enemies[1]) * num_destroyed
        if self.num_destroyed > 1 and self.num_destroyed % (self.num_enemies[0] * self.num_enemies[1]) == 0:
            reward += 50
            self.reset_aliens()
        if player_hit:
            reward -= 10
        if self.player.lives == 0:
            reward -= 100
        enemy_reached_end = map(lambda x: x.contains(x.pos[0], self.shape[1] - 1), self.enemies)
        return np.array(obs), reward, self.player.lives == 0 or self.timestep == self.max_time, {}

    def reset_aliens(self):
        self.dir_x = -1
        self.dir_y = 0
        every_i = ceil((self.shape[0] - int(0.1 * self.shape[0]) - self.alien_shape[0] * self.num_enemies[0])
                       / (self.num_enemies[0] - 1))
        every_j = ceil((self.shape[1] - int(0.3 * self.shape[1]) - self.alien_shape[1] * self.num_enemies[1])
                       / (self.num_enemies[1] - 1))
        i = int(0.07 * self.shape[0])
        for _ in range(self.num_enemies[0]):
            j = int(0.03 * self.shape[1])
            for _ in range(self.num_enemies[1]):
                j += every_j
                self.enemies.append(Alien((i, j), self.shape, size=self.alien_shape))
            i += every_i

    def reset_bunkers(self):
        spacing = (self.shape[0] - self.bunker_shape[1] * self.num_bunkers) // (self.num_bunkers + 1)
        i = 0
        for _ in range(self.num_bunkers):
            i += spacing
            for x in range(self.bunker_shape[0]):
                for y in range(self.bunker_shape[1]):
                    self.bunkers.append((i + x, self.shape[1] - (2 + self.player_shape[1] + self.bunker_shape[1]) + y))
            i += self.bunker_shape[0]

    def reset(self):
        self.reset_aliens()
        self.reset_bunkers()
        self.player = Player(self.shape, self.player_shape)
        self.prev_obs = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        obs = [self.prev_obs, self.get_state()]
        self.prev_obs = obs[1]
        self.enemy_projectiles = []
        self.player_projectiles = []
        self.num_destroyed = 0
        self.dir_x = -1
        self.dir_y = 0
        self.timestep = 0
        return np.array(obs)

    def get_state(self):
        obs = [[0 for i in range(self.shape[0])] for j in range(self.shape[1])]
        try:
            for pos in self.player.positions():
                obs[pos[1]][pos[0]] = 1
            for pos in self.bunkers:
                obs[pos[1]][pos[0]] = 2
            for enemy in self.enemies:
                for pos in enemy.positions():
                    obs[pos[1]][pos[0]] = 3
            for proj in self.enemy_projectiles:
                if proj.in_flash:
                    obs[proj.pos[1]][proj.pos[0]] = 4
            for proj in self.player_projectiles:
                if proj.in_flash:
                    obs[proj.pos[1]][proj.pos[0]] = 5
        except IndexError:
            pass
        obs[0][0] = self.player.lives

        return obs

    def render(self, mode="human"):
        obs = self.get_state()
        for row in obs:
            temp_str = ''
            for val in row:
                temp_str += str(val)
            print(temp_str)
        print()

    def __init__(self, config: Dict):
        """
        :param config:
        """
        self.num_enemies = config['enemies']
        self.num_bunkers = config['bunkers']
        self.shape = config['shape']
        self.alien_shape = config['alien_shape']
        self.player_shape = config['player_shape']
        self.bunker_shape = config['bunker_shape']
        self.player = Player(self.shape, size=self.player_shape)
        self.observation_space = spaces.Box(0, 5, shape=(2, self.shape[1], self.shape[0]), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.prev_obs = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        self.enemies = []
        self.enemy_projectiles = []
        self.player_projectiles = []
        self.bunkers = []
        self.num_destroyed = 0
        self.dir_x = -1
        self.dir_y = 0
        self.timestep = 0
        self.time_before_fire = 0
        self.max_time = 500


if __name__ == '__main__':
    env = SpaceInvaders({'enemies': (8, 6), 'bunkers': 3, 'shape': (20, 30), 'alien_shape': (1, 2),
                         'player_shape': (2, 2), 'bunker_shape': (4, 4)})
    env.reset()
    for _ in range(1000):
        next_state, rew, done, info = env.step(np.random.choice([0, 1, 2]))
        # if done:
        #     break
        env.render()
        print(_)
