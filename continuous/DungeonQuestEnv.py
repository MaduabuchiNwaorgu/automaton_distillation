import numpy as np
import gym
from gym.envs.registration import register

import copy

from ZoneEnvBase import ZoneEnvBase, Zone

# from genv.safety.safety_gym.envs.engine import *
import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), "safety-gym"))

key = Zone.Yellow
sword = Zone.Blue
shield = Zone.Green
dragon = Zone.Red
visited = Zone.JetBlack

class DungeonQuestEnv(ZoneEnvBase):
    def __init__(self, config):
        self.new_zone_reached = None
        self.zones_dirty = None
        config = copy.deepcopy(config)

        self.time_saved_reward = config.pop("time_saved_reward", 0.01)

        if "reward_goal" in self.DEFAULT:
            self.DEFAULT.pop("reward_goal")

        config.update({"continue_goal": False})

        self.zone_types = [key, sword, shield, dragon, visited]
        self.zones = [key, sword, shield, dragon]
        self.high_only_keys = ["remaining"] + [f"zones_lidar_{i}" for i in range(len(self.zones))]

        super().__init__(zones=self.zones, config=config)

    def build_zone_observation_space(self):
        for i, zone in enumerate(self.zones):
            space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)  # 3 = x,y,vis
            self.obs_space_dict.update({f"zones_lidar_{i}": space})

    def obs_zones(self, obs):
        for i, z in enumerate(self.zones):
            pos = self.data.get_body_xpos(f"zone{i}").copy()[:2] / 3.
            vis = z == Zone.White
            obs[f"zones_lidar_{i}"] = np.concatenate([[vis], pos])

    def reward_goal(self):
        return (self.num_steps - self.steps) * self.time_saved_reward

    def reward(self):
        return 1 if self.new_zone_reached else 0

    def step(self, action):
        self.zones_dirty = True
        self.new_zone_reached = False

        return super().step(action)

    def set_mocaps(self):
        if not self.zones_dirty: return

        for i, pos in enumerate(self.zones_pos):
            if self.zones[i] != visited:
                dist = self.dist_xy(pos)
                if dist <= self.zones_size:
                    self.zones[i] = visited
                    body_id = self.sim.model.geom_name2id(f"zone{i}")
                    self.new_zone_reached = True

                    break

        self.zones_dirty = False

    def goal_met(self):
        return len([z for z in self.zones if z != visited]) == 0

    def reset(self):
        self.zones = [key, sword, shield, dragon]

        return super().reset()

config_point = {
    'robot_base': 'xmls/point.xml',
    'walled': True,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 2000
}

register(id="PointDQ-v0", entry_point="genv.DungeonQuestEnv:DungeonQuestEnv", kwargs={"config": config_point})