import gym
import numpy as np

import DungeonQuestEnv

def l2(v):
    return np.square(a).sum() ** 0.5

def f(action, steps=10):
    for i in range(steps):
        o,r,d,i = env.step(action)
        env.render()

env = gym.make("PointDQ-v0")

obs = env.reset()
env.render()

# zone = 0
# for i in range(1000):
#     robot_pos = env.world.robot_pos()
#     cur_vec = obs['robot_dir']
#     zone_pos = env.zones_pos[zone][:2]
    
#     target_vec = zone_pos - robot_pos
#     rot = -np.dot(cur_vec, target_vec) / (l2(cur_vec) * l2(target_vec))
    
#     action = [1, rot]
#     obs, reward, done, info = env.step(action)
#     env.render()
    
#     if env.zones[zone] == env.visited:
#         zone += 1