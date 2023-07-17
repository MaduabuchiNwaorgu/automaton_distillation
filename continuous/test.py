import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import DungeonQuestEnv

def l2(v):
    return np.square(v).sum() ** 0.5

def f(action, steps=10):
    for i in range(steps):
        o,r,d,i = env.step(action)
        env.render()

def dungeon_quest_controller(env, render=True):
    zone = 0
    total_reward = 0
    for i in range(10000):
        robot_pos = obs['robot_pos']
        cur_vec = obs['robot_dir']
        zone_pos = obs[f'zones_lidar_{zone}'][1:]
        
        target_vec = zone_pos - robot_pos
        rot = np.cross(cur_vec, target_vec) / (l2(cur_vec) * l2(target_vec))
        
        action = [0.03, rot]
        obs, reward, done, info = env.step(action)
        if render: env.render()

        total_reward += reward
        
        if env.zones[zone] == DungeonQuestEnv.visited:
            zone += 1
        
        if done:
            break

    return total_reward

checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path="./logs/",
  name_prefix="dq_ppo",
)

env = gym.make("PointDQ-v0")

# Manual controller to verify environment rules
# dungeon_quest_controller(env)

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./dq_ppo_log")
model.learn(total_timesteps=1000000, callback=checkpoint_callback)


# Evaluate PPO agent
obs = env.reset()
for i in range(10000):
    action, state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action[0])
    env.render()
    
    if done:
        break