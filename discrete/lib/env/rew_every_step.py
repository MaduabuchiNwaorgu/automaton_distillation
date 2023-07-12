import gym


class RewEveryStep(gym.Wrapper):
    """Add a given reward to every step of the environment"""
    def __init__(self, env, rew_per_step: float):
        super().__init__(env)
        self.rew_per_step = rew_per_step

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew + self.rew_per_step, done, info
