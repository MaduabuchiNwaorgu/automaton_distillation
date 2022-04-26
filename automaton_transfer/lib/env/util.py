import gym
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from automaton_transfer.lib.config import EnvConfig


def element_add(a, b):
    """
    Add two tuples, elementwise
    """
    return tuple(x + y for x, y in zip(a, b))


# EnvConfig specifies how to make an environment, the next two functions execute these instructions

def make_env(config: EnvConfig) -> gym.Env:
    env = gym.make(config.env_name, **config.kwargs)
    if config.wrapper_cls:
        env = config.wrapper_cls(env, **config.wrapper_kwargs)
    return env


def make_vec_env(config: EnvConfig, num_envs: int) -> VecEnv:
    # TODO change to SubprocVecEnv instead of DummyVecEnv if the environment is computationally intensive
    if config.wrapper_cls:
        return sb3_make_vec_env(config.env_name, num_envs,
                                vec_env_cls=DummyVecEnv, env_kwargs=config.kwargs,
                                wrapper_class=config.wrapper_cls, wrapper_kwargs=config.wrapper_kwargs)
    else:
        return sb3_make_vec_env(config.env_name, num_envs,
                                vec_env_cls=DummyVecEnv, env_kwargs=config.kwargs)
