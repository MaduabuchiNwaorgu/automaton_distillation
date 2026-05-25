#!/usr/bin/env python3
"""Environment factory for automaton-wrapped RL tasks.

Supported env_type values:
    HalfCheetah:
        patrol, obstacles, obstacles_strict
    FlatWorld (2-D continuous navigation, coloured circles):
        flatworld_patrol, flatworld_sequence
    Safety Gymnasium Zones (MuJoCo point-mass with zone navigation):
        zones_patrol, zones_sequence
    Misc:
        cartpole  (smoke-test — no automaton)
"""

import gym
from typing import Dict, Optional


def make_env(env_type: str = "patrol",
             reward_shaping: bool = True,
             max_episode_steps: int = 1000,
             ap_config: Optional[Dict[str, Dict]] = None,
             ltlf_formula: Optional[str] = None,
             reward_mapping: Optional[Dict] = None,
             run_name: str = "default"):
    """Create an automaton-wrapped env based on ``env_type``.

    Returns an env whose API is Gymnasium-style:
        reset()  → (obs, info)
        step(a)  → (obs, reward, terminated, truncated, info)
    and exposes ``.automaton``, ``.get_events()`` for the distillation pipeline.
    """

    # ── HalfCheetah variants ──
    if env_type in ("patrol", "obstacles", "obstacles_strict"):
        from .envs.mujoco.half_cheetah_env import (
            HalfCheetahPatrol, HalfCheetahObstacles, HalfCheetahObstaclesStrict
        )
        if env_type == "patrol":
            return HalfCheetahPatrol(
                reward_shaping=reward_shaping,
                max_episode_steps=max_episode_steps,
                ap_config=ap_config,
            )
        elif env_type == "obstacles":
            return HalfCheetahObstacles()
        elif env_type == "obstacles_strict":
            return HalfCheetahObstaclesStrict()

    # ── FlatWorld variants ──
    elif env_type in ("flatworld_patrol", "flatworld", "flatworld_sequence"):
        from .envs.flatworld.flatworld_env import (
            FlatWorldPatrol, FlatWorldSequence,
        )
        if env_type in ("flatworld_patrol", "flatworld"):
            return FlatWorldPatrol(
                reward_shaping=reward_shaping,
                max_episode_steps=max_episode_steps,
                ap_config=ap_config,
            )
        else:
            return FlatWorldSequence(
                reward_shaping=reward_shaping,
                max_episode_steps=max_episode_steps,
                ap_config=ap_config,
            )

    # ── Safety Gymnasium Zones variants ──
    elif env_type in ("zones_patrol", "zones", "zones_sequence"):
        from .envs.zones.zones_env import (
            ZonesPatrol, ZonesSequence,
        )
        # Allow overriding the Safety-Gymnasium env id via ap_config
        sg_env_id = None
        if isinstance(ap_config, dict):
            sg_env_id = ap_config.get("safety_gym_id", None)
        if env_type in ("zones_patrol", "zones"):
            kwargs = dict(reward_shaping=reward_shaping, max_episode_steps=max_episode_steps, ap_config=ap_config)
            if sg_env_id:
                kwargs["env_id"] = sg_env_id
            return ZonesPatrol(**kwargs)
        else:
            kwargs = dict(reward_shaping=reward_shaping, max_episode_steps=max_episode_steps, ap_config=ap_config)
            if sg_env_id:
                kwargs["env_id"] = sg_env_id
            return ZonesSequence(**kwargs)

    # ── Misc ──
    elif env_type == "cartpole":
        try:
            return gym.make('CartPole-v1')
        except Exception:
            import gymnasium as gymn
            return gymn.make('CartPole-v1')
    else:
        raise ValueError(
            f"Unknown environment type: {env_type}\n"
            f"Supported: patrol, obstacles, obstacles_strict, "
            f"flatworld_patrol, flatworld_sequence, "
            f"zones_patrol, zones_sequence, cartpole"
        )


if __name__ == "__main__":
    env = make_env()
    print(f"Created env: {env}")
    print(f"  obs space: {env.observation_space}")
    print(f"  act space: {env.action_space}")
    if hasattr(env, 'automaton'):
        print(f"  automaton states: {env.automaton.num_states}")
