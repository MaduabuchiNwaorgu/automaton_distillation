"""
Vectorized env utilities – lightweight, self-contained.
No external dependency on stable-baselines3.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import gym
from gym import spaces as gspaces
import numpy as np

# Our local factory
from .train_env import make_env as make_single_env


# ---------------------------------------------------------------------------
#  Lightweight VecEnv base (replaces SB3 VecEnvWrapper / DummyVecEnv / etc.)
# ---------------------------------------------------------------------------

class _VecEnv:
    """Minimal vectorized-env interface matching the subset we actually use.

    API contract (old-style Gym):
        reset()                -> obs   (n_envs, *obs_shape)
        step(actions)          -> obs, rewards, dones, infos
        .num_envs, .observation_space, .action_space
    """

    def __init__(self, num_envs: int, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


class DummyVecEnv(_VecEnv):
    """Run *n* envs sequentially in the current process."""

    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        env0 = self.envs[0]
        super().__init__(
            num_envs=len(self.envs),
            observation_space=env0.observation_space,
            action_space=env0.action_space,
        )

    def reset(self):
        obs_list = []
        for env in self.envs:
            o = env.reset()
            obs_list.append(np.asarray(o, dtype=np.float32))
        return np.stack(obs_list)

    def step(self, actions):
        obs_list, rews, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, d, info = env.step(actions[i])
            if d:
                # Auto-reset on done (standard VecEnv semantics)
                info["terminal_observation"] = np.asarray(o, dtype=np.float32)
                o = env.reset()
            obs_list.append(np.asarray(o, dtype=np.float32))
            rews.append(float(r))
            dones.append(bool(d))
            infos.append(info)
        return (
            np.stack(obs_list),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )


class _VecEnvWrapper(_VecEnv):
    """Base class for VecEnv wrappers (replaces SB3 VecEnvWrapper)."""

    def __init__(self, venv: _VecEnv):
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        self.venv = venv

    def reset(self):
        return self.venv.reset()

    def step(self, actions):
        return self.venv.step(actions)


class VecMonitor(_VecEnvWrapper):
    """Log episode returns / lengths to a CSV file (replaces SB3 VecMonitor)."""

    def __init__(self, venv: _VecEnv, filename: Optional[str] = None):
        super().__init__(venv)
        self._ep_returns = np.zeros(venv.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(venv.num_envs, dtype=np.int64)
        self._t_start = time.time()
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None
        if filename is not None:
            self._csv_path = filename + ".monitor.csv"
            self._csv_file = open(self._csv_path, "w", newline="")
            # SB3-compatible header
            self._csv_file.write(f"#{{\"t_start\": {self._t_start:.2f}}}\n")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["r", "l", "t"])

    def reset(self):
        obs = self.venv.reset()
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        return obs

    def step(self, actions):
        obs, rews, dones, infos = self.venv.step(actions)
        self._ep_returns += rews
        self._ep_lengths += 1
        for i in range(self.num_envs):
            if dones[i]:
                ep_info = {
                    "r": round(float(self._ep_returns[i]), 6),
                    "l": int(self._ep_lengths[i]),
                    "t": round(time.time() - self._t_start, 6),
                }
                infos[i]["episode"] = ep_info
                if self._csv_writer is not None:
                    self._csv_writer.writerow(
                        [ep_info["r"], ep_info["l"], ep_info["t"]]
                    )
                    self._csv_file.flush()
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
        return obs, rews, dones, infos

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
        try:
            self.venv.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Per-env API adapters (Gymnasium -> old Gym 4-tuple)
# ---------------------------------------------------------------------------

class _GymSpaceWrapper(gym.Wrapper):
    """Ensure spaces are gym.spaces.Box(float32) even when the inner env
    uses gymnasium.spaces under the hood."""

    def __init__(self, env):
        super().__init__(env)
        try:
            import gymnasium as gymn

            if isinstance(env.action_space, gymn.spaces.Box):
                self.action_space = gspaces.Box(
                    low=np.asarray(env.action_space.low, dtype=np.float32),
                    high=np.asarray(env.action_space.high, dtype=np.float32),
                    dtype=np.float32,
                )
            if isinstance(env.observation_space, gymn.spaces.Box):
                self.observation_space = gspaces.Box(
                    low=np.asarray(env.observation_space.low, dtype=np.float32),
                    high=np.asarray(env.observation_space.high, dtype=np.float32),
                    dtype=np.float32,
                )
        except Exception:
            pass


class _APIBridge(gym.Wrapper):
    """Bridge Gymnasium API  reset->(obs,info), step->(obs,r,term,trunc,info)
    to old Gym API  reset->obs, step->(obs,r,done,info)."""

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]
        return result

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated or truncated), info
        return result


# ---------------------------------------------------------------------------
#  VecEnv-level wrappers
# ---------------------------------------------------------------------------

class _VecGymSpaceWrapper(_VecEnvWrapper):
    """Coerce vec-level observation/action spaces to gym.spaces.Box(float32)."""

    def __init__(self, venv: _VecEnv):
        super().__init__(venv)
        self.observation_space = _to_gym_box(venv.observation_space)
        self.action_space = _to_gym_box(venv.action_space)


def _to_gym_box(space) -> gspaces.Box:
    """Best-effort conversion of any Box-like space to gym.spaces.Box(float32)."""
    if isinstance(space, gspaces.Box):
        return gspaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            dtype=np.float32,
        )
    try:
        import gymnasium as gymn

        if isinstance(space, gymn.spaces.Box):
            return gspaces.Box(
                low=np.asarray(space.low, dtype=np.float32),
                high=np.asarray(space.high, dtype=np.float32),
                dtype=np.float32,
            )
    except Exception:
        pass
    # Fallback: return as-is
    return space


class VecObsNorm(_VecEnvWrapper):
    """Running-mean/var observation normalizer for VecEnv.

    Saves/loads stats via .save(path) / .load(path).
    """

    def __init__(self, venv: _VecEnv, clip_obs: float = 10.0, epsilon: float = 1e-8):
        super().__init__(venv)
        shape = self.observation_space.shape
        if shape is None:
            raise AssertionError("VecObsNorm expects Box-like observation space with shape")
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.observation_space = gspaces.Box(
            low=np.full(shape, -np.inf, dtype=np.float32),
            high=np.full(shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    # --- Welford running stats ---
    def _update_rms(self, x: np.ndarray):
        x = x.astype(np.float64)
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var + self.epsilon
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            tot = self.count + batch_count
            self.mean += delta * (batch_count / tot)
            M2 = (self.var * self.count
                   + batch_var * batch_count
                   + np.square(delta) * (self.count * batch_count / tot))
            self.var = M2 / tot + self.epsilon
            self.count = tot

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        std = np.sqrt(self.var).astype(np.float32)
        x_norm = (x - self.mean.astype(np.float32)) / (std + 1e-8)
        return np.clip(x_norm, -self.clip_obs, self.clip_obs)

    def reset(self):
        obs = self.venv.reset()
        self._update_rms(obs)
        return self._normalize(obs)

    def step(self, actions):
        obs, rewards, dones, infos = self.venv.step(actions)
        self._update_rms(obs)
        return self._normalize(obs), rewards, dones, infos

    # --- Persistence ---
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path,
                            count=int(self.count),
                            mean=self.mean,
                            var=self.var,
                            clip_obs=self.clip_obs,
                            epsilon=self.epsilon)

    def load(self, path: str):
        d = np.load(path)
        self.count = int(d["count"])
        self.mean = d["mean"]
        self.var = d["var"]
        self.clip_obs = float(d["clip_obs"])
        self.epsilon = float(d["epsilon"])


# ---------------------------------------------------------------------------
#  Public factory
# ---------------------------------------------------------------------------

def make_vec_env(
    env_type: str,
    n_envs: int = 1,
    seed: Optional[int] = None,
    reward_shaping: bool = True,
    max_episode_steps: int = 1000,
    ap_config: Optional[Dict] = None,
    ltlf_formula: Optional[str] = None,
    reward_mapping: Optional[Dict] = None,
    run_name: str = "halfcheetah",
    force_dummy: bool = False,
    monitor_dir: Optional[str] = None,
    normalize: bool = False,
    norm_kwargs: Optional[Dict] = None,
):
    """Create a vectorized env with automaton wrapper (no SB3 dependency)."""

    def make_thunk(rank: int):
        def _init():
            base_seed = None if seed is None else int(seed)
            env_seed = None if base_seed is None else (base_seed + rank)
            if env_seed is not None:
                np.random.seed(env_seed)
            env = make_single_env(
                env_type=env_type,
                reward_shaping=reward_shaping,
                max_episode_steps=max_episode_steps,
                ap_config=ap_config,
                run_name=f"{run_name}_{rank}",
            )
            env = _GymSpaceWrapper(env)
            env = _APIBridge(env)
            try:
                if env_seed is not None:
                    env.reset(seed=env_seed)
            except Exception:
                pass
            return env
        return _init

    # Always use DummyVecEnv (SubprocVecEnv removed with SB3 dep)
    vec = DummyVecEnv([make_thunk(i) for i in range(n_envs)])
    vec = _VecGymSpaceWrapper(vec)

    norm_ref = None
    if normalize:
        clip_obs = 10.0
        if norm_kwargs and "clip_obs" in norm_kwargs:
            clip_obs = float(norm_kwargs["clip_obs"])
        vec = VecObsNorm(vec, clip_obs=clip_obs)
        norm_ref = vec

    if monitor_dir is None:
        monitor_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(monitor_dir, exist_ok=True)
    monitor_path = os.path.join(monitor_dir, run_name)
    monitor = VecMonitor(vec, filename=monitor_path)

    # Attach a reference to VecObsNorm so callers can save stats
    if norm_ref is not None:
        setattr(monitor, "_vecnormalize_ref", norm_ref)
    return monitor
