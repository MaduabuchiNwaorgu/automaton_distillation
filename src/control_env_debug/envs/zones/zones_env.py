"""from __future__ import annotationsSafety Gymnasium (Zones) environment for the automaton-distillation pipeline.

Wraps the Safety-Gymnasium ``Point/Car/Racecar`` zone-navigation tasks so they
expose ``get_events()`` for the AutomatonWrapper (same interface as HalfCheetah).

Observations are *flattened* from the Dict space produced by Safety-Gymnasium so
that the TD3 networks receive a 1-D Box input.

Requirements
    pip install safety-gymnasium   (or the vendored copy in deep-ltl)

Levels available in the deep-ltl fork:

    Ltl0 — 1 colour  (green)
    Ltl1 — 2 colours (green, yellow)
    Ltl2 — 4 colours (blue, green, magenta, yellow)
    Ltl3 — 4 colours, fixed layout

This file mirrors the relevant parts of deep-ltl's ``SafetyGymWrapper`` but is
self-contained so it can run inside the TMLR pipeline without importing deep-ltl.
"""

from typing import Any, Dict, Optional, Set
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.spaces import Box
except ImportError:
    import gym
    from gym import spaces
    from gym.spaces import Box


# ─────────────────────── base wrapper ───────────────────────


class ZonesBase(gym.Wrapper):
    """Flatten Safety-Gymnasium zones obs → 1-D Box and expose ``get_events()``.

    The wrapper discovers zone colours from observation-space keys ending in
    ``_zones_lidar``.  Active propositions are read from ``info['cost_zones_<color>']``
    on each step (>0 means agent is inside a zone of that colour).
    """

    def __init__(self, env, wall_sensor: bool = True):
        super().__init__(env)
        # Discover zone colours from obs keys
        obs_keys = env.observation_space.spaces.keys()
        self.colors: set = set()
        for key in obs_keys:
            if key.endswith("zones_lidar"):
                color = key.split("_")[0]
                self.colors.add(color)

        # Build flat observation space
        total_dim = 0
        self._obs_keys_ordered = sorted(env.observation_space.spaces.keys())
        for k in self._obs_keys_ordered:
            sp = env.observation_space.spaces[k]
            total_dim += int(np.prod(sp.shape))

        # Add wall sensor
        self._wall_sensor = wall_sensor
        if wall_sensor:
            total_dim += 4

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float64
        )
        self._active_props: Set[str] = set()

    # ── helpers ──

    def _flatten_obs(self, obs_dict: dict, info: dict) -> np.ndarray:
        parts = []
        for k in self._obs_keys_ordered:
            parts.append(np.asarray(obs_dict[k], dtype=np.float64).ravel())
        if self._wall_sensor:
            ws = info.get("wall_sensor", np.zeros(4, dtype=np.float64))
            if ws is None:
                ws = np.zeros(4, dtype=np.float64)
            parts.append(np.asarray(ws, dtype=np.float64).ravel())
        return np.concatenate(parts)

    # ── propositions interface ──

    def get_events(self) -> str:
        """Return a string of single-char AP labels that are currently true.

        We map sorted colours to letters a, b, c, … so that ``get_ap_id``
        works with the automaton's ``prop_names``.
        """
        sorted_colors = sorted(self.colors)
        return "".join(
            chr(ord("a") + i) for i, c in enumerate(sorted_colors) if c in self._active_props
        )

    def get_propositions(self):
        return sorted(self.colors)

    # ── gym API ──

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self._active_props = set()
        if self._wall_sensor:
            obs_dict["wall_sensor"] = np.zeros(4, dtype=np.float64)
        flat = self._flatten_obs(obs_dict, info)
        info["propositions"] = set()
        # Expose agent (x, y) for trajectory logging
        try:
            info['agent_pos'] = np.asarray(self.env.unwrapped.task.agent.pos[:2], dtype=np.float64).copy()
        except Exception:
            pass
        return flat, info

    def step(self, action):
        # Safety-Gymnasium returns 6 values: (obs, reward, cost, terminated, truncated, info)
        # Standard Gymnasium returns 5:      (obs, reward, terminated, truncated, info)
        raw = self.env.step(action)
        if isinstance(raw, tuple) and len(raw) == 6:
            obs_dict, reward, _cost, terminated, truncated, info = raw
        elif isinstance(raw, tuple) and len(raw) == 5:
            obs_dict, reward, terminated, truncated, info = raw
        else:
            raise ValueError(f"Unexpected step return length: {len(raw)}")

        # Extract active zone colours
        self._active_props = set()
        for c in self.colors:
            key = f"cost_zones_{c}"
            if info.get(key, 0) > 0:
                self._active_props.add(c)

        # Handle wall collision
        if "cost_ltl_walls" in info and info["cost_ltl_walls"] > 0:
            terminated = True
            reward = -1.0

        if self._wall_sensor:
            ws = info.get("wall_sensor", np.zeros(4, dtype=np.float64))
            if ws is None:
                ws = np.zeros(4, dtype=np.float64)
            obs_dict["wall_sensor"] = ws

        flat = self._flatten_obs(obs_dict, info)
        info["propositions"] = self._active_props
        # Expose agent (x, y) for trajectory logging
        try:
            info['agent_pos'] = np.asarray(self.env.unwrapped.task.agent.pos[:2], dtype=np.float64).copy()
        except Exception:
            pass
        return flat, float(reward), bool(terminated), bool(truncated), info


# ────────────────── automaton-wrapped task variants ──────────────────

from ...automaton.automaton_wrapper import AutomatonWrapper


def _make_safety_env(env_id: str, render_mode: Optional[str] = None):
    """Instantiate a Safety-Gymnasium env and wrap it with ``ZonesBase``."""
    try:
        import safety_gymnasium
    except ImportError:
        raise ImportError(
            "safety-gymnasium is required for zones environments.\n"
            "Install with:  pip install safety-gymnasium"
        )
    raw_env = safety_gymnasium.make(env_id, render_mode=render_mode)
    return ZonesBase(raw_env)


class ZonesPatrol(AutomatonWrapper):
    """Safety-Gymnasium zones: visit colour-A then colour-B — F(a & F(b)).

    Default env: PointLtl1-v0 (green + yellow zones).
    Sorted colours: green (a), yellow (b).
    Formula: F(a & F(b)) → visit green then yellow.
    """

    def __init__(
        self,
        env_id: str = "PointLtl1-v0",
        reward_shaping: bool = True,
        max_episode_steps: int = 1000,
        ap_config: Optional[Dict] = None,
    ):
        env = _make_safety_env(env_id)
        n_colors = len(env.get_propositions())
        propositions = {chr(ord("a") + i): None for i in range(n_colors)}

        # 2-colour patrol: F(a & F(b))
        ltlf_formula = "F(a & F(b))"
        rewards = {
            (0, 0): 0.0,
            (0, 1): 100.0,
            (0, 2): 10.0,    # reached first colour
            (1, 1): 0.0,
            (2, 1): 100.0,   # reached second colour → goal
            (2, 2): 0.0,
        }
        super().__init__(
            env=env,
            propositions=propositions,
            ltlf_formula=ltlf_formula,
            reward_mapping=rewards,
            reward_shaping=reward_shaping,
            shaping_scale=0.0,
            step_penalty=0.01,
        )

    def _env_specific_shaping(self, obs, delta, rm_state):
        return 0.0


class ZonesSequence(AutomatonWrapper):
    """Safety-Gymnasium zones: visit 4 colours in order (PointLtl2-v0).

    Sorted colours: blue (a), green (b), magenta (c), yellow (d).
    Formula: F(b & F(a & F(c & F(d))))  → green → blue → magenta → yellow.
    """

    def __init__(
        self,
        env_id: str = "PointLtl2-v0",
        reward_shaping: bool = True,
        max_episode_steps: int = 1000,
        ap_config: Optional[Dict] = None,
    ):
        env = _make_safety_env(env_id)
        n_colors = len(env.get_propositions())
        propositions = {chr(ord("a") + i): None for i in range(n_colors)}

        # 4-colour sequence: green(b) → blue(a) → magenta(c) → yellow(d)
        ltlf_formula = "F(b & F(a & F(c & F(d))))"
        # Actual automaton states: 0=init, 1=saw-green-then-blue, 2=accept,
        #                          3=saw-green, 4=saw-green-blue-magenta
        # Correct path: 0 →(b)→ 3 →(a)→ 1 →(c)→ 4 →(d)→ 2
        rewards = {
            (0, 0): 0.0,
            (0, 3): 10.0,      # saw green (b)
            (0, 1): 25.0,      # saw green+blue simultaneously
            (0, 4): 45.0,      # saw green+blue+magenta simultaneously
            (0, 2): 145.0,     # saw all four simultaneously
            (3, 3): 0.0,
            (3, 1): 15.0,      # saw blue after green (a)
            (3, 4): 35.0,      # saw blue+magenta simultaneously after green
            (3, 2): 135.0,     # saw blue+magenta+yellow after green
            (1, 1): 0.0,
            (1, 4): 20.0,      # saw magenta after green,blue (c)
            (1, 2): 120.0,     # saw magenta+yellow simultaneously
            (4, 4): 0.0,
            (4, 2): 100.0,     # saw yellow (d) — GOAL
            (2, 2): 0.0,
        }
        super().__init__(
            env=env,
            propositions=propositions,
            ltlf_formula=ltlf_formula,
            reward_mapping=rewards,
            reward_shaping=reward_shaping,
            shaping_scale=0.0,
            step_penalty=0.01,
        )

    def _env_specific_shaping(self, obs, delta, rm_state):
        return 0.0


# ── gym registration ──
try:
    gym.register(id="ZonesPatrol-v0", entry_point=ZonesPatrol, max_episode_steps=1000)
    gym.register(id="ZonesSequence-v0", entry_point=ZonesSequence, max_episode_steps=1000)
except Exception:
    pass
