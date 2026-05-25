"""
FlatWorld environment adapted from deep-ltl for the automaton-distillation pipeline.

The world is a 2-D continuous plane ([-2, 2]^2) with coloured circular regions
and optional rectangular wall obstacles.
Propositions correspond to the colours the agent is currently touching.
Actions are 2-D continuous vectors clipped to [-1, 1].

This file is self-contained (no dependency on deep-ltl) so that it can be used
inside the TMLR training scripts directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# ─────────────────────────────── base env ───────────────────────────────

@dataclass
class Circle:
    center: np.ndarray
    radius: float
    color: str


@dataclass
class Wall:
    """Axis-aligned rectangular obstacle.  Agent cannot pass through."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


# ── pre-defined circle layouts ──

CIRCLES_DEFAULT: List[Circle] = [
    Circle(center=np.array([-1.4,  0.55]), radius=0.40, color="red"),
    Circle(center=np.array([-1.1,  1.10]), radius=0.50, color="magenta"),
    Circle(center=np.array([-1.0, -1.20]), radius=0.30, color="yellow"),
    Circle(center=np.array([-1.53,-0.50]), radius=0.32, color="orange"),
    Circle(center=np.array([ 0.1,  0.00]), radius=0.80, color="blue"),
    Circle(center=np.array([ 0.5, -1.30]), radius=0.35, color="red"),
    Circle(center=np.array([ 0.7,  0.70]), radius=0.50, color="green"),
    Circle(center=np.array([ 1.5, -0.75]), radius=0.40, color="green"),
    Circle(center=np.array([ 0.8,  0.20]), radius=0.30, color="aqua"),
]

CIRCLES_SHIFTED: List[Circle] = [
    # Same colours but different positions — transfer challenge
    Circle(center=np.array([ 1.2,  1.10]), radius=0.35, color="red"),
    Circle(center=np.array([ 0.5, -1.40]), radius=0.45, color="magenta"),
    Circle(center=np.array([ 1.0,  0.80]), radius=0.30, color="yellow"),
    Circle(center=np.array([ 1.50, 0.20]), radius=0.30, color="orange"),
    Circle(center=np.array([-1.0, -0.60]), radius=0.70, color="blue"),
    Circle(center=np.array([-0.8,  1.20]), radius=0.35, color="red"),
    Circle(center=np.array([-1.3, -1.10]), radius=0.45, color="green"),
    Circle(center=np.array([-0.3,  0.50]), radius=0.40, color="green"),
    Circle(center=np.array([ 0.0, -0.30]), radius=0.25, color="aqua"),
]

# ── pre-defined wall layouts ──

WALLS_NONE: List[Wall] = []

WALLS_CROSS: List[Wall] = [
    # A "+" shaped obstacle in the centre that forces detours
    Wall(x_min=-0.08, y_min=-0.80, x_max=0.08, y_max=0.80),   # vertical bar
    Wall(x_min=-0.80, y_min=-0.08, x_max=0.80, y_max=0.08),   # horizontal bar
]

WALLS_CORRIDORS: List[Wall] = [
    # Two horizontal walls creating a zig-zag corridor
    Wall(x_min=-2.0, y_min=0.40, x_max=0.8, y_max=0.52),   # upper wall (gap on right)
    Wall(x_min=-0.8, y_min=-0.52, x_max=2.0, y_max=-0.40),  # lower wall (gap on left)
]


class FlatWorldEnv(gym.Env):
    """Continuous-action 2-D navigation among coloured circles with optional walls."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        circles: Optional[List[Circle]] = None,
        walls: Optional[List[Wall]] = None,
    ):
        super().__init__()
        self.delta_t = 0.08
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.circles: List[Circle] = circles if circles is not None else list(CIRCLES_DEFAULT)
        self.walls: List[Wall] = walls if walls is not None else list(WALLS_NONE)

        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.agent_pos = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._rng = np.random.default_rng()

    # ── propositions interface (mirrors deep-ltl) ──

    def get_active_propositions(self) -> set:
        props = set()
        for c in self.circles:
            if np.linalg.norm(self.agent_pos - c.center) < c.radius:
                props.add(c.color)
        return props

    def get_propositions(self):
        """Return the sorted list of all possible proposition names."""
        return sorted({c.color for c in self.circles})

    # ── wall collision helpers ──

    def _in_wall(self, pos: np.ndarray) -> bool:
        for w in self.walls:
            if w.x_min <= pos[0] <= w.x_max and w.y_min <= pos[1] <= w.y_max:
                return True
        return False

    # ── gym API ──

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Spawn agent uniformly in the plane but *outside* any circle and wall
        for _ in range(10_000):
            self.agent_pos = self._rng.uniform(-2.0, 2.0, size=(2,))
            if len(self.get_active_propositions()) == 0 and not self._in_wall(self.agent_pos):
                break
        self._step_count = 0
        return self.agent_pos.copy().astype(np.float64), {
            "propositions": set(),
            "agent_pos": self.agent_pos.copy(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()[:2]
        action = np.clip(action, -1.0, 1.0)

        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos + action * self.delta_t
        self._step_count += 1

        terminated = False
        reward = 0.0

        # Wall collision: block movement, apply small penalty
        if self._in_wall(new_pos):
            new_pos = old_pos       # stay in place
            reward = -0.1           # bump penalty

        self.agent_pos = new_pos

        # Out-of-bounds terminates
        if (self.agent_pos < -2.0).any() or (self.agent_pos > 2.0).any():
            self.agent_pos = np.clip(self.agent_pos, -2.0, 2.0)
            terminated = True
            reward = -1.0

        truncated = self._step_count >= self.max_episode_steps
        info = {
            "propositions": self.get_active_propositions(),
            "agent_pos": self.agent_pos.copy(),
        }
        return self.agent_pos.copy().astype(np.float64), reward, terminated, truncated, info


# ─────────────────────── automaton-wrapped variants ─────────────────────────

from ...automaton.automaton_wrapper import AutomatonWrapper


class FlatWorldBase(gym.Wrapper):
    """Thin wrapper that adds ``get_events()`` for the automaton wrapper."""

    def __init__(
        self,
        prop_subset: list[str],
        max_episode_steps: int = 500,
        ap_config: Optional[Dict] = None,
        circles: Optional[List[Circle]] = None,
        walls: Optional[List[Wall]] = None,
    ):
        env = FlatWorldEnv(
            max_episode_steps=max_episode_steps,
            circles=circles,
            walls=walls,
        )
        super().__init__(env)
        self.prop_subset = sorted(prop_subset)
        self._last_props: set = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_props = info.get("propositions", set())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_props = info.get("propositions", set())
        return obs, reward, terminated, truncated, info

    def get_events(self) -> str:
        """Return a string of single-char proposition labels that are true now.

        The automaton wrapper calls this to compute the AP id.  We map
        each tracked colour to a single letter (same order as prop_names)
        so that ``get_ap_id`` works correctly.
        """
        return "".join(
            chr(ord("a") + i) for i, c in enumerate(self.prop_subset) if c in self._last_props
        )


class FlatWorldPatrol(AutomatonWrapper):
    """FlatWorld: visit *red* region then *blue* region — F(b & F(a)).

    Proposition mapping:  a ↔ blue,  b ↔ red  (sorted alphabetically).

    Transfer knobs (via ``ap_config``):
        circles  : "default" | "shifted"   (circle layout preset)
        walls    : "none" | "cross" | "corridors"  (obstacle preset)
    """

    def __init__(self, reward_shaping: bool = True, max_episode_steps: int = 500, ap_config: Optional[Dict] = None):
        cfg = ap_config or {}
        circles = _resolve_circles(cfg.get("circles", "default"))
        walls = _resolve_walls(cfg.get("walls", "none"))

        prop_colors = ["blue", "red"]          # sorted → a=blue, b=red
        env = FlatWorldBase(prop_colors, max_episode_steps=max_episode_steps,
                            ap_config=ap_config, circles=circles, walls=walls)
        propositions = {chr(ord("a") + i): None for i in range(len(prop_colors))}
        # F(b & F(a)):  first visit red (b), then blue (a)
        ltlf_formula = "F(b & F(a))"
        rewards = {
            (0, 0): 0.0,
            (0, 1): 100.0,
            (0, 2): 10.0,   # reached first target colour
            (1, 1): 0.0,
            (2, 1): 100.0,  # reached second target → goal
            (2, 2): 0.0,
        }
        super().__init__(
            env=env,
            propositions=propositions,
            ltlf_formula=ltlf_formula,
            reward_mapping=rewards,
            reward_shaping=reward_shaping,
            shaping_scale=0.0,     # disable x-pos shaping (not meaningful here)
            step_penalty=0.02,     # small step cost encourages efficiency
        )

    # Override unused x-pos shaping logic from AutomatonWrapper
    def _env_specific_shaping(self, obs, delta, rm_state):
        """No env-specific shaping for FlatWorld — rely on RM transitions."""
        return 0.0


class FlatWorldSequence(AutomatonWrapper):
    """FlatWorld: visit red → blue → green — F(c & F(a & F(b))).

    Proposition mapping (sorted): a=blue, b=green, c=red.

    Transfer knobs (via ``ap_config``):
        circles  : "default" | "shifted"
        walls    : "none" | "cross" | "corridors"
    """

    def __init__(self, reward_shaping: bool = True, max_episode_steps: int = 500, ap_config: Optional[Dict] = None):
        cfg = ap_config or {}
        circles = _resolve_circles(cfg.get("circles", "default"))
        walls = _resolve_walls(cfg.get("walls", "none"))

        prop_colors = ["blue", "green", "red"]  # sorted
        env = FlatWorldBase(prop_colors, max_episode_steps=max_episode_steps,
                            ap_config=ap_config, circles=circles, walls=walls)
        propositions = {chr(ord("a") + i): None for i in range(len(prop_colors))}
        # a=blue, b=green, c=red
        # Sequence: red(c) → blue(a) → green(b)
        ltlf_formula = "F(c & F(a & F(b)))"
        # Actual automaton states: 0=init, 1=accept, 2=saw-red-then-blue, 3=saw-red
        # Correct path: 0 →(c)→ 3 →(a)→ 2 →(b)→ 1
        rewards = {
            (0, 0): 0.0,
            (0, 3): 10.0,      # reached red (c)
            (0, 2): 25.0,      # reached red+blue simultaneously (c & a)
            (0, 1): 125.0,     # reached all three simultaneously
            (3, 3): 0.0,
            (3, 2): 15.0,      # reached blue after red (a)
            (3, 1): 115.0,     # reached blue+green simultaneously after red
            (2, 2): 0.0,
            (2, 1): 100.0,     # reached green (b) — GOAL
            (1, 1): 0.0,
        }
        super().__init__(
            env=env,
            propositions=propositions,
            ltlf_formula=ltlf_formula,
            reward_mapping=rewards,
            reward_shaping=reward_shaping,
            shaping_scale=0.0,
            step_penalty=0.02,
        )

    def _env_specific_shaping(self, obs, delta, rm_state):
        return 0.0


# ── layout resolution helpers ──

def _resolve_circles(key: str) -> List[Circle]:
    _presets = {"default": CIRCLES_DEFAULT, "shifted": CIRCLES_SHIFTED}
    if key in _presets:
        return list(_presets[key])
    return list(CIRCLES_DEFAULT)


def _resolve_walls(key: str) -> List[Wall]:
    _presets = {"none": WALLS_NONE, "cross": WALLS_CROSS, "corridors": WALLS_CORRIDORS}
    if key in _presets:
        return list(_presets[key])
    return list(WALLS_NONE)


# ── gym registration (optional, for convenience) ──

try:
    gym.register(id="FlatWorldPatrol-v0", entry_point=FlatWorldPatrol, max_episode_steps=500)
    gym.register(id="FlatWorldSequence-v0", entry_point=FlatWorldSequence, max_episode_steps=500)
except Exception:
    pass
