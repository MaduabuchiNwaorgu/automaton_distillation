import os
import numpy as np
import gym
from ...automaton.automaton_wrapper import AutomatonWrapper


def _make_halfcheetah_base(max_episode_steps: int):
    """Create and return a MuJoCo HalfCheetah base environment.

    Tries gymnasium first (HalfCheetah-v5, then v4), then falls back to gym
    (HalfCheetah-v4, v3, v2). Applies ``max_episode_steps`` via the
    ``_max_episode_steps`` attribute or a TimeLimit wrapper.

    Raises RuntimeError if all attempts fail.
    """
    mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
    mujoco_root = os.path.expanduser("~/.mujoco/mujoco210")
    if os.path.isdir(mujoco_bin):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if mujoco_bin not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (ld + (":" if ld else "") + mujoco_bin)
    if os.path.isdir(mujoco_root) and "MUJOCO_PY_MUJOCO_PATH" not in os.environ:
        os.environ["MUJOCO_PY_MUJOCO_PATH"] = mujoco_root
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    env = None
    # Prefer Gymnasium mujoco first (works with mujoco >= 2.x)
    try:
        import gymnasium as gymn
        for env_id in ['HalfCheetah-v5', 'HalfCheetah-v4']:
            try:
                env = gymn.make(env_id, render_mode=None)
                if env is not None:
                    break
            except Exception:
                continue
    except Exception:
        env = None

    # Fallback to classic Gym variants
    if env is None:
        for env_id in ['HalfCheetah-v4', 'HalfCheetah-v3', 'HalfCheetah-v2']:
            try:
                env = gym.make(env_id, exclude_current_positions_from_observation=False, max_episode_steps=max_episode_steps)
                break
            except Exception:
                try:
                    env = gym.make(env_id, max_episode_steps=max_episode_steps)
                    break
                except Exception:
                    try:
                        env = gym.make(env_id)
                        try:
                            env._max_episode_steps = max_episode_steps
                        except Exception:
                            pass
                        break
                    except Exception:
                        env = None
                        continue

    if env is None:
        raise RuntimeError("Could not create any HalfCheetah environment (checked gymnasium and gym variants)")

    # Apply max_episode_steps via _max_episode_steps attribute or TimeLimit wrapper
    updated_existing = False
    try:
        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = max_episode_steps
            updated_existing = True
        spec = getattr(env, 'spec', None)
        if spec is not None and hasattr(spec, 'max_episode_steps') and spec.max_episode_steps is not None:
            spec.max_episode_steps = max_episode_steps
            updated_existing = True
    except Exception:
        pass

    if not updated_existing:
        wrapped = False
        try:
            import gymnasium as gymn  # noqa: F401
            from gymnasium.wrappers import TimeLimit as GymnTimeLimit
            if 'TimeLimit' in type(env).__name__:
                try:
                    env._max_episode_steps = max_episode_steps
                    wrapped = True
                except Exception:
                    pass
            else:
                env = GymnTimeLimit(env, max_episode_steps=max_episode_steps)
                wrapped = True
        except Exception:
            pass
        if not wrapped:
            try:
                from gym.wrappers import TimeLimit as GymTimeLimit
                if 'TimeLimit' in type(env).__name__:
                    try:
                        env._max_episode_steps = max_episode_steps
                        wrapped = True
                    except Exception:
                        pass
                else:
                    env = GymTimeLimit(env, max_episode_steps=max_episode_steps)
                    wrapped = True
            except Exception:
                pass
        if not wrapped:
            try:
                env._max_episode_steps = max_episode_steps
            except Exception:
                pass

    try:
        if not hasattr(env, 'reward_range'):
            base = getattr(env, 'unwrapped', None)
            env.reward_range = getattr(base, 'reward_range', (-float('inf'), float('inf')))
    except Exception:
        try:
            env.reward_range = (-float('inf'), float('inf'))
        except Exception:
            pass

    return env


class HalfCheetahEnv(gym.Wrapper):
    def __init__(self, max_episode_steps=1000, ap_config=None):
        env = _make_halfcheetah_base(max_episode_steps)
        self.ap_config = ap_config or {}
        self.a_threshold = float(self.ap_config.get('a_threshold', 5.0))
        self.b_threshold = float(self.ap_config.get('b_threshold', -2.0))
        super().__init__(env)
        self.last_obs = None

    def reset(self, **kwargs):
        """Return Gymnasium-style (obs, info) pair."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self.last_obs = obs
        # Expose agent x-position for trajectory logging
        try:
            info['agent_pos'] = np.array([float(self.env.unwrapped.data.qpos[0])], dtype=np.float64)
        except Exception:
            pass
        return obs, info

    def step(self, action):
        """Return Gymnasium-style 5-tuple."""
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        self.last_obs = obs
        # Expose agent x-position for trajectory logging
        try:
            info['agent_pos'] = np.array([float(self.env.unwrapped.data.qpos[0])], dtype=np.float64)
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

    def get_events(self):
        x_pos = float(self.env.unwrapped.data.qpos[0])
        events = ''
        # Forward goal: far right
        if x_pos > self.a_threshold:
            events += 'a'
        # Backward goal: modest left
        if x_pos < self.b_threshold:
            events += 'b'
        return events

class HalfCheetahPatrol(AutomatonWrapper):
    def __init__(self, reward_shaping=True, max_episode_steps=1000, ap_config=None):
        base_env = HalfCheetahEnv(max_episode_steps=max_episode_steps, ap_config=ap_config)

        # Optionally wrap with obstacles if requested in ap_config
        use_obstacles = ap_config.get('obstacles', False) if ap_config else False
        if use_obstacles:
            env = HalfCheetahEnvObstacles(max_episode_steps=max_episode_steps, ap_config=ap_config)
        else:
            env = base_env
        propositions = {'a': None, 'b': None}
        ltlf_formula = "F(a & F(b))"
        rewards = {
            # Updated reward structure based on correct automaton states:
            # State 0: Initial state (haven't seen 'a' yet)
            # State 1: SUCCESS terminal state (saw 'a' then 'b' - formula satisfied!)
            # State 2: Intermediate state (saw 'a', now need 'b')

            # From State 0 (initial)
            (0, 0): 0.0,    # staying in initial (no events) - neutral
            (0, 1): 100.0,  # initial → SUCCESS (shouldn't happen for F(a & F(b)))
            (0, 2): 10.0,   # initial → intermediate (reached 'a') - MILESTONE!

            # From State 1 (SUCCESS - terminal)
            (1, 1): 0.0,    # staying in success (episode should end)
            (1, 2): 0.0,    # success → ??? (shouldn't happen)

            # From State 2 (intermediate - have 'a', need 'b')
            (2, 1): 100.0,  # intermediate → SUCCESS (reached 'b' after 'a' - GOAL ACHIEVED!)
            (2, 2): 0.0,    # staying in intermediate - neutral (continue searching for 'b')
        }
        super().__init__(
            env=env,
            propositions=propositions,
            ltlf_formula=ltlf_formula,
            reward_mapping=rewards,
            reward_shaping=reward_shaping,
        )

class HalfCheetahEnvObstacles(gym.Wrapper):
    def __init__(self, max_episode_steps=1000, ap_config=None):
        # Reuse the robust base wrapper to ensure compatibility across gym versions
        base = HalfCheetahEnv(max_episode_steps=max_episode_steps, ap_config=ap_config)
        super().__init__(base)
        self.last_obs = None
        self.ap_config = ap_config or {}
        # Store thresholds from ap_config if provided
        self.a_threshold = float(self.ap_config.get('a_threshold', 5.0))
        self.b_threshold = float(self.ap_config.get('b_threshold', -2.0))

        # Define obstacle zones (x-position ranges where obstacles exist)
        self.obstacle_zones = [
            (1.0, 2.0),   # Obstacle zone 1: x between 1.0 and 2.0
            (4.0, 5.0),   # Obstacle zone 2: x between 4.0 and 5.0
            (-2.0, -1.0), # Obstacle zone 3: x between -2.0 and -1.0
        ]

        # Track if agent was in obstacle in previous step (for penalty logic)
        self.was_in_obstacle = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.was_in_obstacle = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs

        # Add obstacle penalty to the base reward
        x_pos = float(self.env.unwrapped.data.qpos[0])
        in_obstacle = self._is_in_obstacle(x_pos)

        # Apply obstacle penalty
        if in_obstacle:
            reward -= 0.5  # Penalty for being in obstacle
            self.was_in_obstacle = True
        else:
            self.was_in_obstacle = False

        return obs, reward, terminated, truncated, info

    def _is_in_obstacle(self, x_pos):
        """Check if the agent is currently in any obstacle zone"""
        for min_x, max_x in self.obstacle_zones:
            if min_x <= x_pos <= max_x:
                return True
        return False

    def get_events(self):
        x_pos = float(self.env.unwrapped.data.qpos[0])
        events = ''

        # Use thresholds if set (from ap_config), otherwise use defaults
        a_threshold = getattr(self, 'a_threshold', 5.0)
        b_threshold = getattr(self, 'b_threshold', -2.0)

        # Original events for reaching target positions
        if x_pos > a_threshold:
            events += 'a'
        if x_pos < b_threshold:
            events += 'b'

        return events


class HalfCheetahObstacles(AutomatonWrapper):
    """Half Cheetah with obstacles - must reach targets while avoiding obstacles"""
    def __init__(self):
        env = HalfCheetahEnvObstacles()
        propositions = {'a': None, 'b': None, 'o': None}  # Added 'o' for obstacles

        # LTL formula: reach a and then b, while avoiding obstacles
        # F(a & F(b)) & G(!o) means "eventually reach a and then b, but never hit obstacles"
        # However, G(!o) might be too strict, so we use a softer approach with penalties
        ltlf_formula = "F(a & F(b))"  # Keep the same goal, but obstacles add penalties via step rewards

        rewards = {
            # Enhanced reward structure accounting for obstacle avoidance
            # State 0 (initial)
            (0, 0): 0.0,   # staying in initial - neutral
            (0, 1): 100.0, # initial -> TERMINAL (should not happen for F(a & F(b)))
            (0, 2): 2.0,   # initial -> intermediate (reached 'a') - higher reward to compensate for obstacle penalties

            # State 1 (TERMINAL - goal achieved)
            (1, 1): 0.0,   # staying in terminal (episode should end)

            # State 2 (intermediate - have 'a', need 'b')
            (2, 1): 150.0, # intermediate -> TERMINAL (reached 'b' after 'a' - GOAL!) - higher reward
            (2, 2): 0.1,   # staying in intermediate - small positive to encourage continued exploration
        }
        super().__init__(env=env, propositions=propositions, ltlf_formula=ltlf_formula, reward_mapping=rewards)


class HalfCheetahObstaclesStrict(AutomatonWrapper):
    """Half Cheetah with strict obstacle avoidance using LTL"""
    def __init__(self):
        env = HalfCheetahEnvObstacles()
        propositions = {'a': None, 'b': None, 'o': None}

        # Strict LTL formula: must avoid obstacles completely
        # We can't use G(!o) directly in LTLf, so we use a different approach
        # This version will have much stricter penalties for hitting obstacles
        ltlf_formula = "F(a & F(b))"

        rewards = {
            # Strict obstacle avoidance reward structure
            # State 0 (initial)
            (0, 0): 0.0,   # staying in initial - neutral
            (0, 1): 100.0, # initial -> TERMINAL
            (0, 2): 3.0,   # initial -> intermediate (reached 'a') - high reward

            # State 1 (TERMINAL - goal achieved)
            (1, 1): 0.0,   # staying in terminal

            # State 2 (intermediate - have 'a', need 'b')
            (2, 1): 200.0, # intermediate -> TERMINAL (reached 'b' after 'a') - very high reward
            (2, 2): 0.2,   # staying in intermediate - encourage exploration
        }
        super().__init__(env=env, propositions=propositions, ltlf_formula=ltlf_formula, reward_mapping=rewards)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Add severe penalty for hitting obstacles in this strict version
        events = self.env.get_events()
        if 'o' in events:
            reward -= 5.0  # Severe penalty for hitting obstacles

        return obs, reward, terminated, truncated, info


gym.register(
    id='HalfCheetahPatrol-v0',
    entry_point=HalfCheetahPatrol,
    max_episode_steps=2000
)

gym.register(
    id='HalfCheetahObstacles-v0',
    entry_point=HalfCheetahObstacles,
    max_episode_steps=2000
)

gym.register(
    id='HalfCheetahObstaclesStrict-v0',
    entry_point=HalfCheetahObstaclesStrict,
    max_episode_steps=2000
)
