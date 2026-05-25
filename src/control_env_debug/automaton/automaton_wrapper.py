import gym
import os
import json
import time
from typing import Optional

from .automaton import LTLfAutomaton, get_ap_id

class AutomatonWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        propositions,
        ltlf_formula,
        reward_mapping=None,
        verbose=False,
        # Reward shaping controls
        reward_shaping=True,
        # Shaping/penalties
        step_penalty=0.1,
        shaping_scale=0.01,
        continue_right_penalty_scale=0.02,
    ):
        super().__init__(env)
        self.prop_names = sorted(propositions.keys())

        self.automaton = LTLfAutomaton.from_ltlf(ltlf_formula, propositions)
        self.reward_mapping = reward_mapping if reward_mapping else {}
        self.current_u = self.automaton.reset()
        self.num_states = int(self.automaton.num_states)

        # Config
        self.verbose = verbose
        self.reward_shaping = reward_shaping
        self.step_penalty = float(step_penalty)
        self.shaping_scale = float(shaping_scale)
        self.continue_right_penalty_scale = float(continue_right_penalty_scale)

    def reset(self, **kwargs):
        """Return Gymnasium-style (obs, info) pair."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self.current_u = self.automaton.reset()
        try:
            self.last_pos = float(obs[0])
        except Exception:
            self.last_pos = 0.0
        # Track previous observation for CRM experience generation
        self.prev_obs = obs
        # RewardMachines-style pruning: None => allow all starting states on first step
        self.valid_states = None
        self.episode_idx = getattr(self, "episode_idx", -1) + 1
        self.step_idx = 0
        self.turn_started = False
        info['rm_state'] = self.current_u
        return obs, info

    def step(self, action):
        """Return Gymnasium-style (obs, reward, terminated, truncated, info)."""
        # Keep a copy of previous observation before stepping env
        prev_obs = self.prev_obs
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, _, terminated, truncated, info = result
        else:
            obs, _, done, info = result
            terminated, truncated = bool(done), False

        events_str = self.env.get_events()
        ap_id = get_ap_id(events_str, self.prop_names)

        u_prev = self.current_u
        self.current_u = self.automaton.step(u_prev, ap_id)

        rm_reward = float(self.reward_mapping.get((u_prev, self.current_u), 0.0))
        total_reward = rm_reward - self.step_penalty

        # Apply reward shaping only if enabled
        if self.reward_shaping:
            x_pos = obs[0] if len(obs) > 0 else 0.0
            try:
                delta = float(x_pos) - float(self.last_pos)
            except Exception:
                delta = 0.0

            # Dispatch to env-specific shaping.  Subclasses can override
            # ``_env_specific_shaping`` to return 0.0 for envs where ``obs[0]``
            # is not an x-position (e.g. FlatWorld, Zones).
            env_shaping = self._env_specific_shaping(obs, delta, self.current_u)
            total_reward += env_shaping
        else:
            x_pos = obs[0] if len(obs) > 0 else 0.0

        self.last_pos = float(x_pos)
        # Update prev_obs for next step
        self.prev_obs = obs
        self.step_idx += 1

        if self.current_u in self.automaton.T and rm_reward > 50:
            terminated = True

        info['rm_state'] = self.current_u
        info['prev_rm_state'] = u_prev
        info['rm_reward'] = total_reward

        # Populate counterfactual CRM experiences for all possible RM states
        # Format per experience: (obs, rm_state, action, reward, next_obs, next_rm_state, done)
        try:
            crm_exps = []
            # Build shaping helpers once
            delta = 0.0
            try:
                # delta computed above when shaping enabled; recompute otherwise
                cur_x = float(obs[0]) if len(obs) > 0 else 0.0
                prev_x = float(prev_obs[0]) if prev_obs is not None and len(prev_obs) > 0 else cur_x
                delta = cur_x - prev_x
            except Exception:
                delta = 0.0

            # Determine AP taken this step to apply same transition across counterfactual starts
            ap_taken = ap_id
            env_done_flag = bool(terminated or truncated)
            reachable_states = set()
            for u_can in range(self.num_states):
                try:
                    u_next_can = int(self.automaton.step(u_can, ap_taken))
                except Exception:
                    continue
                # Track states reachable by this transition for next-step pruning
                reachable_states.add(int(u_next_can))

                # Only add experience if starting state was reachable by previous step
                if self.valid_states is None or int(u_can) in self.valid_states:
                    # Base RM reward from mapping
                    rm_rew_can = float(self.reward_mapping.get((int(u_can), int(u_next_can)), 0.0))
                    total_can = rm_rew_can - self.step_penalty
                    if self.reward_shaping:
                        # Use env-specific shaping for counterfactual states
                        total_can += self._env_specific_shaping(obs, delta, u_next_can)
                    done_can = env_done_flag or (u_next_can in self.automaton.T and rm_rew_can > 50)
                    # Append experience
                    crm_exps.append((prev_obs, int(u_can), action, float(total_can), obs, int(u_next_can), bool(done_can)))

            # Update pruning set for next call
            self.valid_states = reachable_states
            if crm_exps:
                info['crm_experiences'] = crm_exps
        except Exception:
            # Best-effort; skip CRM if anything goes wrong
            pass

        return obs, total_reward, bool(terminated), bool(truncated), info

    # ── override-able env-specific shaping ──

    def _env_specific_shaping(self, obs, delta, rm_state):
        """Return additional reward shaping based on the current observation.

        The default implementation applies HalfCheetah-style x-position shaping.
        Subclasses should override this to return 0.0 (or their own potential)
        when ``obs[0]`` is not an x-position.
        """
        shaping = 0.0
        if rm_state == 0:
            shaping = min(0.1, self.shaping_scale * max(delta, 0.0))
        elif rm_state == 2:
            shaping = min(0.1, self.shaping_scale * max(-delta, 0.0))
            shaping -= self.continue_right_penalty_scale * max(delta, 0.0) * 2.0
        return shaping
