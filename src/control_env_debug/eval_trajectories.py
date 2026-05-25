#!/usr/bin/env python3
"""Evaluate a trained TD3 agent and save deterministic trajectories.

Loads the final model weights + VecNormalize stats from a training run,
rolls out deterministic episodes (no action noise), and saves trajectory
data (positions + RM states) in the same format used by plot_trajectories.

Usage
-----
# Evaluate a single run (saves to <run_dir>/eval_trajectories/):
python -m src.control_env_debug.eval_trajectories \
    --run logs/bench_flatworld_sequence_td3_base_s0 \
    --env flatworld_sequence --n_episodes 8

# Evaluate all runs for an environment and generate comparison plots:
python -m src.control_env_debug.eval_trajectories \
    --env flatworld_sequence --logs_root logs --prefix bench \
    --n_episodes 8 --plot
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
#  Imports from the project
# ---------------------------------------------------------------------------
from .train_env import make_env as make_single_env
from .train_agent import make_agent as _make_agent


# ---------------------------------------------------------------------------
#  Core evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: str,
    env_type: str,
    n_episodes: int = 8,
    ap_config: Optional[Dict] = None,
    max_episode_steps: int = 1000,
    reward_shaping: bool = True,
    deterministic: bool = True,
) -> Optional[str]:
    """Load model from *run_dir*, run eval episodes, save trajectories.

    Returns the path to the eval trajectory directory, or None on failure.
    """
    model_path = os.path.join(run_dir, "td3_model")
    if not os.path.isfile(model_path + "_actor.pth"):
        print(f"  [skip] No model found in {run_dir}")
        return None

    # ── Build env + agent ──
    env = make_single_env(
        env_type=env_type,
        reward_shaping=reward_shaping,
        max_episode_steps=max_episode_steps,
        ap_config=ap_config,
    )
    agent = _make_agent(env)
    agent.load(model_path)
    agent.actor.eval()

    # ── Load VecNormalize stats (if available) ──
    norm_stats = None
    norm_path_npz = os.path.join(run_dir, "vecnormalize.pkl.npz")
    norm_path_pkl = os.path.join(run_dir, "vecnormalize.pkl")
    for p in (norm_path_npz, norm_path_pkl):
        if os.path.isfile(p):
            try:
                d = np.load(p)
                norm_stats = {
                    "mean": d["mean"].astype(np.float32),
                    "var": d["var"].astype(np.float64),
                    "clip_obs": float(d["clip_obs"]),
                }
                break
            except Exception as e:
                print(f"  [warn] Failed to load norm stats from {p}: {e}")

    def normalise_obs(obs: np.ndarray) -> np.ndarray:
        if norm_stats is None:
            return obs.astype(np.float32)
        std = np.sqrt(norm_stats["var"]).astype(np.float32)
        x = (obs.astype(np.float32) - norm_stats["mean"]) / (std + 1e-8)
        return np.clip(x, -norm_stats["clip_obs"], norm_stats["clip_obs"])

    # ── Output directory ──
    eval_dir = os.path.join(run_dir, "eval_trajectories")
    os.makedirs(eval_dir, exist_ok=True)

    # ── Save env metadata ──
    env_meta = {"env_type": env_type}
    try:
        inner = getattr(env, "env", env)
        if hasattr(inner, "env") and hasattr(inner.env, "circles"):
            fw = inner.env
            env_meta["circles"] = [
                {"center": c.center.tolist(), "radius": c.radius, "color": c.color}
                for c in fw.circles
            ]
            env_meta["walls"] = [
                {"x_min": w.x_min, "y_min": w.y_min, "x_max": w.x_max, "y_max": w.y_max}
                for w in fw.walls
            ] if hasattr(fw, "walls") else []
        if hasattr(inner, "a_threshold"):
            env_meta["a_threshold"] = inner.a_threshold
            env_meta["b_threshold"] = inner.b_threshold
        # Zones: try to get zone info
        if "zones" in env_type:
            try:
                base = inner.env if hasattr(inner, "env") else inner
                if hasattr(base, "task") and hasattr(base.task, "goal_pos"):
                    pass  # Safety-Gym doesn't expose zones nicely
            except Exception:
                pass
    except Exception:
        pass
    with open(os.path.join(eval_dir, "env_meta.json"), "w") as f:
        json.dump(env_meta, f, indent=2)

    # ── Roll out episodes ──
    accept_states = env.automaton.T if hasattr(env, "automaton") else {1}

    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        rm_state = 0
        positions = []
        rm_states = []
        ep_ret = 0.0

        for t in range(max_episode_steps):
            # Record position
            pos = info.get("agent_pos", None)
            if pos is not None:
                positions.append(np.asarray(pos, dtype=np.float64).copy())
            rm_states.append(int(rm_state))

            # Select action (deterministic = no noise)
            obs_norm = normalise_obs(obs)
            action = agent.get_action(obs_norm, rm_state, add_noise=not deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            rm_state = int(info.get("rm_state", rm_state))
            ep_ret += reward

            if terminated or truncated:
                break

        # Record final step
        pos = info.get("agent_pos", None)
        if pos is not None:
            positions.append(np.asarray(pos, dtype=np.float64).copy())
        rm_states.append(int(rm_state))

        reached = rm_state in accept_states
        status = "GOAL" if reached else "timeout"
        print(f"  Ep {ep_idx}: {status} | {len(positions)} steps | return={ep_ret:.2f} | final_rm={rm_state}")

        # Save
        if positions:
            np.save(os.path.join(eval_dir, f"ep{ep_idx:04d}_pos.npy"), np.array(positions))
            np.save(os.path.join(eval_dir, f"ep{ep_idx:04d}_rm.npy"),
                    np.array(rm_states, dtype=np.int32))

    env.close()
    return eval_dir


# ---------------------------------------------------------------------------
#  Run-discovery (reuse logic from plot_trajectories)
# ---------------------------------------------------------------------------

_RUN_RE = re.compile(
    r"^(?P<prefix>\w+?)_(?P<env>patrol|flatworld_patrol|flatworld_sequence|zones_patrol|zones_sequence)"
    r"_(?P<method>td3_\w+?)_s(?P<seed>\d+)$"
)


def discover_runs(logs_root: str, env_name: str, prefix: str = "bench") -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(logs_root)):
        full = os.path.join(logs_root, name)
        if not os.path.isdir(full):
            continue
        m = _RUN_RE.match(name)
        if m and m.group("prefix") == prefix and m.group("env") == env_name:
            method = m.group("method")
            result.setdefault(method, []).append(full)
    return result


# ---------------------------------------------------------------------------
#  Environment config lookup (mirrors run_all_benchmarks.py)
# ---------------------------------------------------------------------------

ENV_DEFAULTS = {
    "patrol":              {"max_episode_steps": 1000},
    "flatworld_patrol":    {"max_episode_steps": 500},
    "flatworld_sequence":  {"max_episode_steps": 500},
    "zones_patrol":        {"max_episode_steps": 1000},
    "zones_sequence":      {"max_episode_steps": 1000},
}

# ap_config per method (typically student configs for distilled methods)
# For evaluation we use the *student* config (what the model was trained on)
def _get_ap_config(env_name: str, method: str) -> Optional[Dict]:
    """Return ap_config for the given env×method.

    Mirrors the student configs from run_all_benchmarks.py ENV_CONFIGS.
    """
    # FlatWorld student: shifted circles, corridor walls
    if env_name == "flatworld_patrol":
        student_cfg = {"circles": "shifted", "walls": "corridors"}
        if method in ("td3_static", "td3_dynamic", "td3_shaped", "td3_cprep"):
            return student_cfg
        if method in ("td3_base", "td3_crm"):
            return student_cfg  # student env is still the student layout
    if env_name == "flatworld_sequence":
        student_cfg = {"circles": "shifted", "walls": "corridors"}
        return student_cfg  # all methods use student env
    # Zones: student uses CarLtl env
    if env_name == "zones_patrol":
        student_cfg = {"safety_gym_id": "CarLtl1-v0"}
        if method in ("td3_static", "td3_dynamic", "td3_shaped", "td3_cprep"):
            return student_cfg
        if method in ("td3_base", "td3_crm"):
            return student_cfg
    if env_name == "zones_sequence":
        student_cfg = {"safety_gym_id": "CarLtl2-v0"}
        return student_cfg
    # HalfCheetah patrol: teacher/student share same thresholds
    if env_name == "patrol":
        return {"a_threshold": 5.0, "b_threshold": -2.0}
    return None


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, default=None, help="Single run directory to evaluate")
    ap.add_argument("--env", type=str, required=True,
                    help="Environment type (e.g. flatworld_sequence)")
    ap.add_argument("--logs_root", default="logs")
    ap.add_argument("--prefix", default="bench")
    ap.add_argument("--n_episodes", type=int, default=8)
    ap.add_argument("--max_episode_steps", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="Generate trajectory plots after eval")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    defaults = ENV_DEFAULTS.get(args.env, {})
    max_steps = args.max_episode_steps or defaults.get("max_episode_steps", 1000)

    if args.run:
        # ── Single run ──
        # Infer method from directory name
        basename = os.path.basename(args.run)
        m = _RUN_RE.match(basename)
        method = m.group("method") if m else "td3_base"
        ap_config = _get_ap_config(args.env, method)

        print(f"Evaluating {basename} ({args.n_episodes} episodes, deterministic)")
        eval_dir = evaluate_run(
            args.run, args.env, n_episodes=args.n_episodes,
            ap_config=ap_config, max_episode_steps=max_steps,
        )
        if eval_dir and args.plot:
            _plot_eval(args.run, args.env, args.out_dir, args.dpi)
    else:
        # ── All runs for this env ──
        methods = discover_runs(args.logs_root, args.env, prefix=args.prefix)
        if not methods:
            print(f"No runs found for env '{args.env}' in {args.logs_root}")
            return

        all_dirs = []
        for method, dirs in sorted(methods.items()):
            for rd in dirs:
                basename = os.path.basename(rd)
                ap_config = _get_ap_config(args.env, method)
                print(f"\n{'='*60}")
                print(f"Evaluating {basename} ({args.n_episodes} episodes)")
                print(f"{'='*60}")
                eval_dir = evaluate_run(
                    rd, args.env, n_episodes=args.n_episodes,
                    ap_config=ap_config, max_episode_steps=max_steps,
                )
                if eval_dir:
                    all_dirs.append(rd)

        if args.plot and all_dirs:
            _plot_all_eval(all_dirs, args.env, methods, args.out_dir or os.path.join(args.logs_root, "plots"), args.dpi)


def _plot_eval(run_dir: str, env_type: str, out_dir: Optional[str], dpi: int):
    """Generate trajectory plots from eval_trajectories/ instead of trajectories/."""
    from .plot_trajectories import plot_run as _plot_run
    # Temporarily swap trajectory dir name
    eval_dir = os.path.join(run_dir, "eval_trajectories")
    traj_dir = os.path.join(run_dir, "trajectories")
    eval_bak = traj_dir + ".train_bak"

    # Rename: trajectories → trajectories.train_bak, eval_trajectories → trajectories
    if os.path.isdir(traj_dir):
        os.rename(traj_dir, eval_bak)
    os.rename(eval_dir, traj_dir)
    try:
        od = out_dir or os.path.join(os.path.dirname(run_dir), "plots")
        os.makedirs(od, exist_ok=True)
        out = _plot_run(
            run_dir,
            label=f"{os.path.basename(run_dir)} (eval)",
            max_eps=20,
            out=os.path.join(od, f"eval_traj_{os.path.basename(run_dir)}.png"),
            dpi=dpi,
        )
        if out:
            print(f"  Saved {out}")
    finally:
        # Restore original names
        os.rename(traj_dir, eval_dir)
        if os.path.isdir(eval_bak):
            os.rename(eval_bak, traj_dir)


def _plot_all_eval(run_dirs: List[str], env_type: str, methods: Dict[str, List[str]],
                   out_dir: str, dpi: int):
    """Plot eval trajectories for all runs and generate comparison."""
    from .plot_trajectories import (
        plot_run as _plot_run,
        plot_comparison as _plot_comparison,
        METHOD_LABELS,
    )
    os.makedirs(out_dir, exist_ok=True)

    for rd in run_dirs:
        _plot_eval(rd, env_type, out_dir, dpi)

    # Comparison plot: pick first seed of each method
    comp_dirs = []
    comp_labels = []
    for method, dirs in sorted(methods.items()):
        if not dirs:
            continue
        rd = dirs[0]
        eval_dir = os.path.join(rd, "eval_trajectories")
        traj_dir = os.path.join(rd, "trajectories")
        if not os.path.isdir(eval_dir):
            continue
        comp_dirs.append(rd)
        comp_labels.append(METHOD_LABELS.get(method, method))

    if len(comp_dirs) > 1:
        # Temporarily swap dirs for comparison
        backups = {}
        for rd in comp_dirs:
            traj_dir = os.path.join(rd, "trajectories")
            eval_dir = os.path.join(rd, "eval_trajectories")
            bak = traj_dir + ".train_bak"
            if os.path.isdir(traj_dir):
                os.rename(traj_dir, bak)
                backups[rd] = bak
            os.rename(eval_dir, traj_dir)
        try:
            out = _plot_comparison(
                comp_dirs, comp_labels,
                out=os.path.join(out_dir, f"eval_traj_{env_type}_compare.png"),
                dpi=dpi,
            )
            if out:
                print(f"  Saved {out}")
        finally:
            for rd in comp_dirs:
                traj_dir = os.path.join(rd, "trajectories")
                eval_dir = os.path.join(rd, "eval_trajectories")
                os.rename(traj_dir, eval_dir)
                bak = backups.get(rd)
                if bak and os.path.isdir(bak):
                    os.rename(bak, traj_dir)


if __name__ == "__main__":
    main()
