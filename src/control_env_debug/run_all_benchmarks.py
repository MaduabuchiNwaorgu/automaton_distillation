#!/usr/bin/env python3
"""
Run all comparison experiments across all methods and all environments.

Environments (teacher → student transfer):
    patrol              – HalfCheetah patrol  (teacher: a=5,b=-2 → student: a=8,b=-5)
    flatworld_patrol    – FlatWorld 2-colour  (teacher: default circles → student: shifted + cross walls)
    flatworld_sequence  – FlatWorld 3-colour  (teacher: default circles → student: shifted + corridors)
    zones_patrol        – Safety-Gym zones    (teacher: PointLtl1 → student: CarLtl1, different dynamics)
    zones_sequence      – Safety-Gym zones    (teacher: PointLtl2 → student: CarLtl2, different dynamics)

Methods (per environment):
    1) td3_base          – vanilla TD3 (no RM transfer, no CRM)
    2) td3_crm           – TD3 + Counterfactual Replay
    3) td3_static        – TD3 + static Q-distillation from teacher
    4) td3_dynamic       – TD3 + dynamic Q-distillation (anneal)
    5) td3_shaped        – TD3 + teacher reward shaping
    6) td3_cprep         – C-PREP: warm-start from teacher weights

Workflow per environment:
    Phase 1  – Train a teacher on the *source* env configuration
    Phase 2  – Run all student methods on the *target* (harder) env configuration
               using the teacher's Q-automaton for knowledge transfer

Usage:
    # Full run (all envs, 3 seeds, 100k steps)
    python -m src.control_env_debug.run_all_benchmarks

    # Quick smoke test
    python -m src.control_env_debug.run_all_benchmarks \\
        --total_steps 500 --teacher_steps 500 --n_envs 1 --seeds 0

    # Single environment
    python -m src.control_env_debug.run_all_benchmarks \\
        --envs flatworld_patrol --total_steps 50000

    # Skip teacher training (reuse existing)
    python -m src.control_env_debug.run_all_benchmarks \\
        --skip_teacher --envs patrol
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from .train_vectorized_td3 import run_training

# ────────────────────────── environment configs ──────────────────────────
# Each entry has a *teacher* config (source env) and a *student* config
# (target env).  The student environment is harder or different so that
# knowledge transfer is meaningful.

ENV_CONFIGS = {
    "patrol": dict(
        max_episode_steps=1000,
        teacher_steps_multiplier=1.0,
        reward_shaping=True,
        # Teacher: clean HalfCheetah, easy distance (a=5 right, b=-2 left)
        teacher_ap_config={"a_threshold": 5.0, "b_threshold": -2.0, "obstacles": False},
        # Student: HalfCheetah with obstacles + harder distance (a=8, b=-5) - tests both environment and task transfer
        student_ap_config={"a_threshold": 8.0, "b_threshold": -5.0, "obstacles": True},
    ),
    "flatworld_patrol": dict(
        max_episode_steps=500,
        teacher_steps_multiplier=1.0,
        reward_shaping=True,
        # Teacher: default circle layout, no walls
        teacher_ap_config={"circles": "default", "walls": "none"},
        # Student: shifted circles + cross obstacle
        student_ap_config={"circles": "shifted", "walls": "cross"},
    ),
    "flatworld_sequence": dict(
        max_episode_steps=500,
        teacher_steps_multiplier=1.5,
        reward_shaping=True,
        # Teacher: default layout, no walls
        teacher_ap_config={"circles": "default", "walls": "none"},
        # Student: shifted circles + corridor obstacles
        student_ap_config={"circles": "shifted", "walls": "corridors"},
    ),
    "zones_patrol": dict(
        max_episode_steps=1000,
        teacher_steps_multiplier=1.0,
        reward_shaping=True,
        # Teacher: Point robot
        teacher_ap_config={"safety_gym_id": "PointLtl1-v0"},
        # Student: Car robot (different dynamics)
        student_ap_config={"safety_gym_id": "CarLtl1-v0"},
    ),
    "zones_sequence": dict(
        max_episode_steps=1000,
        teacher_steps_multiplier=1.5,
        reward_shaping=True,
        # Teacher: Point robot
        teacher_ap_config={"safety_gym_id": "PointLtl2-v0"},
        # Student: Car robot
        student_ap_config={"safety_gym_id": "CarLtl2-v0"},
    ),
}

# ────────────────────────── method configs ──────────────────────────

def get_methods(teacher_run_name: str, distill_tau: float) -> List[dict]:
    """Return list of (method_name, run_training kwargs) for all methods.

    Methods
    -------
    td3_base      – scratch baseline (no RM distillation)
    td3_crm       – counterfactual RM experiences
    td3_static    – static Q-automaton distillation (fixed weight)
    td3_dynamic   – dynamic distillation with annealing weight decay
    td3_shaped    – teacher reward shaping
    td3_cprep     – C-PREP: init student weights from teacher then fine-tune
    """
    return [
        ("td3_base", dict(
            distill_mode="off",
            use_crm=False,
        )),
        ("td3_crm", dict(
            distill_mode="off",
            use_crm=True,
        )),
        ("td3_static", dict(
            distill_mode="static",
            teacher_run_name=teacher_run_name,
            w0=0.1,
            use_crm=False,
        )),
        ("td3_dynamic", dict(
            distill_mode="dynamic",
            teacher_run_name=teacher_run_name,
            w0=0.5,
            distill_tau=distill_tau,
            w_min=0.0,
            use_crm=False,
        )),
        ("td3_shaped", dict(
            distill_mode="off",
            shape_teacher=True,
            shape_scale=0.1,
            teacher_run_name=teacher_run_name,
            w0=0.05,
            use_crm=False,
        )),
        ("td3_cprep", dict(
            distill_mode="off",
            teacher_run_name=teacher_run_name,
            init_from_teacher=True,
            use_crm=False,
        )),
    ]


# ────────────────────────── runner ──────────────────────────

def run_all(
    envs: List[str],
    total_steps: int,
    teacher_steps: Optional[int],
    n_envs: int,
    seeds: List[int],
    prefix: str,
    normalize: bool,
    skip_teacher: bool,
    methods_to_run: Optional[List[str]],
    teacher_only: bool = False,
    device: str = None,
    train_freq: int = 1,
):
    """Run the full benchmark matrix."""
    _cwd = Path.cwd()
    log_root = _cwd / "logs"
    q_auto_dir = _cwd / "automaton_q"
    results_summary = {}

    total_runs = 0
    completed_runs = 0
    failed_runs = []

    # Count total planned runs
    for env_type in envs:
        n_teacher = 0 if skip_teacher else 1
        if teacher_only:
            total_runs += n_teacher
        else:
            methods = get_methods("dummy", total_steps * 0.2)
            if methods_to_run:
                methods = [(n, c) for n, c in methods if n in methods_to_run]
            n_student = len(methods) * len(seeds)
            total_runs += n_teacher + n_student

    print("=" * 70)
    print(f"  BENCHMARK MATRIX  (teacher -> student TRANSFER)")
    print(f"  Environments : {envs}")
    print(f"  Methods      : {[n for n, _ in get_methods('_', 0)] if not methods_to_run else methods_to_run}")
    print(f"  Seeds        : {seeds}")
    print(f"  Steps/run    : {total_steps}  (teacher: {teacher_steps or 'auto'})")
    print(f"  Envs/run     : {n_envs}")
    print(f"  Total runs   : {total_runs}")
    print(f"  Skip teacher : {skip_teacher}")
    for ename in envs:
        ecfg = ENV_CONFIGS[ename]
        print(f"  [{ename}] teacher: {ecfg['teacher_ap_config']}")
        print(f"  {' '*len(ename)}  student: {ecfg['student_ap_config']}")
    print("=" * 70)

    for env_type in envs:
        env_cfg = ENV_CONFIGS[env_type]
        max_ep = env_cfg["max_episode_steps"]
        t_steps = teacher_steps or int(total_steps * env_cfg["teacher_steps_multiplier"])
        distill_tau_val = float(total_steps) * 0.2  # anneal over ~20% of training

        teacher_run = f"{prefix}_{env_type}_teacher"

        teacher_apc = env_cfg["teacher_ap_config"]
        student_apc = env_cfg["student_ap_config"]

        # ── Phase 1: Teacher ──
        if not skip_teacher:
            print(f"\n{'='*70}")
            print(f"  PHASE 1: Training teacher for [{env_type}]")
            print(f"  Run name  : {teacher_run}")
            print(f"  Steps     : {t_steps}")
            print(f"  ap_config : {teacher_apc}")
            print(f"{'='*70}")
            t0 = time.time()
            try:
                run_training(
                    env_type=env_type,
                    reward_shaping=env_cfg["reward_shaping"],
                    max_episode_steps=max_ep,
                    ap_config=teacher_apc,
                    total_steps=t_steps,
                    n_envs=n_envs,
                    run_name=teacher_run,
                    normalize=normalize,
                    seed=seeds[0],          # teacher uses first seed
                    distill_mode="off",
                    use_crm=False,
                    device=device,
                    train_freq=train_freq,
                )
                dt = time.time() - t0
                print(f"  Teacher done in {dt:.1f}s")
                completed_runs += 1
            except Exception as e:
                dt = time.time() - t0
                print(f"  TEACHER FAILED after {dt:.1f}s: {e}")
                traceback.print_exc()
                failed_runs.append((teacher_run, str(e)))
                completed_runs += 1
                # If teacher fails, skip this env entirely
                print(f"  Skipping students for {env_type} (no teacher)")
                continue
        else:
            # Verify teacher Q-automaton exists
            q_path = q_auto_dir / f"{teacher_run}.json"
            if not q_path.exists():
                print(f"\n  WARNING: Teacher Q-automaton not found at {q_path}")
                print(f"  Students that need a teacher will fail for {env_type}")

        if teacher_only:
            print(f"  --teacher_only: skipping students for {env_type}")
            continue

        # ── Phase 2: Students ──
        all_methods = get_methods(teacher_run, distill_tau_val)
        # Create a fixed mapping of method names to indices (for seed offsetting)
        method_name_to_idx = {n: i for i, (n, _) in enumerate(all_methods)}
        
        methods = all_methods
        if methods_to_run:
            methods = [(n, c) for n, c in methods if n in methods_to_run]

        for method_name, method_cfg in methods:
            for seed in seeds:
                # IMPORTANT: Assign unique offset to each method to prevent deterministic noise collision
                # When run_training calls seed_everything(seed), global RNG gets reset deterministically.
                # Running multiple methods sequentially in same process means they all see identical noise.
                # Solution: Use (seed + method_idx*10000) to ensure different RNG state per method.
                # Use fixed mapping so method index is consistent even when single method is run.
                method_idx = method_name_to_idx.get(method_name, 0)
                method_seed = seed + method_idx * 10000
                
                run_name = f"{prefix}_{env_type}_{method_name}_s{seed}"
                print(f"\n{'─'*60}")
                print(f"  [{env_type}] {method_name} seed={seed} [actual_seed={method_seed}]  →  {run_name}")
                print(f"  student ap_config: {student_apc}")
                print(f"{'─'*60}")

                t0 = time.time()
                try:
                    run_training(
                        env_type=env_type,
                        reward_shaping=env_cfg["reward_shaping"],
                        max_episode_steps=max_ep,
                        ap_config=student_apc,
                        total_steps=total_steps,
                        n_envs=n_envs,
                        run_name=run_name,
                        normalize=normalize,
                        seed=method_seed,
                        device=device,
                        train_freq=train_freq,
                        **method_cfg,
                    )
                    dt = time.time() - t0
                    print(f"  Completed in {dt:.1f}s")

                    # Record summary
                    import numpy as np
                    ret_path = log_root / run_name / "episode_returns.npy"
                    if ret_path.exists():
                        rets = np.load(ret_path)
                        last_n = min(50, len(rets))
                        mean_r = float(np.mean(rets[-last_n:])) if last_n > 0 else 0.0
                        results_summary[run_name] = {
                            "env": env_type, "method": method_name,
                            "seed": seed, "mean_last50": round(mean_r, 3),
                            "n_episodes": len(rets), "time_s": round(dt, 1),
                        }
                except Exception as e:
                    dt = time.time() - t0
                    print(f"  FAILED after {dt:.1f}s: {e}")
                    traceback.print_exc()
                    failed_runs.append((run_name, str(e)))
                completed_runs += 1

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Completed: {completed_runs}/{total_runs}")
    if failed_runs:
        print(f"  Failed ({len(failed_runs)}):")
        for name, err in failed_runs:
            print(f"    - {name}: {err[:80]}")

    if results_summary:
        print(f"\n  {'Run':<50} {'Mean(last50)':<14} {'Episodes'}")
        print(f"  {'─'*50} {'─'*14} {'─'*10}")
        for run_name, info in sorted(results_summary.items()):
            print(f"  {run_name:<50} {info['mean_last50']:<14.3f} {info['n_episodes']}")

    # Save results JSON
    summary_path = log_root / f"{prefix}_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n  Results saved to {summary_path}")
    except Exception:
        pass

    print("=" * 70)
    return results_summary


# ────────────────────────── CLI ──────────────────────────

ALL_ENVS = list(ENV_CONFIGS.keys())
ALL_METHODS = [n for n, _ in get_methods("_", 0)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all comparison experiments across methods and environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available environments: {', '.join(ALL_ENVS)}
Available methods:      {', '.join(ALL_METHODS)}

Examples:
  # Full benchmark (all envs, 3 seeds)
  python -m src.control_env_debug.run_all_benchmarks

  # Quick smoke test
  python -m src.control_env_debug.run_all_benchmarks \\
      --total_steps 500 --teacher_steps 500 --n_envs 1 --seeds 0

  # Single env, specific methods
  python -m src.control_env_debug.run_all_benchmarks \\
      --envs flatworld_patrol --methods td3_base td3_static td3_dynamic
""",
    )
    parser.add_argument(
        "--envs", type=str, nargs="+", default=ALL_ENVS,
        help=f"Environments to benchmark (default: all). Choices: {ALL_ENVS}",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help=f"Methods to run (default: all). Choices: {ALL_METHODS}",
    )
    parser.add_argument("--total_steps", type=int, default=100_000,
                        help="Training steps per student run (default: 100000)")
    parser.add_argument("--teacher_steps", type=int, default=None,
                        help="Training steps for teacher (default: auto, based on env)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of vectorized envs (default: 4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds to run (default: 0 1 2)")
    parser.add_argument("--prefix", type=str, default="bench",
                        help="Prefix for run names (default: bench)")
    parser.add_argument("--normalize", type=lambda x: str(x).lower() in ["1", "true", "yes"],
                        default=True, help="Enable observation normalization (default: true)")
    parser.add_argument("--skip_teacher", action="store_true",
                        help="Skip teacher training (reuse existing Q-automata)")
    parser.add_argument("--teacher_only", action="store_true",
                        help="Train teacher only (no student methods). For parallel workflows.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device: cpu, cuda, cuda:0, cuda:1, etc.")
    parser.add_argument("--train_freq", type=int, default=1,
                        help="Train every N env steps (default: 1)")

    args = parser.parse_args()

    # Validate env names
    for e in args.envs:
        if e not in ENV_CONFIGS:
            print(f"ERROR: Unknown env '{e}'. Choose from: {ALL_ENVS}")
            sys.exit(1)
    # Validate method names
    if args.methods:
        for m in args.methods:
            if m not in ALL_METHODS:
                print(f"ERROR: Unknown method '{m}'. Choose from: {ALL_METHODS}")
                sys.exit(1)

    run_all(
        envs=args.envs,
        total_steps=args.total_steps,
        teacher_steps=args.teacher_steps,
        n_envs=args.n_envs,
        seeds=args.seeds,
        prefix=args.prefix,
        normalize=args.normalize,
        skip_teacher=args.skip_teacher,
        methods_to_run=args.methods,
        teacher_only=args.teacher_only,
        device=args.device,
        train_freq=args.train_freq,
    )
