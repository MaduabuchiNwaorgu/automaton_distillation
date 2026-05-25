#!/usr/bin/env python3
"""Train TD3 agent on our Automaton-wrapped env using our VecEnv helper."""
import os
from pathlib import Path
import argparse
import numpy as np
import torch
import random

# Project root (3 levels up from this file: src/control_env_debug/ → src/ → TMLR/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Outputs go into the current working directory (so you can run from any folder)
_CWD = Path.cwd()
_AUTOMATON_Q_DIR = _CWD / 'automaton_q'
_LOGS_DIR = _CWD / 'logs'

from .vec_env import make_vec_env
from .train_env import make_env as make_single_env
from .train_agent import make_agent as _make_agent


def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    if seed is None:
        return
    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    except Exception:
        pass
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # Make CuDNN deterministic
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        # Prefer deterministic algorithms where supported
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        # Optional: stricter cuBLAS determinism (only if needed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass


def _compute_static_q_from_v(adj_mat, v_arr, disc, max_iter=1000, tol=1e-6):
    ns = adj_mat.shape[0]
    na = adj_mat.shape[1]
    Q = np.zeros((ns, na), dtype=np.float32)
    for _ in range(max_iter):
        Q_old = Q
        next_states = adj_mat
        R = v_arr[:, None] - disc * v_arr[next_states]
        max_next = Q_old[next_states].max(axis=2)
        Q = R + disc * max_next
        if np.max(np.abs(Q - Q_old)) < tol:
            break
    return Q


def run_training(env_type="patrol",
                 reward_shaping=True,
                 max_episode_steps=1000,
                 ap_config=None,
                 total_steps=10_000,
                 n_envs=1,
                 run_name="td3_vec",
                 normalize=True,
                 seed: int = None,
                 # Transfer schedule selector (used for shaping schedule)
                 distill_mode: str = 'off',
                 teacher_run_name: str = None,
                 w0: float = 0.1,
                 distill_tau: float = None,
                 w_min: float = 0.0,
                 # Dynamic automaton target mixing (teacher Qavg + beta)
                 mix_target: bool = False,
                 rho: float = 0.999,
                 min_source_q_count: int = 1,
                 # Teacher reward shaping
                 shape_teacher: bool = False,
                 shape_scale: float = 1.0,
                 shape_min_count: int = 1,
                 # C-PREP style weight transfer: init student from teacher model
                 init_from_teacher: bool = False,
                 # Performance tuning
                 device: str = None,
                 train_freq: int = 1,
                 # Agent hyperparameters (overrides)
                 actor_lr: float = None,
                 critic_lr: float = None,
                 batch_size: int = None,
                 gamma: float = None,
                 tau: float = None,
                 buffer_size: int = None,
                 policy_noise: float = None,
                 noise_clip: float = None,
                 policy_delay: int = None,
                 use_crm: bool = None,
                 no_rm: bool = False):
    # Ensure all RNGs are seeded deterministically before any object creation
    seed_everything(seed)

    # Create a single env just to build the agent (for shapes and automaton)
    single_env = make_single_env(env_type=env_type, reward_shaping=reward_shaping, max_episode_steps=max_episode_steps, ap_config=ap_config)
    # Build agent with optional overrides if supported
    # Avoid passing None values that can cause TypeError inside the factory
    overrides = dict(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        use_crm=use_crm,
        no_rm=no_rm,
        device=device,
    )
    overrides = {k: v for k, v in overrides.items() if v is not None}
    try:
        agent = _make_agent(single_env, **overrides)
    except TypeError:
        agent = _make_agent(single_env)

    # Startup diagnostics banner
    try:
        aut = getattr(single_env, 'automaton', getattr(single_env.unwrapped, 'automaton', None))
        num_rm_states = getattr(aut, 'num_states', None)
    except Exception:
        aut = None
        num_rm_states = None
    print("[init] run=", run_name)
    print(f"[init] env_type={env_type} n_envs={n_envs} max_ep_steps={max_episode_steps} normalize={normalize}")
    # Show AP thresholds only for HalfCheetah envs that use them
    if env_type in ("patrol", "obstacles", "obstacles_strict"):
        print(f"[init] thresholds: a_threshold={ap_config['a_threshold'] if isinstance(ap_config, dict) and 'a_threshold' in ap_config else 'N/A'}, b_threshold={ap_config['b_threshold'] if isinstance(ap_config, dict) and 'b_threshold' in ap_config else 'N/A'}")
    else:
        prop_names = getattr(single_env, 'prop_names', [])
        print(f"[init] propositions: {prop_names}")
    print(f"[init] RM states (env)={num_rm_states} | agent.num_rm_states={getattr(agent, 'num_rm_states', None)} | no_rm={no_rm} | use_crm={getattr(agent, 'use_crm', False)}")

    # ── C-PREP style weight transfer ──
    if init_from_teacher and teacher_run_name:
        teacher_model_path = str(_LOGS_DIR / teacher_run_name / 'td3_model')
        if os.path.exists(teacher_model_path + "_actor.pth"):
            print(f"[C-PREP] Initialising student from teacher weights: {teacher_run_name}")
            n_transferred = agent.load_compatible(teacher_model_path)
            print(f"[C-PREP] Weight transfer complete ({n_transferred} params loaded)")
        else:
            print(f"[C-PREP] WARNING: Teacher model not found at {teacher_model_path}_*.pth — training from scratch")

    # Cache adjacency for AP inference in diagnostics (if available)
    adj_for_log = None
    try:
        if aut is not None and hasattr(aut, 'adj_matrix'):
            adj_for_log = np.array(aut.adj_matrix)
    except Exception:
        adj_for_log = None

    # Initialize teacher context (used by mixing and/or shaping)
    teacher_ctx = None

    # If distillation (static/dynamic target mixing) requested, load teacher automaton
    if distill_mode in ('static', 'dynamic') and teacher_run_name:
        try:
            import json, traceback
            q_path = _AUTOMATON_Q_DIR / f"{teacher_run_name}.json"
            print(f"[mix] Loading teacher automaton from {q_path}")
            if not q_path.exists():
                raise FileNotFoundError(f"Teacher file not found: {q_path}")
            with open(q_path, 'r') as f:
                data = json.load(f)
            aut_total_q = np.array(data['aut_total_q'], dtype=np.float32)
            aut_num_q = np.array(data['aut_num_q'], dtype=np.int64)
            aut_total_v = np.array(data.get('aut_total_v', []), dtype=np.float32) if 'aut_total_v' in data else None
            aut_num_v = np.array(data.get('aut_num_v', []), dtype=np.int64) if 'aut_num_v' in data else None
            aut = getattr(single_env, 'automaton', getattr(single_env.unwrapped, 'automaton', None))
            adj = np.array(aut.adj_matrix)
            # Sanity checks on shapes
            if aut_total_q.ndim != 2 or aut_num_q.ndim != 2:
                raise ValueError(f"Teacher Q arrays must be 2D, got shapes {aut_total_q.shape} and {aut_num_q.shape}")
            if adj.shape != aut_total_q.shape:
                print(f"[mix][warn] Adjacency shape {adj.shape} != teacher Q shape {aut_total_q.shape}. Mixing may be invalid (AP mismatch). Skipping enable.")
            else:
                _static_w = float(w0) if distill_mode == 'static' else None
                _initial_w = float(w0) if distill_mode == 'dynamic' else 1.0
                agent.enable_target_mixing(
                    source_q_total=aut_total_q,
                    source_q_count=aut_num_q,
                    adj=adj,
                    rho=rho,
                    min_source_q_count=int(min_source_q_count),
                    static_weight=_static_w,
                    initial_weight=_initial_w,
                    beta_min=float(w_min) if distill_mode == 'dynamic' else 0.0,
                )
                if _static_w is not None:
                    print(f"Static target mixing enabled with teacher='{teacher_run_name}', β={_static_w}")
                else:
                    print(f"Dynamic target mixing enabled with teacher='{teacher_run_name}', rho={rho}, w_min={w_min}")
            # Ensure teacher_ctx exists for AP inference if shaping also used
            if teacher_ctx is None:
                # For shaping we still need q_hat; compute from totals/counts
                q_hat = np.divide(aut_total_q, np.maximum(1.0, aut_num_q))
                # Attempt to load V as well if present
                v_hat = None
                if aut_total_v is not None and aut_num_v is not None and aut_total_v.size > 0:
                    v_hat = np.divide(aut_total_v, np.maximum(1.0, aut_num_v))
                teacher_ctx = dict(q_hat=q_hat, v_hat=v_hat, adj=adj,
                                   q_count=aut_num_q, v_count=aut_num_v)
                # If static transfer requested and V is available, compute automaton Q via Eq.(10)
                if distill_mode == 'static' and v_hat is not None:
                    try:
                        gamma_agent = float(getattr(agent, 'gamma', 0.99))
                        Q_static = _compute_static_q_from_v(adj, v_hat, gamma_agent)
                        teacher_ctx['q_hat'] = Q_static
                        print("Static transfer: replaced teacher Qavg with Q̂ computed from V via Eq.(10)")
                    except Exception as e:
                        print(f"Warning: static Q-from-V computation failed: {e}")
        except Exception as e:
            print(f"Warning: failed to enable dynamic target mixing: {e}\n{traceback.format_exc()}")
    elif distill_mode in ('static', 'dynamic') and not teacher_run_name:
        print(f"[distill] Requested distill_mode='{distill_mode}' but --teacher_run_name was not provided; skipping.")

    # If reward shaping is requested without distillation/mixing, load teacher automaton
    if shape_teacher and teacher_run_name and teacher_ctx is None:
        try:
            q_hat, v_hat, q_count, v_count = _load_teacher_q_automaton(teacher_run_name)
            aut = getattr(single_env, 'automaton', getattr(single_env.unwrapped, 'automaton', None))
            adj = np.array(aut.adj_matrix)
            teacher_ctx = dict(q_hat=q_hat, v_hat=v_hat, adj=adj, q_count=q_count, v_count=v_count)
            # If static transfer requested and V is available, compute automaton Q via Eq.(10)
            if distill_mode == 'static' and v_hat is not None:
                try:
                    gamma_agent = float(getattr(agent, 'gamma', 0.99))
                    Q_static = _compute_static_q_from_v(adj, v_hat, gamma_agent)
                    teacher_ctx['q_hat'] = Q_static
                    print("Static transfer: replaced teacher Qavg with Q̂ computed from V via Eq.(10)")
                except Exception as e:
                    print(f"Warning: static Q-from-V computation failed: {e}")
            print(f"Teacher reward shaping enabled with teacher='{teacher_run_name}'")
        except Exception as e:
            print(f"Warning: failed to load teacher for reward shaping: {e}")

    # Vectorized env for rollout
    vec = make_vec_env(
        env_type=env_type,
        n_envs=n_envs,
        # Force Dummy to avoid cross-process space/type issues with VecNormalize in this setup
        force_dummy=True,
        run_name=run_name,
        reward_shaping=reward_shaping,
        max_episode_steps=max_episode_steps,
        ap_config=ap_config,
        seed=seed,
        normalize=normalize,
        norm_kwargs=dict(norm_obs=True, norm_reward=False, clip_obs=10.0)
    )
    obs = vec.reset()

    # Track per-env rm_state
    rm_states = np.zeros(n_envs, dtype=np.int32)

    # Shaping weight schedule (reuse distill params for simplicity)
    import math
    def _shape_weight(step: int) -> float:
        if not shape_teacher or teacher_ctx is None:
            return 0.0
        if distill_mode == 'dynamic' and distill_tau is not None:
            return max(w_min, w0 * math.exp(-float(step) / max(1e-9, distill_tau)))
        if distill_mode in ('static', 'off'):
            return w0
        return 0.0

    # Output dirs
    out_dir = str(_LOGS_DIR / run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'td3_model')

    ep_returns = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)
    completed_returns = []
    completed_lengths = []
    # Episode diagnostics CSV — keep file handles open for the entire run
    eps_csv_path = os.path.join(out_dir, 'episodes.csv')
    _eps_csv_f = open(eps_csv_path, 'w', buffering=1)  # line-buffered for live monitoring
    _eps_csv_f.write('step,env,ep_length,ep_return,truncated\n')
    eps_ex_csv_path = os.path.join(out_dir, 'episodes_ex.csv')
    _eps_ex_csv_f = open(eps_ex_csv_path, 'w', buffering=1)
    _eps_ex_csv_f.write('step,env,ep_length,ep_return,truncated,final_rm_state,last_ap\n')
    crm_csv_path = os.path.join(out_dir, 'crm_stats.csv')
    _crm_csv_f = open(crm_csv_path, 'w', buffering=1)
    _crm_csv_f.write('step,crm_last_synth,crm_total_real,crm_total_synth,ratio_synth_per_real\n')
    mix_csv_path = os.path.join(out_dir, 'mix_stats.csv')
    _mix_csv_f = open(mix_csv_path, 'w', buffering=1)
    _mix_csv_f.write('step,valid_frac,beta_mean,tq_mean,td_target_mean\n')
    shaping_csv_path = os.path.join(out_dir, 'shaping_stats.csv')
    _shaping_csv_f = open(shaping_csv_path, 'w', buffering=1)
    _shaping_csv_f.write('step,applied_count,added_sum\n')
    shaping_added_sum = 0.0
    shaping_applied = 0
    shaping_logged_banner = False

    # ── Trajectory logging (positions + RM states per step) ──
    # We collect at most MAX_TRAJ_EPISODES full episodes for visualisation.
    MAX_TRAJ_EPISODES = 20
    traj_dir = os.path.join(out_dir, 'trajectories')
    os.makedirs(traj_dir, exist_ok=True)
    # Per-env accumulators: list of (pos, rm_state) per step
    _traj_pos = [[] for _ in range(n_envs)]
    _traj_rm = [[] for _ in range(n_envs)]
    _traj_count = 0  # how many completed episodes saved so far

    # Save region metadata for the trajectory visualiser
    import json as _json
    _env_meta = {'env_type': env_type}
    try:
        inner = getattr(single_env, 'env', single_env)
        # FlatWorld: circles + walls
        if hasattr(inner, 'env') and hasattr(inner.env, 'circles'):
            fw = inner.env
            _env_meta['circles'] = [
                {'center': c.center.tolist(), 'radius': c.radius, 'color': c.color}
                for c in fw.circles
            ]
            _env_meta['walls'] = [
                {'x_min': w.x_min, 'y_min': w.y_min, 'x_max': w.x_max, 'y_max': w.y_max}
                for w in fw.walls
            ] if hasattr(fw, 'walls') else []
        # HalfCheetah: thresholds
        if hasattr(inner, 'a_threshold'):
            _env_meta['a_threshold'] = inner.a_threshold
            _env_meta['b_threshold'] = inner.b_threshold
    except Exception:
        pass
    with open(os.path.join(traj_dir, 'env_meta.json'), 'w') as _f:
        _json.dump(_env_meta, _f, indent=2)

    for step in range(total_steps):
        # Select actions — single batched forward pass
        actions = agent.get_actions_batch(obs, rm_states, add_noise=True)

        next_obs, rewards, dones, infos = vec.step(actions)

        for i in range(n_envs):
            info = infos[i]
            next_rm_state = int(info.get('rm_state', rm_states[i]))
            prev_rm_state = int(info.get('prev_rm_state', rm_states[i]))
            # Teacher reward shaping (optional)
            base_r = float(rewards[i])
            r = base_r
            if teacher_ctx is not None and shape_teacher:
                w = _shape_weight(step)
                if w > 0.0:
                    if not shaping_logged_banner:
                        print(f"Teacher shaping active: mode={distill_mode}, w0={w0}, distill_tau={distill_tau}, w_min={w_min}, scale={shape_scale}, min_count={shape_min_count}")
                        shaping_logged_banner = True
                    try:
                        # Prefer potential-based shaping: V(u) - gamma * V(u')
                        if 'v_hat' in teacher_ctx and teacher_ctx['v_hat'] is not None:
                            # gate on counts if available
                            vc = teacher_ctx.get('v_count', None)
                            if vc is None or (int(vc[prev_rm_state]) >= int(shape_min_count) and int(vc[next_rm_state]) >= int(shape_min_count)):
                                vhat = teacher_ctx['v_hat']
                                r += w * shape_scale * (float(vhat[prev_rm_state]) - float(getattr(agent, 'gamma', 0.99)) * float(vhat[next_rm_state]))
                        else:
                            # Fallback: Q-hat shaping on (u, sigma)
                            row = teacher_ctx['adj'][prev_rm_state]
                            ap_idx = int(np.where(row == next_rm_state)[0][0]) if np.any(row == next_rm_state) else -1
                            if ap_idx >= 0:
                                qc = teacher_ctx.get('q_count', None)
                                if qc is None or int(qc[prev_rm_state, ap_idx]) >= int(shape_min_count):
                                    qhat = float(teacher_ctx['q_hat'][prev_rm_state, ap_idx])
                                    r += w * shape_scale * qhat
                        # Track shaping contribution
                        delta = r - base_r
                        if delta != 0.0:
                            shaping_added_sum += float(delta)
                            shaping_applied += 1
                    except Exception:
                        pass
            agent.store_experience(obs[i], int(prev_rm_state), actions[i], r, next_obs[i], next_rm_state, bool(dones[i]), info)

            # ── Trajectory recording ──
            if _traj_count < MAX_TRAJ_EPISODES:
                pos = info.get('agent_pos', None)
                if pos is not None:
                    _traj_pos[i].append(np.asarray(pos, dtype=np.float64).copy())
                    _traj_rm[i].append(int(next_rm_state))

            ep_returns[i] += r
            ep_lengths[i] += 1
            if dones[i]:
                # ── Save completed trajectory ──
                if _traj_count < MAX_TRAJ_EPISODES and len(_traj_pos[i]) > 0:
                    ep_idx = _traj_count
                    np.save(os.path.join(traj_dir, f'ep{ep_idx:04d}_pos.npy'),
                            np.array(_traj_pos[i]))
                    np.save(os.path.join(traj_dir, f'ep{ep_idx:04d}_rm.npy'),
                            np.array(_traj_rm[i], dtype=np.int32))
                    _traj_count += 1
                _traj_pos[i].clear()
                _traj_rm[i].clear()
                # Detect truncation (time-limit) if present in info
                truncated_flag = False
                try:
                    truncated_flag = bool(info.get('TimeLimit.truncated', False)) or ('terminal_observation' in info)
                except Exception:
                    truncated_flag = False
                completed_returns.append(ep_returns[i])
                completed_lengths.append(ep_lengths[i])
                # Log to episodes.csv
                try:
                    _eps_csv_f.write(f"{step+1},{i},{int(ep_lengths[i])},{float(ep_returns[i])},{int(truncated_flag)}\n")
                    # Also log extended RM/AP info
                    # Infer last AP from adjacency row of prev_rm_state leading to next_rm_state
                    last_ap = -1
                    try:
                        if adj_for_log is not None:
                            row = adj_for_log[int(prev_rm_state)]
                            idx = np.where(row == int(next_rm_state))[0]
                            last_ap = int(idx[0]) if idx.size > 0 else -1
                    except Exception:
                        last_ap = -1
                    _eps_ex_csv_f.write(f"{step+1},{i},{int(ep_lengths[i])},{float(ep_returns[i])},{int(truncated_flag)},{int(next_rm_state)},{last_ap}\n")
                except Exception:
                    pass
                rm_states[i] = 0
                ep_returns[i] = 0.0
                ep_lengths[i] = 0
            else:
                rm_states[i] = next_rm_state

        obs = next_obs

        if (step + 1) % train_freq == 0:
            train_info = agent.train()
        else:
            train_info = None
        # Log CRM counters periodically
        if train_info is not None and agent.crm_enabled:
            last = int(train_info.get('crm_last_synth', 0))
            tot_r = max(1, int(train_info.get('crm_total_real', 0)))
            tot_s = int(train_info.get('crm_total_synth', 0))
            ratio = float(tot_s) / float(tot_r) if tot_r > 0 else 0.0
            if (step + 1) % 100 == 0:
                _crm_csv_f.write(f"{step+1},{last},{tot_r},{tot_s},{ratio}\n")
        # Log mixing diagnostics periodically (if provided by agent)
        if train_info is not None and (step + 1) % 100 == 0:
            if 'mix_valid_frac' in train_info:
                try:
                    _mix_csv_f.write(f"{step+1},{float(train_info.get('mix_valid_frac', 0.0))},{float(train_info.get('mix_beta_mean', 0.0))},{float(train_info.get('mix_tq_mean', 0.0))},{float(train_info.get('mix_td_mean', 0.0))}\n")
                except Exception:
                    pass

        if (step + 1) % 1000 == 0:
            recent_mean = np.mean(completed_returns[-max(1, min(100, len(completed_returns))):]) if completed_returns else 0.0
            if agent.crm_enabled:
                tot_r = max(1, int(getattr(agent, 'crm_real_steps', 0)))
                tot_s = int(getattr(agent, 'crm_synth_steps', 0))
                ratio = float(tot_s) / float(tot_r)
                print(f"Step {step+1}/{total_steps} | envs={n_envs} | episodes={len(completed_returns)} | recent mean return={recent_mean:.2f} | CRM synth/real={ratio:.2f} ({tot_s}/{tot_r})")
            else:
                print(f"Step {step+1}/{total_steps} | envs={n_envs} | episodes={len(completed_returns)} | recent mean return={recent_mean:.2f}")
            # Flush shaping stats
            try:
                _shaping_csv_f.write(f"{step+1},{int(shaping_applied)},{float(shaping_added_sum)}\n")
            except Exception:
                pass
            shaping_applied = 0
            shaping_added_sum = 0.0

    # Close CSV file handles
    for _fh in (_eps_csv_f, _eps_ex_csv_f, _crm_csv_f, _mix_csv_f, _shaping_csv_f):
        try:
            _fh.close()
        except Exception:
            pass

    agent.save(model_path)
    print(f"Saved TD3 model to {model_path}_*.pth")

    # Save VecNormalize stats if enabled
    norm_ref = getattr(vec, "_vecnormalize_ref", None)
    if norm_ref is not None:
        try:
            norm_path = os.path.join(out_dir, 'vecnormalize.pkl')
            norm_ref.save(norm_path)
            print(f"Saved VecNormalize stats to {norm_path}")
        except Exception as e:
            print(f"Warning: failed to save VecNormalize stats: {e}")

    # Save episodic logs for plotting
    try:
        np.save(os.path.join(out_dir, 'episode_returns.npy'), np.asarray(completed_returns, dtype=np.float32))
        np.save(os.path.join(out_dir, 'episode_lengths.npy'), np.asarray(completed_lengths, dtype=np.int32))
        print(f"Saved episodic logs to {out_dir}")
    except Exception as e:
        print(f"Warning: failed to save episodic logs: {e}")

    # Post-training: save CRM summary
    try:
        if getattr(agent, 'crm_enabled', False):
            tot_r = int(getattr(agent, 'crm_real_steps', 0))
            tot_s = int(getattr(agent, 'crm_synth_steps', 0))
            ratio = float(tot_s) / float(max(1, tot_r))
            with open(os.path.join(out_dir, 'crm_summary.txt'), 'w') as f:
                f.write(f"total_real={tot_r}\n")
                f.write(f"total_synth={tot_s}\n")
                f.write(f"ratio_synth_per_real={ratio}\n")
    except Exception:
        pass

    # Post-training: construct Q-automaton from replay buffer (mirror of discrete)
    try:
        import importlib
        _construct_q_auto = importlib.import_module('src.control_env_debug.construct_q_automaton').construct_q_automaton
        # Save replay buffer for reproducible post-analysis
        try:
            buf_path = os.path.join(out_dir, 'replay_buffer.pkl.gz')
            agent.buffer.save(buf_path)
            print(f"Saved replay buffer to {buf_path}")
        except Exception as be:
            print(f"Warning: failed to save replay buffer: {be}")
        # Build a single env to access the automaton definition consistent with training
        single_env_for_auto = make_single_env(env_type=env_type, reward_shaping=reward_shaping, max_episode_steps=max_episode_steps, ap_config=ap_config)
        automaton = getattr(single_env_for_auto, 'automaton', getattr(single_env_for_auto.unwrapped, 'automaton', None))
        if automaton is None:
            raise RuntimeError('Automaton not available on environment')
        # Save under TMLR/automaton_q
        q_out_dir = _AUTOMATON_Q_DIR
        _construct_q_auto(agent, agent.buffer.as_list(), automaton, run_name, q_out_dir)
    except Exception as e:
        print(f"Warning: Q-automaton construction skipped/failed: {e}")


def _load_teacher_q_automaton(run_name: str):
    import json
    from pathlib import Path
    q_path = _AUTOMATON_Q_DIR / f"{run_name}.json"
    with open(q_path, 'r') as f:
        data = json.load(f)
    # Counts
    aut_num_q = np.array(data['aut_num_q'], dtype=np.int64)
    aut_total_q = np.array(data['aut_total_q'], dtype=np.float32)
    # Avoid division by zero for Q
    q_hat = np.divide(aut_total_q, np.maximum(1.0, aut_num_q))
    # V-values and counts are optional; fall back to None if missing
    v_hat = None
    aut_num_v = None
    try:
        aut_num_v = np.array(data['aut_num_v'], dtype=np.int64)
        aut_total_v = np.array(data['aut_total_v'], dtype=np.float32)
        v_hat = np.divide(aut_total_v, np.maximum(1.0, aut_num_v))
    except Exception:
        v_hat = None
        aut_num_v = None
    return q_hat, v_hat, aut_num_q, aut_num_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument('--env_type', type=str, default=os.environ.get('ENV_TYPE', 'patrol'),
                        help='Environment type: patrol, obstacles, obstacles_strict, '
                             'flatworld_patrol, flatworld_sequence, '
                             'zones_patrol, zones_sequence, cartpole')
    parser.add_argument('--reward_shaping', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--a_threshold', type=float, default=None)
    parser.add_argument('--b_threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    # Training args
    parser.add_argument('--total_steps', type=int, default=int(os.environ.get('STEPS', '5000')))
    parser.add_argument('--n_envs', type=int, default=int(os.environ.get('N_ENVS', '1')))
    parser.add_argument('--run_name', type=str, default='td3_vec')
    parser.add_argument('--normalize', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True)
    # Transfer schedule
    parser.add_argument('--distill_mode', type=str, default='off', choices=['off','static','dynamic'])
    parser.add_argument('--teacher_run_name', type=str, default=None)
    parser.add_argument('--w0', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--distill_tau', type=float, default=None)
    parser.add_argument('--w_min', type=float, default=0.0)
    # Dynamic target mixing
    parser.add_argument('--mix_target', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=False)
    parser.add_argument('--rho', type=float, default=0.999)
    parser.add_argument('--min_source_q_count', type=int, default=1)
    # Teacher reward shaping
    parser.add_argument('--shape_teacher', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=False)
    parser.add_argument('--shape_scale', type=float, default=1.0)
    parser.add_argument('--shape_min_count', type=int, default=1)
    # C-PREP weight transfer
    parser.add_argument('--init_from_teacher', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=False,
                        help='Initialise student network weights from teacher model (C-PREP style)')
    # Agent overrides
    parser.add_argument('--actor_lr', type=float, default=None)
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--buffer_size', type=int, default=None)
    parser.add_argument('--policy_noise', type=float, default=None)
    parser.add_argument('--noise_clip', type=float, default=None)
    parser.add_argument('--policy_delay', type=int, default=None)
    parser.add_argument('--use_crm', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=None)
    parser.add_argument('--no_rm', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=False)
    # Performance
    parser.add_argument('--device', type=str, default=None, help='torch device: cpu, cuda, cuda:0, cuda:1, etc.')
    parser.add_argument('--train_freq', type=int, default=1, help='Train every N env steps (default: 1)')

    args = parser.parse_args()
    # Build ap_config if thresholds provided
    ap_cfg = None
    if args.a_threshold is not None or args.b_threshold is not None:
        ap_cfg = {}
        if args.a_threshold is not None:
            ap_cfg['a_threshold'] = args.a_threshold
        if args.b_threshold is not None:
            ap_cfg['b_threshold'] = args.b_threshold

    run_training(
        env_type=args.env_type,
        reward_shaping=args.reward_shaping,
        max_episode_steps=args.max_episode_steps,
        ap_config=ap_cfg,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        run_name=args.run_name,
        normalize=args.normalize,
        seed=args.seed,
        distill_mode=args.distill_mode,
        teacher_run_name=args.teacher_run_name,
        w0=args.w0,
        distill_tau=args.distill_tau,
        w_min=args.w_min,
        mix_target=args.mix_target,
        rho=args.rho,
        min_source_q_count=args.min_source_q_count,
        shape_teacher=args.shape_teacher,
        shape_scale=args.shape_scale,
        shape_min_count=args.shape_min_count,
        init_from_teacher=args.init_from_teacher,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        use_crm=args.use_crm,
        no_rm=args.no_rm,
        device=args.device,
        train_freq=args.train_freq,
    )
