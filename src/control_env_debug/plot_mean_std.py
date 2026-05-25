#!/usr/bin/env python3
"""Aggregate multiple runs' episode_returns.npy and plot mean with std bands.

Supports both single-pattern mode and multi-group overlay mode.

Usage examples:
    # Single pattern (aggregates across seeds matching the glob)
    python -m src.control_env_debug.plot_mean_std --logs_root logs --pattern 'td3_vec_s*' --out logs/td3_vec_agg.png

    # Multi-group overlay (compare methods), each --group is label:glob
    python -m src.control_env_debug.plot_mean_std \
        --logs_root logs \
        --group 'Dynamic:bench_patrol_td3_dynamic_*' \
        --group 'Static:bench_patrol_td3_static_*' \
        --group 'Scratch:bench_patrol_td3_base_*' \
        --group 'C-PREP:bench_patrol_td3_cprep_*' \
        --out logs/compare_methods.png --window 10
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Colorblind-friendly palette (Okabe-Ito)
OKABE_ITO = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#CC79A7',  # purple
    '#F0E442',  # yellow
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
    '#000000',  # black
    '#0099CC',  # teal-ish
    '#999999',  # gray
]


def apply_style(style):
    style = (style or 'default').lower()
    if style == 'paper':
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'lines.linewidth': 2.2,
            'savefig.bbox': 'tight',
        })
    elif style == 'talk':
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 12,
            'lines.linewidth': 2.6,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'grid.linestyle': '--',
            'savefig.bbox': 'tight',
        })
    else:
        # default
        plt.rcParams.update({'savefig.bbox': 'tight'})


def load_episode_returns(run_dirs):
    runs = []
    paths = []
    for d in run_dirs:
        p = os.path.join(d, 'episode_returns.npy')
        if os.path.exists(p):
            try:
                arr = np.load(p)
                if arr.size > 0:
                    runs.append(arr)
                    paths.append(p)
            except Exception:
                pass
    return runs, paths


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')


def load_steps_array(run_dir: str):
    """Load per-episode global training step from episodes.csv if present.
    Returns a 1D numpy array sorted by step, or None if missing/unreadable.
    """
    p_csv = os.path.join(run_dir, 'episodes.csv')
    if not os.path.exists(p_csv):
        return None
    try:
        data = np.genfromtxt(p_csv, delimiter=',', names=True, dtype=None, encoding='utf-8')
        if getattr(data, 'size', 0) == 0 or not hasattr(data, 'dtype') or not data.dtype.names:
            return None
        names = {n.lower(): n for n in data.dtype.names}
        if 'step' not in names:
            return None
        step_col = names['step']
        steps = np.array(data[step_col], dtype=float)
        # Ensure ascending by step
        order = np.argsort(steps)
        return steps[order]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_root', type=str, default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs'))
    ap.add_argument('--pattern', type=str, default=None, help="Glob pattern for run directories under logs_root (single-mode)")
    ap.add_argument('--group', action='append', default=None, help="Multi-group overlay: 'label:glob' under logs_root; can be repeated")
    ap.add_argument('--out', type=str, default=None, help="Output PNG path; default under logs_root")
    ap.add_argument('--window', type=int, default=1, help="Smoothing window for mean/std display")
    ap.add_argument('--format', type=str, default=None, choices=['png', 'pdf', 'svg'], help="Override output format (defaults from --out extension or png)")
    ap.add_argument('--dpi', type=int, default=300, help="DPI for raster formats (png). Ignored for vector (pdf/svg)")
    ap.add_argument('--style', type=str, default='paper', choices=['default', 'paper', 'talk'], help="Plot style preset")
    ap.add_argument('--width', type=float, default=None, help="Figure width in inches (overrides defaults)")
    ap.add_argument('--height', type=float, default=None, help="Figure height in inches (overrides defaults)")
    ap.add_argument('--no_band', action='store_true', help="Disable std fill bands")
    ap.add_argument('--ymin', type=float, default=None, help="Y-axis min")
    ap.add_argument('--ymax', type=float, default=None, help="Y-axis max")
    ap.add_argument('--x_axis', type=str, default='steps', choices=['episodes', 'steps'], help="X-axis: episode index or global steps (from episodes.csv)")
    ap.add_argument('--per_group_lengths', action='store_true',
                    help='Plot each group up to its own min run length instead of truncating all groups to a global minimum length.')
    args = ap.parse_args()

    apply_style(args.style)
    logs_root = os.path.abspath(args.logs_root)
    if args.group:
        # Multi-group overlay mode
        groups = []  # list of (label, [run_dirs])
        for spec in args.group:
            if ':' in spec:
                label, pat = spec.split(':', 1)
                label = label.strip() or pat.strip()
                pat = pat.strip()
            else:
                label = spec.strip()
                pat = spec.strip()
            cand = sorted(glob.glob(os.path.join(logs_root, pat)))
            run_dirs = [d for d in cand if os.path.isdir(d)]
            groups.append((label, run_dirs))

        # Load runs; optionally find global min length for alignment
        all_runs = []
        loaded_groups = []  # list of (label, [np.ndarray])
        loaded_step_groups = []  # list of (label, [np.ndarray or None])
        for label, run_dirs in groups:
            runs, run_paths = load_episode_returns(run_dirs)
            if runs:
                loaded_groups.append((label, runs))
                all_runs.extend(runs)
                # Steps arrays aligned per run (may be None if missing)
                if args.x_axis == 'steps':
                    steps_list = []
                    for d in [rd for rd in run_dirs if os.path.isdir(rd)]:
                        if os.path.exists(os.path.join(d, 'episode_returns.npy')):
                            steps_list.append(load_steps_array(d))
                    if steps_list:
                        loaded_step_groups.append((label, steps_list))
        if not all_runs:
            raise SystemExit(f"No episode_returns.npy found under {logs_root} for given groups")
        global_min_len = min(map(len, all_runs))
        # Use steps x-axis if requested; will fallback per-group if steps missing
        use_steps = (args.x_axis == 'steps')

        fig_w = args.width if args.width is not None else 10
        fig_h = args.height if args.height is not None else 5
        plt.figure(figsize=(fig_w, fig_h))
        colors = OKABE_ITO
        ci = 0
        # Construct per-group plotting
        for gi, (label, runs) in enumerate(loaded_groups):
            # Choose per-group min length if requested; otherwise use global
            group_min_len = min(map(len, runs))
            cut_len = group_min_len if args.per_group_lengths else global_min_len
            runs_t = [r[:cut_len] for r in runs]
            arr = np.stack(runs_t, axis=0)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            mean_s = moving_average(mean, args.window)
            std_s = moving_average(std, args.window)
            if use_steps:
                # Pick the steps series with the highest final step for this group
                steps_list = None
                if gi < len(loaded_step_groups) and loaded_step_groups[gi][0] == label:
                    steps_list = loaded_step_groups[gi][1]
                x = None
                if steps_list:
                    best = None
                    best_last = -np.inf
                    for s in steps_list:
                        if s is None or len(s) < cut_len:
                            continue
                        last_val = s[cut_len-1]
                        if last_val > best_last:
                            best_last = last_val
                            best = s
                    if best is not None:
                        x_base = best[:cut_len]
                        x = moving_average(x_base, args.window)
                if x is None:
                    x = np.arange(len(mean_s))
            else:
                x = np.arange(len(mean_s))
            color = colors[ci % len(colors)]
            ci += 1
            # plt.plot(x, mean_s, label=f"{label} (n={arr.shape[0]})", color=color)
            plt.plot(x, mean_s, label=f"{label}", color=color)
            if not args.no_band:
                plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.15, color=color)
        plt.xlabel('Timesteps' if use_steps else 'Episode')
        plt.ylabel('Reward')
        # plt.title('TD3 methods: mean ± std across seeds')
        plt.legend(loc='upper left')
        if args.ymin is not None or args.ymax is not None:
            ymin, ymax = plt.ylim()
            plt.ylim(args.ymin if args.ymin is not None else ymin,
                     args.ymax if args.ymax is not None else ymax)
        plt.tight_layout()

        out = args.out or os.path.join(logs_root, 'td3_vec_compare.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # Determine format
        ext = os.path.splitext(out)[1][1:].lower() if args.out else 'png'
        fmt = args.format or ext or 'png'
        save_kwargs = {}
        if fmt == 'png':
            save_kwargs['dpi'] = args.dpi
        plt.savefig(out, format=fmt, **save_kwargs)
        print(f"Saved: {out} ({fmt}{', dpi='+str(save_kwargs['dpi']) if 'dpi' in save_kwargs else ''})")
    else:
        # Single-pattern mode (back-compat)
        pat = args.pattern or 'td3_vec_s*'
        cand = sorted(glob.glob(os.path.join(logs_root, pat)))
        run_dirs = [d for d in cand if os.path.isdir(d)]
        runs, paths = load_episode_returns(run_dirs)

        if not runs:
            raise SystemExit(f"No episode_returns.npy found under {logs_root}/{pat}")

        min_len = min(map(len, runs))
        runs = [r[:min_len] for r in runs]
        arr = np.stack(runs, axis=0)

        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        # Optional smoothing
        mean_s = moving_average(mean, args.window)
        std_s = moving_average(std, args.window)
        if args.x_axis == 'steps':
            # Pick the steps series with the highest final step among eligible runs
            x = None
            best = None
            best_last = -np.inf
            for d in run_dirs:
                if os.path.exists(os.path.join(d, 'episode_returns.npy')):
                    steps = load_steps_array(d)
                    if steps is not None and len(steps) >= min_len:
                        last_val = steps[min_len-1]
                        if last_val > best_last:
                            best_last = last_val
                            best = steps
            if best is not None:
                x_base = best[:min_len]
                x = moving_average(x_base, args.window)
            else:
                x = np.arange(len(mean_s))
        else:
            x = np.arange(len(mean_s))

        fig_w = args.width if args.width is not None else 9
        fig_h = args.height if args.height is not None else 4
        plt.figure(figsize=(fig_w, fig_h))
        color = OKABE_ITO[0]
        plt.plot(x, mean_s, label=f'mean (n={arr.shape[0]})', color=color)
        if not args.no_band:
            plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2, label='std', color=color)
        plt.xlabel('Steps' if args.x_axis == 'steps' else 'Episode')
        plt.ylabel('Return')
        plt.title('TD3 across seeds: mean ± std')
        plt.legend()
        if args.ymin is not None or args.ymax is not None:
            ymin, ymax = plt.ylim()
            plt.ylim(args.ymin if args.ymin is not None else ymin,
                     args.ymax if args.ymax is not None else ymax)
        plt.tight_layout()

        out = args.out or os.path.join(logs_root, 'td3_vec_agg.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ext = os.path.splitext(out)[1][1:].lower() if args.out else 'png'
        fmt = args.format or ext or 'png'
        save_kwargs = {}
        if fmt == 'png':
            save_kwargs['dpi'] = args.dpi
        plt.savefig(out, format=fmt, **save_kwargs)
        print(f"Saved: {out} ({fmt}{', dpi='+str(save_kwargs['dpi']) if 'dpi' in save_kwargs else ''})")


if __name__ == '__main__':
    main()
