#!/usr/bin/env python3
"""Publication-quality plots: one per environment with specified colors.

Usage:
    python plot_paper.py                    # All environments
    python plot_paper.py --envs patrol      # One environment
    python plot_paper.py --out ./results/   # Output directory
"""
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────

LOGS_ROOT = "logs"
PREFIX = "bench"

# User-specified colors for methods
# METHOD_COLORS = {
#     "td3_base":    "#8B4513",  # Brown
#     "td3_crm":     "#008000",  # Green
#     "td3_static":  "#FF8C00",  # Orange
#     "td3_dynamic": "#0000FF",  # Blue
#     "td3_shaped":  "#800080",  # Purple
#     "td3_cprep":   "#FF0000",  # Red
# }

METHOD_COLORS = {
    "td3_base":    "#8B4513",  # Brown
    "td3_crm":     "#0000FF",  # Green
    "td3_static":  "#FF8C00",  # Orange
    "td3_dynamic": "#008000",  # Blue
    "td3_shaped":  "#800080",  # Purple
    "td3_cprep":   "#FF0000",  # Red
}


# Labels
METHOD_LABELS = {
    "td3_base":    "Vanilla TD3",
    "td3_static":  "Static Distill",
    "td3_dynamic": "CRM",
    "td3_crm":     "Dynamic Distill",
    "td3_shaped":  "ProductMDP",
    "td3_cprep":   "CPREP",
}

# Plot order (consistent across plots)
METHOD_ORDER = ["td3_base", "td3_crm", "td3_static", "td3_dynamic", "td3_shaped", "td3_cprep"]

ENV_TITLES = {
    "patrol":              "HalfCheetah Patrol",
    "flatworld_patrol":    "FlatWorld Patrol",
    "flatworld_sequence":  "FlatWorld Sequence",
    "zones_sequence":      "Zones Sequence (Point→Car)",
}

ENV_MAX_STEPS = {
    "flatworld_patrol":   1_000_000,
    "patrol":             2_000_000,
    "flatworld_sequence": 1_000_000,
    "zones_sequence":     2_000_000,
}

_RUN_RE = re.compile(
    r"^(?P<prefix>\w+?)_(?P<env>patrol|flatworld_patrol|flatworld_sequence|zones_patrol|zones_sequence)"
    r"_(?P<method>td3_\w+?)_s(?P<seed>\d+)$"
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def ema(x, alpha):
    """Vectorized EMA for speed."""
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def load_csv(path):
    """Load episodes.csv → (steps, returns) arrays, or None."""
    if not os.path.isfile(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        if data.size < 2:
            return None
        return np.asarray(data["step"], dtype=np.float64), np.asarray(data["ep_return"], dtype=np.float64)
    except Exception:
        return None


def load_monitor_csv(path, n_envs=4):
    """Load a Gym monitor.csv → (steps, returns), or None."""
    if not os.path.isfile(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1, encoding="utf-8")
        if data.size < 2:
            return None
        returns = np.asarray(data["r"], dtype=np.float64)
        lengths = np.asarray(data["l"], dtype=np.float64)
        steps = np.cumsum(lengths) / n_envs
        return steps, returns
    except Exception:
        return None


def discover(logs_root, prefix, n_envs=4):
    """Return {env: {method: [(steps, returns), ...]}}."""
    tree = defaultdict(lambda: defaultdict(list))
    for name in sorted(os.listdir(logs_root)):
        m = _RUN_RE.match(name)
        if not m or m.group("prefix") != prefix:
            continue
        csv_path = os.path.join(logs_root, name, "episodes.csv")
        result = load_csv(csv_path)
        if result is None:
            monitor_path = os.path.join(logs_root, name + ".monitor.csv")
            result = load_monitor_csv(monitor_path, n_envs=n_envs)
        if result is not None:
            tree[m.group("env")][m.group("method")].append(result)
    return tree


def aggregate_method(seed_data, window=50, max_steps=None, n_interp=1200, min_progress=0.5):
    """Aggregate seeds, skip incomplete runs."""
    if not seed_data:
        return None

    alpha = 2.0 / (window + 1)
    processed = []
    end_steps = []

    for steps, returns in seed_data:
        if len(returns) < 2 or len(steps) < 2:
            continue

        if max_steps is not None:
            mask = steps <= max_steps
            if not mask.any():
                continue
            steps = steps[mask]
            returns = returns[mask]
            if len(steps) < 2:
                continue

        smoothed_returns = ema(returns, alpha)
        smoothed_steps = ema(steps, alpha)

        uniq_steps, uniq_idx = np.unique(smoothed_steps, return_index=True)
        if len(uniq_steps) < 2:
            continue
        uniq_returns = smoothed_returns[uniq_idx]

        # Skip incomplete seeds
        if max_steps is not None and float(uniq_steps[-1]) < max_steps * min_progress:
            continue

        processed.append((uniq_steps, uniq_returns))
        end_steps.append(float(uniq_steps[-1]))

    if not processed:
        return None

    common_end = min(end_steps)
    if common_end <= 0:
        return None

    x = np.linspace(0.0, common_end, n_interp)
    mat = np.stack([np.interp(x, s, r) for s, r in processed], axis=0)

    n = mat.shape[0]
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1 if n > 1 else 0)

    if n > 1:
        from scipy.stats import t as t_dist
        t_val = t_dist.ppf(0.95, df=n - 1)
        sem = std / np.sqrt(n)
        lo, hi = mean - t_val * sem, mean + t_val * sem
    else:
        lo, hi = mean - std, mean + std

    return {"x": x, "mean": mean, "lo": lo, "hi": hi, "n": n}


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_env_paper(env, method_aggs, figsize=(10.8, 8.8)):
    """Create a publication-quality plot for one environment with full box frame."""
    fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI handled at save time

    for method in METHOD_ORDER:
        agg = method_aggs.get(method)
        if agg is None:
            continue
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        ax.plot(agg["x"], agg["mean"], color=color, label=label, linewidth=2.2, zorder=3)
        ax.fill_between(agg["x"], agg["lo"], agg["hi"], color=color, alpha=0.12, zorder=1)

    title = ENV_TITLES.get(env, env)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(" Timesteps", fontsize=30, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=30, fontweight="bold")
    
    ax.legend(loc="best", fontsize=26, framealpha=0.95, edgecolor="black", fancybox=False, 
              frameon=True, borderpad=0.8)
    
    # Make all spines visible to form complete box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.3)
        spine.set_color("black")
    
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=30)
    
    fig.tight_layout(pad=0.8)
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description="Publication-quality per-environment plots")
    ap.add_argument("--logs_root", default="logs")
    ap.add_argument("--prefix", default="bench")
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--format", default="pdf", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--n_interp", type=int, default=500, help="Number of interpolation points (lower=faster)")
    ap.add_argument("--out", default="plots", help="Output directory")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 30,
    })

    Path(args.out).mkdir(parents=True, exist_ok=True)

    tree = discover(args.logs_root, args.prefix, n_envs=args.n_envs)
    if args.envs:
        tree = {e: tree[e] for e in args.envs if e in tree}

    if not tree:
        print("No data found.")
        return

    envs = sorted(tree.keys())

    for env in envs:
        max_steps = ENV_MAX_STEPS.get(env)
        method_aggs = {}
        for method, seeds in tree[env].items():
            method_aggs[method] = aggregate_method(seeds, window=args.window, max_steps=max_steps, n_interp=args.n_interp)

        fig, ax = plot_env_paper(env, method_aggs)
        
        out_file = os.path.join(args.out, f"{env}.{args.format}")
        fig.savefig(out_file, format=args.format, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")

        # Print summary
        print(f"\n  {ENV_TITLES.get(env, env)}:")
        for method in METHOD_ORDER:
            seeds = tree[env].get(method, [])
            if seeds:
                completed = sum(1 for s, r in seeds if max_steps is None or s[-1] >= max_steps * 0.5)
                max_step = max(s[-1] for s, r in seeds if max_steps is None or s[-1] >= max_steps * 0.5) if completed > 0 else 0
                if completed > 0:
                    print(f"    {METHOD_LABELS.get(method, method):20s}  {completed} seeds, up to step {int(max_step):,}")


if __name__ == "__main__":
    main()
