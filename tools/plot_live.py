#!/usr/bin/env python3
"""Quick live-progress plot from episodes.csv files (no need to wait for training to finish).

Usage:
    python plot_live.py                          # plot all available data
    python plot_live.py --envs flatworld_patrol   # one env only
    python plot_live.py --format pdf              # save as pdf
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

METHOD_LABELS = {
    "td3_base":    "Vanilla TD3",
    "td3_crm":     "CRM",
    "td3_static":  "Static Transfer",
    "td3_dynamic": "Dynamic Transfer",
    "td3_shaped":  "Product MDP",
    "td3_cprep":   "CPREP",
}

# Plot order (so legend is consistent)
METHOD_ORDER = ["td3_base", "td3_crm", "td3_cprep", "td3_shaped", "td3_static", "td3_dynamic"]

COLORS = {
    "td3_base":    "#0072B2",
    "td3_crm":     "#D55E00",
    "td3_cprep":   "#009E73",
    "td3_shaped":  "#CC79A7",
    "td3_static":  "#E69F00",
    "td3_dynamic": "#56B4E9",
}

LINESTYLES = {
    "td3_base":    "-",
    "td3_crm":     "-",
    "td3_cprep":   "--",
    "td3_shaped":  "-",
    "td3_static":  "-",
    "td3_dynamic": "-.",
}

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
    """Load a Gym monitor.csv (r,l,t with comment header) → (steps, returns), or None.
    
    Monitor logs episode lengths across all vectorized envs, so cumsum(l)
    over-counts by ~n_envs. We divide by n_envs to get approximate training steps.
    """
    if not os.path.isfile(path):
        return None
    try:
        # Skip the first line (JSON comment starting with #)
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
            # Fallback: check for monitor.csv at logs/<run_name>.monitor.csv
            monitor_path = os.path.join(logs_root, name + ".monitor.csv")
            result = load_monitor_csv(monitor_path, n_envs=n_envs)
        if result is not None:
            tree[m.group("env")][m.group("method")].append(result)
    return tree


def aggregate_method(seed_data, window=50, max_steps=None, n_interp=1200, min_progress=0.5):
    """Given list of (steps, returns), return dict with x, mean, lo, hi or None.

    Curves are aligned on a shared step grid (not episode index), so seeds with
    different episode counts do not truncate each other prematurely.
    
    Seeds that reach less than min_progress (default 50%) of max_steps are excluded
    to avoid incomplete early runs truncating the aggregation.
    """
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

        # Interpolation requires strictly increasing x-values.
        uniq_steps, uniq_idx = np.unique(smoothed_steps, return_index=True)
        if len(uniq_steps) < 2:
            continue
        uniq_returns = smoothed_returns[uniq_idx]

        # Skip seeds that are far from completion (< min_progress of max_steps)
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
        t_val = t_dist.ppf(0.95, df=n - 1)  # 90% CI
        sem = std / np.sqrt(n)
        lo, hi = mean - t_val * sem, mean + t_val * sem
    else:
        lo, hi = mean - std, mean + std

    return {"x": x, "mean": mean, "lo": lo, "hi": hi, "n": n}


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_env(env, method_aggs, ax):
    for method in METHOD_ORDER:
        agg = method_aggs.get(method)
        if agg is None:
            continue
        color = COLORS.get(method, "#333333")
        ls = LINESTYLES.get(method, "-")
        label = METHOD_LABELS.get(method, method)
        ax.plot(agg["x"], agg["mean"], color=color, label=label, linewidth=2, linestyle=ls)
        ax.fill_between(agg["x"], agg["lo"], agg["hi"], color=color, alpha=0.15)

    title = ENV_TITLES.get(env, env)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Return")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")


def main():
    ap = argparse.ArgumentParser(description="Quick live-progress plot")
    ap.add_argument("--logs_root", default="logs")
    ap.add_argument("--prefix", default="bench")
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=4, help="Number of vectorized envs (for monitor.csv step scaling)")
    ap.add_argument("--out", default=None, help="Output filename (default: live_plot.<format>)")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
    })

    tree = discover(args.logs_root, args.prefix, n_envs=args.n_envs)
    if args.envs:
        tree = {e: tree[e] for e in args.envs if e in tree}

    if not tree:
        print("No data found. Check --logs_root and --prefix.")
        return

    envs = sorted(tree.keys())
    n_envs = len(envs)

    fig, axes = plt.subplots(1, n_envs, figsize=(7 * n_envs, 4.5), squeeze=False)

    for idx, env in enumerate(envs):
        max_steps = ENV_MAX_STEPS.get(env)
        method_aggs = {}
        for method, seeds in tree[env].items():
            method_aggs[method] = aggregate_method(seeds, window=args.window, max_steps=max_steps)
        plot_env(env, method_aggs, axes[0][idx])

    fig.tight_layout(w_pad=3.0)

    out = args.out or f"live_plot.{args.format}"
    fig.savefig(out, format=args.format, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary
    for env in envs:
        print(f"\n  {ENV_TITLES.get(env, env)}:")
        for method in METHOD_ORDER:
            seeds = tree[env].get(method, [])
            if seeds:
                max_step = max(s[-1] for s, r in seeds)
                print(f"    {METHOD_LABELS.get(method, method):20s}  {len(seeds)} seeds, up to step {int(max_step):,}")


if __name__ == "__main__":
    main()
