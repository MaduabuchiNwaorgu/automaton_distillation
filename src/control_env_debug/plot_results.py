#!/usr/bin/env python3
"""Generate publication-quality learning-curve plots with 90% confidence bands.

Auto-discovers environments and methods from log directory names produced by
``run_all_benchmarks.py``.  Produces one figure per environment overlaying all
methods, plus an optional combined grid figure.

Usage
-----
# After running benchmarks:
python -m src.control_env_debug.plot_results          # all defaults
python -m src.control_env_debug.plot_results --logs_root logs --ci 0.90 --window 10
python -m src.control_env_debug.plot_results --envs patrol flatworld_patrol --format pdf
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ── aesthetics ──────────────────────────────────────────────────────────────

OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

METHOD_LABELS = {
    "td3_base": "Vanilla TD3",
    "td3_crm": "CRM",
    "td3_static": "Static Transfer",
    "td3_dynamic": "Dynamic Transfer",
    "td3_shaped": "Product MDP",
    "td3_cprep": "CPREP",
}

ENV_TITLES = {
    "patrol": "HalfCheetah Patrol",
    "flatworld_patrol": "FlatWorld Patrol",
    "flatworld_sequence": "FlatWorld Sequence",
    "zones_patrol": "Zones Patrol (Point→Car)",
    "zones_sequence": "Zones Sequence (Point→Car)",
}


def _paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "figure.figsize": (7, 4.2),
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.05,
    })


# ── data helpers ────────────────────────────────────────────────────────────

_RUN_RE = re.compile(
    r"^(?P<prefix>\w+?)_(?P<env>patrol|flatworld_patrol|flatworld_sequence|zones_patrol|zones_sequence)"
    r"_(?P<method>td3_\w+?)_s(?P<seed>\d+)$"
)


def discover_runs(logs_root: str, prefix: str = "bench") -> Dict[str, Dict[str, List[str]]]:
    """Return ``{env_name: {method: [run_dir, ...]}}`` from log folder."""
    tree: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for name in sorted(os.listdir(logs_root)):
        full = os.path.join(logs_root, name)
        if not os.path.isdir(full):
            continue
        m = _RUN_RE.match(name)
        if m and m.group("prefix") == prefix:
            tree[m.group("env")][m.group("method")].append(full)
    return tree


def load_returns(run_dirs: List[str]) -> List[np.ndarray]:
    """Load returns from ``episode_returns.npy`` or fall back to ``episodes.csv``."""
    runs = []
    for d in run_dirs:
        p = os.path.join(d, "episode_returns.npy")
        if os.path.isfile(p):
            arr = np.load(p)
            if arr.size > 0:
                runs.append(arr)
                continue
        # Fallback: read ep_return column from episodes.csv (for in-progress runs)
        csv_path = os.path.join(d, "episodes.csv")
        if os.path.isfile(csv_path):
            try:
                data = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8")
                if data.size > 0:
                    runs.append(np.asarray(data["ep_return"], dtype=np.float64))
            except Exception:
                pass
    return runs


def load_steps(run_dir: str) -> Optional[np.ndarray]:
    """Load per-episode step index from ``episodes.csv``."""
    p = os.path.join(run_dir, "episodes.csv")
    if not os.path.isfile(p):
        return None
    try:
        data = np.genfromtxt(p, delimiter=",", names=True, encoding="utf-8")
        if data.size == 0:
            return None
        return np.sort(np.asarray(data["step"], dtype=np.float64))
    except Exception:
        return None


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average (forward pass). alpha ∈ (0,1]; larger = less smooth."""
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def smooth(x: np.ndarray, w: int, method: str = "ema") -> np.ndarray:
    """Smooth a 1-D signal.

    method='ema':  Exponential moving average with alpha = 2/(w+1).
    method='sma':  Simple moving average (old behaviour).
    """
    if w <= 1:
        return x.copy()
    if method == "ema":
        alpha = 2.0 / (w + 1)
        return _ema(x, alpha)
    # Simple moving average fallback
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def aggregate(
    runs: List[np.ndarray],
    window: int = 1,
    ci: float = 0.90,
    x_steps: Optional[List[Optional[np.ndarray]]] = None,
    smooth_type: str = "ema",
) -> Optional[dict]:
    """Compute mean, CI bands, and x-axis for a list of per-seed return arrays.

    Smooths each seed independently *before* computing statistics so the
    confidence bands are smooth too — much cleaner for publication figures.

    Returns ``None`` when fewer than 1 seed have data.
    """
    if len(runs) < 1:
        return None
    min_len = min(len(r) for r in runs)
    if min_len < 2:
        return None
    # Smooth each seed independently, then compute cross-seed stats
    smoothed = np.stack(
        [smooth(r[:min_len].astype(np.float64), window, method=smooth_type) for r in runs],
        axis=0,
    )  # (n_seeds, T) — T = min_len for EMA, shorter for SMA
    T = smoothed.shape[1]
    n = smoothed.shape[0]
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0, ddof=1 if n > 1 else 0)

    # CI via t-distribution
    if n > 1:
        t_val = sp_stats.t.ppf(1 - (1 - ci) / 2, df=n - 1)
        sem = std / np.sqrt(n)
        lo = mean - t_val * sem
        hi = mean + t_val * sem
    else:
        lo = mean - std
        hi = mean + std

    # X-axis: prefer steps from CSV, fallback to episode index
    x = np.arange(T)
    if x_steps:
        for s in x_steps:
            if s is not None and len(s) >= min_len:
                xs = smooth(s[:min_len].astype(np.float64), window, method=smooth_type)
                if len(xs) >= T:
                    x = xs[:T]
                break

    return {"x": x, "mean": mean, "lo": lo, "hi": hi, "n": n}


# ── plotting ────────────────────────────────────────────────────────────────

def plot_env(
    env_name: str,
    method_data: Dict[str, dict],
    ax: Optional[plt.Axes] = None,
    ci_level: float = 0.90,
) -> plt.Figure:
    """Plot one environment's learning curves on *ax* (creates figure if None)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    ci_pct = int(ci_level * 100)

    for i, (method, agg) in enumerate(sorted(method_data.items())):
        if agg is None:
            continue
        color = OKABE_ITO[i % len(OKABE_ITO)]
        label = METHOD_LABELS.get(method, method)
        n = agg["n"]
        lab = f"{label} (n={n})"
        ax.plot(agg["x"], agg["mean"], color=color, label=lab)
        ax.fill_between(agg["x"], agg["lo"], agg["hi"], color=color, alpha=0.18)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode return")
    title = ENV_TITLES.get(env_name, env_name)
    ax.set_title(f"{title}  ({ci_pct}% CI)")
    ax.legend(loc="best", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        fig.tight_layout()
    return fig


def plot_all_envs(
    all_agg: Dict[str, Dict[str, dict]],
    ci_level: float = 0.90,
) -> plt.Figure:
    """Grid of subplots — one per environment."""
    envs = sorted(all_agg.keys())
    n = len(envs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    for idx, env_name in enumerate(envs):
        r, c = divmod(idx, cols)
        plot_env(env_name, all_agg[env_name], ax=axes[r][c], ci_level=ci_level)
    # Hide leftover axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    return fig


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs_root", default="logs", help="Root log dir (default: ./logs)")
    ap.add_argument("--prefix", default="bench", help="Run-name prefix (default: bench)")
    ap.add_argument("--envs", nargs="*", default=None, help="Filter environments (default: all discovered)")
    ap.add_argument("--methods", nargs="*", default=None, help="Filter methods (default: all discovered)")
    ap.add_argument("--ci", type=float, default=0.90, help="Confidence level for bands (default: 0.90)")
    ap.add_argument("--window", type=int, default=50, help="Smoothing window (episodes, default 50)")
    ap.add_argument("--smooth_type", default="ema", choices=["ema", "sma"],
                    help="Smoothing method: ema (exponential) or sma (simple moving avg)")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <logs_root>/plots)")
    ap.add_argument("--x_axis", default="steps", choices=["steps", "episodes"])
    ap.add_argument("--combined", action="store_true", help="Also produce a combined grid figure")
    args = ap.parse_args()

    _paper_style()
    logs_root = os.path.abspath(args.logs_root)
    out_dir = args.out_dir or os.path.join(logs_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    tree = discover_runs(logs_root, prefix=args.prefix)
    if not tree:
        raise SystemExit(f"No benchmark runs found in {logs_root} with prefix '{args.prefix}'")

    if args.envs:
        tree = {e: tree[e] for e in args.envs if e in tree}

    all_agg: Dict[str, Dict[str, dict]] = {}

    for env_name, methods in sorted(tree.items()):
        method_agg = {}
        for method, dirs in sorted(methods.items()):
            if args.methods and method not in args.methods:
                continue
            runs = load_returns(dirs)
            x_steps = [load_steps(d) for d in dirs] if args.x_axis == "steps" else None
            method_agg[method] = aggregate(runs, window=args.window, ci=args.ci,
                                            x_steps=x_steps, smooth_type=args.smooth_type)
        if any(v is not None for v in method_agg.values()):
            all_agg[env_name] = method_agg

    if not all_agg:
        raise SystemExit("No plottable data found (need ≥1 seed with ≥2 episodes)")

    # Per-env figures
    for env_name, method_agg in all_agg.items():
        fig = plot_env(env_name, method_agg, ci_level=args.ci)
        fname = os.path.join(out_dir, f"{env_name}.{args.format}")
        fig.savefig(fname, format=args.format, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {fname}")

    # Combined grid
    if args.combined and len(all_agg) > 1:
        fig = plot_all_envs(all_agg, ci_level=args.ci)
        fname = os.path.join(out_dir, f"all_envs.{args.format}")
        fig.savefig(fname, format=args.format, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {fname}")

    print(f"\nDone — {len(all_agg)} environment(s) plotted to {out_dir}/")


if __name__ == "__main__":
    main()
