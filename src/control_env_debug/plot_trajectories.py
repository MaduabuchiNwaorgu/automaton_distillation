#!/usr/bin/env python3
"""Visualise agent trajectories with RM-state colouring.

Loads trajectory data saved by ``train_vectorized_td3.py`` (per-episode position
and RM-state arrays) and draws deep-ltl–style path plots.

For 2-D environments (FlatWorld, Zones) this produces bird's-eye-view plots with
circles/zones drawn.  For 1-D envs (HalfCheetah) it plots x-position over time
with threshold lines.

Usage
-----
# Single run:
python -m src.control_env_debug.plot_trajectories \
    --run logs/bench_flatworld_patrol_td3_dynamic_s0

# Compare multiple runs side-by-side:
python -m src.control_env_debug.plot_trajectories \
    --runs logs/bench_flatworld_patrol_td3_base_s0 \
           logs/bench_flatworld_patrol_td3_dynamic_s0 \
    --labels "Scratch" "Dynamic"

# Auto-discover all runs for an environment:
python -m src.control_env_debug.plot_trajectories \
    --env flatworld_patrol --logs_root logs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgba

# ── RM-state colour palette ────────────────────────────────────────────────

RM_COLORS = [
    "#4caf50",  # state 0 — green   (initial)
    "#f44336",  # state 1 — red     (accept / terminal)
    "#2196f3",  # state 2 — blue
    "#ff9800",  # state 3 — orange
    "#9c27b0",  # state 4 — purple
    "#00bcd4",  # state 5 — teal
    "#795548",  # state 6 — brown
    "#607d8b",  # state 7 — grey
]

ZONE_COLORS = {
    "red":     "#e53935",
    "blue":    "#1e88e5",
    "green":   "#43a047",
    "yellow":  "#fdd835",
    "magenta": "#ab47bc",
    "orange":  "#fb8c00",
    "aqua":    "#00acc1",
    "cyan":    "#00acc1",
}

METHOD_LABELS = {
    "td3_base": "TD3 (scratch)",
    "td3_crm": "CRM",
    "td3_static": "Static distill",
    "td3_dynamic": "Dynamic distill",
    "td3_shaped": "Shaped",
    "td3_cprep": "C-PREP",
}


# ── data loading ────────────────────────────────────────────────────────────

def load_trajectories(traj_dir: str, max_eps: int = 20) -> List[dict]:
    """Return list of ``{pos: ndarray(T,d), rm: ndarray(T,)}`` dicts."""
    episodes = []
    for i in range(max_eps):
        pos_path = os.path.join(traj_dir, f"ep{i:04d}_pos.npy")
        rm_path = os.path.join(traj_dir, f"ep{i:04d}_rm.npy")
        if not os.path.isfile(pos_path):
            break
        pos = np.load(pos_path)
        rm = np.load(rm_path) if os.path.isfile(rm_path) else np.zeros(len(pos), dtype=np.int32)
        episodes.append({"pos": pos, "rm": rm})
    return episodes


def load_env_meta(traj_dir: str) -> dict:
    meta_path = os.path.join(traj_dir, "env_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ── drawing helpers ─────────────────────────────────────────────────────────

def _setup_2d_axis(ax, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=7)


def draw_circle(ax, center, radius, color, alpha=0.35, label=None):
    c = plt.Circle(center, radius, fc=to_rgba(ZONE_COLORS.get(color, color), alpha),
                   ec=ZONE_COLORS.get(color, color), linewidth=1.2)
    ax.add_patch(c)
    ax.text(center[0], center[1], color[0].upper(), ha="center", va="center",
            fontsize=7, fontweight="bold", color=ZONE_COLORS.get(color, color))


def draw_wall(ax, wall: dict, color="#555555", alpha=0.5):
    rect = mpatches.Rectangle(
        (wall["x_min"], wall["y_min"]),
        wall["x_max"] - wall["x_min"],
        wall["y_max"] - wall["y_min"],
        fc=to_rgba(color, alpha), ec=color, linewidth=1.0,
    )
    ax.add_patch(rect)


def draw_regions(ax, meta: dict):
    """Draw circles and walls from env metadata."""
    for circ in meta.get("circles", []):
        draw_circle(ax, circ["center"], circ["radius"], circ["color"])
    for wall in meta.get("walls", []):
        draw_wall(ax, wall)


def draw_start_marker(ax, pos, size=0.12):
    """Orange diamond at start position."""
    x, y = pos[0], pos[1]
    diamond = plt.Polygon(
        [(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
        facecolor="#ff9800", edgecolor="black", linewidth=0.5, zorder=10,
    )
    ax.add_patch(diamond)


def draw_coloured_path(ax, positions: np.ndarray, rm_states: np.ndarray,
                       linewidth: float = 2.0, alpha: float = 0.85):
    """Draw path segments coloured by RM state."""
    if len(positions) < 2:
        return
    # Build segments for LineCollection
    points = positions[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [to_rgba(RM_COLORS[int(s) % len(RM_COLORS)], alpha) for s in rm_states[:-1]]
    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)


# ── 2-D trajectory plot (FlatWorld / Zones) ─────────────────────────────────

def plot_2d_trajectories(
    episodes: List[dict],
    meta: dict,
    title: str = "",
    max_per_row: int = 4,
) -> plt.Figure:
    """Grid of 2-D trajectory subplots, one per episode."""
    n = len(episodes)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    cols = min(n, max_per_row)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)

    env_type = meta.get("env_type", "")
    is_zones = "zones" in env_type

    for idx, ep in enumerate(episodes):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        if is_zones:
            _setup_2d_axis(ax, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))
        else:
            _setup_2d_axis(ax)

        draw_regions(ax, meta)
        pos = ep["pos"]
        rm = ep["rm"]

        if pos.shape[1] >= 2:
            draw_start_marker(ax, pos[0])
            draw_coloured_path(ax, pos, rm, linewidth=2.5)
            # End marker
            ax.plot(pos[-1, 0], pos[-1, 1], "x", color="black", markersize=8,
                    markeredgewidth=2, zorder=11)

        # Determine if goal was reached (RM state 1 = accept)
        reached = np.any(rm == 1)
        status = "✓ Goal" if reached else "✗ Timeout"
        ax.set_title(f"Ep {idx}  ({status}, {len(pos)} steps)", fontsize=9)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # RM state legend
    unique_states = sorted(set(int(s) for ep in episodes for s in ep["rm"]))
    handles = [mpatches.Patch(color=RM_COLORS[s % len(RM_COLORS)], label=f"RM state {s}")
               for s in unique_states]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               fontsize=8, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── 1-D trajectory plot (HalfCheetah) ──────────────────────────────────────

def plot_1d_trajectories(
    episodes: List[dict],
    meta: dict,
    title: str = "",
    max_per_row: int = 4,
) -> plt.Figure:
    """Plot x-position over time with threshold lines."""
    n = len(episodes)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    cols = min(n, max_per_row)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    a_thresh = meta.get("a_threshold", 5.0)
    b_thresh = meta.get("b_threshold", -2.0)

    for idx, ep in enumerate(episodes):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        pos = ep["pos"].ravel()
        rm = ep["rm"]
        T = len(pos)
        t = np.arange(T)

        # Colour segments by RM state
        for seg_start in range(T - 1):
            color = RM_COLORS[int(rm[seg_start]) % len(RM_COLORS)]
            ax.plot(t[seg_start:seg_start + 2], pos[seg_start:seg_start + 2],
                    color=color, linewidth=1.8)
        # Threshold lines
        ax.axhline(a_thresh, color="#d32f2f", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"a={a_thresh}")
        ax.axhline(b_thresh, color="#1565c0", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"b={b_thresh}")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

        reached = np.any(rm == 1)
        status = "✓ Goal" if reached else "✗"
        ax.set_title(f"Ep {idx}  ({status}, {T} steps)", fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("x position", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


# ── high-level entry points ─────────────────────────────────────────────────

def plot_run(run_dir: str, label: str = "", max_eps: int = 8, out: str = None,
             fmt: str = "png", dpi: int = 300) -> Optional[str]:
    """Plot trajectories from a single run. Returns saved path or None."""
    traj_dir = os.path.join(run_dir, "trajectories")
    if not os.path.isdir(traj_dir):
        return None
    episodes = load_trajectories(traj_dir, max_eps=max_eps)
    if not episodes:
        return None
    meta = load_env_meta(traj_dir)
    env_type = meta.get("env_type", "")
    # Infer env_type from directory name if metadata is missing
    if not env_type:
        basename = os.path.basename(run_dir)
        for candidate in ("flatworld_patrol", "flatworld_sequence",
                           "zones_patrol", "zones_sequence", "patrol"):
            if candidate in basename:
                env_type = candidate
                meta["env_type"] = env_type
                break

    title = label or os.path.basename(run_dir)

    is_2d = "flatworld" in env_type or "zones" in env_type
    # Also infer from position dimensionality as fallback
    if not env_type and episodes:
        is_2d = episodes[0]["pos"].ndim == 2 and episodes[0]["pos"].shape[1] >= 2

    if is_2d:
        fig = plot_2d_trajectories(episodes, meta, title=title)
    else:
        fig = plot_1d_trajectories(episodes, meta, title=title)

    if out is None:
        out = os.path.join(run_dir, f"trajectories.{fmt}")
    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_comparison(
    run_dirs: List[str],
    labels: List[str],
    ep_idx: int = 0,
    out: str = "trajectory_comparison.png",
    fmt: str = "png",
    dpi: int = 300,
) -> Optional[str]:
    """Overlay one episode from each run on a single figure for comparison."""
    all_eps = []
    all_labels = []
    meta = {}
    for rd, lab in zip(run_dirs, labels):
        traj_dir = os.path.join(rd, "trajectories")
        eps = load_trajectories(traj_dir, max_eps=ep_idx + 1)
        if eps and ep_idx < len(eps):
            all_eps.append(eps[ep_idx])
            all_labels.append(lab)
            if not meta:
                meta = load_env_meta(traj_dir)
    if not all_eps:
        return None

    env_type = meta.get("env_type", "")
    is_2d = "flatworld" in env_type or "zones" in env_type

    if is_2d:
        fig, ax = plt.subplots(figsize=(7, 7))
        is_zones = "zones" in env_type
        _setup_2d_axis(ax,
                       xlim=(-3.2, 3.2) if is_zones else (-2.2, 2.2),
                       ylim=(-3.2, 3.2) if is_zones else (-2.2, 2.2))
        draw_regions(ax, meta)
        path_colors = ["#2196f3", "#f44336", "#4caf50", "#ff9800", "#9c27b0", "#00bcd4"]
        for i, (ep, lab) in enumerate(zip(all_eps, all_labels)):
            pos = ep["pos"]
            col = path_colors[i % len(path_colors)]
            if pos.shape[1] >= 2:
                ax.plot(pos[:, 0], pos[:, 1], color=col, linewidth=2.2, alpha=0.8, label=lab)
                draw_start_marker(ax, pos[0])
        ax.legend(fontsize=9)
        ax.set_title(f"Trajectory comparison (episode {ep_idx})", fontsize=12, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        a_thresh = meta.get("a_threshold", 5.0)
        b_thresh = meta.get("b_threshold", -2.0)
        path_colors = ["#2196f3", "#f44336", "#4caf50", "#ff9800", "#9c27b0", "#00bcd4"]
        for i, (ep, lab) in enumerate(zip(all_eps, all_labels)):
            pos = ep["pos"].ravel()
            col = path_colors[i % len(path_colors)]
            ax.plot(pos, color=col, linewidth=1.8, alpha=0.8, label=lab)
        ax.axhline(a_thresh, color="#d32f2f", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(b_thresh, color="#1565c0", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("x position")
        ax.legend(fontsize=9)
        ax.set_title(f"Trajectory comparison (episode {ep_idx})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── auto-discover runs for an environment ───────────────────────────────────

_RUN_RE = re.compile(
    r"^(?P<prefix>\w+?)_(?P<env>patrol|flatworld_patrol|flatworld_sequence|zones_patrol|zones_sequence)"
    r"_(?P<method>td3_\w+?)_s(?P<seed>\d+)$"
)


def discover_runs(logs_root: str, env_name: str, prefix: str = "bench") -> Dict[str, List[str]]:
    """Return ``{method: [run_dir, ...]}`` for a given environment."""
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


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, default=None, help="Single run directory")
    ap.add_argument("--runs", nargs="+", default=None, help="Multiple run dirs for comparison")
    ap.add_argument("--labels", nargs="+", default=None, help="Labels for --runs")
    ap.add_argument("--env", type=str, default=None,
                    help="Auto-discover runs for this env (e.g. flatworld_patrol)")
    ap.add_argument("--logs_root", default="logs")
    ap.add_argument("--prefix", default="bench")
    ap.add_argument("--max_eps", type=int, default=8, help="Max episodes to show per run")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out_dir", default=None, help="Output directory for auto-discover mode")
    args = ap.parse_args()

    if args.run:
        # Single run
        out = plot_run(args.run, max_eps=args.max_eps, fmt=args.format, dpi=args.dpi)
        if out:
            print(f"  Saved {out}")
        else:
            print(f"  No trajectory data in {args.run}")

    elif args.runs:
        # Comparison overlay
        labels = args.labels or [os.path.basename(d) for d in args.runs]
        if len(labels) < len(args.runs):
            labels += [os.path.basename(d) for d in args.runs[len(labels):]]
        out_dir = args.out_dir or os.path.join(args.logs_root, "plots")
        os.makedirs(out_dir, exist_ok=True)

        # Individual trajectory grids
        for rd, lab in zip(args.runs, labels):
            out = plot_run(rd, label=lab, max_eps=args.max_eps,
                           out=os.path.join(out_dir, f"traj_{os.path.basename(rd)}.{args.format}"),
                           fmt=args.format, dpi=args.dpi)
            if out:
                print(f"  Saved {out}")

        # Comparison figure
        out = plot_comparison(
            args.runs, labels,
            out=os.path.join(out_dir, f"traj_comparison.{args.format}"),
            fmt=args.format, dpi=args.dpi,
        )
        if out:
            print(f"  Saved {out}")

    elif args.env:
        # Auto-discover all runs for this environment
        logs_root = os.path.abspath(args.logs_root)
        out_dir = args.out_dir or os.path.join(logs_root, "plots")
        os.makedirs(out_dir, exist_ok=True)
        methods = discover_runs(logs_root, args.env, prefix=args.prefix)
        if not methods:
            raise SystemExit(f"No runs found for env '{args.env}' in {logs_root}")

        all_dirs = []
        all_labels = []
        for method, dirs in sorted(methods.items()):
            for rd in dirs:
                label = METHOD_LABELS.get(method, method)
                seed = os.path.basename(rd).split("_s")[-1]
                lab = f"{label} (s{seed})"
                out = plot_run(
                    rd, label=lab, max_eps=args.max_eps,
                    out=os.path.join(out_dir, f"traj_{os.path.basename(rd)}.{args.format}"),
                    fmt=args.format, dpi=args.dpi,
                )
                if out:
                    print(f"  Saved {out}")
                    all_dirs.append(rd)
                    all_labels.append(lab)

        # Comparison: pick first seed of each method
        comp_dirs = []
        comp_labels = []
        for method, dirs in sorted(methods.items()):
            if dirs:
                comp_dirs.append(dirs[0])
                comp_labels.append(METHOD_LABELS.get(method, method))
        if len(comp_dirs) > 1:
            out = plot_comparison(
                comp_dirs, comp_labels,
                out=os.path.join(out_dir, f"traj_{args.env}_compare.{args.format}"),
                fmt=args.format, dpi=args.dpi,
            )
            if out:
                print(f"  Saved {out}")

        print(f"\nDone — trajectory plots for '{args.env}' saved to {out_dir}/")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
