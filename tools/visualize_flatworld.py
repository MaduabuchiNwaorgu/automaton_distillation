#!/usr/bin/env python3
"""Visualize FlatWorld teacher and student configurations for paper.

Uses the same rendering style as trajectory plots with semi-transparent circles,
colored edges, and wall obstacles.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import numpy as np
from pathlib import Path

# ── FlatWorld configurations ─────────────────────────────────────────────

CIRCLES_DEFAULT = [
    {"center": np.array([-1.4,  0.55]), "radius": 0.40, "color": "red"},
    {"center": np.array([-1.1,  1.10]), "radius": 0.50, "color": "magenta"},
    {"center": np.array([-1.0, -1.20]), "radius": 0.30, "color": "yellow"},
    {"center": np.array([-1.53,-0.50]), "radius": 0.32, "color": "orange"},
    {"center": np.array([ 0.1,  0.00]), "radius": 0.80, "color": "blue"},
    {"center": np.array([ 0.5, -1.30]), "radius": 0.35, "color": "red"},
    {"center": np.array([ 0.7,  0.70]), "radius": 0.50, "color": "green"},
    {"center": np.array([ 1.5, -0.75]), "radius": 0.40, "color": "green"},
    {"center": np.array([ 0.8,  0.20]), "radius": 0.30, "color": "cyan"},
]

CIRCLES_SHIFTED = [
    {"center": np.array([ 1.2,  1.10]), "radius": 0.35, "color": "red"},
    {"center": np.array([ 0.5, -1.40]), "radius": 0.45, "color": "magenta"},
    {"center": np.array([ 1.0,  0.80]), "radius": 0.30, "color": "yellow"},
    {"center": np.array([ 1.50, 0.20]), "radius": 0.30, "color": "orange"},
    {"center": np.array([-1.0, -0.60]), "radius": 0.70, "color": "blue"},
    {"center": np.array([-0.8,  1.20]), "radius": 0.35, "color": "red"},
    {"center": np.array([-1.3, -1.10]), "radius": 0.45, "color": "green"},
    {"center": np.array([-0.3,  0.50]), "radius": 0.40, "color": "green"},
    {"center": np.array([ 0.0, -0.30]), "radius": 0.25, "color": "cyan"},
]

WALLS_CROSS = [
    {"x_min": -0.08, "y_min": -0.80, "x_max": 0.08, "y_max": 0.80},   # vertical
    {"x_min": -0.80, "y_min": -0.08, "x_max": 0.80, "y_max": 0.08},   # horizontal
]

WALLS_CORRIDORS = [
    {"x_min": -2.0, "y_min":  0.40, "x_max": 0.8, "y_max": 0.52},   # upper wall
    {"x_min": -0.8, "y_min": -0.52, "x_max": 2.0, "y_max": -0.40},  # lower wall
]

# Color palette matching trajectory plots
ZONE_COLORS = {
    "red":     "#e53935",
    "blue":    "#1e88e5",
    "green":   "#43a047",
    "yellow":  "#fdd835",
    "magenta": "#ab47bc",
    "orange":  "#fb8c00",
    "cyan":    "#00acc1",
}


def draw_circle(ax, center, radius, color, alpha=0.35, label_alpha=1.0):
    """Draw a proposition circle matching trajectory plot style."""
    edge_color = ZONE_COLORS.get(color, color)
    face_color = to_rgba(edge_color, alpha)
    
    circle = patches.Circle(center, radius, 
                           fc=face_color, 
                           ec=edge_color, 
                           linewidth=1.5, 
                           zorder=2)
    ax.add_patch(circle)
    
    # Label with first letter
    label = color[0].upper()
    ax.text(center[0], center[1], label, 
           ha='center', va='center',
           fontsize=10, fontweight='bold', 
           color=edge_color, zorder=3)


def draw_wall(ax, wall, color="#555555", alpha=0.6):
    """Draw a wall rectangle matching trajectory plot style."""
    rect = patches.Rectangle(
        (wall["x_min"], wall["y_min"]),
        wall["x_max"] - wall["x_min"],
        wall["y_max"] - wall["y_min"],
        fc=to_rgba(color, alpha),
        ec=color,
        linewidth=1.2,
        zorder=1)
    ax.add_patch(rect)


def setup_2d_axis(ax):
    """Setup axis style matching trajectory plots."""
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect('equal')
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    
    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('X', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', fontsize=10, fontweight='bold')


def plot_transfer_pair(circles_teacher, circles_student, walls_teacher, walls_student, 
                       task_name, output_path):
    """Create side-by-side comparison of teacher and student using trajectory plot style."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Teacher (left)
    setup_2d_axis(axes[0])
    axes[0].set_title(f"Teacher\n(Source Task)", 
                     fontsize=12, fontweight='bold', pad=10)
    for circle in circles_teacher:
        draw_circle(axes[0], circle['center'], circle['radius'], circle['color'])
    for wall in walls_teacher:
        draw_wall(axes[0], wall)
    
    # Student (right)
    setup_2d_axis(axes[1])
    axes[1].set_title(f"Student\n(Target Task)", 
                     fontsize=12, fontweight='bold', pad=10)
    for circle in circles_student:
        draw_circle(axes[1], circle['center'], circle['radius'], circle['color'])
    for wall in walls_student:
        draw_wall(axes[1], wall)
    
    fig.suptitle(f'FlatWorld {task_name}: Teacher→Student Transfer', 
                fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2, w_pad=0.8)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    return fig


def main():
    output_dir = Path("env_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # FlatWorld Patrol: default circles + no walls → shifted circles + cross walls
    print("Generating FlatWorld Patrol visualization...")
    plot_transfer_pair(
        circles_teacher=CIRCLES_DEFAULT,
        circles_student=CIRCLES_SHIFTED,
        walls_teacher=[],  # Teacher has no walls
        walls_student=WALLS_CROSS,  # Student has cross-shaped obstacle
        task_name="Patrol",
        output_path=str(output_dir / "flatworld_patrol_transfer.pdf")
    )
    
    # FlatWorld Sequence: default circles + no walls → shifted circles + corridor walls
    print("\nGenerating FlatWorld Sequence visualization...")
    plot_transfer_pair(
        circles_teacher=CIRCLES_DEFAULT,
        circles_student=CIRCLES_SHIFTED,
        walls_teacher=[],  # Teacher has no walls
        walls_student=WALLS_CORRIDORS,  # Student has corridors
        task_name="Sequence",
        output_path=str(output_dir / "flatworld_sequence_transfer.pdf")
    )
    
    print("\nVisualizations complete!")


if __name__ == "__main__":
    main()
