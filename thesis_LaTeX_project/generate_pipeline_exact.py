#!/usr/bin/env python3
"""
Causal Discovery Benchmarking Pipeline
Round shafts + proper arrowheads (robust, publication-ready)
"""

import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# =========================
# Matplotlib defaults
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.22,
})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# Geometry helpers
# =========================
def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (0.0, 0.0) if n == 0 else (dx / n, dy / n)

# =========================
# Shaft drawing
# =========================
def draw_shaft_line(ax, start, end, *, color, lw, dashed=False, z=10):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=z
    )

def draw_shaft_curve(ax, start, end, *, color, lw, rad=0.25, dashed=False, z=10):
    (x0, y0), (x1, y1) = start, end
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0

    ux, uy = _unit(-dy, dx)
    dist = math.hypot(dx, dy)
    cx, cy = mx + ux * (rad * dist), my + uy * (rad * dist)

    path = Path(
        [(x0, y0), (cx, cy), (x1, y1)],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    )

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        capstyle="round",
        joinstyle="round",
        zorder=z
    )
    if dashed:
        patch.set_linestyle((0, (4, 3)))
    ax.add_patch(patch)

    return cx, cy

# =========================
# Arrowhead (drawn separately)
# =========================
def draw_head(ax, start, end, *, color, scale, z):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=0,
            mutation_scale=scale,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=z
    )

# =========================
# Composite arrows
# =========================
def arrow_straight(ax, start, end, *, color, lw, dashed,
                   head_scale, head_len,
                   z_shaft=10, z_head=30):
    dx, dy = end[0] - start[0], end[1] - start[1]
    ux, uy = _unit(dx, dy)

    shaft_end = (end[0] - ux * head_len, end[1] - uy * head_len)
    draw_shaft_line(ax, start, shaft_end,
                    color=color, lw=lw, dashed=dashed, z=z_shaft)

    head_start = (end[0] - ux * head_len * 1.1,
                  end[1] - uy * head_len * 1.1)
    draw_head(ax, head_start, end,
              color=color, scale=head_scale, z=z_head)

def arrow_curve(ax, start, end, *, color, lw, dashed,
                rad, head_scale, head_len,
                z_shaft=10, z_head=30):
    cx, cy = draw_shaft_curve(
        ax, start, end,
        color=color, lw=lw, dashed=dashed, rad=rad, z=z_shaft
    )

    ux, uy = _unit(end[0] - cx, end[1] - cy)
    head_start = (end[0] - ux * head_len,
                  end[1] - uy * head_len)
    draw_head(ax, head_start, end,
              color=color, scale=head_scale, z=z_head)

# =========================
# Figure
# =========================
def create_pipeline_figure():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    y, w, h = 2.5, 1.6, 1.2

    stages = [
        (1.0,  "Benchmark\nNetworks",  "\nAsia, Sachs,\nAlarm, Child,\nInsurance", "#e8f5e9"),
        (3.2,  "Data\nGeneration",     "n samples\nper network",               "#e3f2fd"),
        (5.4,  "Causal\nDiscovery",    "\nPC, GES,\nNOTEARS,\nCOSMO",              "#fff9c4"),
        (7.6,  "Learned\nGraph",       "Estimated\nDAG",                        "#f3e5f5"),
        (9.8,  "Evaluation\nMetrics",  "\nF1, SHD,\nPrecision,\nRecall",           "#e0f7fa"),
        (12.0, "Results",              "Tables &\nFigures",                    "#fce4ec"),
    ]

    # Boxes
    for x, label, sub, color in stages:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="#333333",
            linewidth=2,
            zorder=5
        ))
        ax.text(x + w/2, y + h*0.65, label,
                ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=6)
        ax.text(x + w/2, y + h*0.28, sub,
                ha="center", va="center",
                fontsize=9, style="italic", zorder=6)

    # Main pipeline arrows
    gap = 0.14
    for i in range(len(stages) - 1):
        arrow_straight(
            ax,
            (stages[i][0] + w + gap, y + h/2),
            (stages[i+1][0] - gap, y + h/2),
            color="#333333",
            lw=2.8,
            dashed=False,
            head_scale=24,
            head_len=0.22
        )

    # Bootstrap box
    ax.add_patch(FancyBboxPatch(
        (2.5, 4.5), 2.2, 0.8,
        boxstyle="round,pad=0.08",
        facecolor="#bbdefb",
        edgecolor="#1976d2",
        linewidth=2,
        linestyle="--"
    ))
    ax.text(3.6, 4.9, "Bootstrap × k runs",
            ha="center", va="center",
            fontsize=10, style="italic", fontweight="bold", color="#0d47a1")

    # True graph box
    ax.add_patch(FancyBboxPatch(
        (7.8, 4.5), 2.2, 0.8,
        boxstyle="round,pad=0.08",
        facecolor="#c8e6c9",
        edgecolor="#388e3c",
        linewidth=2,
        linestyle="--"
    ))
    ax.text(8.9, 4.95, "True Graph\n(Ground Truth)",
            ha="center", va="center", fontsize=10, color="#1b5e20")

    # Sensitivity box
    ax.add_patch(FancyBboxPatch(
        (7.0, 0.5), 3.0, 1.0,
        boxstyle="round,pad=0.1",
        facecolor="#fff59d",
        edgecolor="#f57f17",
        linewidth=2
    ))
    ax.text(8.5, 1.05, "Sensitivity Analysis\n(Mis-specification)",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#827717")

    # Validation arrows
    arrow_straight(ax, (3.6, 4.45), (3.6, 3.75),
                   color="#1976d2", lw=2.2, dashed=True,
                   head_scale=20, head_len=0.20)

    arrow_curve(ax, (4.65, 4.45), (5.55, 3.78),
                color="#1976d2", lw=2.2, dashed=True,
                rad=0.28, head_scale=20, head_len=0.22)

    arrow_straight(ax, (8.85, 4.45), (8.35, 3.78),
                   color="#388e3c", lw=2.2, dashed=True,
                   head_scale=20, head_len=0.20)

    arrow_straight(ax, (8.40, 2.45), (8.40, 1.55),
                   color="#f57f17", lw=2.2, dashed=True,
                   head_scale=20, head_len=0.20)

    arrow_curve(ax, (10.00, 1.05), (12.00, 2.55),
                color="#f57f17", lw=2.2, dashed=True,
                rad=-0.22, head_scale=20, head_len=0.22)

    # Title + caption
    ax.text(7.0, 5.8, "Causal Discovery Benchmarking Pipeline",
            ha="center", va="center",
            fontsize=15, fontweight="bold")

    ax.text(7.0, 0.1,
            "Solid arrows: main pipeline flow • Dashed arrows: validation loops and comparisons",
            ha="center", va="center",
            fontsize=10, style="italic", color="#666666")

    # Save
    plt.savefig(os.path.join(FIGURES_DIR, "pipeline.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "pipeline.pdf"))
    plt.close(fig)

if __name__ == "__main__":
    create_pipeline_figure()
