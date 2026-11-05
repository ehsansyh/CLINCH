#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the CLINCH pipeline schematic used in the README.

Two rows. The upper row is how the interaction signal is built and learned, the lower row
is how molecules are produced from it and checked. The curved arrow between them is the
reward, which is the only channel through which the predictor reaches the generator, and
it is also the channel the in-distribution constraints exist to protect.

    python assets/make_schematic.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent

NAVY = "#003060"
TEAL = "#149094"
NAVY_FILL = "#EAF0F6"
TEAL_FILL = "#E3F2F2"
GREY = "#53606E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

# row, x centre, title, body lines, accent
BOXES = [
    (1, 1.55, "Docked pose ensemble",
     ["3,949 wild-type ligands", "45 poses each, 142,160 poses"], NAVY),
    (1, 4.75, "POSEPRINT",
     ["per-pose hydrogen bonds", "and hydrophobic contacts"], NAVY),
    (1, 7.95, "Distributional fingerprints",
     ["pose counts become a", "probability over two classes"], NAVY),
    (1, 11.15, "Interaction predictor",
     ["ligand graph plus ESM-2,", "supervised cross-attention"], NAVY),
    (0, 1.55, "Constrained policy",
     ["SMILES transformer, PPO,", "held on the active manifold"], TEAL),
    (0, 4.75, "Harmonic-mean reward",
     ["credit only when both", "interactions are secured"], TEAL),
    (0, 7.95, "Selection",
     ["43,943 generated, gate to", "15,955, NSGA-II to 500"], TEAL),
    (0, 11.15, "Validation",
     ["re-docking, 200 ns dynamics,", "MM/GBSA against darunavir"], TEAL),
]

BW, BH = 2.72, 1.30


def box(ax, cx, cy, title, lines, accent, fill):
    ax.add_patch(FancyBboxPatch(
        (cx - BW / 2, cy - BH / 2), BW, BH,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.5, edgecolor=accent, facecolor=fill, zorder=2))
    ax.plot([cx - BW / 2 + 0.07, cx - BW / 2 + 0.07],
            [cy - BH / 2 + 0.12, cy + BH / 2 - 0.12],
            color=accent, lw=3.2, solid_capstyle="round", zorder=3)
    ax.text(cx, cy + 0.33, title, ha="center", va="center",
            fontsize=10.2, fontweight="bold", color=accent, zorder=4)
    for i, line in enumerate(lines):
        ax.text(cx, cy - 0.02 - i * 0.30, line, ha="center", va="center",
                fontsize=8.1, color=GREY, zorder=4)


def arrow(ax, x0, y0, x1, y1, colour, rad=0.0, lw=1.7):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), connectionstyle="arc3,rad=%.3f" % rad,
        arrowstyle="-|>,head_width=4.2,head_length=7.5",
        linewidth=lw, color=colour, zorder=1,
        shrinkA=0, shrinkB=0, mutation_scale=1.0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_top, y_bot = 3.30, 1.00
    for row, cx, title, lines, accent in BOXES:
        cy = y_top if row == 1 else y_bot
        box(ax, cx, cy, title, lines, accent,
            NAVY_FILL if accent == NAVY else TEAL_FILL)

    # within-row flow
    xs = [1.55, 4.75, 7.95, 11.15]
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + BW / 2 + 0.06, y_top, b - BW / 2 - 0.06, y_top, NAVY)
        arrow(ax, a + BW / 2 + 0.06, y_bot, b - BW / 2 - 0.06, y_bot, TEAL)

    # the reward, routed as a bus through the band between the rows so that it
    # never runs along the edge of a box
    mid_y = (y_top - BH / 2 + y_bot + BH / 2) / 2 + 0.06
    ax.plot([11.15, 11.15], [y_top - BH / 2 - 0.04, mid_y],
            color=TEAL, lw=2.0, solid_capstyle="round", zorder=1)
    ax.plot([11.15, 1.55], [mid_y, mid_y],
            color=TEAL, lw=2.0, solid_capstyle="round", zorder=1)
    arrow(ax, 1.55, mid_y, 1.55, y_bot + BH / 2 + 0.04, TEAL, lw=2.0)
    ax.text(6.35, mid_y, "reward", ha="center", va="center", fontsize=10.2,
            fontweight="bold", color=TEAL,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="none"), zorder=5)
    ax.text(6.35, mid_y - 0.34,
            "the only channel from the predictor to the generator",
            ha="center", va="center", fontsize=8.0, color=GREY, zorder=5)

    # row labels
    ax.text(-0.26, y_top, "measure and learn", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=NAVY, rotation=90)
    ax.text(-0.26, y_bot, "generate and check", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=TEAL, rotation=90)

    ax.set_xlim(-0.70, 12.70)
    ax.set_ylim(0.20, 4.12)
    ax.axis("off")
    fig.tight_layout(pad=0.35)
    out = HERE / "clinch_schematic.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print("  wrote %s" % out.name)


if __name__ == "__main__":
    main()
