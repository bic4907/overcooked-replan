"""Plot Table 2 with XP bars and SP reference markers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from plot_baseline_performance import COLORS, DATA, LAYOUTS, METHODS


OUT_DIR = Path(__file__).resolve().parent
DISPLAY_LAYOUTS = {
    "Split-0": "Partition-0",
    "Split-1": "Partition-1",
    "Outage-0": "Blackout-0",
    "Outage-1": "Blackout-1",
    "Distance-0": "Inversion-0",
    "Distance-1": "Inversion-1",
}


def draw_panel(ax, metric, panel_label):
    x = np.arange(len(LAYOUTS))
    width = 0.235
    offsets = np.array([-width, 0.0, width])

    for method, offset in zip(METHODS, offsets):
        sp = np.array([DATA[metric][layout][method][0] for layout in LAYOUTS])
        xp = np.array([DATA[metric][layout][method][1] for layout in LAYOUTS])
        xpos = x + offset
        color = COLORS[method]

        for pos, sp_value, xp_value in zip(xpos, sp, xp):
            if metric == "Episode return" and sp_value == 0 and xp_value == 0:
                ax.text(
                    pos,
                    4.0,
                    "0",
                    color=color,
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                )
                continue
            if not np.isfinite(sp_value) and not np.isfinite(xp_value):
                ax.text(
                    pos,
                    0.02 if metric == "Drop" else 2.0,
                    "N/A",
                    color=color,
                    fontsize=5.0,
                    ha="center",
                    va="bottom",
                    rotation=90,
                )
                continue
            sp_style = {
                "height": sp_value,
                "color": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "hatch": "////",
            }
            xp_style = {
                "height": xp_value,
                "color": color,
                "edgecolor": "white",
                "linewidth": 0.35,
                "alpha": 0.94,
            }
            if sp_value >= xp_value:
                back, front = sp_style, xp_style
            else:
                back, front = xp_style, sp_style

            ax.bar(pos, width=width * 0.88, zorder=1, **back)
            ax.bar(pos, width=width * 0.58, zorder=3, **front)

    ax.set_title(f"{panel_label} {metric}", loc="left", fontweight="bold", fontsize=9.0, pad=7)
    ax.set_xticks(
        x,
        [DISPLAY_LAYOUTS[layout] for layout in LAYOUTS],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="both", labelsize=6.9, length=2.5)
    ax.grid(axis="y", color="#DDE2E7", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#98A1AA")

    if metric == "Episode return":
        ax.set_ylim(0, 600)
        ax.set_ylabel("Return (higher is better)", fontsize=7.2)
    elif metric == "Drop":
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel("Drop (lower is better)", fontsize=7.2)
    else:
        ax.set_ylim(0, 80)
        ax.set_ylabel("Steps (lower is better)", fontsize=7.2)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Libertinus Serif", "Linux Libertine O", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    for ax, metric, label in zip(
        axes,
        ["Episode return", "Drop", "Recovery (steps)"],
        ["(a)", "(b)", "(c)"],
    ):
        draw_panel(ax, metric, label)

    method_handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS]
    condition_handles = [
        Patch(facecolor="white", edgecolor="#626A73", hatch="////", label="SP"),
        Patch(facecolor="#626A73", edgecolor="white", label="XP"),
    ]
    fig.legend(
        handles=method_handles + condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=5,
        frameon=False,
        fontsize=7.3,
        handlelength=1.5,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.79, bottom=0.24, wspace=0.34)
    fig.savefig(OUT_DIR / "baseline_performance_bars.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "baseline_performance_bars.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
