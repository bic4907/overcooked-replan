"""Plot the point estimates reported in Table 2 as paired SP--XP dumbbells."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


OUT_DIR = Path(__file__).resolve().parent

LAYOUTS = ["Split-0", "Split-1", "Outage-0", "Outage-1", "Distance-0", "Distance-1"]
METHODS = ["IPPO-0", "IPPO-RNN", "FCP"]
COLORS = {"IPPO-0": "#4C78A8", "IPPO-RNN": "#2A9D55", "FCP": "#D95F76"}
OFFSETS = {"IPPO-0": -0.22, "IPPO-RNN": 0.0, "FCP": 0.22}

# Values are rounded point estimates from the 2026-09-21 W&B evaluation runs:
# overcooked-v3-{ippo,ippo-rnn,fcp}-0921_eval. Undefined transition metrics
# are represented as NaN and omitted from the figure.
NAN = float("nan")
DATA = {
    "Episode return": {
        "Split-0": {"IPPO-0": (223.3, 162.0), "IPPO-RNN": (260.0, 220.0), "FCP": (236.7, 239.3)},
        "Split-1": {"IPPO-0": (350.0, 130.7), "IPPO-RNN": (353.3, 181.3), "FCP": (210.0, 288.0)},
        "Outage-0": {"IPPO-0": (253.3, 172.7), "IPPO-RNN": (390.0, 254.7), "FCP": (0.0, 0.0)},
        "Outage-1": {"IPPO-0": (153.3, 54.7), "IPPO-RNN": (146.7, 83.3), "FCP": (0.0, 0.0)},
        "Distance-0": {"IPPO-0": (163.3, 65.3), "IPPO-RNN": (193.3, 88.7), "FCP": (183.3, 180.7)},
        "Distance-1": {"IPPO-0": (300.0, 292.7), "IPPO-RNN": (466.7, 426.0), "FCP": (410.0, 413.3)},
    },
    "Drop": {
        "Split-0": {"IPPO-0": (0.399, 0.395), "IPPO-RNN": (0.263, 0.313), "FCP": (0.291, 0.308)},
        "Split-1": {"IPPO-0": (0.162, 0.677), "IPPO-RNN": (0.192, 0.534), "FCP": (0.190, 0.237)},
        "Outage-0": {"IPPO-0": (0.263, 0.452), "IPPO-RNN": (0.129, 0.381), "FCP": (NAN, NAN)},
        "Outage-1": {"IPPO-0": (0.230, 0.557), "IPPO-RNN": (0.218, 0.446), "FCP": (NAN, NAN)},
        "Distance-0": {"IPPO-0": (0.756, 0.886), "IPPO-RNN": (0.600, 0.656), "FCP": (0.272, 0.237)},
        "Distance-1": {"IPPO-0": (0.203, 0.231), "IPPO-RNN": (0.104, 0.198), "FCP": (0.143, 0.100)},
    },
    "Recovery (steps)": {
        "Split-0": {"IPPO-0": (38.0, 43.6), "IPPO-RNN": (31.4, 34.9), "FCP": (33.3, 31.8)},
        "Split-1": {"IPPO-0": (35.5, 58.4), "IPPO-RNN": (36.2, 54.8), "FCP": (36.8, 42.5)},
        "Outage-0": {"IPPO-0": (50.5, 52.3), "IPPO-RNN": (41.0, 51.2), "FCP": (NAN, NAN)},
        "Outage-1": {"IPPO-0": (39.2, 50.3), "IPPO-RNN": (39.7, 50.7), "FCP": (NAN, NAN)},
        "Distance-0": {"IPPO-0": (61.2, 67.0), "IPPO-RNN": (52.9, 48.2), "FCP": (33.0, 32.0)},
        "Distance-1": {"IPPO-0": (35.2, 34.1), "IPPO-RNN": (31.8, 36.3), "FCP": (36.7, 35.0)},
    },
}


def draw_panel(ax, metric, panel_label):
    for idx, layout in enumerate(LAYOUTS):
        for method in METHODS:
            sp, xp = DATA[metric][layout][method]
            y = idx + OFFSETS[method]
            color = COLORS[method]
            ax.plot([sp, xp], [y, y], color=color, linewidth=1.35, alpha=0.72, zorder=1)
            ax.scatter(sp, y, s=25, facecolor="white", edgecolor=color, linewidth=1.25, zorder=3)
            ax.scatter(xp, y, s=28, marker="D", facecolor=color, edgecolor="white", linewidth=0.45, zorder=4)

    ax.set_title(f"{panel_label} {metric}", loc="left", fontweight="bold", fontsize=9.2, pad=7)
    ax.grid(axis="x", color="#DDE2E7", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(LAYOUTS) - 0.4)
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=7.4, length=2.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#98A1AA")
    ax.tick_params(axis="y", length=0)

    if metric == "Episode return":
        ax.set_xlim(0, 600)
        ax.set_xlabel("Higher is better", fontsize=7.2)
    elif metric == "Drop":
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("Lower is better", fontsize=7.2)
    else:
        ax.set_xlim(0, 155)
        ax.set_xlabel("Lower is better", fontsize=7.2)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "Linux Libertine O", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.05), sharey=True)
    for ax, metric, label in zip(
        axes,
        ["Episode return", "Drop", "Recovery (steps)"],
        ["(a)", "(b)", "(c)"],
    ):
        draw_panel(ax, metric, label)

    axes[0].set_yticks(range(len(LAYOUTS)), LAYOUTS)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)

    method_handles = [
        Line2D([0], [0], color=COLORS[m], linewidth=2, label=m) for m in METHODS
    ]
    condition_handles = [
        Line2D([0], [0], marker="o", markersize=5.4, markerfacecolor="white",
               markeredgecolor="#3F454B", linewidth=0, label="SP"),
        Line2D([0], [0], marker="D", markersize=5.0, markerfacecolor="#3F454B",
               markeredgecolor="white", linewidth=0, label="XP"),
    ]
    fig.legend(
        handles=method_handles + condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.015),
        ncol=5,
        frameon=False,
        fontsize=7.5,
        handlelength=1.7,
        columnspacing=1.2,
    )
    fig.text(
        0.5,
        0.01,
        "Lines connect self-play (SP) and cross-play (XP) point estimates for the same method and layout.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#505860",
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.83, bottom=0.19, wspace=0.18)
    fig.savefig(OUT_DIR / "baseline_performance_dumbbell.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "baseline_performance_dumbbell.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
