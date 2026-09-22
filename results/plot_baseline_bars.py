"""Plot policy self-play behind cross-play with the simulated human H0.

The source of truth is ``outputs/figures/h0_partner_cells.csv``.  For each
learned policy, the wide hatched bar is its ordinary self-play return and the
narrow solid bar in front is the return obtained when that policy partners H0.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "outputs/figures/h0_partner_cells.csv"

LAYOUTS = (
    "split_0",
    "split_1",
    "outage_0",
    "outage_1",
    "distance_0",
    "distance_1",
)
DISPLAY_LAYOUTS = {
    "split_0": "Partition-0",
    "split_1": "Partition-1",
    "outage_0": "Blackout-0",
    "outage_1": "Blackout-1",
    "distance_0": "Inversion-0",
    "distance_1": "Inversion-1",
}
METHODS = ("IPPO-0", "IPPO-RNN", "FCP")
CSV_PARTNERS = {"IPPO-0": "cnn", "IPPO-RNN": "rnn", "FCP": "fcp"}
COLORS = {"IPPO-0": "#4878a8", "IPPO-RNN": "#3f9b52", "FCP": "#9068b0"}
H0_COLOR = "#d1495b"


def load_returns(path: Path = SOURCE_CSV):
    """Load learned-policy returns and H0 self-play from the source table."""
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream) :
            if row["human"] != "h0":
                continue
            if row["partner"] == "h0":
                rows[(row["layout"], "h0")] = float(row["return_mean"])
            elif row["partner"] in CSV_PARTNERS.values():
                rows[(row["layout"], row["partner"])] = (
                    float(row["partner_self_play"]),
                    float(row["return_mean"]),
                )

    missing = [
        f"{layout}/{CSV_PARTNERS[method]}"
        for layout in LAYOUTS
        for method in METHODS
        if (layout, CSV_PARTNERS[method]) not in rows
    ]
    missing.extend(
        f"{layout}/h0" for layout in LAYOUTS if (layout, "h0") not in rows
    )
    if missing:
        raise ValueError(f"Missing H0 partner rows in {path}: {', '.join(missing)}")
    return rows


def draw(axes, returns):
    x = np.arange(len(LAYOUTS))
    width = 0.235
    offsets = np.array([-width, 0.0, width])

    for method, offset in zip(METHODS, offsets):
        partner = CSV_PARTNERS[method]
        positions = x + offset
        self_play = [returns[(layout, partner)][0] for layout in LAYOUTS]
        h0_crossplay = [returns[(layout, partner)][1] for layout in LAYOUTS]
        color = COLORS[method]

        # SP is always the wide rear bar; H0 cross-play is always in front.
        axes.bar(
            positions,
            self_play,
            width=width * 0.88,
            color="white",
            edgecolor=color,
            linewidth=0.8,
            hatch="////",
            zorder=1,
        )
        axes.bar(
            positions,
            h0_crossplay,
            width=width * 0.58,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.94,
            zorder=3,
        )

    for index, layout in enumerate(LAYOUTS):
        axes.hlines(
            returns[(layout, "h0")],
            index - 1.5 * width,
            index + 1.5 * width,
            color=H0_COLOR,
            linewidth=1.4,
            zorder=5,
        )

    axes.set_title("Episode return", loc="left", fontweight="bold", fontsize=9.0, pad=7)
    axes.set_xticks(
        x,
        [DISPLAY_LAYOUTS[layout] for layout in LAYOUTS],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    axes.set_ylim(0, 600)
    axes.set_ylabel("Return (higher is better)", fontsize=7.2)
    axes.tick_params(axis="both", labelsize=6.9, length=2.5)
    axes.grid(axis="y", color="#DDE2E7", linewidth=0.65)
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color("#98A1AA")


def main():
    returns = load_returns()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Libertinus Serif", "Linux Libertine O", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match the width of one panel in the former three-panel figure.
    figure, axes = plt.subplots(figsize=(7.15 / 3, 2.45))
    draw(axes, returns)

    method_handles = [
        Patch(facecolor=COLORS[method], edgecolor="none", label=method)
        for method in METHODS
    ]
    condition_handles = [
        Patch(facecolor="white", edgecolor="#626A73", hatch="////", label="SP"),
        Patch(facecolor="#626A73", edgecolor="white", label="XP + H0"),
        Line2D([0], [0], color=H0_COLOR, linewidth=1.4, label="H0 SP"),
    ]
    # Matplotlib fills a two-row legend by columns. Interleave the handles so
    # the methods occupy row one and the two evaluation conditions row two.
    legend_handles = [
        method_handles[0],
        condition_handles[0],
        method_handles[1],
        condition_handles[1],
        method_handles[2],
        condition_handles[2],
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=3,
        frameon=False,
        fontsize=5.6,
        handlelength=1.1,
        columnspacing=0.65,
    )
    figure.subplots_adjust(left=0.18, right=0.99, top=0.69, bottom=0.24)
    figure.savefig(OUT_DIR / "baseline_performance_bars.pdf")
    figure.savefig(OUT_DIR / "baseline_performance_bars.png", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    main()
