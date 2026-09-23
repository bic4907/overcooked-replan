"""Render the Section 6.3 result (paper Figure 8): the uninformed partner's
behavioural divergence against evaluation Drop and against episode return,
one point per kitchen and arm, the no-notice control joined to the
asymmetric-notice arm.

    python experiment/notice/plot_behavior_notice.py --arm rnn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"control": "#AAB4C2", "notice": "#087F8C"}
LABELS = {"control": "No alert", "notice": "A-only alert"}
DISPLAY = {
    "split_0": "Partition-0", "split_1": "Partition-1",
    "outage_0": "Blackout-0", "outage_1": "Blackout-1",
    "distance_0": "Inversion-0", "distance_1": "Inversion-1",
}


def panel(ax, layouts, metric, ylabel, invert, title):
    for name, s in layouts.items():
        x0, x1 = s["jsd_control"], s["jsd_notice"]
        y0, y1 = s[f"{metric}_control"], s[f"{metric}_notice"]
        if not all(np.isfinite(v) for v in (x0, x1, y0, y1)):
            continue
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#9AA6B5", lw=0.9, shrinkA=4, shrinkB=4),
            zorder=2,
        )
        ax.scatter(x0, y0, s=34, color=COLORS["control"], edgecolor="#465569", linewidth=0.6, zorder=3)
        ax.scatter(x1, y1, s=34, color=COLORS["notice"], edgecolor="#465569", linewidth=0.6, zorder=3)
        ax.text(x1, y1, "  " + DISPLAY.get(name, name), fontsize=6.4, color="#243247", va="center", ha="left", zorder=4)
    ax.set_xlabel("B behavior divergence (JSD)", fontsize=7.0)
    ax.set_ylabel(ylabel, fontsize=7.0)
    ax.set_title(title, loc="left", fontsize=8.7, fontweight="bold")
    if invert:
        ax.invert_yaxis()
    ax.grid(color="#E5E9EF", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA6B5")
    ax.tick_params(length=3, color="#9AA6B5", labelsize=6.6)
    ax.margins(x=0.25, y=0.15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="rnn")
    parser.add_argument("--in-dir", default=str(ROOT / "outputs/notice"))
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    summary = json.loads((Path(args.in_dir) / f"behavior_summary_{args.arm}.json").read_text())
    layouts = summary["layouts"]
    across = summary.get("across_layouts", {})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.1, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    panel(axes[0], layouts, "drop", "Evaluation drop ↓", True, "(a) B behavior and evaluation Drop")
    panel(axes[1], layouts, "return", "Episode return ↑", False, "(b) B behavior and episode return")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS[k], markeredgecolor="#465569", markersize=5.5, label=LABELS[k])
        for k in ("control", "notice")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.1)
    if across:
        fig.text(
            0.5, 0.01,
            f"Across layout means: r(ΔJSD, Drop reduction) = {across['r_delta_jsd_vs_drop_reduction']:.3f}, "
            f"exact permutation p = {across['p_perm_drop']:.3f};  "
            f"r(ΔJSD, return gain) = {across['r_delta_jsd_vs_return_gain']:.3f}, p = {across['p_perm_return']:.3f}",
            ha="center", fontsize=6.4, color="#49576A",
        )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.22, wspace=0.3)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"behavior_as_notice_{args.arm}"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    print(out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
