"""Render advance-notice intervention endpoint rates (paper Figure 7).

Copied from the paper's figure script; reads the records written by
summarize_probe.py and draws one panel per scenario for one endpoint and
one kitchen variant:

    python experiment/notice/plot_alert_preparation.py --endpoint initiation --variant 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "outputs/notice/alert_preparation_data.json"
OUT_DIR = ROOT / "results"
DATA = []
COLORS = {"OFF": "#AAB4C2", "ON": "#087F8C"}
SUBTITLES = {
    "initiation": {
        "Split": "Crosses before closure and remains separated",
        "Outage": "Acquires a fresh plate before dispensers vanish",
        "Distance": "Initiates plate work before the change",
    },
    "completion": {
        "Split": "Crosses before closure and remains separated",
        "Outage": "Stores the plate and it survives the boundary",
        "Distance": "Plate pickups dominate after the change as well",
    },
}
DISPLAY_NAMES = {
    "Split": "Partition",
    "Outage": "Blackout",
    "Distance": "Inversion",
}


#: Which seat is the informed agent per scenario. In the Inversion kitchens
#: seat 1 is the supplier, whose switch to plate work the paper's endpoint reads.
SEATS = {"Split": 0, "Outage": 0, "Distance": 1}


def select(scenario, method, endpoint, variant):
    return next(
        item
        for item in DATA
        if item["scenario"] == scenario
        and item["algorithm"] == method
        and item["endpoint"] == endpoint
        and item["layout"].endswith(f"_{variant}")
        and item["informed_seat"] == SEATS[scenario]
    )


def draw_panel(ax, scenario: str, panel_index: int, endpoint: str, variant: str, ymax: float) -> None:
    for method_index, method in enumerate(("IPPO-RNN", "FCP")):
        row = select(scenario, method, endpoint, variant)
        n_seeds = len(row["seed_rates"])
        for arm_index, (arm, offset) in enumerate((("OFF", -0.18), ("ON", 0.18))):
            mean = 100 * row[arm]
            lo, hi = 100 * np.asarray(row[f"{arm}_ci95"])
            seed_values = 100 * np.asarray(row["seed_rates"])[:, arm_index]
            x = method_index + offset
            ax.bar(
                x,
                mean,
                0.31,
                color=COLORS[arm],
                label=(
                    "Notice masked" if arm == "OFF" else "Notice available"
                ) if method_index == 0 else None,
                zorder=2,
            )
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                color="#243247",
                linewidth=0.9,
                capsize=2.5,
                zorder=3,
            )
            ax.scatter(
                x + np.linspace(-0.06, 0.06, n_seeds),
                seed_values,
                s=10,
                facecolor="white",
                edgecolor="#465569",
                linewidth=0.75,
                zorder=4,
            )
            ax.text(
                x,
                max(hi, seed_values.max()) + 1.4,
                f"{mean:.2f}%",
                ha="center",
                fontsize=6.6,
                fontweight="bold",
                color="#182338",
            )

        p_text = f"{row['p_raw']:.5f}" if row["p_raw"] < 1 else "1.00000"
        ax.text(
            method_index,
            -0.17,
            f"$\\Delta$ {100 * row['delta']:+.2f} pp\n$p$ = {p_text}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.4,
            color="#243247",
        )

    ax.set_title(
        f"({chr(97 + panel_index)}) {DISPLAY_NAMES[scenario]}-{variant} (A = seat {SEATS[scenario]})",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        pad=35,
    )
    ax.text(
        0.5,
        1.04,
        SUBTITLES[endpoint][scenario],
        transform=ax.transAxes,
        ha="center",
        fontsize=6.4,
        color="#49576A",
    )
    ax.set_xticks((0, 1), ("IPPO-RNN", "FCP"))
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1, 10))
    ax.grid(axis="y", color="#E5E9EF", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA6B5")
    ax.tick_params(length=3, color="#9AA6B5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="initiation", choices=("initiation", "completion"))
    parser.add_argument("--variant", default="0", help="Kitchen variant suffix: 0 or 1.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    DATA.extend(json.loads(Path(args.data).read_text()))
    tops = [
        100 * max(max(item["ON_ci95"][1], item["OFF_ci95"][1]), max(max(r) for r in item["seed_rates"]))
        for scenario in SUBTITLES[args.endpoint]
        for method in ("IPPO-RNN", "FCP")
        for item in [select(scenario, method, args.endpoint, args.variant)]
    ]
    ymax = float(10 * np.ceil((max(tops) + 8) / 10))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharey=True)
    for index, (axis, scenario) in enumerate(
        zip(axes, ("Split", "Outage", "Distance"))
    ):
        draw_panel(axis, scenario, index, args.endpoint, args.variant, ymax)
    axes[0].set_ylabel("Endpoint rate (% of fixed probes)", labelpad=6, fontsize=7.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7.1,
    )
    fig.subplots_adjust(left=0.07, right=0.993, top=0.70, bottom=0.29, wspace=0.25)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"alert_preparation_{args.endpoint}_{args.variant}"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    print(out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
