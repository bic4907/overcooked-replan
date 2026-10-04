"""Render advance-notice intervention endpoint rates (paper Figure 7).

The paper's figure script, reading the records summarize_probe.py writes:
one panel per scenario, one contrast per baseline, the two kitchens of a
scenario averaged within each training seed.

    python experiment/notice/plot_alert_preparation.py --endpoint initiation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "outputs/notice/alert_preparation_data.json"
OUT_DIR = ROOT / "results"
DATA = []
COLORS = {"OFF": "#AAB4C2", "ON": "#087F8C"}
#: Optional, for figures that colour the bars by baseline instead: a colour per
#: method, with the masked arm drawn as that colour's hatching. Empty keeps
#: the paper bar plot's two arm colours.
METHOD_COLORS: dict = {}
MASKED_HATCH = "//"
#: Optional light tint per method for the masked arm, drawn instead of the
#: hatching where a method has one.
MASKED_TINTS: dict = {}
#: Lift the second of two equal-height values clear of the first. Off where
#: the labels fit side by side, so a pair of empty bars reads level.
LIFT_TIES = True
#: The outline of a tinted bar pair; "none" for no outline.
TINT_EDGE = "none"
#: The rule marking how much of what was begun was carried through.
COMPLETED = "#D1495B"
#: Two ways to draw this figure. "paper" is what goes in the manuscript --
#: bars, intervals and values, nothing else. "working" keeps the reading
#: aids: every seed as a dot, the endpoint sentence, and the paired
#: difference with its p-value under each contrast.
STYLES = {
    "paper": dict(dots=False, stats=False, subtitle=False, panel_letters=True,
                  variant_in_title=False, overlay=False, ylabel="Endpoint rate (%)",
                  scope="family",
                  font="times", figsize=(7.15, 2.4), bottom=0.14,
                  sizes={"title": 9.0, "tick": 8.0, "value": 7.2, "stat": 7.2,
                         "axis": 8.5, "legend": 8.5}),
    "working": dict(dots=True, stats=True, subtitle=True, panel_letters=True,
                    variant_in_title=True, overlay=False, scope="kitchen",
                    ylabel="Endpoint rate (% of fixed probes)",
                    font="sans", figsize=(9.2, 2.95), bottom=0.30,
                    sizes={"title": 8.7, "tick": 7.0, "value": 6.6, "stat": 6.4,
                           "axis": 7.0, "legend": 7.1}),
}
STYLE = dict(STYLES["paper"])
#: Point sizes, filled in by main() once the style and --scale are known.
SIZES = dict(STYLES["paper"]["sizes"])
SUBTITLES = {
    "initiation": {
        "Split": "Crosses before closure and remains separated",
        "Outage": "Picks up a plate before the dispensers vanish",
        "Distance": "Waits where the supply is about to open",
    },
    "completion": {
        "Split": "Crosses before closure and remains separated",
        "Outage": "Stores the plate and it survives the boundary",
        "Distance": "…and draws from it once it opens",
    },
}
#: Which seat is the informed agent per scenario. In the Inversion kitchens
#: seat 1 is the supplier, whose switch to plate work the paper's endpoint reads.
SEATS = {"Split": 0, "Outage": 0, "Distance": 1}
#: Which kitchen of each family the figure shows, as the paper reports one
#: kitchen per scenario. Overridden by --variants.
VARIANTS = {"Split": "0", "Outage": "0", "Distance": "1"}
DISPLAY_NAMES = {
    "Split": "Partition",
    "Outage": "Blackout",
    "Distance": "Inversion",
}


def select(scenario, method, endpoint, scope="kitchen", variant="0", seat=None):
    """One contrast.

    The paper's figure is one kitchen per scenario -- its 1,800 pairs are
    100 states x 6 seeds x 3 partner streams -- so that is the default.
    ``scope="family"`` instead takes the scenario's two kitchens averaged
    within each training seed, which doubles the pairs.
    """
    matches = [
        item
        for item in DATA
        if item["scenario"] == scenario
        and item["algorithm"] == method
        and item["endpoint"] == endpoint
        and item.get("scope") == scope
        and (scope == "family" or item["layout"].endswith(f"_{variant}"))
    ]
    if not matches:
        raise SystemExit(f"no record for {scenario} / {method} / {endpoint} / {scope}")
    # The scenario's own seat where the arm has it: the alert-A policies only
    # ever saw a warning in seat 0, so they are only probed there.
    wanted = SEATS[scenario] if seat is None else seat
    preferred = [item for item in matches if item["informed_seat"] == wanted]
    return (preferred or matches)[0]


def darker(colour, factor=0.62):
    """The same hue, further down -- the completed endpoint inside its own bar."""
    red, green, blue = matplotlib.colors.to_rgb(colour)
    return (red * factor, green * factor, blue * factor)


def draw_panel(ax, scenario, panel_index, endpoint, ymax, scope, variant, methods, seat) -> None:
    for method_index, method in enumerate(methods):
        row = select(scenario, method, endpoint, scope, variant, seat)
        # Completion is a subset of initiation -- the agent that stored the
        # plate is one that took it -- so it goes inside the same bar rather
        # than in a figure of its own.
        inner = (
            select(scenario, method, "completion", scope, variant, seat)
            if STYLE["overlay"] and endpoint == "initiation"
            else None
        )
        # Partition's endpoint is already a completed one -- crossing and
        # still being apart at the boundary -- so its rule would sit on the
        # top of every bar and say nothing.
        if inner is not None and all(inner[arm] == row[arm] for arm in ("OFF", "ON")):
            inner = None
        tops = []
        for arm_index, (arm, offset) in enumerate((("OFF", -0.18), ("ON", 0.18))):
            mean = 100 * row[arm]
            lo, hi = 100 * np.asarray(row[f"{arm}_ci95"])
            x = method_index + offset
            label = ("Notice masked" if arm == "OFF" else "Notice available") if method_index == 0 else None
            if method in METHOD_COLORS:
                colour = METHOD_COLORS[method]
                if arm == "OFF" and method in MASKED_TINTS:
                    ax.bar(x, mean, 0.31, facecolor=MASKED_TINTS[method], edgecolor=TINT_EDGE,
                           linewidth=0.6, label=label, zorder=2)
                elif arm == "OFF":
                    # The method's colour with sparse white hatching; the white
                    # is the hatch's colour, the outline itself not drawn.
                    ax.bar(x, mean, 0.31, facecolor=colour, edgecolor="white", hatch=MASKED_HATCH,
                           linewidth=0, label=label, zorder=2)
                else:
                    edge = TINT_EDGE if method in MASKED_TINTS else colour
                    width = 0.6 if method in MASKED_TINTS else 0.9
                    ax.bar(x, mean, 0.31, facecolor=colour, edgecolor=edge, linewidth=width,
                           label=label, zorder=2)
            else:
                ax.bar(x, mean, 0.31, color=COLORS[arm], label=label, zorder=2)
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                color="#243247",
                linewidth=0.9,
                capsize=2.5,
                zorder=6,  # in front of the completed rule
            )
            if inner is not None:
                # How much of what was begun was carried through: a rule
                # drawn across the bar at the completed rate. One bar, one
                # line, and the gap between the line and the top is what the
                # agent started and did not finish.
                ax.hlines(
                    100 * inner[arm],
                    x - 0.155,
                    x + 0.155,
                    color=COMPLETED,
                    linewidth=1.5,
                    linestyles=(0, (2.6, 1.3)),
                    zorder=5,
                )
            top = max(hi, mean)
            if STYLE["dots"]:
                seed_values = 100 * np.asarray(row["seed_rates"])[:, arm_index]
                ax.scatter(
                    x + np.linspace(-0.06, 0.06, len(seed_values)),
                    seed_values,
                    s=10,
                    facecolor="white",
                    edgecolor="#465569",
                    linewidth=0.75,
                    zorder=5,
                )
                top = max(top, seed_values.max())
            tops.append((x, top, mean))

        # Two bars of the same height put their values on top of each other,
        # which is what a pair of empty ones does; lift the second clear.
        (x_off, top_off, mean_off), (x_on, top_on, mean_on) = tops
        lift = 0.075 * ymax if LIFT_TIES and abs(top_on - top_off) < 0.07 * ymax else 0.0
        for x, top, mean, extra in (
            (x_off, top_off, mean_off, 0.0), (x_on, top_on, mean_on, lift)
        ):
            ax.text(
                x,
                top + 0.045 * ymax + extra,
                f"{mean:.2f}%",
                ha="center",
                fontsize=SIZES["value"],
                fontweight="bold",
                color="#182338",
            )

        if STYLE["stats"]:
            p_text = f"{row['p_raw']:.5f}" if row["p_raw"] < 1 else "1.00000"
            ax.text(
                method_index,
                -0.17,
                f"$\\Delta$ {100 * row['delta']:+.2f} pp\n$p$ = {p_text}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=SIZES["stat"],
                color="#243247",
            )

    letter = f"({chr(97 + panel_index)}) " if STYLE["panel_letters"] else ""
    ax.set_title(
        letter + DISPLAY_NAMES[scenario]
        + ("" if scope == "family" or not STYLE["variant_in_title"] else f"-{variant}"),
        fontsize=SIZES["title"],
        fontweight="bold",
        pad=26 if STYLE["subtitle"] else 10,
        loc="left" if STYLE["panel_letters"] else "center",
    )
    if STYLE["subtitle"]:
        ax.text(
            0.5, 1.04, SUBTITLES[endpoint][scenario], transform=ax.transAxes,
            ha="center", fontsize=SIZES["stat"], color="#49576A",
        )
    ax.set_xticks(range(len(methods)), methods, fontsize=SIZES["tick"])
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1, 10))
    ax.grid(axis="y", color="#E5E9EF", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA6B5")
    ax.tick_params(length=3, color="#9AA6B5", labelsize=SIZES["tick"], pad=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="initiation", choices=("initiation", "completion"))
    parser.add_argument(
        "--scope", default=None, choices=("kitchen", "family"),
        help="kitchen: one kitchen per scenario, as the paper's 1,800-pair contrasts are; "
        "family: the scenario's two kitchens averaged within each seed.",
    )
    parser.add_argument(
        "--variants", default="0,0,1",
        help="Kitchen per scenario, Partition,Blackout,Inversion; used when --scope kitchen.",
    )
    parser.add_argument(
        "--seat", type=int, default=None,
        help="Force one informed seat for every baseline; by default each scenario's own "
        "(seat 1 for Inversion), falling back to seat 0 where an arm has no run in it.",
    )
    parser.add_argument(
        "--methods", default="IPPO-RNN,FCP",
        help="Baselines to draw, comma separated; the paper's figure holds the two recurrent ones.",
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--style", default=None, choices=("paper", "working"),
        help="paper: the manuscript figure; working: dots, endpoint sentence and statistics. "
        "By default, paper when --out-dir ends in 'paper' and working otherwise.",
    )
    parser.add_argument("--font", default=None, help="times, sans (Pretendard), serif (Libertine), or a family name.")
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply every font size.")
    args = parser.parse_args()
    style = args.style or ("paper" if Path(args.out_dir).name == "paper" else "working")
    STYLE.clear()
    STYLE.update(STYLES[style])
    SIZES.clear()
    SIZES.update(STYLES[style]["sizes"])
    font = args.font or STYLE["font"]
    scope = args.scope or STYLE["scope"]
    DATA.extend(json.loads(Path(args.data).read_text()))
    for scenario, variant in zip(("Split", "Outage", "Distance"), args.variants.split(",")):
        VARIANTS[scenario] = variant.strip()
    tops = [
        100 * max(
            max(item["ON_ci95"][1], item["OFF_ci95"][1]),
            max(max(r) for r in item["seed_rates"]) if STYLE["dots"] else max(item["ON"], item["OFF"]),
        )
        for scenario in SUBTITLES[args.endpoint]
        for method in args.methods.split(",")
        for item in [select(scenario, method, args.endpoint, scope, VARIANTS[scenario], args.seat)]
    ]
    pad = 9 if STYLE["dots"] else 6
    ymax = float(10 * np.ceil((max(tops) + pad) / 10))
    families = {
        "times": ("serif", ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]),
        "sans": ("sans-serif", ["Pretendard", "Nimbus Sans", "DejaVu Sans"]),
        "serif": ("serif", ["Linux Libertine O", "Libertinus Serif", "Nimbus Roman", "DejaVu Serif"]),
    }
    family, names = families.get(font, ("serif", [font, "DejaVu Serif"]))
    for key in SIZES:
        SIZES[key] *= args.scale
    plt.rcParams.update(
        {
            "font.family": family,
            f"font.{'serif' if family == 'serif' else 'sans-serif'}": names,
            "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
            "font.size": (8.0 if style == "paper" else 7.1) * args.scale,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharey=True)
    for index, (axis, scenario) in enumerate(
        zip(axes, ("Split", "Outage", "Distance"))
    ):
        draw_panel(axis, scenario, index, args.endpoint, ymax, scope, VARIANTS[scenario], args.methods.split(","), args.seat)
    axes[0].set_ylabel(STYLE["ylabel"], labelpad=8, fontsize=SIZES["axis"])
    handles, labels = axes[0].get_legend_handles_labels()
    if STYLE["overlay"] and args.endpoint == "initiation":
        handles = handles + [
            plt.Line2D([], [], color=COMPLETED, linewidth=1.5, linestyle=(0, (2.6, 1.3)),
                       label="Completed")
        ]
        labels = labels + ["Completed"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=SIZES["legend"],
    )
    fig.subplots_adjust(left=0.07, right=0.993, top=0.70, bottom=0.29, wspace=0.25)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The manuscript takes one bar plot; the working copies keep the
    # endpoint in the name so the two do not overwrite each other.
    stem = "alert_preparation_barplot" if style == "paper" else f"alert_preparation_{args.endpoint}"
    if style == "paper" and args.endpoint != "initiation":
        stem += f"_{args.endpoint}"
    if args.methods != "IPPO-RNN,FCP":
        stem += "_" + "-".join(m.split("-")[-1].lower() for m in args.methods.split(","))
    if args.seat is not None:
        stem += f"_seat{args.seat}"
    if style != "paper" and scope == "kitchen" and args.variants.replace(" ", "") != "0,0,1":
        stem += "_k" + args.variants.replace(",", "").replace(" ", "")
    if style != "paper" and scope == "family":
        stem += "_pooled"

    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    print(out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
