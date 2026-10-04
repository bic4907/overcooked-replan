"""Render the Section 6.3 result (paper Figure 8): the uninformed partner's
behavioural divergence against evaluation Drop and against episode return.

One figure per panel, so the paper can place them as separate subfigures.
Colour is the kitchen, as the paper draws it; an open circle is the
no-notice control and a filled diamond the asymmetric-notice arm, joined by
an arrow. Whiskers are a 95% bootstrap interval over the informed training
seeds, on both axes.

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
#: The paper's per-kitchen colours, an Okabe-Ito set with a purple and a gold.
COLORS = {
    "split_0": "#0072B2",
    "split_1": "#D55E00",
    "outage_0": "#009E73",
    "outage_1": "#CC79A7",
    "distance_0": "#6F51A1",
    "distance_1": "#9B7500",
}
ORDER = ("distance_1", "distance_0", "split_0", "outage_1", "split_1", "outage_0")
MARKERS = {"control": "o", "notice": "D"}
LABELS = {"control": "No alert", "notice": "A-only alert"}
#: The manuscript's own words for the two arms, as its caption reads them.
#: "Notice masked" as in Figure 6: the same alert-trained policies with the
#: warning hidden from both, not policies trained without it.
PAPER_LABELS = {"control": "Notice masked", "notice": "One-sided notice"}
#: Where each panel's legend sits; the drop panel's bottom right corner is
#: the one part of it no kitchen reaches.
LEGEND_LOC = {"drop": "lower right", "return": "best"}
#: Two ways to draw this figure. "paper" is what goes in the manuscript: one
#: panel per file, Times, names where they were asked for, nothing else on
#: the canvas. "working" is the reading copy: both panels side by side with
#: the across-layout statistics under them and the names placed automatically.
STYLES = {
    "paper": dict(split=True, caption=False, pinned=True, titles=False,
                  font="times", normalize=True, legend=True, errorbars=False,
                  figsize=(4.8, 4.0),
                  sizes=dict(axis=19.0, title=16.5, tick=17.0, label=16.0, legend=17.0),
                  base=15.5),
    "working": dict(split=False, caption=True, pinned=False, titles=True,
                    font="sans", normalize=False, legend=True, errorbars=True,
                    figsize=(7.15, 2.75),
                    sizes=dict(axis=7.0, title=7.6, tick=6.4, label=6.4, legend=6.4),
                    base=7.1),
}
#: The best self-play return each kitchen reached under the alert-both
#: policies, the maximum over the ten-seed 09-22 evaluations of IPPO-RNN,
#: FCP and IPPO-0. Used to read return as a share of what the kitchen
#: affords rather than in points.
OPTIMAL = {
    "split_0": 260.0, "split_1": 356.0, "outage_0": 392.0,
    "outage_1": 150.0, "distance_0": 202.0, "distance_1": 460.0,
}
#: What each panel's file is called in the manuscript folder, numbered the
#: way the subfigures are ordered.
PAPER_STEMS = {
    "drop": "behavior_drop",
    "return": "behavior_return",
    "return_norm": "behavior_return",
}
PANELS = {
    "drop": ("Drop ↓", True, "B behavior and evaluation drop"),
    "return": ("Episode return ↑", False, "B behavior and episode return"),
    "return_norm": ("Normalized return (%) ↑", False, "B behavior and episode return"),
}
#: Radii and directions a label may sit at, near first. Colour already ties
#: a label to its kitchen, so a name is free to move well away from its
#: marker rather than sit on somebody else's data.
RADII = (7, 11, 16, 23, 31, 41, 53)
DIRECTIONS = tuple(
    (round(np.cos(np.pi * k / 8), 4), round(np.sin(np.pi * k / 8), 4)) for k in range(16)
)
#: Where a kitchen's name goes when the automatic search is not what the
#: figure wants: an offset in points from one end of its arrow, with the
#: alignment that keeps the name on that side, and optionally which end --
#: "notice" (the diamond, the default) or "control" (the circle the arrow
#: starts from). Fixed, not searched, so the figure does not move between
#: renders.
PLACEMENT = {
    "return": {
        "split_0": (-8, 3, "right", "bottom"),      # upper left, clear of the key
        "split_1": (8, 3, "left", "bottom"),
        "outage_1": (0, -26, "center", "top", "control"),  # below its arrow's error bar
        "distance_0": (10, -8, "left", "top"),      # lower right of its own arrow head
        "distance_1": (8, 13, "left", "bottom"),    # upper right, clear of outage_1's bar
        "outage_0": (-8, 3, "right", "bottom"),     # upper left of where its arrow ends
    },
    "drop": {
        "split_0": (8, 3, "left", "bottom"),        # upper right
        "split_1": (9, -3, "left", "top"),          # lower right
        "outage_0": (-6, 3, "right", "bottom"),     # upper left
        "outage_1": (-8, -20, "right", "top", "control"),  # lower left of where its arrow starts
        "distance_0": (10, 6, "left", "center"),    # right of its own arrow head, a touch higher
        "distance_1": (-24, 24, "center", "bottom"), # up and left, past the axis numbers
    },
}

#: The same, for the copy without intervals: with no whiskers to clear, each
#: name sits right beside its own marker.
BARE_PLACEMENT = {
    "return": {
        "split_0": (0, -8, "center", "top", "control"),     # under its circle
        "split_1": (8, 0, "left", "center"),
        "outage_0": (-6, 4, "right", "bottom"),             # upper left
        "outage_1": (0, -8, "center", "top", "control"),    # under its circle
        "distance_0": (8, -3, "left", "center"),
        "distance_1": (20, 8, "center", "bottom"),          # above, clear of the axis numbers
    },
    "drop": {
        "split_0": (8, 0, "left", "center"),
        "split_1": (8, 0, "left", "center"),
        "outage_0": (-6, 3, "right", "bottom"),             # upper left
        "outage_1": (-12, -8, "center", "top", "control"),  # under its circle
        "distance_0": (8, 0, "left", "center"),
        "distance_1": (-24, 24, "center", "bottom"),        # up and left, past the axis numbers
    },
}


def bars(summary, key, value):
    """Half-widths of a 95% seed-bootstrap interval, as errorbar wants them."""
    lo, hi = summary.get(f"{key}_ci95", [np.nan, np.nan])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return [[max(0.0, value - lo)], [max(0.0, hi - value)]]


def obstacle_boxes(ax, spans):
    """Display-space boxes a label must keep off: whiskers, markers, arrows."""
    boxes = []
    for (x0, y0, xe0, ye0), (x1, y1, xe1, ye1) in spans:
        for x, y, xe, ye in ((x0, y0, xe0, ye0), (x1, y1, xe1, ye1)):
            left, bottom = ax.transData.transform((x - xe, y - ye))
            right, top = ax.transData.transform((x + xe, y + ye))
            boxes.append((min(left, right) - 3, min(bottom, top) - 3,
                          max(left, right) + 3, max(bottom, top) + 3))
        # the arrow between the pair, sampled so a label cannot sit on it
        for t in (0.25, 0.5, 0.75):
            px, py = ax.transData.transform((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            boxes.append((px - 5, py - 5, px + 5, py + 5))
    return boxes


def pair_box(ax, span):
    """Display box around one kitchen's two points and their whiskers."""
    (x0, y0, xe0, ye0), (x1, y1, xe1, ye1) = span
    xs = (x0 - xe0, x0 + xe0, x1 - xe1, x1 + xe1)
    ys = (y0 - ye0, y0 + ye0, y1 - ye1, y1 + ye1)
    left, bottom = ax.transData.transform((min(xs), min(ys)))
    right, top = ax.transData.transform((max(xs), max(ys)))
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def place_labels(fig, ax, marks, fontsize, obstacles=(), spans=None, placement=None, gap=5.0):
    """Put each kitchen's name where it covers no other mark.

    A kitchen listed in ``placement`` sits at the offset given there and
    nowhere else. The rest are searched: growing
    distances in sixteen directions, first placement whose box touches
    nothing -- marker, whisker, arrow or earlier label -- and failing that
    the one that overlaps least.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    spans = spans or {}
    placement = placement or {}
    taken = list(obstacles)
    for x, y, _, _ in marks:
        px, py = ax.transData.transform((x, y))
        taken.append((px - 9, py - 9, px + 9, py + 9))

    scale = fig.dpi / 72
    labels = []
    for x, y, name, colour in marks:
        probe = ax.annotate(
            name, (x, y), textcoords="offset points", xytext=(0, 0),
            fontsize=fontsize, color=colour, fontweight="bold", zorder=5, family="monospace",
        )
        fig.canvas.draw()
        box = probe.get_window_extent(renderer)
        width, height = box.width, box.height
        probe.remove()
        px, py = ax.transData.transform((x, y))

        pinned = placement.get(name)
        if pinned:
            dx, dy, ha, va = pinned[:4]
            anchor = pinned[4] if len(pinned) > 4 else "notice"
            ax_, ay_ = (x, y)
            if anchor == "control" and name in spans:
                ax_, ay_ = spans[name][0][0], spans[name][0][1]
            px, py = ax.transData.transform((ax_, ay_))
            left = px + dx * scale - (width if ha == "right" else width / 2 if ha == "center" else 0)
            bottom = py + dy * scale - (height if va == "top" else height / 2 if va == "center" else 0)
            taken.append((left, bottom, left + width, bottom + height))
            labels.append(ax.annotate(
                name, (ax_, ay_), textcoords="offset points", xytext=(dx, dy),
                fontsize=fontsize, color=colour, fontweight="bold", zorder=5, ha=ha, va=va,
                family="monospace",
            ))
            px, py = ax.transData.transform((x, y))
            continue

        best, best_cost, best_box = (RADII[0], 0.0, 0.0), None, None
        for radius in RADII:
            for ux, uy in DIRECTIONS:
                dx, dy = radius * ux, radius * uy
                left = px + dx * scale - (width if ux < -0.3 else width / 2 if abs(ux) <= 0.3 else 0)
                bottom = py + dy * scale - height / 2
                candidate = (left, bottom, left + width, bottom + height)
                cost = sum(
                    max(0.0, min(candidate[2], o[2]) - max(candidate[0], o[0]))
                    * max(0.0, min(candidate[3], o[3]) - max(candidate[1], o[1]))
                    for o in taken
                )
                axes_box = ax.get_window_extent(renderer)
                outside = (
                    max(0.0, axes_box.x0 - candidate[0]) + max(0.0, candidate[2] - axes_box.x1)
                    + max(0.0, axes_box.y0 - candidate[1]) + max(0.0, candidate[3] - axes_box.y1)
                )
                cost += outside * 40
                if best_cost is None or cost < best_cost:
                    best, best_cost, best_box = (dx, dy, ux), cost, candidate
                if cost == 0.0:
                    break
            if best_cost == 0.0:
                break
        dx, dy, ux = best
        taken.append(best_box)
        labels.append(ax.annotate(
            name, (x, y), textcoords="offset points", xytext=(dx, dy),
            fontsize=fontsize, color=colour, fontweight="bold", zorder=5,
            family="monospace", va="center",
            ha="right" if ux < -0.3 else "center" if abs(ux) <= 0.3 else "left",
        ))
    return labels


def fit_axes(fig, ax, spans, pad=10.0):
    """Shrink the axes onto the data, with a small margin.

    Matplotlib's margins are a fraction of the data range, which leaves a
    lot of empty panel once the points cluster. This takes the union of the
    markers and their whiskers and pads it by a few points. The kitchen
    names are deliberately left out: the figure is saved with a tight
    bounding box, so a name that sits outside the axes is still in the
    image, and making the axes grow for it would squeeze the data.
    """
    fig.canvas.draw()
    boxes = [pair_box(ax, span) for span in spans]
    if not boxes:
        return
    scale = pad * fig.dpi / 72
    left = min(b[0] for b in boxes) - scale
    bottom = min(b[1] for b in boxes) - scale
    right = max(b[2] for b in boxes) + scale
    top = max(b[3] for b in boxes) + scale
    # The corners come back in display order, so these are already the
    # values at the left/bottom and right/top of the axes -- an inverted
    # axis stays inverted without any swapping.
    (nx0, ny0), (nx1, ny1) = ax.transData.inverted().transform([(left, bottom), (right, top)])
    ax.set_xlim(nx0, nx1)
    ax.set_ylim(ny0, ny1)


def normalise_returns(layouts):
    """Return as a share of the kitchen's best both-alert self-play."""
    scaled = {}
    for name, entry in layouts.items():
        item = dict(entry)
        best = OPTIMAL.get(name)
        if best:
            for key in ("return_control", "return_notice"):
                item[key] = 100.0 * entry[key] / best
                bounds = entry.get(f"{key}_ci95")
                if bounds:
                    item[f"{key}_ci95"] = [100.0 * b / best for b in bounds]
        scaled[name] = item
    return scaled


def draw_panel(fig, ax, layouts, metric, sizes, style, gap=5.0):
    """One scatter: the kitchens' two arms, joined, with their names."""
    ylabel, invert, title = PANELS[metric]
    metric = "return" if metric == "return_norm" else metric
    marks = []
    spans = {}
    for name in [n for n in ORDER if n in layouts] + [n for n in layouts if n not in ORDER]:
        s = layouts[name]
        colour = COLORS.get(name, "#465569")
        x0, x1 = s["jsd_control"], s["jsd_notice"]
        y0, y1 = s[f"{metric}_control"], s[f"{metric}_notice"]
        if not all(np.isfinite(v) for v in (x0, x1, y0, y1)):
            continue
        for arm, x, y in (("control", x0, y0), ("notice", x1, y1)) if style["errorbars"] else ():
            ax.errorbar(
                x, y,
                xerr=bars(s, f"jsd_{arm}", x),
                yerr=bars(s, f"{metric}_{arm}", y),
                fmt="none", ecolor=colour, elinewidth=1.0, capsize=2.4, alpha=0.85, zorder=2,
            )
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4, shrinkA=5, shrinkB=6),
            zorder=3,
        )
        ax.scatter(x0, y0, s=58, marker=MARKERS["control"], facecolor="white",
                   edgecolor=colour, linewidth=1.4, zorder=4)
        ax.scatter(x1, y1, s=64, marker=MARKERS["notice"], color=colour, zorder=4)
        marks.append((x1, y1, name, colour))
        # The intervals still set the frame when they are not drawn, so the
        # panel keeps the axes, and the names their places, of the drawn one.
        half = lambda key, value: (  # noqa: E731
            0.0 if bars(s, key, value) is None else max(bars(s, key, value)[0][0], bars(s, key, value)[1][0])
        )
        spans[name] = (
            (x0, y0, half("jsd_control", x0), half(f"{metric}_control", y0)),
            (x1, y1, half("jsd_notice", x1), half(f"{metric}_notice", y1)),
        )

    ax.set_xlabel("B behavior divergence (JSD)", fontsize=sizes["axis"])
    ax.set_ylabel(ylabel, fontsize=sizes["axis"])
    # The manuscript panels carry their caption in the text, so the title
    # would only repeat the two axis labels back at the reader.
    if style.get("titles", True):
        ax.set_title(title, fontsize=sizes["title"], color="#182338", pad=8)
    if invert:
        ax.invert_yaxis()
    if style["split"] and metric == "drop":
        # Larger tick labels thin the automatic ticks to every 0.2; keep 0.1.
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(color="#E5E9EF", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA6B5")
    ax.tick_params(length=3.5, color="#9AA6B5", labelsize=sizes["tick"])
    ax.margins(x=0.06 if style["split"] else 0.22, y=0.08 if style["split"] else 0.16)
    handles = [
        plt.Line2D([], [], linestyle="", marker=MARKERS[arm],
                   markerfacecolor="white" if arm == "control" else "#243247",
                   markeredgecolor="#243247", markersize=8 if style["split"] else 5.5,
                   label=(PAPER_LABELS if style["split"] else LABELS)[arm])
        for arm in ("control", "notice")
    ]
    # The two panels are subfigures of one figure, so the manuscript copy
    # carries the key once, on the return panel.
    if style["legend"] and not (style["split"] and metric == "drop"):
        # Inside the plot, up and to the left, just clear of split_0, so both
        # panels keep the same plot area.
        if style["split"]:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.36, 1.03),
                      frameon=False, fontsize=sizes["legend"], handletextpad=0.5, borderaxespad=0.0)
        else:
            ax.legend(handles=handles, loc=LEGEND_LOC.get(metric, "best"), frameon=False,
                      fontsize=sizes["legend"], handletextpad=0.5, borderaxespad=0.4)
    return marks, spans


def finish(fig, ax, marks, spans, metric, sizes, style, gap):
    metric = metric.replace("_norm", "")
    if style["split"] and not style["errorbars"]:
        # Without the intervals the pinned places, made to clear whiskers,
        # leave names far from their marks; these sit beside them instead.
        fit_axes(fig, ax, spans.values())
        place_labels(fig, ax, marks, sizes["label"], spans=spans,
                     placement=BARE_PLACEMENT.get(metric), gap=gap)
        return
    labels = place_labels(
        fig, ax, marks, sizes["label"], obstacle_boxes(ax, spans.values()),
        spans=spans, placement=PLACEMENT.get(metric) if style["pinned"] else None, gap=gap,
    )
    if style["split"]:
        fit_axes(fig, ax, spans.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="rnn")
    parser.add_argument("--in-dir", default=str(ROOT / "outputs/notice"))
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply every font size.")
    parser.add_argument("--gap", type=float, default=5.0, help="Points between a label and its data.")
    parser.add_argument(
        "--style", default=None, choices=("paper", "working"),
        help="paper: one panel per file, as the manuscript takes them; working: both panels and "
        "the across-layout statistics. By default, paper when --out-dir ends in 'paper'.",
    )
    parser.add_argument(
        "--errorbars", action=argparse.BooleanOptionalAction, default=None,
        help="Draw the seed-bootstrap intervals; off for the paper figure, on for the working copy.",
    )
    parser.add_argument("--font", default=None, help="times, sans (Pretendard), serif (Libertine), or a family name.")
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=None,
        help="Draw the return panel as a share of the kitchen's best both-alert self-play; "
        "on for the paper figure, off for the working copy.",
    )
    args = parser.parse_args()
    summary = json.loads((Path(args.in_dir) / f"behavior_summary_{args.arm}.json").read_text())
    layouts = summary["layouts"]
    across = summary.get("across_layouts", {})

    style = dict(STYLES[args.style or ("paper" if Path(args.out_dir).name == "paper" else "working")])
    if args.errorbars is not None:
        style["errorbars"] = args.errorbars
    sizes = {k: v * args.scale for k, v in style["sizes"].items()}

    # Pretendard for the prose, a typewriter face for the kitchen names --
    # the paper sets those as code and the figure follows it. The names
    # after each are the fallbacks this machine actually has.
    families = {
        "times": ("serif", ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]),
        "sans": ("sans-serif", ["Pretendard", "Nimbus Sans", "DejaVu Sans"]),
        "serif": ("serif", ["Linux Libertine O", "Libertinus Serif", "Nimbus Roman",
                            "Times New Roman", "DejaVu Serif"]),
    }
    font = args.font or style["font"]
    family, names = families.get(font, ("sans-serif", [font, "DejaVu Sans"]))
    plt.rcParams.update(
        {"font.family": family, f"font.{'serif' if family == 'serif' else 'sans-serif'}": names,
         "font.monospace": ["Nimbus Mono PS", "Liberation Mono", "DejaVu Sans Mono"],
         "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
         "font.size": style["base"] * args.scale, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )

    out_dir = Path(args.out_dir)
    if style["split"] and out_dir.name == "paper":
        out_dir = out_dir / "fig7"   # the manuscript's panels, in their figure's folder
    out_dir.mkdir(parents=True, exist_ok=True)
    if style["split"]:
        normalize = style["normalize"] if args.normalize is None else args.normalize
        # The manuscript copy comes both ways: without the intervals, and with
        # them as <stem>_errorbars, unless --errorbars/--no-errorbars picks one.
        variants = (True, False) if args.errorbars is None else (args.errorbars,)
        for errorbars in variants:
            style["errorbars"] = errorbars
            for metric in ("drop", "return_norm" if normalize else "return"):
                fig, ax = plt.subplots(figsize=style["figsize"])
                panel_layouts = normalise_returns(layouts) if metric == "return_norm" else layouts
                marks, spans = draw_panel(fig, ax, panel_layouts, metric, sizes, style, args.gap)
                fig.tight_layout()
                finish(fig, ax, marks, spans, metric, sizes, style, args.gap)
                stem = (
                    PAPER_STEMS[metric] + ("_errorbars" if errorbars else "") if style["split"]
                    else f"behavior_as_notice_{args.arm}_{metric.replace('_norm', '')}"
                )
                fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
                fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
                plt.close(fig)
                print(out_dir / f"{stem}.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=style["figsize"])
    for index, (ax, metric) in enumerate(zip(axes, ("drop", "return"))):
        marks, spans = draw_panel(fig, ax, layouts, metric, sizes, style, args.gap)
        ax.set_title(
            f"({chr(97 + index)}) " + PANELS[metric][2], fontsize=sizes["title"],
            loc="left", fontweight="bold", color="#182338", pad=6,
        )
        finish(fig, ax, marks, spans, metric, sizes, style, args.gap)
    if style["caption"] and across:
        fig.text(
            0.5, 0.01,
            "Bars: 95% bootstrap interval over the informed training seeds.  "
            f"r(\u0394JSD, Drop reduction) = {across['r_delta_jsd_vs_drop_reduction']:.3f}, "
            f"exact permutation p = {across['p_perm_drop']:.3f};  "
            f"r(\u0394JSD, return gain) = {across['r_delta_jsd_vs_return_gain']:.3f}, "
            f"p = {across['p_perm_return']:.3f}",
            ha="center", fontsize=sizes["label"], color="#49576A",
        )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.24, wspace=0.28)
    stem = f"behavior_as_notice_{args.arm}"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
