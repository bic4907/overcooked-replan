"""The advance-notice bar plot with a paired trajectory under each scenario.

Top row: experiment/notice/plot_alert_preparation.py's paper figure, the
endpoint rates with the notice masked and available. Under each scenario, one
probe state from the same experiment played out twice, notice available and
notice masked, drawn by results/explore_notice_traces_plot.py.

    python results/plot_notice_combined.py

Writes Figure 6 into results/paper/fig6/: alert_preparation_combined (the
whole figure as one image) and its parts for a LaTeX layout -- a, b, c (each
scenario's pair of trajectories), a_masked ... c_available (each kitchen
alone), tabulation (the bars, with the figure's key under them) and overview
(the parts placed as the LaTeX layout sets them).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))
sys.path.insert(0, str(ROOT / "experiment/notice"))

import explore_notice_traces_plot as traces  # noqa: E402
import plot_alert_preparation as bars  # noqa: E402

OUT = ROOT / "results/paper/fig6"
PARTS = OUT                         # the separate images sit beside it
SCENARIOS = ("Split", "Outage", "Distance")
#: The trajectory under each bar panel, and what Table 1 scores there.
TRACES = {
    "Split": ("split_0_s6_r0_p6", "cross"),
    "Outage": ("outage_0_s5_r0_p22", "plate"),
    "Distance": ("distance_0_s7_r0_p20", "wait"),
}
ROWS = (("OFF", "Notice masked"), ("ON", "With notice"))  # as the bars are ordered

WIDTH = 7.15            # one full text width, as the bar plot is
LEFT, RIGHT = 0.55, 0.05
COL_GAP = 0.22          # between bar panels, as in the bar plot
BAR_H = 0.95
PAIR_GAP = 0.05         # between the two trajectories of one scenario
OVERHANG = 0.10         # how far Blackout's pair may reach past its panel, each side
#: The bars are coloured by baseline, from the Google News palette: solid for
#: the notice available, the same colour hatched for the notice masked.
METHOD_COLORS = {"IPPO-RNN": "#34A853", "FCP": "#EA4335"}
#: A frame round each kitchen, so a trajectory says which arm it is without a
#: label. Google palette: Light Yellow for the notice masked, Yellow for the
#: notice available -- light and strong, as the bars are.
ARM_GREYS = {"OFF": None, "ON": "#FBBC04"}
#: Only the kitchens with notice are framed; the masked ones go bare.
FRAME_PT = 1.0   # the frame's width, in points


def setup_bars():
    """The bar plot module in its paper style, as its main() would leave it."""
    bars.STYLE.clear()
    bars.STYLE.update(bars.STYLES["paper"])
    bars.SIZES.clear()
    bars.SIZES.update(bars.STYLES["paper"]["sizes"])
    bars.DATA.extend(json.loads(Path(bars.DATA_PATH).read_text()))
    scope = bars.STYLE["scope"]
    methods = ["IPPO-RNN", "FCP"]
    tops = [
        100 * max(item["ON_ci95"][1], item["OFF_ci95"][1], item["ON"], item["OFF"])
        for scenario in SCENARIOS
        for method in methods
        for item in [bars.select(scenario, method, "initiation", scope, bars.VARIANTS[scenario])]
    ]
    ymax = float(10 * np.ceil((max(tops) + 6) / 10))
    return scope, methods, ymax


KEY_H = 0.40   # the key's two rows under the figure
#: The key's squares: their pitch in points, and a handle long enough for the
#: longest row of them (three), so no row reaches into the gap before it and
#: the second row's entries stand as far apart as the first row's.
PITCH = 11.0
#: Every swatch is drawn wide, as a short bar on its side, so a bar's
#: hatching shows; the frame swatch matches them. Width in multiples of its height.
BAR_SWATCH_W = 2.2
SWATCH_GAP = 3.0         # points between two swatches
ROW_LEN = 3 * BAR_SWATCH_W * PITCH * 0.75 / (8.5 - 1.2) + 2 * SWATCH_GAP / (8.5 - 1.2)


def key(fig):
    """The figure's key, in two rows along its foot: the marks of the
    trajectories, then the two arms: each as its two bars, framed as its
    kitchens are."""
    height = fig.get_size_inches()[1]
    size = dict(frameon=False, fontsize=bars.SIZES["legend"] - 1.2, handlelength=1.3,
                columnspacing=1.2, handletextpad=0.4, borderaxespad=0.0)
    fig.legend(handles=[
        Line2D([0], [0], color=traces.RED, lw=1.6),
        Line2D([0], [0], color=traces.BLUE, lw=1.6, ls=(0, (2.4, 1.6))),
        Line2D([0], [0], marker="s", ls="none", ms=6, mfc="none", mec=traces.FRAME, mew=1.0),
    ], labels=["informed agent", "uninformed agent", "changing tile"],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.21 / height), **size)
    # Two entries, each its kitchens' frame then its bars: hatched for the
    # notice masked, solid for the notice available.
    from matplotlib.legend_handler import HandlerBase

    class Row(HandlerBase):
        """Squares side by side at one fixed pitch, whatever their number,
        set against the label so each row ends the same distance from it."""

        def create_artists(self, legend, handles, xdescent, ydescent, width, height, fontsize, trans):
            side = 0.85 * height
            out = []
            right = width - xdescent
            for h in reversed(handles):
                w = side * BAR_SWATCH_W
                out.append(Rectangle((right - w, (height - side) / 2 - ydescent), w, side,
                                     facecolor=h.get_facecolor(), edgecolor=h.get_edgecolor(),
                                     hatch=h.get_hatch(), linewidth=h.get_linewidth(),
                                     linestyle=h.get_linestyle(), transform=trans))
                right -= w + SWATCH_GAP
            return out

    def frame_swatch(colour, masked=False):
        return Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=colour, linewidth=1.2,
                         linestyle=(0, (1.6, 1.0)) if masked else "-")

    def bar_swatch(colour, masked):
        return Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor="white" if masked else colour,
                         hatch="///" if masked else None, linewidth=0)

    fig.legend(handles=[tuple(bar_swatch(c, True) for c in METHOD_COLORS.values()),
                        (frame_swatch(ARM_GREYS["ON"]),)
                        + tuple(bar_swatch(c, False) for c in METHOD_COLORS.values())],
               labels=["Notice masked", "With notice"],
               handler_map={tuple: Row()}, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 0.03 / height), **{**size, "handlelength": ROW_LEN, "handletextpad": 0.5})


def frame(ax, arm):
    """The arm's colour round a kitchen, just outside its edge.

    Drawn as a filled block behind the kitchen, the kitchen's size plus the
    frame's width on every side, rather than as an outline: the kitchen then
    covers it right to its edge, so no sliver of page shows between the two at
    any resolution, in the PNG or the PDF.
    """
    if ARM_GREYS[arm] is None:
        return
    fig = ax.figure
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    band = FRAME_PT / 72
    fig.patches.append(Rectangle(
        (pos.x0 - band / W, pos.y0 - band / H), pos.width + 2 * band / W, pos.height + 2 * band / H,
        transform=fig.transFigure, facecolor=ARM_GREYS[arm], edgecolor="none", zorder=0))
    # The kitchen over the block, whatever order the figure draws them in.
    ax.set_zorder(1)
    ax.patch.set_visible(False)


def save(fig, stem, where=OUT):
    where.mkdir(parents=True, exist_ok=True)
    fig.savefig(where / f"{stem}.pdf")
    fig.savefig(where / f"{stem}.png", dpi=320)
    plt.close(fig)
    print("figure ->", where / f"{stem}.png")


def separate(records, shape, col_w, scope, methods, ymax):
    """The figure's parts as images of their own.

    c keeps the combined figure's tile, its pair as wide as one bar panel; b
    is drawn so its pair is exactly as wide as c's; a shares b's tile.
    """
    widest = shape["Distance"][1]
    # Room for the frame round each of the pair, outside the kitchens.
    inner = col_w - PAIR_GAP - 4 * FRAME_PT / 72
    tile = {"Distance": inner / (2 * widest)}
    tile["Outage"] = inner / (2 * shape["Outage"][1])
    tile["Split"] = tile["Outage"]
    label_h, pad = 0.0, 0.01
    for i, scenario in enumerate(SCENARIOS):
        record, (h, w), t = records[scenario], shape[scenario], tile[scenario]
        traces.TILE_IN, traces.PT = t, t * 72
        pair_gap = PAIR_GAP
        width, height = 2 * w * t + pair_gap + 2 * pad, label_h + h * t + pad
        fig = plt.figure(figsize=(width, height))
        for j, (arm, _) in enumerate(ROWS):
            x = pad + j * (w * t + pair_gap)
            ax = fig.add_axes((x / width, pad / height, w * t / width, h * t / height))
            image = traces.boundary_frame(record, arm, partner=True, step=0, cooks_at=record["pre"])
            traces.panel(ax, record, image, arm, TRACES[scenario][1], "line")
            frame(ax, arm)
        save(fig, chr(97 + i), PARTS)
        # And each kitchen on its own, the same tile and only the frame's width
        # round it -- the frame on the available, blank on the masked -- so a
        # layout can set the six independently and the kitchens match in size.
        band = FRAME_PT / 72
        for arm, name in (("OFF", "masked"), ("ON", "available")):
            width, height = w * t + 2 * band, h * t + 2 * band
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_axes((band / width, band / height, w * t / width, h * t / height))
            image = traces.boundary_frame(record, arm, partner=True, step=0, cooks_at=record["pre"])
            traces.panel(ax, record, image, arm, TRACES[scenario][1], "line")
            frame(ax, arm)
            save(fig, f"{chr(97 + i)}_{name}", PARTS)

    # The bars, as in the combined figure, with their key under them.
    tick_h, legend_h, top = 0.22, KEY_H, 0.03   # little above the bars: the subcaptions sit there
    height = top + BAR_H + tick_h + legend_h
    fig = plt.figure(figsize=(WIDTH, height))
    first = None
    for i, scenario in enumerate(SCENARIOS):
        x0 = LEFT + i * (col_w + COL_GAP)
        ax = fig.add_axes((x0 / WIDTH, (tick_h + legend_h) / height, col_w / WIDTH, BAR_H / height),
                          sharey=first)
        bars.draw_panel(ax, scenario, i, "initiation", ymax, scope, bars.VARIANTS[scenario], methods, None)
        ax.set_title("", loc="left")
        ax.set_yticks(np.arange(0, ymax + 1, 20))
        if first is None:
            first = ax
            ax.set_ylabel(bars.STYLE["ylabel"], labelpad=4, fontsize=bars.SIZES["axis"])
        else:
            ax.tick_params(labelleft=False)
    key(fig)
    save(fig, "tabulation", PARTS)
    overview(col_w)


def overview(col_w, dpi=320, caption_h=0.20, gap=0.06):
    """The saved parts, at their own sizes, where the LaTeX layout puts them.

    The three pairs each centred over its bar panel with its subcaption under
    it, and the bars, with the figure's key, at the bottom. Read back from the PNGs,
    so it shows the files themselves rather than a redrawing of them.
    """
    import matplotlib.image as mpimg
    part = {k: mpimg.imread(PARTS / f"{k}.png") for k in ("a", "b", "c", "tabulation")}
    size = {k: (im.shape[1] / dpi, im.shape[0] / dpi) for k, im in part.items()}
    pairs_h = max(size[k][1] for k in "abc")
    height = pairs_h + caption_h + gap + size["tabulation"][1]
    fig = plt.figure(figsize=(WIDTH, height))

    def place(key, x, y_top):
        w, h = size[key]
        ax = fig.add_axes((x / WIDTH, (y_top - h) / height, w / WIDTH, h / height))
        ax.imshow(part[key], interpolation="none")
        ax.axis("off")

    y = height
    for i, (key, scenario) in enumerate(zip("abc", SCENARIOS)):
        x0 = LEFT + i * (col_w + COL_GAP)
        place(key, x0 + (col_w - size[key][0]) / 2, y)
        fig.text((x0 + col_w / 2) / WIDTH, (y - pairs_h - 0.03) / height,
                 f"({key}) {bars.DISPLAY_NAMES[scenario]}", ha="center", va="top",
                 fontsize=bars.SIZES["title"])
    y -= pairs_h + caption_h + gap
    place("tabulation", 0.0, y)
    save(fig, "overview", PARTS)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    scope, methods, ymax = setup_bars()
    bars.METHOD_COLORS.update(METHOD_COLORS)
    # The masked arm is told apart by hatching, not by a lighter tint.
    # Two near-empty bars keep their values level; side by side they fit.
    bars.LIFT_TIES = False
    plt.rcParams["hatch.linewidth"] = 1.0
    # The narrow Partition column puts its value labels close; a size down.
    bars.SIZES["value"] = 6.4
    records = {s: json.loads((traces.TRACES / f"{TRACES[s][0]}.json").read_text()) for s in SCENARIOS}
    shape = {s: (len(r["static_a"]), len(r["static_a"][0])) for s, r in records.items()}
    # The bar panels keep the bar plot's equal widths. Every kitchen is drawn
    # at one tile size, the one at which the widest pair just spans its
    # panel; the other pairs sit centred over theirs.
    col_w = (WIDTH - LEFT - RIGHT - 2 * COL_GAP) / 3
    one = min((col_w - PAIR_GAP) / (2 * w) for h, w in shape.values())
    tile = {s: one for s in SCENARIOS}
    # Blackout's kitchen is long and low; let its pair run a little past its
    # panel on both sides so it is not the smallest thing in the figure.
    tile["Outage"] = (col_w + 2 * OVERHANG - PAIR_GAP) / (2 * shape["Outage"][1])
    # Partition's small square kitchen is drawn at Blackout's tile.
    tile["Split"] = tile["Outage"]
    map_h = max(shape[s][0] * tile[s] for s in SCENARIOS)
    map_h = max(h for h, w in shape.values()) * one

    top_legend, title_h, label_h, gap, tick_h, legend_h = 0.02, 0.17, 0.03, 0.08, 0.22, KEY_H
    height = top_legend + title_h + label_h + map_h + gap + BAR_H + tick_h + legend_h
    fig = plt.figure(figsize=(WIDTH, height))

    def box(x, y, w, h):
        return (x / WIDTH, y / height, w / WIDTH, h / height)

    map_y = height - top_legend - title_h - label_h - map_h
    bar_y = map_y - gap - BAR_H
    first = None
    for i, scenario in enumerate(SCENARIOS):
        x0 = LEFT + i * (col_w + COL_GAP)
        ax = fig.add_axes(box(x0, bar_y, col_w, BAR_H), sharey=first)
        bars.draw_panel(ax, scenario, i, "initiation", ymax, scope, bars.VARIANTS[scenario], methods, None)
        ax.set_title("", loc="left")
        # A shorter panel than the bar plot's own: every other gridline.
        ax.set_yticks(np.arange(0, ymax + 1, 20))
        if first is None:
            first = ax
            ax.set_ylabel(bars.STYLE["ylabel"], labelpad=4, fontsize=bars.SIZES["axis"])
        else:
            ax.tick_params(labelleft=False)
        # The scenario's name heads its column, over the trajectories.
        fig.text(x0 / WIDTH, (height - top_legend - 0.01) / height, f"({chr(97 + i)}) {bars.DISPLAY_NAMES[scenario]}",
                 ha="left", va="top", fontsize=bars.SIZES["title"], fontweight="bold")

        kind = TRACES[scenario][1]
        record = records[scenario]
        h, w = shape[scenario]
        t = tile[scenario]
        # The trajectory module sizes every stroke by its tile.
        traces.TILE_IN, traces.PT = t, t * 72
        for j, (arm, _) in enumerate(ROWS):
            pair_gap = PAIR_GAP
            mx = x0 + (col_w - 2 * w * t - pair_gap) / 2 + j * (w * t + pair_gap)
            mx_ax = fig.add_axes(box(mx, map_y + map_h - h * t, w * t, h * t))
            image = traces.boundary_frame(record, arm, partner=True, step=0, cooks_at=record["pre"])
            traces.panel(mx_ax, record, image, arm, kind, "line")
            frame(mx_ax, arm)

    # One key for the whole figure, under it.
    key(fig)

    # Everything drawn, slide it sideways as one so the outer margins match:
    # the y label on the left and the last trajectory on the right sit the
    # same distance from the edges, every gap between them untouched.
    fig.canvas.draw()
    extent = fig.get_tightbbox(fig.canvas.get_renderer())
    shift = ((WIDTH - extent.x1) - extent.x0) / 2 / WIDTH
    for axis in fig.axes:
        pos = axis.get_position()
        axis.set_position((pos.x0 + shift, pos.y0, pos.width, pos.height))
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x + shift, y))

    OUT.mkdir(parents=True, exist_ok=True)
    save(fig, "alert_preparation_combined")
    separate(records, shape, col_w, scope, methods, ymax)


if __name__ == "__main__":
    main()
