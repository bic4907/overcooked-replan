"""Figure 2 as parts for a LaTeX layout: the group bars, and a kitchen per group.

One column: the bars on the left, and on the right one kitchen per group, one
over the other, drawn as the alert-preparation figure draws them -- the
kitchen before the edit, its agents where they start, the tiles the edit will
change framed in orange. The
bars are results/plot_alert_groups.py's paper figure without the OvercookedV2
group, recoloured for the recurrent IPPO policies it reports with the light
and full green of the alert-preparation figure.

    python results/plot_alert_groups_parts.py

Writes results/paper/fig2/: a and b (each kitchen alone, no margin),
bars, overview (the parts as figure.tex sets them) and figure.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import explore_notice_traces_plot as traces  # noqa: E402
import plot_alert_groups as groups_figure  # noqa: E402
import results.explore_same_edit_pairs as kitchens  # noqa: E402

PARTS = ROOT / "results/paper/fig2"
#: Alert off and on: the Google palette's Light Green and Green, as the
#: IPPO-RNN bars of the alert-preparation figure.
COLORS = {"off": "#CEEAD6", "on": "#34A853"}
#: One kitchen per group, in the bars' order, and what it is called there.
#: The OvercookedV2 kitchens are left out, as in alert_groups_noV2: their
#: cross-play sits at the ceiling, so they say nothing about the alert.
#: The third field is the kitchen's subcaption: the edit's action, and where
#: it lands -- anywhere, or where it divides the kitchen.
EXAMPLES = (
    ("a", "rs_v1_coord_ring_block", "Add wall at random"),
    ("b", "split_0", "Add wall as divider"),
)
GROUPS = ("V1", "Structured")
#: The bars' axis labels, lettered as the kitchens beside them.
#: The groups are named under their bars; the kitchens beside them carry the
#: letter alone, so nothing is said twice.
TICKS = {"V1": "(a) Carroll et al.,\nrandom edit", "Structured": "(b) Allocation-\naware edit"}
WIDTH = 3.33            # one column
#: The x range: the two groups sit close together, the slack at either end.
XLIM = (-0.5, 1.5)
BARS_W = 2.20           # the bar plot, narrowed to leave the kitchens a column
BARS_H = 2.10           # its height with the significance marks
GAP = 0.05              # between the bars and the kitchens
CAPTION_H = 0.15        # room under a kitchen for its one-line subcaption
MAP_GAP = 0.02          # between a subcaption and the kitchen under it
#: Bar width on the page, in inches: Figure 5's bars measure 0.105 in.
BAR_IN = 0.16
DPI = 320


def shapes():
    return {name: (len(kitchens.phase_grids(name)[0]), len(kitchens.phase_grids(name)[0][0]))
            for _, name, _ in EXAMPLES}


def tile_size(shape):
    """One tile for both kitchens: the largest at which each fits the column
    right of the bars and the two, with their subcaptions, stand no taller
    than the bars."""
    col = WIDTH - BARS_W - GAP
    stack = BARS_H - len(EXAMPLES) * CAPTION_H - (len(EXAMPLES) - 1) * MAP_GAP
    return min(col / max(w for _, w in shape.values()),
               stack / sum(h for h, _ in shape.values()))


def kitchen(ax, name, tile):
    """The kitchen before the edit, as the alert-preparation figure draws one:
    the viewer's render with the tiles the edit changes framed in orange."""
    before, after = kitchens.phase_grids(name)
    h, w = len(before), len(before[0])
    px = kitchens.RENDER_PX
    ax.imshow(kitchens.sharp_frame(name, 0), interpolation="none", extent=(0, w * px, h * px, 0))
    pt = tile * 72
    for y, (ra, rb) in enumerate(zip(before, after)):
        for x, (was, now) in enumerate(zip(ra, rb)):
            if was != now:
                ax.add_patch(Rectangle((x * px, y * px), px, px, fill=False, edgecolor=traces.FRAME,
                                       linewidth=0.07 * pt, zorder=5))
    ax.set_xlim(0, w * px)
    ax.set_ylim(h * px, 0)
    ax.axis("off")


def draw_kitchens(shape, tile):
    for key, name, _ in EXAMPLES:
        h, w = shape[name]
        fig = plt.figure(figsize=(w * tile, h * tile))
        kitchen(fig.add_axes((0, 0, 1, 1)), name, tile)
        fig.savefig(PARTS / f"{key}.pdf")
        fig.savefig(PARTS / f"{key}.png", dpi=DPI)
        plt.close(fig)
        print("figure ->", PARTS / f"{key}.png")


def overview(shape, tile):
    """The saved parts at their own sizes, where figure.tex sets them."""
    keys = [key for key, _, _ in EXAMPLES]
    part = {k: mpimg.imread(PARTS / f"{k}.png") for k in ["bars", *keys]}
    bars_h = part["bars"].shape[0] / part["bars"].shape[1] * BARS_W
    fig = plt.figure(figsize=(WIDTH, bars_h))

    def place(k, x, y, w, h):
        ax = fig.add_axes((x / WIDTH, y / bars_h, w / WIDTH, h / bars_h))
        ax.imshow(part[k], interpolation="none")
        ax.axis("off")

    place("bars", 0, 0, BARS_W, bars_h)
    col_x, col_w = BARS_W + GAP, WIDTH - BARS_W - GAP
    stack = (sum(shape[n][0] * tile + CAPTION_H for _, n, _ in EXAMPLES)
             + MAP_GAP * (len(EXAMPLES) - 1))
    y = (bars_h + stack) / 2
    for key, name, caption in EXAMPLES:
        h, w = shape[name]
        y -= h * tile
        place(key, col_x + (col_w - w * tile) / 2, y, w * tile, h * tile)
        # The subcaption under its kitchen, as LaTeX sets it.
        fig.text((col_x + col_w / 2) / WIDTH, (y - 0.03) / bars_h,
                 f"({key}) {caption}", ha="center", va="top",
                 fontsize=7.0, linespacing=1.1)
        y -= CAPTION_H + MAP_GAP
    fig.savefig(PARTS / "overview.png", dpi=DPI)
    fig.savefig(PARTS / "overview.pdf")
    plt.close(fig)
    print("figure ->", PARTS / "overview.png")


def write_tex(shape, tile):
    """A column figure: the bars, and beside them the two kitchens one over the
    other, sized as fractions of the column so their tiles stay equal."""
    col = WIDTH - BARS_W - GAP
    lines = [
        "% Needs \\usepackage{graphicx} and \\usepackage{subcaption}.",
        "\\begin{figure}[t]",
        "  \\centering",
        # Subcaptions at 8 pt, the size of the bars' axis labels.
        "  \\captionsetup[subfigure]{font=footnotesize,justification=centering}",
        f"  \\begin{{minipage}}[c]{{{BARS_W / WIDTH:.4f}\\linewidth}}",
        "    \\includegraphics[width=\\linewidth]{figures/fig2/bars.pdf}",
        "  \\end{minipage}\\hfill",
        f"  \\begin{{minipage}}[c]{{{col / WIDTH:.4f}\\linewidth}}",
        "    \\centering",
    ]
    for i, (key, name, caption) in enumerate(EXAMPLES):
        h, w = shape[name]
        lines += [
            "    \\begin{subfigure}{\\linewidth}",
            "      \\centering",
            f"      \\includegraphics[width={w * tile / col:.4f}\\linewidth]{{figures/fig2/{key}}}",
            # What the edit does; the bars' axis names the group.
            "      \\caption{" + caption + "}",
            f"      \\label{{fig:alert-group-{key}}}",
            "    \\end{subfigure}",
        ]
        if i < len(EXAMPLES) - 1:
            lines.append(f"    \\par\\vspace{{{MAP_GAP / col:.4f}\\linewidth}}")
    lines += [
        "  \\end{minipage}",
        "  \\caption{Effect of advance notice under random and allocation-aware edits. Bars show",
        "  normalized cross-play return with and without notice, and brackets mark the paired change.",
        "  The kitchens show an example edit from each group, the edited tile in orange.}",
        "  \\label{fig:alert-groups}",
        "\\end{figure}",
    ]
    (PARTS / "figure.tex").write_text("\n".join(lines) + "\n")
    print("latex ->", PARTS / "figure.tex")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    PARTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["hatch.linewidth"] = 1.0
    # Every word in the figure at 8 pt, the size LaTeX sets the subcaptions in.
    groups_figure.SIZES.update(tick=8.0, value=8.0, axis=8.0, legend=8.0)
    groups = {k: v for k, v in groups_figure.normalized(groups_figure.load_cells()).items()
              if k in GROUPS}
    groups_figure.draw(groups, dots=False, stem="bars", ylim=(0.4, 0.8), width_in=BARS_W,
                       marks=True, colors=COLORS, out_dir=PARTS,
                       # Narrowed, the bars take their key under the axis
                       # labels and keep room on the right for the last one.
                       legend_below=True, right_in=0.10,
                       # Without / with, not masked / available: the two bars are
                       # policies trained apart, one never shown the notice, rather
                       # than one policy with the notice hidden as in Figure 6.
                       arm_labels=("Without notice", "With notice"),
                       # Without notice hatched, as Figure 6 tells its arms apart.
                       hatch_off="//",   # sparse white hatching, as Figure 6
                       # Lettered as the kitchens beside them, so each bar group
                       # names its kitchen in the figure itself.
                       # The letter leads the name; Allocation-aware breaks at its
                       # hyphen so the two labels clear each other.
                       tick_labels=TICKS,
                       # Thin bars as in Figure 5, and no values over them.
                       # In axis units, for BAR_IN on the page (draw keeps 92% of it).
                       # The two groups drawn close, with room at the plot's ends.
                       xlim=XLIM,
                       height_in=BARS_H,   # taller, so the kitchens beside it can grow
                       ytick_size=9.0,   # the axis numbers a size up
                       bar_width=BAR_IN / 0.92 / ((BARS_W - 0.586 - 0.10) / (XLIM[1] - XLIM[0])),
                       show_values=False,
                       # The narrower plot takes a tighter key.
                       legend_kw=dict(fontsize=8.0, columnspacing=0.6, handlelength=1.5, handletextpad=0.3),
                       extra_bottom=0.0)
    shape = shapes()
    tile = tile_size(shape)
    draw_kitchens(shape, tile)
    overview(shape, tile)
    write_tex(shape, tile)


if __name__ == "__main__":
    main()
