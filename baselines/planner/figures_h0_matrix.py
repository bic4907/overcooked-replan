"""Draw what each partner is worth to the simulated human H0.

The matrix ran as one W&B project per column -- BR, H0, IPPO-CNN, IPPO-RNN and
FCP, each six kitchens of twelve games -- which is the right shape for running
it and the wrong shape for looking at it. This gathers the thirty cells, puts
each arm's own self-play score beside them, and draws the two pictures:

    per kitchen   bars for what H0 scores with each learned partner, BR hatched
                  beside them as the planner reference, and a red rule where H0
                  sits with another of its own kind
    overall       one bar per partner over all six kitchens, with the spread

    python -m baselines.planner.figures_h0_matrix

Add --wandb-mode disabled to keep it local; it writes the numbers out as CSV
and Markdown either way.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: One project per column of the H0 row.
CELL_PROJECTS = {
    "br": "overcooked-v3-br-h0partner-0919-eval",
    "h0": "overcooked-v3-h0-h0partner-0919-eval",
    "cnn": "overcooked-v3-ippo-cnn-h0partner-0919-eval",
    "rnn": "overcooked-v3-ippo-rnn-h0partner-0919-eval",
    "fcp": "overcooked-v3-fcp-h0partner-0919-eval",
}
#: Where each learned arm's own self-play and crossplay scores were measured.
EVAL_PROJECTS = {
    "cnn": "overcooked-v3-ippo-cnn-75step-plate-0915_eval",
    "rnn": "overcooked-v3-ippo-rnn-75step-plate-0915_eval",
    "fcp": "overcooked-v3-fcp-75step-plate-0915_eval",
}
LAYOUTS = ("split_0", "split_1", "outage_0", "outage_1", "distance_0", "distance_1")
#: What the kitchens are called in the paper: what each one does to the floor.
KITCHENS = {
    "split_0": "Partition-0",
    "split_1": "Partition-1",
    "outage_0": "Blackout-0",
    "outage_1": "Blackout-1",
    "distance_0": "Distance-0",
    "distance_1": "Distance-1",
}
#: The bars, left to right: only the learned arms get one.
COLUMNS = ("cnn", "rnn", "fcp")
LABELS = {"br": "BR", "h0": "H0", "cnn": "IPPO-CNN", "rnn": "IPPO-RNN", "fcp": "FCP"}
COLORS = {"cnn": "#4878a8", "rnn": "#3f9b52", "fcp": "#9068b0"}
#: The two planner partners are references rather than competitors, so each is
#: a rule across the bars: BR is the best this person gets from anyone, H0 what
#: another of their own kind is worth.
BR_COLOR = "#13395f"
H0_COLOR = "#d1495b"
LEGEND = dict(
    loc="upper center", ncols=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02)
)


def legend_handles(human):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        *(Patch(color=COLORS[arm], label=LABELS[arm]) for arm in COLUMNS),
        Line2D([0], [0], color=BR_COLOR, linewidth=1.8, label="BR partner"),
        Line2D(
            [0], [0], color=H0_COLOR, linewidth=1.8, label=f"{LABELS[human]} partner"
        ),
    ]


def mean_over_kitchens(cells, arm):
    """One arm's mean over every game it played, all six kitchens together."""
    found = [cells[(layout, arm)]["mean"] for layout in LAYOUTS if (layout, arm) in cells]
    return float(np.mean(found))


def style(axes):
    """The house look: a recessive y grid and no box around the plot."""
    axes.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(labelsize=9.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked"))
    parser.add_argument("--human", default="h0", help="The row of the matrix to draw.")
    parser.add_argument("--output", default="outputs/figures")
    parser.add_argument("--prefix", default="h0_partner")
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument(
        "--project",
        default="overcooked-v3-h0partner-0919-report",
        help="Where the figures are logged when W&B is on.",
    )
    return parser.parse_args(argv)


def collect_cells(entity, human):
    """One record per cell: the kitchen, the partner, the mean and the spread."""
    import wandb

    api = wandb.Api()
    cells = {}
    for arm, project in CELL_PROJECTS.items():
        for run in api.runs(f"{entity}/{project}", per_page=200):
            if run.state != "finished" or "return_mean" not in run.summary:
                continue
            if str(run.summary.get("human", run.config.get("human"))) != human:
                continue
            layout = str(run.summary.get("layout", run.config.get("layout")))
            cells[(layout, arm)] = dict(
                mean=float(run.summary["return_mean"]),
                std=float(run.summary.get("return_std", float("nan"))),
                games=int(run.summary.get("games", 0)),
                run=run.id,
            )
    return cells


def collect_self_play(entity):
    """Each learned arm's own self-play and crossplay return, per kitchen."""
    import wandb

    api = wandb.Api()
    scores = {}
    for arm, project in EVAL_PROJECTS.items():
        for run in api.runs(f"{entity}/{project}", per_page=200):
            config = run.config
            layout = (
                (config.get("ENV_KWARGS") or {}).get("layout")
                or config.get("layout")
                or config.get("target_layout")
            )
            if layout is None or "SP" not in run.summary:
                continue
            scores[(str(layout), arm)] = dict(
                sp=float(run.summary["SP"]), xp=float(run.summary["XP"])
            )
    return scores


def pooled(values, spreads):
    """Mean and standard deviation over equally sized groups of games.

    Averaging the cell means gives the mean over every game; the spread over
    every game is the spread within the kitchens plus the spread between them,
    and both halves matter -- a partner can be steady in each kitchen and still
    be worth four times as much in one of them as in another.
    """
    values, spreads = np.asarray(values), np.asarray(spreads)
    return float(values.mean()), float(np.sqrt((spreads**2).mean() + values.var()))


def by_kitchen_figure(cells, _self_play, path, human):
    """Bars per kitchen, BR laid behind them, H0 as a red rule across the group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(figsize=(10.5, 4.0))
    span, width = 0.78, 0.78 / len(COLUMNS)
    places = np.arange(len(LAYOUTS))

    for slot, arm in enumerate(COLUMNS):
        offset = -span / 2 + width * (slot + 0.5)
        means = [cells.get((layout, arm), {}).get("mean", np.nan) for layout in LAYOUTS]
        axes.bar(
            places + offset,
            means,
            width=width * 0.86,
            color=COLORS[arm],
            zorder=3,
        )

    for arm, colour in (("br", BR_COLOR), ("h0", H0_COLOR)):
        for index, layout in enumerate(LAYOUTS):
            cell = cells.get((layout, arm))
            if cell is None:
                continue
            axes.hlines(
                cell["mean"],
                places[index] - span / 2,
                places[index] + span / 2,
                color=colour,
                linewidth=1.8,
                zorder=5,
            )

    axes.set_xticks(places, [KITCHENS[layout] for layout in LAYOUTS])
    axes.set_ylabel("Return (higher is better)")
    axes.set_ylim(0, 380)
    axes.set_xlim(places[0] - 0.55, places[-1] + 0.55)
    style(axes)
    figure.legend(handles=legend_handles(human), **LEGEND)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return save(figure, path)


def overall_figure(cells, path, human):
    """One bar per learned partner over all six kitchens, BR behind them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    for slot, arm in enumerate(COLUMNS):
        mean = mean_over_kitchens(cells, arm)
        axes.bar(slot, mean, width=0.66, color=COLORS[arm], zorder=3)
        axes.text(slot, mean + 6, f"{mean:.0f}", ha="center", fontsize=10)

    ceiling = mean_over_kitchens(cells, "br")
    level = mean_over_kitchens(cells, "h0")
    for value, colour in ((ceiling, BR_COLOR), (level, H0_COLOR)):
        axes.axhline(value, color=colour, linewidth=1.8, zorder=5)
        axes.text(
            len(COLUMNS) - 0.52, value + 6, f"{value:.0f}", fontsize=10, color=colour
        )

    axes.set_xticks(range(len(COLUMNS)), [LABELS[arm] for arm in COLUMNS])
    axes.set_xlim(-0.55, len(COLUMNS) - 0.45)
    axes.set_ylim(0, max(ceiling, level) * 1.12)
    axes.set_ylabel("Return (higher is better)")
    style(axes)
    figure.legend(handles=legend_handles(human), **LEGEND)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return save(figure, path)


def save(figure, path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path.with_suffix(".png")


def write_tables(cells, self_play, output, prefix, human):
    """The same numbers as text, since a figure is not a number anyone can cite."""
    rows = []
    for layout in LAYOUTS:
        for arm in ("h0",) + COLUMNS:
            cell = cells.get((layout, arm))
            if cell is None:
                continue
            score = self_play.get((layout, arm), {})
            rows.append(
                dict(
                    kitchen=KITCHENS[layout],
                    layout=layout,
                    human=human,
                    partner=arm,
                    return_mean=round(cell["mean"], 1),
                    return_std=round(cell["std"], 1),
                    games=cell["games"],
                    partner_self_play=score.get("sp", ""),
                    partner_crossplay=score.get("xp", ""),
                    run=cell["run"],
                )
            )
    path = output / f"{prefix}_cells.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    order = ("h0", "br", "cnn", "rnn", "fcp")
    lines = [
        f"### {LABELS[human]} at the table, mean return over twelve games",
        "",
        "| kitchen | " + " | ".join(LABELS[a] for a in order) + " |",
        "|---" * (len(order) + 1) + "|",
    ]
    for layout in LAYOUTS:
        values = [
            f"{cells[(layout, arm)]['mean']:.0f}" if (layout, arm) in cells else "-"
            for arm in order
        ]
        lines.append(f"| {KITCHENS[layout]} | " + " | ".join(values) + " |")
    summary = []
    for arm in order:
        found = [cells[(layout, arm)] for layout in LAYOUTS if (layout, arm) in cells]
        mean, spread = pooled(
            [item["mean"] for item in found], [item["std"] for item in found]
        )
        summary.append(f"**{mean:.1f}** ±{spread:.0f}")
    lines += ["| **all six** | " + " | ".join(summary) + " |", ""]
    (output / f"{prefix}_tables.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines), path


def main(argv=None):
    args = parse_args(argv)
    cells = collect_cells(args.entity, args.human)
    if not cells:
        raise SystemExit(f"No finished cells for human {args.human}")
    self_play = collect_self_play(args.entity)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    figures = {
        "by_kitchen": by_kitchen_figure(
            cells, self_play, output / f"{args.prefix}_by_kitchen", args.human
        ),
        "overall": overall_figure(cells, output / f"{args.prefix}_overall", args.human),
    }
    document, table_path = write_tables(
        cells, self_play, output, args.prefix, args.human
    )
    print(document)
    print(f"saved {table_path} and {len(figures)} figures under {output}")

    if args.wandb_mode == "disabled":
        return 0

    import wandb

    with wandb.init(
        entity=args.entity,
        project=args.project,
        mode=args.wandb_mode,
        name=f"figures-{args.human}",
        job_type="report",
        config=dict(human=args.human, cells=len(cells)),
    ) as run:
        run.log({f"figure/{key}": wandb.Image(str(p)) for key, p in figures.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
