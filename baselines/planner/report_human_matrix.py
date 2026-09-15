"""Draw the human-model matrix out of the cells the sweep ran.

Every cell of the sweep is its own W&B run, which is the right shape for
running it and the wrong shape for reading it. This gathers them into one run:
a table of every cell, one heatmap per kitchen, and a bar chart of what each
partner is worth averaged over the simulated humans.

    python -m baselines.planner.report_human_matrix --sweep pd504s7p
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HUMANS = ("br", "h0", "h1", "h2")
PARTNERS = ("br", "h0", "h1", "h2", "cnn", "rnn", "fcp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked"))
    parser.add_argument("--project", default="overcooked-v3-obp-human-matrix")
    parser.add_argument("--sweep", default=None, help="Read one sweep instead of the project.")
    parser.add_argument("--output", default="outputs/human_matrix", help="Markdown and CSV go here.")
    parser.add_argument(
        "--merge",
        action="append",
        default=[],
        help="A cells.csv from an earlier run, for cells this sweep does not "
        "cover; runs read from W&B win where both have a cell.",
    )
    parser.add_argument("--name", default="matrix", help="Name of the run this writes.")
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    return parser.parse_args(argv)


def _as_int(value):
    """A count, or zero when the key holds something that is not one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def collect(args):
    """One row per finished cell: kitchen, human, partner, mean return."""
    import wandb

    api = wandb.Api()
    if args.sweep:
        source = api.sweep(f"{args.entity}/{args.project}/{args.sweep}").runs
    else:
        source = api.runs(f"{args.entity}/{args.project}", per_page=500)
    rows = []
    for run in source:
        if run.state != "finished":
            continue
        summary = run.summary
        if "return_mean" not in summary:
            continue
        config = run.config
        rows.append(
            dict(
                layout=str(summary.get("layout", config.get("layout"))),
                human=str(summary.get("human", config.get("human"))),
                partner=str(summary.get("partner", config.get("partner"))),
                return_mean=float(summary["return_mean"]),
                return_std=float(summary.get("return_std", float("nan"))),
                games=_as_int(summary.get("games")),
            )
        )
    return rows


def matrix(rows, layout=None):
    """Humans down the side, partners across, means in the cells."""
    grid = np.full((len(HUMANS), len(PARTNERS)), np.nan)
    for row in rows:
        if layout is not None and row["layout"] != layout:
            continue
        if row["human"] in HUMANS and row["partner"] in PARTNERS:
            here = (HUMANS.index(row["human"]), PARTNERS.index(row["partner"]))
            # Averaged when several kitchens land in the same cell.
            grid[here] = (
                row["return_mean"]
                if np.isnan(grid[here])
                else (grid[here] + row["return_mean"]) / 2
            )
    return grid


def as_one_table(rows, layouts, humans, partners):
    """Every kitchen in a single table: a block of rows per kitchen.

    Ten tables of four rows each is ten things to scroll past; one table with
    the kitchen in the first column is one thing to read, and the columns line
    up so a partner can be followed down the page.
    """
    lines = [
        "| kitchen | human | " + " | ".join(partners) + " |",
        "|---|---" + "|---:" * len(partners) + "|",
    ]
    cells = {(r["layout"], r["human"], r["partner"]): r["return_mean"] for r in rows}
    for layout in layouts:
        for index, human in enumerate(humans):
            name = layout if index == 0 else ""
            values = [
                "-" if (layout, human, partner) not in cells
                else f"{cells[(layout, human, partner)]:.0f}"
                for partner in partners
            ]
            lines.append(f"| {name} | **{human}** | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def as_markdown(grid, title):
    lines = [f"### {title}", "", "| human | " + " | ".join(PARTNERS) + " |", "|---" * (len(PARTNERS) + 1) + "|"]
    for index, human in enumerate(HUMANS):
        cells = [
            "-" if np.isnan(value) else f"{value:.0f}" for value in grid[index]
        ]
        lines.append(f"| {human} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def heatmap(grid, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(1.1 * len(PARTNERS) + 2, 1.0 * len(HUMANS) + 1.5))
    image = axes.imshow(grid, cmap="viridis")
    axes.set_xticks(range(len(PARTNERS)), PARTNERS)
    axes.set_yticks(range(len(HUMANS)), HUMANS)
    axes.set_xlabel("partner")
    axes.set_ylabel("simulated human")
    axes.set_title(title)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not np.isnan(grid[y, x]):
                axes.text(x, y, f"{grid[y, x]:.0f}", ha="center", va="center", color="w")
    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main(argv=None):
    args = parse_args(argv)
    rows = collect(args)
    seen = {(row["layout"], row["human"], row["partner"]) for row in rows}
    for path in args.merge:
        import csv as _csv

        for row in _csv.DictReader(open(path)):
            key = (row["layout"], row["human"], row["partner"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                dict(
                    layout=row["layout"],
                    human=row["human"],
                    partner=row["partner"],
                    return_mean=float(row["return_mean"]),
                    return_std=float(row.get("return_std", "nan")),
                    games=_as_int(row.get("games")),
                )
            )
    if not rows:
        raise SystemExit("No finished cells to report yet")
    layouts = sorted({row["layout"] for row in rows})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    overall = matrix(rows)
    played_humans = [h for h in HUMANS if any(r["human"] == h for r in rows)]
    played_partners = [p for p in PARTNERS if any(r["partner"] == p for r in rows)]
    document = [
        as_markdown(overall, f"All kitchens ({len(layouts)})"),
        "### Kitchen by kitchen",
        "",
        as_one_table(rows, layouts, played_humans, played_partners),
    ]
    images = {"matrix/all": heatmap(overall, "all kitchens", output / "all.png")}
    for layout in layouts:
        images[f"matrix/{layout}"] = heatmap(
            matrix(rows, layout), layout, output / f"{layout}.png"
        )
    (output / "tables.md").write_text("\n".join(document), encoding="utf-8")

    import csv

    with (output / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(document))
    print(f"saved {output}/tables.md, cells.csv and {len(images)} heatmaps")

    if args.wandb_mode == "disabled":
        return 0

    import wandb

    with wandb.init(
        entity=args.entity,
        project=args.project,
        mode=args.wandb_mode,
        name=args.name,
        job_type="report",
        config=dict(sweep=args.sweep, layouts=layouts, cells=len(rows)),
    ) as run:
        run.log(
            {key: wandb.Image(str(path)) for key, path in images.items()}
        )
        table = wandb.Table(
            columns=list(rows[0].keys()), data=[list(row.values()) for row in rows]
        )
        run.log({"cells": table})
        # One bar per partner, averaged over the simulated humans, which is the
        # question the matrix was run to answer.
        by_partner = wandb.Table(columns=["partner", "return_mean"])
        for index, partner in enumerate(PARTNERS):
            column = overall[:, index]
            if not np.all(np.isnan(column)):
                by_partner.add_data(partner, float(np.nanmean(column)))
        run.log(
            {
                "by_partner": wandb.plot.bar(
                    by_partner, "partner", "return_mean", title="Mean return by partner"
                )
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
