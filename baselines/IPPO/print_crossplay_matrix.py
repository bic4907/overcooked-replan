"""Print a terminal matrix from W&B cross-play evaluation summaries."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the current cross-play mean-return matrix."
    )
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-width", type=int, default=12)
    parser.add_argument("--pair-mode", choices=("all", "cross-only"), default="all")
    return parser.parse_args()


def short_run_id(run_id):
    return run_id.strip("/").split("/")[-1]


def compact_label(label, width):
    if width < 5:
        raise ValueError("label width must be at least 5")
    if len(label) <= width:
        return label
    left = (width - 1) // 2
    right = width - left - 1
    return f"{label[:left]}…{label[-right:]}"


def read_pair_metrics(metrics_dir):
    records = []
    for path in sorted(Path(metrics_dir).glob("pair_*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def build_return_matrix(records, run_ids):
    labels = [short_run_id(run_id) for run_id in run_ids]
    indices = {run_id: index for index, run_id in enumerate(labels)}
    if len(indices) != len(labels):
        raise ValueError("Cross-play run IDs must be unique")

    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for record in records:
        row = indices.get(record.get("agent_0_id"))
        column = indices.get(record.get("agent_1_id"))
        if row is not None and column is not None:
            matrix[row, column] = float(record["eval/mean_return"])
    return labels, matrix


def format_return_matrix(records, run_ids, label_width=12, pair_mode="all"):
    labels, matrix = build_return_matrix(records, run_ids)
    display_labels = [compact_label(label, label_width) for label in labels]
    value_width = max(label_width, 10)
    completed = int(np.isfinite(matrix).sum())
    expected = matrix.size if pair_mode == "all" else matrix.size - len(labels)
    row_header = "agent_0 \\ agent_1"
    row_width = max(label_width, len(row_header))

    lines = [
        "",
        f"Cross-play mean return ({completed}/{expected})",
        f"{row_header.rjust(row_width)} │ "
        + " │ ".join(label.rjust(value_width) for label in display_labels),
        "─" * row_width + "─┼─" + "─┼─".join("─" * value_width for _ in labels),
    ]
    for row, label in enumerate(display_labels):
        values = []
        for column, value in enumerate(matrix[row]):
            if pair_mode == "cross-only" and row == column:
                text = "×"
            else:
                text = "·" if not np.isfinite(value) else f"{value:.2f}"
            values.append(text.rjust(value_width))
        lines.append(f"{label.rjust(row_width)} │ {' │ '.join(values)}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    records = read_pair_metrics(args.metrics_dir)
    output = format_return_matrix(
        records,
        args.run_ids,
        label_width=args.label_width,
        pair_mode=args.pair_mode,
    )
    print(output)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")
        if sys.stdout.isatty():
            print(f"Matrix snapshot: {args.output}")


if __name__ == "__main__":
    main()
