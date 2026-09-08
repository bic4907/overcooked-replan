"""Summarize Easy-1 SP/XP evaluation sweeps and screen their gaps.

The script is intentionally read-only: it queries W&B and prints a Markdown
table.  Exit status is zero only when every expected map has a finished run and
passes the configured absolute and relative SP-XP gap thresholds.
"""

from __future__ import annotations

import argparse
import math
import os

import wandb


DEFAULT_ENTITY = "cilab-overcooked"
DEFAULT_LAYOUTS = ("split_1", "outage_1", "distance_switch_1")
DEFAULT_SWEEPS = {
    "IPPO": ("overcooked-v3-ippo-easy1_eval", "j9y1n577"),
    "IPPO-RNN": ("overcooked-v3-ippo-rnn-easy1_eval", "7wgsgrh5"),
    "FCP": ("overcooked-v3-fcp-easy1_eval", "040dx6ha"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report per-map SP, XP, and SP-XP gaps from Easy-1 eval sweeps."
    )
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", DEFAULT_ENTITY)
    )
    parser.add_argument("--layouts", nargs="+", default=list(DEFAULT_LAYOUTS))
    parser.add_argument(
        "--min-absolute-gap",
        type=float,
        default=10.0,
        help="Minimum SP-XP return gap counted as clearly visible (default: 10).",
    )
    parser.add_argument(
        "--min-relative-gap",
        type=float,
        default=0.05,
        help="Minimum (SP-XP)/abs(SP) counted as clearly visible (default: 0.05).",
    )
    return parser.parse_args()


def finite_number(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def classify(sp: float, xp: float, min_absolute: float, min_relative: float):
    gap = sp - xp
    relative_gap = gap / max(abs(sp), 1.0)
    if gap <= 0:
        decision = "FAIL (XP>=SP)"
    elif gap < min_absolute or relative_gap < min_relative:
        decision = "REVIEW (small gap)"
    else:
        decision = "PASS"
    return gap, relative_gap, decision


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    rows = []
    all_pass = True

    for algorithm, (project, sweep_id) in DEFAULT_SWEEPS.items():
        sweep = api.sweep(f"{args.entity}/{project}/{sweep_id}")
        runs_by_layout = {}
        for run in sweep.runs:
            layout = (run.config or {}).get("layout")
            if layout in args.layouts:
                runs_by_layout.setdefault(layout, []).append(run)

        for layout in args.layouts:
            runs = runs_by_layout.get(layout, [])
            finished = [run for run in runs if run.state == "finished"]
            if not finished:
                state = runs[0].state if runs else sweep.state.lower()
                rows.append((algorithm, layout, state, None, None, None, None, "INCOMPLETE"))
                all_pass = False
                continue

            run = finished[0]
            summary = run.summary or {}
            sp = finite_number(summary.get("SP"))
            xp = finite_number(summary.get("XP"))
            if sp is None or xp is None:
                rows.append((algorithm, layout, run.state, sp, xp, None, None, "INVALID"))
                all_pass = False
                continue

            gap, relative_gap, decision = classify(
                sp, xp, args.min_absolute_gap, args.min_relative_gap
            )
            rows.append(
                (algorithm, layout, run.state, sp, xp, gap, relative_gap, decision)
            )
            all_pass = all_pass and decision == "PASS"

    print("| Algorithm | Map | State | SP | XP | SP-XP | Gap / SP | Decision |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for algorithm, layout, state, sp, xp, gap, relative_gap, decision in rows:
        values = [
            algorithm,
            layout,
            state,
            "-" if sp is None else f"{sp:.2f}",
            "-" if xp is None else f"{xp:.2f}",
            "-" if gap is None else f"{gap:.2f}",
            "-" if relative_gap is None else f"{100 * relative_gap:.1f}%",
            decision,
        ]
        print("| " + " | ".join(values) + " |")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
