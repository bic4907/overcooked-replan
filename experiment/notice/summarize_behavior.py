"""Aggregate the Section 6.3 paired episodes: layout means, the
across-layout correlation, and the cluster scores for the partial-r
sign-flip test.

Reads outputs/notice/behavior_<layout>_<arm>.jsonl (window rows) and the
matching .episodes.jsonl, and writes for one arm:

    behavior_summary_<arm>.json          layout means and paired seed tests
    partial_r_cluster_scores_<arm>.csv   one row per (layout, informed seed)
                                         for partial_r_signflip.py

    python experiment/notice/summarize_behavior.py --arm rnn
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import BENCHMARK_LAYOUTS, ROOT, sign_flip_p  # noqa: E402

ARMS = ("control", "notice", "reference")


def load(in_dir, layout, arm):
    path = Path(in_dir) / f"behavior_{layout}_{arm}.jsonl"
    if not path.exists():
        return None, None
    rows = [json.loads(line) for line in path.open()]
    episodes = [
        json.loads(line)
        for line in path.with_name(f"behavior_{layout}_{arm}.episodes.jsonl").open()
    ]
    return rows, episodes


def finite_mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def macro_drop(rows, name):
    """Direction-macro-averaged mean Drop, the paper's convention."""
    by_direction = defaultdict(list)
    for r in rows:
        if r[f"drop_{name}"] is not None:
            by_direction[r[f"direction_{name}"]].append(r[f"drop_{name}"])
    return finite_mean([np.mean(v) for v in by_direction.values()])


def layout_summary(rows, episodes):
    out = {}
    for name in ("control", "notice"):
        out[f"jsd_{name}"] = finite_mean(r[f"jsd_{name}"] for r in rows)
        out[f"jsd_{name}_n"] = int(sum(r[f"jsd_{name}"] is not None for r in rows))
    for name in ARMS:
        out[f"drop_{name}"] = macro_drop(rows, name)
        out[f"drop_{name}_n"] = int(sum(r[f"drop_{name}"] is not None for r in rows))
        out[f"recovery_{name}"] = finite_mean(r[f"recovery_{name}"] for r in rows)
        out[f"return_{name}"] = finite_mean(e[f"return_{name}"] for e in episodes)
    out["episodes"] = len(episodes)
    out["windows"] = len(rows)
    out["eligible_pairs"] = int(sum(r["eligible_control"] and r["eligible_notice"] for r in rows))
    # Seed-level paired tests, the informed policy's training seed as the unit.
    seeds = sorted({r["informed_seed"] for r in rows})
    per_seed = defaultdict(list)
    for seed in seeds:
        mine = [r for r in rows if r["informed_seed"] == seed]
        eps = [e for e in episodes if e["informed_seed"] == seed]
        per_seed["jsd"].append(
            finite_mean(r["jsd_notice"] for r in mine) - finite_mean(r["jsd_control"] for r in mine)
        )
        per_seed["drop"].append(macro_drop(mine, "notice") - macro_drop(mine, "control"))
        per_seed["return"].append(
            finite_mean(e["return_notice"] for e in eps) - finite_mean(e["return_control"] for e in eps)
        )
    for key, deltas in per_seed.items():
        deltas = [float(d) for d in deltas if np.isfinite(d)]
        out[f"delta_{key}_by_seed"] = deltas
        out[f"delta_{key}"] = float(np.mean(deltas)) if deltas else float("nan")
        out[f"delta_{key}_p"] = float(sign_flip_p(deltas)) if len(deltas) > 1 else float("nan")
    return out


def exact_permutation_p(x, y):
    """Two-sided exact permutation p of Pearson r over all orderings of y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(np.corrcoef(x, y)[0, 1])
    count = total = 0
    for perm in itertools.permutations(range(len(y))):
        r = abs(np.corrcoef(x, y[list(perm)])[0, 1])
        count += r >= observed - 1e-12
        total += 1
    return count / total


def cluster_scores(rows_by_layout):
    """Within-cluster centered sums for the partial-r sign-flip test."""
    scores = []
    for layout, rows in rows_by_layout.items():
        for seed in sorted({r["informed_seed"] for r in rows}):
            pairs = [
                r
                for r in rows
                if r["informed_seed"] == seed
                and r["eligible_control"]
                and r["eligible_notice"]
                and r["drop_control"] is not None
                and r["drop_notice"] is not None
            ]
            dx = np.array([r["jsd_notice"] - r["jsd_control"] for r in pairs])
            dy = np.array([r["drop_control"] - r["drop_notice"] for r in pairs])
            dx = dx - dx.mean() if dx.size else dx
            dy = dy - dy.mean() if dy.size else dy
            scores.append(
                dict(
                    layout=layout,
                    training_seed=seed,
                    eligible_episode_pairs=int(dx.size),
                    sum_centered_delta_jsd_sq=float(np.sum(dx * dx)),
                    sum_centered_drop_reduction_sq=float(np.sum(dy * dy)),
                    covariance_score=float(np.sum(dx * dy)),
                )
            )
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", default="rnn")
    parser.add_argument("--in-dir", default=str(ROOT / "outputs/notice"))
    args = parser.parse_args(argv)
    in_dir = Path(args.in_dir)

    layouts = {}
    rows_by_layout = {}
    for layout in BENCHMARK_LAYOUTS:
        rows, episodes = load(in_dir, layout, args.arm)
        if rows is None:
            print(f"  {layout}: no data")
            continue
        rows_by_layout[layout] = rows
        layouts[layout] = layout_summary(rows, episodes)

    summary = dict(arm=args.arm, layouts=layouts)
    # A kitchen where the arm never delivers has no Drop and says nothing
    # about the relation; it is listed but left out of the across-layout part.
    names = [
        l
        for l in layouts
        if np.isfinite(layouts[l]["drop_control"]) and np.isfinite(layouts[l]["drop_notice"])
    ]
    summary["layouts_with_drop"] = names
    if len(names) >= 3:
        across = {}
        for key in (
            "jsd_control",
            "jsd_notice",
            "drop_control",
            "drop_notice",
            "return_control",
            "return_notice",
            "eligible_pairs",
        ):
            across[key] = float(np.mean([layouts[l][key] for l in names]))
        d_jsd = [layouts[l]["jsd_notice"] - layouts[l]["jsd_control"] for l in names]
        d_drop = [layouts[l]["drop_control"] - layouts[l]["drop_notice"] for l in names]
        d_ret = [layouts[l]["return_notice"] - layouts[l]["return_control"] for l in names]
        across["n_layouts_jsd_up"] = int(sum(d > 0 for d in d_jsd))
        across["n_layouts_drop_down"] = int(sum(d > 0 for d in d_drop))
        across["n_layouts_return_up"] = int(sum(d > 0 for d in d_ret))
        across["r_delta_jsd_vs_drop_reduction"] = float(np.corrcoef(d_jsd, d_drop)[0, 1])
        across["p_perm_drop"] = float(exact_permutation_p(d_jsd, d_drop))
        across["r_delta_jsd_vs_return_gain"] = float(np.corrcoef(d_jsd, d_ret)[0, 1])
        across["p_perm_return"] = float(exact_permutation_p(d_jsd, d_ret))
        summary["across_layouts"] = across

    scores = cluster_scores(rows_by_layout)
    csv_path = in_dir / f"partial_r_cluster_scores_{args.arm}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0].keys()))
        writer.writeheader()
        writer.writerows(scores)
    xx = sum(s["sum_centered_delta_jsd_sq"] for s in scores)
    yy = sum(s["sum_centered_drop_reduction_sq"] for s in scores)
    xy = sum(s["covariance_score"] for s in scores)
    summary["partial_r"] = dict(
        clusters=len(scores),
        eligible_episode_pairs=int(sum(s["eligible_episode_pairs"] for s in scores)),
        partial_r=float(xy / math.sqrt(xx * yy)) if xx > 0 and yy > 0 else float("nan"),
        csv=str(csv_path),
    )

    out = in_dir / f"behavior_summary_{args.arm}.json"
    out.write_text(json.dumps(summary, indent=1))

    print(
        f"{'layout':<11}{'JSD ctl':>8}{'JSD ntc':>8}{'Drop ctl':>9}{'Drop ntc':>9}"
        f"{'Ret ctl':>8}{'Ret ntc':>8}{'p_jsd':>8}{'p_drop':>8}{'p_ret':>8}{'pairs':>6}"
    )
    for l in layouts:
        s = layouts[l]
        print(
            f"{l:<11}{s['jsd_control']:>8.3f}{s['jsd_notice']:>8.3f}{s['drop_control']:>9.3f}"
            f"{s['drop_notice']:>9.3f}{s['return_control']:>8.1f}{s['return_notice']:>8.1f}"
            f"{s['delta_jsd_p']:>8.3f}{s['delta_drop_p']:>8.3f}{s['delta_return_p']:>8.3f}"
            f"{s['eligible_pairs']:>6}"
        )
    if "across_layouts" in summary:
        a = summary["across_layouts"]
        print(
            f"{'mean':<11}{a['jsd_control']:>8.3f}{a['jsd_notice']:>8.3f}{a['drop_control']:>9.3f}"
            f"{a['drop_notice']:>9.3f}{a['return_control']:>8.1f}{a['return_notice']:>8.1f}"
        )
        print(
            f"across layouts: r(dJSD, Drop reduction) = {a['r_delta_jsd_vs_drop_reduction']:.3f}, "
            f"exact permutation p = {a['p_perm_drop']:.3f}; "
            f"r(dJSD, return gain) = {a['r_delta_jsd_vs_return_gain']:.3f}, p = {a['p_perm_return']:.3f}"
        )
        print(
            f"JSD up in {a['n_layouts_jsd_up']}, Drop down in {a['n_layouts_drop_down']}, "
            f"return up in {a['n_layouts_return_up']} of {len(names)} layouts with defined Drop "
            f"({', '.join(names)})"
        )
    print(
        f"partial r over {summary['partial_r']['clusters']} clusters / "
        f"{summary['partial_r']['eligible_episode_pairs']} pairs: {summary['partial_r']['partial_r']:.4f}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
