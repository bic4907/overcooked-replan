"""Aggregate the Section 6.2 probes into the figure's data file.

Reads outputs/notice/probe_<layout>_<arm>_seat<k>.jsonl and writes
outputs/notice/alert_preparation_data.json: one record per (layout,
algorithm, endpoint) with the six-seed mean endpoint rate under masked and
available notice, 95% seed-bootstrap intervals, the seed means, the paired
difference and the exact two-sided sign-flip p-value over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, seed_bootstrap_ci, sign_flip_p  # noqa: E402

SCENARIO_KEY = {"split": "Split", "outage": "Outage", "distance": "Distance"}
ENDPOINTS = ("initiation", "completion")


def summarize(rows, endpoint):
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {arm: [] for arm in ("OFF", "ON")}
    for seed in seeds:
        for arm in per_seed:
            values = [r[endpoint] for r in rows if r["seed"] == seed and r["notice"] == arm]
            per_seed[arm].append(float(np.mean(values)))
    off = np.array(per_seed["OFF"])
    on = np.array(per_seed["ON"])
    delta = on - off
    off_lo, off_hi = seed_bootstrap_ci(off)
    on_lo, on_hi = seed_bootstrap_ci(on)
    d_lo, d_hi = seed_bootstrap_ci(delta)
    return dict(
        OFF=float(off.mean()),
        ON=float(on.mean()),
        OFF_ci95=[float(off_lo), float(off_hi)],
        ON_ci95=[float(on_lo), float(on_hi)],
        seed_rates=[[float(a), float(b)] for a, b in zip(off, on)],
        delta=float(delta.mean()),
        delta_ci95=[float(d_lo), float(d_hi)],
        p_raw=float(sign_flip_p(delta)),
        n_pairs=int(sum(1 for r in rows if r["notice"] == "ON")),
        seeds=len(seeds),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in-dir", default=str(ROOT / "outputs/notice"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    in_dir = Path(args.in_dir)
    out = Path(args.out) if args.out else in_dir / "alert_preparation_data.json"

    records = []
    for path in sorted(in_dir.glob("probe_*_seat*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        if not rows:
            continue
        head = rows[0]
        extras = {}
        if head["family"] == "split":
            extras = {"partner_stayed": "partner_stayed"}
        if head["family"] == "distance":
            extras = {"role_initiation": 1, "role_completion": 1}
        for endpoint in ENDPOINTS + tuple(extras):
            stats = summarize(rows, endpoint)
            records.append(
                dict(
                    scenario=SCENARIO_KEY[head["family"]],
                    scenario_name=head["scenario"],
                    layout=head["layout"],
                    algorithm=head["algorithm"],
                    arm=head["arm"],
                    endpoint=endpoint,
                    informed_seat=int(path.stem.split("seat")[-1]),
                    switch=sorted({r.get("switch", "") for r in rows}),
                    **stats,
                )
            )
    out.write_text(json.dumps(records, indent=1))

    print(
        f"{'layout':<11}{'seat':<5}{'algorithm':<10}{'endpoint':<16}"
        f"{'masked':>8}{'notice':>8}{'delta':>8}{'p':>9}{'pairs':>7}"
    )
    for r in records:
        print(
            f"{r['layout']:<11}{r['informed_seat']:<5}{r['algorithm']:<10}{r['endpoint']:<16}"
            f"{100 * r['OFF']:>7.2f}%{100 * r['ON']:>7.2f}%{100 * r['delta']:>+7.2f}"
            f"{r['p_raw']:>9.5f}{r['n_pairs']:>7}"
        )
    print(f"wrote {len(records)} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
