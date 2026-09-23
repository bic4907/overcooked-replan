#!/usr/bin/env python3
"""The post-hoc partial-r cluster sign-flip p-value (paper Section 6.3).

Copied from the paper's reproduction script, with the input taken from the
command line and the paper's expected values dropped: it now reports the
statistic of whatever cluster file summarize_behavior.py wrote.

    python experiment/notice/partial_r_signflip.py outputs/notice/partial_r_cluster_scores_rnn.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


import sys

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/notice/partial_r_cluster_scores_rnn.csv")
LAYOUTS = (
    "split_0",
    "split_1",
    "outage_0",
    "outage_1",
    "distance_0",
    "distance_1",
)
TRAINING_SEEDS = tuple(range(6))
MONTE_CARLO_DRAWS = 200_000
RNG_SEED = 20_260_918


def load_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with INPUT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    expected_order = [(layout, seed) for layout in LAYOUTS for seed in TRAINING_SEEDS]
    actual_order = [(row["layout"], int(row["training_seed"])) for row in rows]
    if actual_order != expected_order:
        raise AssertionError("cluster rows are missing, duplicated, or out of canonical order")

    xx = np.asarray([float(row["sum_centered_delta_jsd_sq"]) for row in rows])
    yy = np.asarray([float(row["sum_centered_drop_reduction_sq"]) for row in rows])
    xy = np.asarray([float(row["covariance_score"]) for row in rows])
    eligible_pairs = sum(int(row["eligible_episode_pairs"]) for row in rows)
    return xx, yy, xy, eligible_pairs


def main() -> None:
    xx, yy, scores, eligible_pairs = load_scores()
    signed_statistic = float(scores.sum())
    observed_statistic = abs(signed_statistic)
    partial_r = signed_statistic / math.sqrt(float(xx.sum() * yy.sum()))

    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(MONTE_CARLO_DRAWS, len(scores)))
    null_statistics = np.abs(signs @ scores)
    exceedances = int(np.count_nonzero(null_statistics >= observed_statistic))
    add_one_p = (exceedances + 1) / (MONTE_CARLO_DRAWS + 1)

    print(
        json.dumps(
            {
                "input": INPUT.name,
                "clusters": len(scores),
                "eligible_episode_pairs": eligible_pairs,
                "signed_observed_covariance_statistic": signed_statistic,
                "absolute_observed_covariance_statistic": observed_statistic,
                "partial_r": partial_r,
                "rng": f"numpy.random.default_rng({RNG_SEED})",
                "numpy_version": np.__version__,
                "monte_carlo_draws": MONTE_CARLO_DRAWS,
                "two_sided_criterion": "abs(null) >= abs(observed)",
                "exceedances": exceedances,
                "p_value_formula": "(exceedances + 1) / (draws + 1)",
                "add_one_p_value": add_one_p,
                "reported_p_value_3dp": round(add_one_p, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
