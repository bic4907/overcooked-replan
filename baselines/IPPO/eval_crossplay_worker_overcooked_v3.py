"""Internal long-lived GPU worker for Overcooked V3 cross-play pairs."""

import argparse
import json
import logging
from pathlib import Path

try:
    from .eval_wandb_crossplay_matrix_overcooked_v3 import (
        _atomic_write_json,
        evaluate_pair_task,
    )
except ImportError:  # Direct execution from the repository root.
    from eval_wandb_crossplay_matrix_overcooked_v3 import (
        _atomic_write_json,
        evaluate_pair_task,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-label", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | {args.worker_label} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("crossplay-worker")
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    records = []
    runtime_cache = {}
    params_cache = {}
    for task in tasks:
        logger.info(
            "[%d/%d] evaluating %s %s x %s",
            task["progress_index"],
            task["total_pairs"],
            task["pair_type"],
            task["agent_0_label"],
            task["agent_1_label"],
        )
        record = evaluate_pair_task(task, runtime_cache, params_cache)
        records.append(record)
        _atomic_write_json(args.output, records)
        logger.info(
            "[%d/%d] result mean=%.2f std=%.2f",
            task["progress_index"],
            task["total_pairs"],
            record["mean_return"],
            record["std_return"],
        )


if __name__ == "__main__":
    main()
