"""Wait-safe checkpoint inventory for the split observer-none RNN training."""

import json
import sys
from pathlib import Path

import wandb


PROJECT = "cilab-overcooked/overcooked-v3-ippo-rnn-observer-none-0921_train"
REVISION = "dual-handoff-recipe-priority-11x7-v1"


def main():
    output = Path(sys.argv[1])
    api = wandb.Api(timeout=90)
    filters = {
        "config.ENV_KWARGS.layout": "distance_0",
        "config.ENV_KWARGS.transition_observer": "none",
        "config.LAYOUT_REVISION": REVISION,
        "state": "finished",
    }
    by_seed = {}
    for run in api.runs(PROJECT, filters=filters, per_page=100):
        config = run.config
        seed = config.get("SEED")
        if (seed not in range(10) or config.get("TOTAL_TIMESTEPS") != 30000000
                or config.get("ARCHITECTURE") != "rnn"):
            raise ValueError(f"Unexpected finished distance_0 run {run.path}")
        if not any(a.type == "checkpoint" for a in run.logged_artifacts()):
            # An interrupted run can settle to finished without uploading its model.
            continue
        if seed in by_seed:
            raise ValueError(f"Duplicate complete seed {seed}: {by_seed[seed]['id']} and {run.id}")
        by_seed[seed] = {"id": run.id, "seed": seed, "url": run.url}
    if set(by_seed) != set(range(10)):
        print(f"WAIT complete checkpoint seeds={sorted(by_seed)}", flush=True)
        return 10
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"project": PROJECT, "layout_revision": REVISION,
                                  "models": [by_seed[i] for i in range(10)]}, indent=2) + "\n")
    print(f"READY 10 unique checkpoint seeds: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
