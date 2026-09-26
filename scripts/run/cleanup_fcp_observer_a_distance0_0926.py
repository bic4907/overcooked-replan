"""Audit exact old observer-a FCP distance_0 IDs before optional W&B removal."""

import argparse
import json
import os
from pathlib import Path

import wandb


ENTITY = "cilab-overcooked"
REVISION = "distance-inversion-detour-13x6-v4"
TRAIN_PROJECT = "overcooked-v3-fcp-observer-a-0921_train"
POP_PROJECT = "overcooked-v3-fcp-observer-a-0921_population"
EVAL_PROJECT = "overcooked-v3-fcp-observer-a-0921_eval"


def path(entry):
    return f"{ENTITY}/{entry['project']}/{entry['id']}"


def read_receipts(path_):
    if not path_.exists():
        return set()
    return {json.loads(line)["run"] for line in path_.read_text().splitlines() if line}


def record(path_, entry):
    path_.parent.mkdir(parents=True, exist_ok=True)
    with path_.open("a") as handle:
        handle.write(json.dumps({"run": path(entry), "revision": REVISION}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text())["runs"]
    if len(entries) != 21 or len({path(entry) for entry in entries}) != 21:
        raise ValueError("Expected exactly 21 distinct historical run IDs")
    groups = {project: [entry for entry in entries if entry["project"] == project]
              for project in (TRAIN_PROJECT, POP_PROJECT, EVAL_PROJECT)}
    if [len(groups[project]) for project in (TRAIN_PROJECT, POP_PROJECT, EVAL_PROJECT)] != [10, 10, 1]:
        raise ValueError("Wrong project distribution in historical manifest")
    if {entry["project"] for entry in entries} != set(groups):
        raise ValueError("Manifest includes another project")
    for project in (TRAIN_PROJECT, POP_PROJECT):
        group = groups[project]
        if {entry["seed"] for entry in group} != set(range(10)):
            raise ValueError(f"Wrong seed inventory: {project}")
        if {entry["layout_revision"] for entry in group} != {REVISION}:
            raise ValueError(f"Wrong historical revision: {project}")

    backup = args.backup_root
    if not (backup / "COMPLETE").is_file() or json.loads((backup / "backup_errors.json").read_text()):
        raise ValueError("Historical backup is incomplete")
    for entry in entries:
        base = backup / entry["project"] / entry["id"]
        if not (base / "COMPLETE").is_file() or not (base / "SHA256.json").is_file():
            raise ValueError(f"Historical backup lacks run {path(entry)}")

    receipts = read_receipts(args.receipt)
    if receipts - {path(entry) for entry in entries}:
        raise ValueError("Receipt contains a run outside the manifest")
    api = wandb.Api(timeout=120)
    checked = []
    old_train_ids = {entry["id"] for entry in groups[TRAIN_PROJECT]}
    for entry in entries:
        if path(entry) in receipts:
            continue
        run = api.run(path(entry))
        if run.id != entry["id"] or run.name != entry["name"] or run.state != "finished":
            raise ValueError(f"Live run differs from manifest: {path(entry)}")
        config = run.config
        if entry["project"] == EVAL_PROJECT:
            selected = config.get("selected_models") or []
            selected_ids = {item.get("run", "").rsplit("/", 1)[-1] for item in selected}
            if (config.get("layout") != "distance_0" or config.get("episodes") != 20
                    or config.get("transition_observer") != "agent_0"
                    or len(selected) != 10 or selected_ids != old_train_ids):
                raise ValueError(f"Historical eval provenance mismatch: {path(entry)}")
        else:
            env = config.get("ENV_KWARGS") or {}
            if (config.get("CONDITION") != "distance_0"
                    or config.get("LAYOUT_REVISION") != REVISION
                    or config.get("SEED") != entry["seed"]
                    or env.get("transition_observer") != "agent_0"):
                raise ValueError(f"Historical train/pop provenance mismatch: {path(entry)}")
        checked.append((entry, run))
        print(f"VERIFIED {path(entry)}", flush=True)

    print(f"AUDIT_OK checked={len(checked)} already_deleted={len(receipts)}", flush=True)
    if not args.apply:
        return
    for entry, run in sorted(checked, key=lambda pair: pair[0]["project"] != EVAL_PROJECT):
        run.delete(delete_artifacts=False)
        record(args.receipt, entry)
        print(f"DELETED {path(entry)}", flush=True)


if __name__ == "__main__":
    main()
