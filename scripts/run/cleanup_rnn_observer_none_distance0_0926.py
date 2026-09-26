"""Audit and optionally remove only archived old observer-none RNN distance_0 runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import wandb


ENTITY = "cilab-overcooked"
OLD_REVISION = "distance-inversion-detour-13x6-v4"
NEW_REVISION = "dual-handoff-recipe-priority-11x7-v1"
TRAIN = "overcooked-v3-ippo-rnn-observer-none-0921_train"
TEST = "overcooked-v3-ippo-rnn-observer-none-0921_test"
TEST10 = "overcooked-v3-ippo-rnn-observer-none-0921_test10"
PROJECT_COUNTS = {TRAIN: 10, TEST: 1, TEST10: 1}


def run_path(entry):
    return f"{ENTITY}/{entry['project']}/{entry['id']}"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_backup(entries, root):
    if not (root / "COMPLETE").is_file() or json.loads((root / "backup_errors.json").read_text()):
        raise ValueError("Historical backup is incomplete")
    for entry in entries:
        base = root / entry["project"] / entry["id"]
        if not (base / "COMPLETE").is_file():
            raise ValueError(f"Missing backup for {run_path(entry)}")
        checksums = json.loads((base / "SHA256.json").read_text())
        for name, expected in checksums.items():
            if digest(base / name) != expected:
                raise ValueError(f"Backup checksum mismatch: {run_path(entry)} {name}")


def selected_ids(run):
    return {item.get("run", "").rsplit("/", 1)[-1]
            for item in run.config.get("selected_models", [])}


def verify_replacements(api, old_ids):
    new_train = {}
    for run in api.runs(f"{ENTITY}/{TRAIN}"):
        c = run.config
        if c.get("CONDITION") != "distance_0" or run.id in old_ids:
            continue
        env = c.get("ENV_KWARGS") or {}
        if (run.state != "finished" or c.get("LAYOUT_REVISION") != NEW_REVISION
                or env.get("transition_observer") != "none"
                or c.get("TOTAL_TIMESTEPS") != 30000000
                or c.get("SEED") not in range(10)):
            raise ValueError(f"Unexpected new distance_0 train run {run.path}")
        if c["SEED"] in new_train:
            raise ValueError(f"Duplicate new seed {c['SEED']}")
        if not any(artifact.type == "checkpoint" for artifact in run.logged_artifacts()):
            raise ValueError(f"Missing train checkpoint artifact {run.path}")
        new_train[c["SEED"]] = run.id
    if set(new_train) != set(range(10)):
        raise ValueError(f"New training seeds incomplete: {sorted(new_train)}")

    for project, seeds in ((TEST, range(6)), (TEST10, range(10))):
        matches = []
        for run in api.runs(f"{ENTITY}/{project}"):
            if run.config.get("layout") == "distance_0" and run.id not in old_ids:
                matches.append(run)
        if len(matches) != 1:
            raise ValueError(f"Expected one replacement distance_0 eval in {project}")
        run = matches[0]
        c = run.config
        if (run.state != "finished" or c.get("transition_observer") != "none"
                or c.get("episodes") != 20 or c.get("training_seeds") != list(seeds)
                or selected_ids(run) != {new_train[seed] for seed in seeds}):
            raise ValueError(f"Replacement eval provenance mismatch: {run.path}")
    print("REPLACEMENTS_OK train=10 test=6x6x20 test10=10x10x20", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())["runs"]
    paths = {run_path(entry) for entry in entries}
    if len(entries) != 12 or len(paths) != 12:
        raise ValueError("Expected exactly 12 distinct historical runs")
    if {project: sum(e["project"] == project for e in entries)
            for project in PROJECT_COUNTS} != PROJECT_COUNTS:
        raise ValueError("Historical project distribution mismatch")
    if {e["project"] for e in entries} != set(PROJECT_COUNTS):
        raise ValueError("Manifest includes a different project")
    old_train = [e for e in entries if e["project"] == TRAIN]
    if {e["seed"] for e in old_train} != set(range(10)):
        raise ValueError("Old training seed inventory mismatch")
    if {e["layout_revision"] for e in old_train} != {OLD_REVISION}:
        raise ValueError("Old training revision mismatch")
    verify_backup(entries, args.backup_root)

    receipts = set()
    if args.receipt.is_file():
        receipts = {json.loads(line)["run"] for line in args.receipt.read_text().splitlines() if line}
    if receipts - paths:
        raise ValueError("Receipt contains an unlisted run")

    api = wandb.Api(timeout=120)
    verify_replacements(api, {e["id"] for e in entries})
    old_train_ids = {e["id"] for e in old_train}
    pending = []
    for entry in entries:
        if run_path(entry) in receipts:
            continue
        run = api.run(run_path(entry))
        c = run.config
        if run.name != entry["name"] or run.state != "finished":
            raise ValueError(f"Historical identity/state mismatch: {run.path}")
        if entry["project"] == TRAIN:
            env = c.get("ENV_KWARGS") or {}
            if (c.get("CONDITION") != "distance_0" or c.get("LAYOUT_REVISION") != OLD_REVISION
                    or c.get("SEED") != entry["seed"] or env.get("transition_observer") != "none"):
                raise ValueError(f"Historical training provenance mismatch: {run.path}")
        else:
            count = 6 if entry["project"] == TEST else 10
            if (c.get("layout") != "distance_0" or c.get("episodes") != 20
                    or c.get("transition_observer") != "none"
                    or c.get("training_seeds") != list(range(count))
                    or selected_ids(run) != {e["id"] for e in old_train if e["seed"] in range(count)}):
                raise ValueError(f"Historical eval provenance mismatch: {run.path}")
        pending.append((entry, run))
        print(f"VERIFIED {run_path(entry)}", flush=True)
    print(f"AUDIT_OK pending={len(pending)} already_deleted={len(receipts)}", flush=True)
    if not args.apply:
        return
    for entry, run in sorted(pending, key=lambda pair: pair[0]["project"] != TEST and pair[0]["project"] != TEST10):
        run.delete(delete_artifacts=False)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("a") as handle:
            handle.write(json.dumps({"run": run_path(entry), "revision": OLD_REVISION}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"DELETED {run_path(entry)}", flush=True)


if __name__ == "__main__":
    main()
