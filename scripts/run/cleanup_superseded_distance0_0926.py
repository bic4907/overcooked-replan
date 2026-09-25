"""Audit and selectively remove archived, superseded 0921 distance_0 runs.

The manifest IDs are the only deletion targets. Run without --apply to inspect
all live runs first; --apply records each completed deletion for safe retries.
"""

import argparse
import json
import os
from pathlib import Path

import wandb


ENTITY = "cilab-overcooked"
REVISIONS = {
    "distance-inversion-detour-13x6-v4",
    "dual-handoff-private-pots-11x7-v1",
}
MAIN_PROJECTS = {
    "overcooked-v3-ippo-0921_train",
    "overcooked-v3-ippo-rnn-0921_train",
    "overcooked-v3-fcp-0921_train",
    "overcooked-v3-fcp-0921-population",
    "overcooked-v3-ippo-0921_eval",
    "overcooked-v3-ippo-rnn-0921_eval",
    "overcooked-v3-fcp-0921_eval",
}
OBSERVER_PROJECTS = {
    "overcooked-v3-ippo-rnn-observer-a-0921_train",
    "overcooked-v3-ippo-rnn-observer-a-0921_eval",
}


def read_manifest(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError(f"Invalid manifest: {path}")
    return data["runs"]


def run_path(entry):
    return f"{ENTITY}/{entry['project']}/{entry['id']}"


def check_training(run, entry, observer):
    config = run.config
    env = config.get("ENV_KWARGS") or {}
    if config.get("CONDITION") != "distance_0":
        raise ValueError(f"Wrong condition: {run_path(entry)}")
    if config.get("LAYOUT_REVISION") != entry["layout_revision"]:
        raise ValueError(f"Wrong revision: {run_path(entry)}")
    if config.get("SEED") != entry["seed"]:
        raise ValueError(f"Wrong seed: {run_path(entry)}")
    if env.get("transition_observer", "both") != observer:
        raise ValueError(f"Wrong observer: {run_path(entry)}")


def check_eval(run, entry, allowed_train_ids, observer):
    config = run.config
    selected = config.get("selected_models") or []
    model_ids = {item.get("run", "").rsplit("/", 1)[-1] for item in selected}
    if config.get("layout") != "distance_0":
        raise ValueError(f"Wrong eval layout: {run_path(entry)}")
    if config.get("episodes") != 20:
        raise ValueError(f"Wrong eval episode count: {run_path(entry)}")
    if len(selected) != 10 or len(model_ids) != 10:
        raise ValueError(f"Wrong eval model count: {run_path(entry)}")
    if model_ids != allowed_train_ids:
        raise ValueError(f"Eval references unexpected models: {run_path(entry)}")
    if (config.get("transition_observer") or "both") != observer:
        raise ValueError(f"Wrong eval observer: {run_path(entry)}")


def load_receipts(path):
    if not path.exists():
        return set()
    return {json.loads(line)["run"] for line in path.read_text().splitlines() if line}


def record_receipt(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"run": run_path(entry), "revision": entry["layout_revision"]}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--privatepot-manifest", required=True)
    parser.add_argument("--scope", choices=("main", "observer-a"), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    entries = read_manifest(args.legacy_manifest) + read_manifest(args.privatepot_manifest)
    projects = MAIN_PROJECTS if args.scope == "main" else OBSERVER_PROJECTS
    entries = [entry for entry in entries if entry["project"] in projects]
    expected_count = 78 if args.scope == "main" else 12
    if len(entries) != expected_count:
        raise ValueError(f"Expected {expected_count} {args.scope} runs; got {len(entries)}")
    paths = [run_path(entry) for entry in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate run IDs in manifests")
    expected_revisions = REVISIONS if args.scope == "main" else {"distance-inversion-detour-13x6-v4"}
    actual_revisions = {entry["layout_revision"] for entry in entries if entry["layout_revision"]}
    if actual_revisions != expected_revisions:
        raise ValueError("Unexpected manifest revisions")

    receipts = load_receipts(args.receipt)
    if receipts - set(paths):
        raise ValueError("Receipt contains a run outside selected manifests")
    api = wandb.Api(timeout=120)
    checked = []
    for entry in entries:
        path = run_path(entry)
        if path in receipts:
            continue
        run = api.run(path)
        if run.id != entry["id"] or run.name != entry["name"] or run.state != "finished":
            raise ValueError(f"Manifest/live run mismatch: {path}")
        observer = "agent_0" if args.scope == "observer-a" else "both"
        if entry["project"].endswith("_eval"):
            source_project = run.config.get("source_project", "").rsplit("/", 1)[-1]
            model_ids = {
                item.get("run", "").rsplit("/", 1)[-1]
                for item in (run.config.get("selected_models") or [])
            }
            matching = [
                other for other in entries
                if other["id"] in model_ids and other["project"] == source_project
                and other["project"].endswith("_train")
            ]
            revisions = {other["layout_revision"] for other in matching}
            if len(matching) != 10 or len(revisions) != 1:
                raise ValueError(f"Cannot establish eval source revision: {path}")
            if next(iter(revisions)) not in expected_revisions:
                raise ValueError(f"Unexpected eval source revision: {path}")
            allowed = {other["id"] for other in matching}
            check_eval(run, entry, allowed, observer)
        else:
            check_training(run, entry, observer)
        checked.append((entry, run))
        print(f"VERIFIED {path}", flush=True)

    print(f"AUDIT_OK scope={args.scope} checked={len(checked)} already_deleted={len(receipts)}", flush=True)
    if not args.apply:
        return
    # Remove evals first so live eval provenance never points to a removed run.
    for entry, run in sorted(checked, key=lambda pair: not pair[0]["project"].endswith("_eval")):
        run.delete(delete_artifacts=False)
        record_receipt(args.receipt, entry)
        print(f"DELETED {run_path(entry)}", flush=True)


if __name__ == "__main__":
    main()
