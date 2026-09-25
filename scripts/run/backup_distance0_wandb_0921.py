"""Archive the exact legacy distance_0 W&B runs before selective deletion."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import wandb


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_run(api, record, root):
    project, run_id = record["project"], record["id"]
    destination = root / project / run_id
    if (destination / "COMPLETE").exists():
        print(f"skip complete {project}/{run_id}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    run = api.run(f"cilab-overcooked/{project}/{run_id}")
    config = dict(run.config)
    env = config.get("ENV_KWARGS") or {}
    layout = config.get("CONDITION") or env.get("layout") or config.get("layout")
    if layout != "distance_0":
        raise ValueError(f"Refusing non-distance_0 run: {project}/{run_id}: {layout}")
    if config.get("LAYOUT_REVISION") != record["layout_revision"]:
        raise ValueError(f"Layout revision changed for {project}/{run_id}")

    write_json(
        destination / "metadata.json",
        {
            "entity": "cilab-overcooked",
            "project": project,
            "id": run_id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "created_at": run.created_at,
            "config": config,
            "summary": dict(run.summary),
            "tags": list(run.tags),
        },
    )
    with (destination / "history.jsonl").open("w") as output:
        for row in run.scan_history(page_size=1000):
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    for file in run.files():
        file.download(root=str(destination / "run_files"), replace=True)
    artifacts = []
    for artifact in run.logged_artifacts():
        safe_name = artifact.name.replace("/", "_").replace(":", "__")
        artifact.download(root=str(destination / "artifacts" / safe_name))
        artifacts.append(
            {
                "name": artifact.name,
                "type": artifact.type,
                "size": artifact.size,
                "digest": artifact.digest,
            }
        )
    write_json(destination / "artifacts.json", artifacts)
    checksums = {
        str(path.relative_to(destination)): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name not in {"SHA256.json", "COMPLETE"}
    }
    write_json(destination / "SHA256.json", checksums)
    (destination / "COMPLETE").write_text("archived and checksummed\n")
    print(
        f"archived {project}/{run_id}: {len(checksums)} files, "
        f"{len(artifacts)} artifacts",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    api = wandb.Api(timeout=90)
    errors = []
    for record in manifest["runs"]:
        try:
            archive_run(api, record, args.output)
        except Exception as error:
            errors.append((record["project"], record["id"], str(error)))
            print(f"FAILED {record['project']}/{record['id']}: {error}", file=sys.stderr)
    write_json(args.output / "backup_errors.json", errors)
    if errors:
        raise SystemExit(f"Incomplete W&B backup: {len(errors)} run(s) failed")
    (args.output / "COMPLETE").write_text(f"{len(manifest['runs'])} runs archived\n")


if __name__ == "__main__":
    main()
