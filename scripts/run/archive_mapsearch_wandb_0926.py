"""Inventory, archive, verify, and selectively remove the 0925 mapsearch projects."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ENTITY = "cilab-overcooked"
PREFIX = "overcooked-v3-mapsearch-0925-d10-priority-"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(api):
    projects = []
    runs = []
    for project in api.projects(entity=ENTITY):
        if "mapsearch" not in project.name:
            continue
        if not project.name.startswith(PREFIX):
            raise RuntimeError(f"Unexpected mapsearch project: {project.name}")
        items = []
        for run in api.runs(f"{ENTITY}/{project.name}"):
            items.append({"project": project.name, "id": run.id, "name": run.name,
                          "state": run.state, "created_at": run.created_at})
        items.sort(key=lambda item: item["id"])
        projects.append({"name": project.name, "id": project.id,
                         "run_ids": [item["id"] for item in items]})
        runs.extend(items)
    projects.sort(key=lambda item: item["name"])
    runs.sort(key=lambda item: (item["project"], item["id"]))
    return {"entity": ENTITY, "prefix": PREFIX,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "projects": projects, "runs": runs}


def archive_run(api, item, root):
    destination = root / "runs" / item["project"] / item["id"]
    if (destination / "COMPLETE").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    run = api.run(f"{ENTITY}/{item['project']}/{item['id']}")
    if run.id != item["id"] or run.state != item["state"]:
        raise RuntimeError(f"Run identity/state changed: {item['project']}/{item['id']}")
    write_json(destination / "metadata.json", {
        "entity": ENTITY, "project": item["project"], "id": run.id,
        "name": run.name, "state": run.state, "url": run.url,
        "created_at": run.created_at, "config": dict(run.config),
        "summary": dict(run.summary), "tags": list(run.tags),
    })
    with (destination / "history.jsonl").open("w") as output:
        for row in run.scan_history(page_size=1000):
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    for file in run.files():
        file.download(root=str(destination / "run_files"), replace=True)
    artifacts = []
    for artifact in run.logged_artifacts():
        safe_name = artifact.name.replace("/", "_").replace(":", "__")
        artifact.download(root=str(destination / "artifacts" / safe_name))
        artifacts.append({"name": artifact.name, "type": artifact.type,
                          "size": artifact.size, "digest": artifact.digest,
                          "id": artifact.id})
    write_json(destination / "artifacts.json", artifacts)
    checksums = {str(path.relative_to(destination)): digest(path)
                 for path in sorted(destination.rglob("*"))
                 if path.is_file() and path.name not in {"SHA256.json", "COMPLETE"}}
    write_json(destination / "SHA256.json", checksums)
    (destination / "COMPLETE").write_text("archived and checksummed\n")
    print(f"archived {item['project']}/{item['id']}: {len(checksums)} files, {len(artifacts)} artifacts", flush=True)


def verify(root, manifest):
    if not (root / "COMPLETE").exists():
        raise RuntimeError("Archive is not marked COMPLETE")
    if json.loads((root / "backup_errors.json").read_text()):
        raise RuntimeError("Backup has unresolved errors")
    if json.loads((root / "manifest.json").read_text())["projects"] != manifest["projects"]:
        raise RuntimeError("Manifest differs from backup")
    count = 0
    for item in manifest["runs"]:
        folder = root / "runs" / item["project"] / item["id"]
        if not (folder / "COMPLETE").exists():
            raise RuntimeError(f"Incomplete run {item['project']}/{item['id']}")
        checksums = json.loads((folder / "SHA256.json").read_text())
        actual = {str(path.relative_to(folder)): digest(path)
                  for path in sorted(folder.rglob("*"))
                  if path.is_file() and path.name not in {"SHA256.json", "COMPLETE"}}
        if actual != checksums:
            raise RuntimeError(f"Checksum mismatch {item['project']}/{item['id']}")
        count += len(actual)
    print(f"VERIFIED {len(manifest['projects'])} projects, {len(manifest['runs'])} runs, {count} files", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inventory", "backup", "verify", "delete"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root
    if args.mode == "verify":
        api = None
    else:
        import wandb
        api = wandb.Api(timeout=120)
    if args.mode == "inventory":
        current = inventory(api)
        if len(current["projects"]) != 18 or len(current["runs"]) != 56:
            raise RuntimeError(f"Unexpected inventory size: {len(current['projects'])} projects / {len(current['runs'])} runs")
        if any(item["state"] == "running" for item in current["runs"]):
            raise RuntimeError("A mapsearch run is still running")
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", current)
        print(f"INVENTORIED {len(current['projects'])} projects, {len(current['runs'])} runs", flush=True)
        return
    manifest = json.loads((root / "manifest.json").read_text())
    if args.mode == "backup":
        errors = []
        for item in manifest["runs"]:
            try:
                archive_run(api, item, root)
            except Exception as error:
                errors.append({"project": item["project"], "id": item["id"], "error": str(error)})
                print(f"FAILED {item['project']}/{item['id']}: {error}", file=sys.stderr, flush=True)
        write_json(root / "backup_errors.json", errors)
        if errors:
            raise RuntimeError(f"Incomplete backup: {len(errors)} runs")
        (root / "COMPLETE").write_text(f"{len(manifest['runs'])} runs archived\n")
        verify(root, manifest)
        return
    verify(root, manifest)
    if args.mode == "verify":
        return
    current = inventory(api)
    if current["projects"] != manifest["projects"] or current["runs"] != manifest["runs"]:
        raise RuntimeError("W&B projects/runs changed since inventory; refusing deletion")
    print(f"DELETE AUDIT PASSED: {len(current['projects'])} exact mapsearch projects, {len(current['runs'])} exact runs", flush=True)
    if not args.apply:
        return
    query = "mutation DeleteModel($input: DeleteModelInput!) { deleteModel(input: $input) { success } }"
    receipt = root / "deletion_receipt.jsonl"
    already = set()
    if receipt.exists():
        already = {json.loads(line)["name"] for line in receipt.read_text().splitlines() if line}
    for project in manifest["projects"]:
        if project["name"] in already:
            continue
        response = api._service_api.execute_graphql(query, {"input": {"id": project["id"]}})
        if not response["deleteModel"]["success"]:
            raise RuntimeError(f"Delete failed: {project['name']}")
        item = {"name": project["name"], "id": project["id"],
                "run_ids": project["run_ids"], "deleted_at": datetime.now(timezone.utc).isoformat(),
                "success": True}
        with receipt.open("a") as output:
            output.write(json.dumps(item) + "\n")
            output.flush()
            os.fsync(output.fileno())
        print(f"DELETED {project['name']}", flush=True)
    # The project list on the deletion client can be stale; use a fresh API client.
    remaining = {item.name for item in wandb.Api(timeout=120).projects(entity=ENTITY)}
    if remaining.intersection(item["name"] for item in manifest["projects"]):
        raise RuntimeError("Some mapsearch projects remain visible after deletion")
    print("DELETION VERIFIED", flush=True)


if __name__ == "__main__":
    main()
