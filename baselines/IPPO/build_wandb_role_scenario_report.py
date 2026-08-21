"""Build a per-map W&B report combining rollout videos and XP matrices."""

import argparse
import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path

import wandb
import yaml

from jaxmarl._env import load_project_env


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = ROOT / "experiment/self_play/train.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download one training rollout and the latest cross-play matrix for "
            "each selected layout, then create a local and W&B summary report."
        )
    )
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked")
    )
    parser.add_argument("--training-project", default="overcooked-v3-ippo_train")
    parser.add_argument("--crossplay-project", default="overcooked-v3-ippo_eval")
    parser.add_argument(
        "--output-project",
        default="overcooked-v3-ippo_eval",
        help="Project receiving the combined media table.",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        help="Layouts to include. Defaults to the training sweep selection.",
    )
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument(
        "--video-seed",
        type=int,
        default=0,
        help="Preferred training seed for the sample rollout (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to saves/reports/role-scenarios-<timestamp>.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return parser.parse_args(argv)


def selected_layouts(path):
    sweep = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(sweep["parameters"]["scenario"]["values"])


def _run_seed(run):
    config = dict(getattr(run, "config", {}) or {})
    value = config.get("SEED", config.get("seed"))
    return None if value is None else int(value)


def _media_file(run, suffix, namespace):
    return next(
        (
            file
            for file in run.files()
            if namespace in file.name and file.name.lower().endswith(suffix)
        ),
        None,
    )


def select_training_video_run(runs, preferred_seed=0):
    """Select the preferred seed without silently returning a run lacking video."""
    candidates = []
    for run in runs:
        media = _media_file(run, ".mp4", "visualization/final_episode")
        if media is not None:
            candidates.append((run, media))
    preferred = [item for item in candidates if _run_seed(item[0]) == preferred_seed]
    return (
        preferred[0] if preferred else (candidates[0] if candidates else (None, None))
    )


def select_crossplay_matrix_run(runs):
    """Select the newest run containing both aggregate metrics and a matrix PNG."""
    for run in runs:
        summary = dict(getattr(run, "summary", {}) or {})
        if any(summary.get(key) is None for key in ("SP", "XP", "SP-XP_gap")):
            continue
        media = _media_file(run, ".png", "matrices/models")
        if media is None:
            media = _media_file(run, ".png", "matrices/algorithms")
        if media is not None:
            return run, media
    return None, None


def _download(file, root):
    root = Path(root)
    file.download(root=str(root), replace=True)
    path = root / file.name
    if not path.is_file():
        raise FileNotFoundError(f"W&B download did not create {path}")
    return path


def _query_runs(api, project_path, filters):
    return list(
        api.runs(project_path, filters=filters, order="-created_at", per_page=50)
    )


def collect_layout(api, args, layout, output_dir):
    training_runs = _query_runs(
        api,
        f"{args.entity}/{args.training_project}",
        {"config.ENV_KWARGS.layout": layout, "state": "finished"},
    )
    training_run, video_file = select_training_video_run(training_runs, args.video_seed)
    if training_run is None:
        raise FileNotFoundError(f"No final-episode video found for {layout}")

    crossplay_runs = _query_runs(
        api,
        f"{args.entity}/{args.crossplay_project}",
        {"config.layout": layout, "state": "finished"},
    )
    crossplay_run, matrix_file = select_crossplay_matrix_run(crossplay_runs)
    if crossplay_run is None:
        raise FileNotFoundError(f"No completed cross-play matrix found for {layout}")

    media_dir = Path(output_dir) / "media" / layout
    video_path = _download(video_file, media_dir / "training")
    matrix_path = _download(matrix_file, media_dir / "crossplay")
    summary = dict(crossplay_run.summary or {})
    return {
        "map": layout,
        "video_seed": _run_seed(training_run),
        "video_run": training_run.id,
        "video_path": video_path,
        "matrix_run": crossplay_run.id,
        "matrix_path": matrix_path,
        "SP": float(summary["SP"]),
        "XP": float(summary["XP"]),
        "SP-XP_gap": float(summary["SP-XP_gap"]),
    }


def write_csv(path, records):
    columns = (
        "map",
        "SP",
        "XP",
        "SP-XP_gap",
        "video_seed",
        "video_run",
        "matrix_run",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record[key] for key in columns} for record in records)


def write_html(path, records):
    path = Path(path)
    rows = []
    for record in records:
        video = os.path.relpath(record["video_path"], path.parent)
        matrix = os.path.relpath(record["matrix_path"], path.parent)
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(record['map'])}</strong></td>"
            f'<td><video controls preload="metadata" src="{html.escape(video)}"></video>'
            f"<small>seed={record['video_seed']} · run={record['video_run']}</small></td>"
            f'<td><img src="{html.escape(matrix)}" alt="{html.escape(record["map"])} matrix">'
            f"<small>run={record['matrix_run']}</small></td>"
            f"<td>{record['SP']:.2f}</td><td>{record['XP']:.2f}</td>"
            f"<td>{record['SP-XP_gap']:.2f}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Overcooked V3 role scenarios</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
th {{ position: sticky; top: 0; background: #20242b; color: white; }}
video, img {{ display: block; width: min(420px, 38vw); margin: auto; }}
small {{ display: block; margin-top: 6px; color: #666; }}
</style></head><body>
<h1>Overcooked V3 role-scenario SP/XP report</h1>
<table><thead><tr><th>Map</th><th>Sample rollout</th><th>Payoff matrix</th>
<th>SP</th><th>XP</th><th>SP−XP</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></body></html>"""
    path.write_text(document, encoding="utf-8")


def log_wandb_report(args, records, output_dir):
    columns = ["map", "sample_rollout", "payoff_matrix", "SP", "XP", "SP-XP_gap"]
    data = [
        [
            record["map"],
            wandb.Video(str(record["video_path"]), format="mp4"),
            wandb.Image(str(record["matrix_path"])),
            record["SP"],
            record["XP"],
            record["SP-XP_gap"],
        ]
        for record in records
    ]
    with wandb.init(
        entity=args.entity,
        project=args.output_project,
        mode=args.wandb_mode,
        name="role-scenario-sp-xp-report",
        job_type="cross-play-report",
        dir=str(output_dir),
        config={
            "training_project": args.training_project,
            "crossplay_project": args.crossplay_project,
            "layouts": [record["map"] for record in records],
            "preferred_video_seed": args.video_seed,
        },
    ) as run:
        run.log({"report/map_results": wandb.Table(columns=columns, data=data)})
        artifact = wandb.Artifact(
            f"role-scenario-report-{run.id}", type="crossplay-report"
        )
        artifact.add_file(str(Path(output_dir) / "index.html"), name="index.html")
        artifact.add_file(
            str(Path(output_dir) / "map_metrics.csv"), name="map_metrics.csv"
        )
        artifact.add_file(str(Path(output_dir) / "report.json"), name="report.json")
        run.log_artifact(artifact, aliases=["latest"])


def main(argv=None):
    load_project_env()
    args = parse_args(argv)
    layouts = args.layouts or selected_layouts(args.sweep_config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir or Path("saves/reports") / f"role-scenarios-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    records = []
    for index, layout in enumerate(layouts, start=1):
        print(f"[{index}/{len(layouts)}] collecting {layout}", flush=True)
        records.append(collect_layout(api, args, layout, output_dir))

    write_csv(output_dir / "map_metrics.csv", records)
    write_html(output_dir / "index.html", records)
    (output_dir / "report.json").write_text(
        json.dumps(
            [
                {
                    **record,
                    "video_path": str(record["video_path"]),
                    "matrix_path": str(record["matrix_path"]),
                }
                for record in records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    log_wandb_report(args, records, output_dir)
    print(f"Report: {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
