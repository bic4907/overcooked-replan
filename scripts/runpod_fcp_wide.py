"""Run the three wide FCP experiments on one Pod and release its GPUs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ("split_wide", "outage_wide", "distance_switch_wide")
TAG = "wide-final-20260910"


def run_command(command, deadline):
    print("RUN", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        result = process.wait(timeout=max(1, deadline - time.monotonic()))
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if result:
        raise RuntimeError(f"Command exited with status {result}: {command[0]}")


def verify_sweep(reference, seeds):
    expected = {(layout, seed) for layout in LAYOUTS for seed in seeds}
    revisions = {
        layout: yaml.safe_load((ROOT / f"conf/scenario/{layout}.yaml").read_text())[
            "LAYOUT_REVISION"
        ]
        for layout in LAYOUTS
    }
    for attempt in range(10):
        runs = list(wandb.Api(timeout=30).sweep(reference).runs)
        actual = {
            (run.config.get("ENV_KWARGS", {}).get("layout"), run.config.get("SEED"))
            for run in runs
        }
        complete = len(runs) == len(expected) and actual == expected
        if complete:
            complete = all(
                run.state == "finished"
                and run.config.get("LAYOUT_REVISION")
                == revisions[run.config["ENV_KWARGS"]["layout"]]
                and any("final" in artifact.aliases for artifact in run.logged_artifacts())
                for run in runs
            )
        if complete:
            print(f"Verified {reference}: {len(runs)} finished runs", flush=True)
            return
        if attempt < 9:
            time.sleep(20)
    raise RuntimeError(f"Incomplete sweep, wrong revision or missing artifacts: {reference}")


def archive_directory(args, directory, label):
    if not directory.is_dir():
        raise RuntimeError(f"Missing output directory: {directory}")
    with wandb.init(
        entity=args.entity,
        project="overcooked-v3-fcp-population",
        job_type="checkpoint-backup",
        name=f"fcp-wide-{label}-{args.pod_id}",
        tags=["FCP", TAG, label],
        config={
            "layouts": list(LAYOUTS),
            "pod_id": args.pod_id,
            "source_commit": args.source_commit,
            "population_sweep": args.population_sweep,
            "train_sweep": args.train_sweep,
        },
    ) as run:
        artifact = wandb.Artifact(f"fcp-wide-{label}-{args.pod_id}", type="checkpoint")
        artifact.add_dir(str(directory))
        run.log_artifact(artifact, aliases=["latest"]).wait()
        run.summary["checkpoint_count"] = len(list(directory.rglob("*.safetensors")))


def release_pod(args, action):
    key = args.control_key_file.read_text().strip()
    for attempt in range(6):
        try:
            response = requests.post(
                f"https://api.runpod.io/v2/pods/{args.pod_id}/action",
                headers={"Authorization": f"Bearer {key}"},
                json={"action": action},
                timeout=30,
            )
            if response.status_code in (200, 204, 404):
                return
            print(f"Pod {action}: HTTP {response.status_code}", flush=True)
        except requests.RequestException as error:
            print(f"Pod {action}: {type(error).__name__}", flush=True)
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Could not {action} pod {args.pod_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-sweep", required=True)
    parser.add_argument("--train-sweep", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--control-key-file", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--entity", default="cilab-overcooked")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--max-hours", type=float, default=8)
    args = parser.parse_args()
    if not args.control_key_file.is_file() or args.max_hours <= 0:
        raise ValueError("Control key file and positive max-hours are required")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required; offline experiments are disabled")
    os.environ.update(
        GPUS=" ".join(args.gpus),
        WANDB_MODE="online",
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONPATH=str(ROOT),
        WANDB_DISABLE_CODE="true",
    )
    os.environ.pop("LD_LIBRARY_PATH", None)
    deadline = time.monotonic() + args.max_hours * 3600
    saves_dir = ROOT / "saves/fcp_wide"
    population_dir = saves_dir / "fcp_population"

    def set_status(stage, **extra):
        status = {
            "stage": stage,
            "layouts": list(LAYOUTS),
            "pod_id": args.pod_id,
            "source_commit": args.source_commit,
            "updated_at": time.time(),
            **extra,
        }
        (ROOT / "pipeline_status.json").write_text(json.dumps(status, indent=2))
        print(f"STAGE {stage}", flush=True)

    def timeout_handler(signum, frame):
        raise TimeoutError("FCP pipeline exceeded its wall-clock limit")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.alarm(int(args.max_hours * 3600))
    success = False
    try:
        set_status("population")
        run_command(
            ["bash", "experiment/run_agents_sequential.sh", args.population_sweep],
            deadline,
        )
        verify_sweep(args.population_sweep, range(3))
        run_command(
            [sys.executable, "scripts/verify_easy1_fcp_population.py",
             str(population_dir), "--layouts", *LAYOUTS],
            deadline,
        )
        archive_directory(args, population_dir, "population")

        set_status("best_response")
        run_command(
            ["bash", "experiment/run_agents_sequential.sh", args.train_sweep],
            deadline,
        )
        verify_sweep(args.train_sweep, range(6))
        archive_directory(args, saves_dir, "all-checkpoints")

        for layout in LAYOUTS:
            set_status("evaluation", layout=layout)
            evaluation_dir = ROOT / "evaluation/fcp_wide" / layout
            run_command(
                [sys.executable, "baselines/IPPO/eval_crossplay_overcooked_v3.py",
                 f"{args.entity}/overcooked-v3-fcp_train", "--algorithms", "FCP",
                 "--layout", layout, "--seeds", "0", "1", "2", "3", "4", "5",
                 "--episodes", "20", "--max-steps", "450", "--gpus", *args.gpus,
                 "--workers-per-gpu", "2", "--output-project",
                 f"{args.entity}/overcooked-v3-fcp_eval", "--output-dir",
                 str(evaluation_dir), "--save-adaptation-traces", "--run-label", "FCP-wide"],
                deadline,
            )
            if not (evaluation_dir / "summary.json").is_file():
                raise RuntimeError(f"Evaluation summary is missing: {layout}")
            archive_directory(args, evaluation_dir, f"{layout}-evaluation")
        success = True
        set_status("complete")
    except BaseException as error:
        set_status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        release_pod(args, "terminate" if success else "stop")


if __name__ == "__main__":
    main()
