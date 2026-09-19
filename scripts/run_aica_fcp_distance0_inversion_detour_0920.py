#!/usr/bin/env python3
"""Run both-warning FCP on the 13x6 asymmetric-detour distance_0 map."""

import hashlib
import json
import os
import subprocess
import sys
import time
from netrc import netrc
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
PYTHON = "/home/inchang/overcooked-replan-venv/bin/python"
ENTITY = "cilab-overcooked"
SOURCE_BASE_COMMIT = "2ba0a476f3d9503bab957cab5e5cee08db1701bc"
LAYOUT_REVISION = "distance-inversion-detour-13x6-v4"
GPUS = ["6", "7"]
LAYOUTS = ["distance_0"]
OBSERVERS = ["both"]
SEEDS = list(range(6))
PILOT_CELL = ("distance_0", "both")
POP_PROJECT = "overcooked-v3-fcp-distance0-inversion-detour-0920_population"
TRAIN_PROJECT = "overcooked-v3-fcp-distance0-inversion-detour-0920_train"
EVAL_PROJECT = "overcooked-v3-fcp-distance0-inversion-detour-0920_eval"
POPULATION_ROOT = ROOT / "population"
FCP_SAVES = ROOT / "fcp_saves"
EVAL_ROOT = ROOT / "evaluation"
LOG_ROOT = ROOT / "logs/jobs"
STATUS_PATH = ROOT / "status.json"

WANDB_AUTH = netrc().authenticators("api.wandb.ai")
if not WANDB_AUTH or not WANDB_AUTH[2]:
    raise RuntimeError("No W&B API key found for api.wandb.ai in ~/.netrc")
WANDB_API_KEY = WANDB_AUTH[2]


def observer_label(observer):
    return {
        "none": "None", "agent_0": "A", "agent_1": "B", "both": "Both"
    }[observer]


def stable_id(stage, layout, observer, seed=None):
    raw = f"fcp-distance0-inversion-detour-0920|{stage}|{layout}|{observer}|{seed}".encode()
    return hashlib.sha256(raw).hexdigest()[:8]


def key(stage, layout, observer, seed=None):
    suffix = "eval" if seed is None else f"s{seed}"
    return f"{stage}/{layout}/{observer}/{suffix}"


def make_jobs():
    cells = [PILOT_CELL] + [
        (layout, observer)
        for layout in LAYOUTS
        for observer in OBSERVERS
        if (layout, observer) != PILOT_CELL
    ]
    jobs = []
    for cell_index, (layout, observer) in enumerate(cells):
        base = 0 if cell_index == 0 else 10 + 3 * (cell_index - 1)
        pop_keys = [key("population", layout, observer, seed) for seed in SEEDS]
        fcp_keys = [key("fcp", layout, observer, seed) for seed in SEEDS]
        for seed in SEEDS:
            jobs.append({
                "key": key("population", layout, observer, seed),
                "stage": "population", "layout": layout, "observer": observer,
                "observer_label": observer_label(observer), "seed": seed,
                "priority": base, "dependencies": [], "status": "pending",
                "run_id": stable_id("population", layout, observer, seed),
            })
        for seed in SEEDS:
            jobs.append({
                "key": key("fcp", layout, observer, seed),
                "stage": "fcp", "layout": layout, "observer": observer,
                "observer_label": observer_label(observer), "seed": seed,
                "priority": base + 1, "dependencies": pop_keys, "status": "pending",
                "run_id": stable_id("fcp", layout, observer, seed),
            })
        jobs.append({
            "key": key("eval", layout, observer),
            "stage": "eval", "layout": layout, "observer": observer,
            "observer_label": observer_label(observer), "seed": None,
            "priority": base + 2, "dependencies": fcp_keys, "status": "pending",
            "run_id": stable_id("eval", layout, observer),
        })
    return jobs


def initial_state():
    return {
        "campaign": ROOT.name,
        "source_base_commit": SOURCE_BASE_COMMIT,
        "local_branch": "codex/distance-0-large-inversion",
        "layout_revision": LAYOUT_REVISION,
        "layout_sha256": hashlib.sha256(
            (SOURCE / "jaxmarl/environments/overcooked_v3/dynamic_layout_data.py")
            .read_bytes()
        ).hexdigest(),
        "scenario_sha256": hashlib.sha256(
            (SOURCE / "conf/scenario/distance_0.yaml").read_bytes()
        ).hexdigest(),
        "allowed_gpus": GPUS,
        "layouts": LAYOUTS,
        "observers": OBSERVERS,
        "seeds": SEEDS,
        "projects": {
            "population": POP_PROJECT,
            "fcp_train": TRAIN_PROJECT,
            "fcp_eval": EVAL_PROJECT,
        },
        "eval_workers_per_run": 8,
        "started_at": time.time(),
        "stage": "running",
        "pilot_cell": {"layout": PILOT_CELL[0], "observer": PILOT_CELL[1]},
        "pilot_complete": False,
        "jobs": make_jobs(),
    }


def save_state(state):
    counts = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    state["counts"] = counts
    state["updated_at"] = time.time()
    temp = STATUS_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True))
    temp.replace(STATUS_PATH)


def population_files(layout, observer, seed):
    folder = POPULATION_ROOT / observer
    pattern = f"ippo_rnn_overcooked_v3_{layout}_seed{seed}_vmap0*.safetensors"
    return sorted(folder.rglob(pattern)) if folder.exists() else []


def validate_population(layout, observer):
    counts = {seed: len(population_files(layout, observer, seed)) for seed in SEEDS}
    if counts != {seed: 3 for seed in SEEDS}:
        raise RuntimeError(
            f"Invalid population for {layout}/{observer}: expected 3 snapshots per seed, got {counts}"
        )


def command_for(job, gpu):
    layout = job["layout"]
    observer = job["observer"]
    label = job["observer_label"]
    seed = job["seed"]
    common_tags = (
        f"[0920,75-step,distance0-inversion-detour,13x6,FCP,default-warning,"
        f"observer-{label},aica]"
    )
    if job["stage"] == "population":
        return [
            PYTHON, "baselines/IPPO/ippo_overcooked_v3.py",
            "--config-name", "fcp_population_transition_window_overcooked_v3",
            f"scenario={layout}", f"TRANSITION_OBSERVER={observer}", f"SEED={seed}",
            "TOTAL_TIMESTEPS=30000000", f"ENTITY={ENTITY}", f"PROJECT={POP_PROJECT}",
            f"POPULATION_ROOT={POPULATION_ROOT}", "EXPERIMENT_FOLDER=null",
            "CHECKPOINT_INTERVAL=0", "CHECKPOINT_FRACTIONS=[0.1,0.5,1.0]",
            "upload_final_checkpoint=true",
            "recording=disabled", f"WANDB_TAGS={common_tags}",
            f"WANDB_GROUP=fcp-population-{layout}-observer-{label}",
            f"RUN_NAME=fcp-population-{layout}-observer-{label}-s{seed}",
        ]
    if job["stage"] == "fcp":
        validate_population(layout, observer)
        return [
            PYTHON, "baselines/FCP/fcp_overcooked_v3.py",
            "--config-name", "fcp_transition_window_overcooked_v3",
            f"scenario={layout}", f"TRANSITION_OBSERVER={observer}", f"SEED={seed}",
            "ARCHITECTURE=rnn",
            "TOTAL_TIMESTEPS=30000000", f"ENTITY={ENTITY}", f"PROJECT={TRAIN_PROJECT}",
            f"POPULATION_ROOT={POPULATION_ROOT}", f"SAVES_DIR={FCP_SAVES}",
            "EXPERIMENT_FOLDER=null", "FCP.snapshots_per_policy=3",
            "FCP.minimum_population_size=18", "FCP.max_population_size=null",
            "upload_final_checkpoint=true", "recording=disabled", f"WANDB_TAGS={common_tags}",
            f"WANDB_GROUP=fcp-{layout}-observer-{label}",
            f"RUN_NAME=fcp-train-{layout}-observer-{label}-s{seed}",
        ]
    output = EVAL_ROOT / observer / layout / "greedy"
    return [
        PYTHON, "baselines/IPPO/eval_crossplay_overcooked_v3.py",
        f"{ENTITY}/{TRAIN_PROJECT}", "--algorithms", "FCP",
        "--layout", layout, "--transition-observer", observer,
        "--seeds", *[str(seed) for seed in SEEDS],
        "--episodes", "20", "--max-steps", "450",
        "--gpus", gpu, "--workers-per-gpu", "8",
        "--output-project", f"{ENTITY}/{EVAL_PROJECT}",
        "--output-dir", str(output), "--save-adaptation-traces",
        "--adaptation-horizon", "75", "--adaptation-window", "30",
        "--drop-baseline-window", "60", "--drop-horizon", "60",
        "--recovery-threshold", "0.9", "--recovery-persistence", "5",
        "--run-label", f"FCP-greedy-distance0-inversion-detour-0920-{layout}-{label}",
    ]


def launch(job, gpu):
    command = command_for(job, gpu)
    log_path = LOG_ROOT / (job["key"].replace("/", "__") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("WANDB_SWEEP_ID", None)
    # A parent process that used wandb.Api may leave a private service socket
    # token in its environment. That socket belongs to the parent and cannot be
    # reused by independently launched training jobs.
    for name in tuple(env):
        if "WANDB" in name and "SERVICE" in name:
            env.pop(name, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HYDRA_FULL_ERROR": "1",
        "WANDB_ENTITY": ENTITY,
        "WANDB_API_KEY": WANDB_API_KEY,
        "WANDB_MODE": "online",
        "WANDB_RUN_ID": job["run_id"],
        "WANDB_RESUME": "allow",
        "PYTHONPATH": str(SOURCE) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    })
    process = subprocess.Popen(
        command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"process": process, "log": log, "gpu": gpu, "command": command, "log_path": str(log_path)}


def archive_population(state):
    state["stage"] = "archiving_population"
    save_state(state)
    run = wandb.init(
        entity=ENTITY, project=POP_PROJECT, id="popd0h20", resume="allow",
        name="population-archive-fcp-distance0-inversion-detour-0920",
        job_type="population-archive",
        tags=["0920", "75-step", "distance0-inversion-detour", "13x6",
              "FCP-Population", "default-warning", "observer-Both", "archive"],
        config={"layouts": LAYOUTS, "observers": OBSERVERS, "seeds": SEEDS,
                "snapshots_per_seed": 3, "campaign": ROOT.name},
        dir=str(ROOT / "wandb_archive"), settings=wandb.Settings(init_timeout=180),
    )
    try:
        files = list(POPULATION_ROOT.rglob("*.safetensors"))
        expected_files = len(LAYOUTS) * len(OBSERVERS) * len(SEEDS) * 3
        if len(files) != expected_files:
            raise RuntimeError(
                f"Expected {expected_files} population checkpoints, found {len(files)}"
            )
        artifact = wandb.Artifact(
            "fcp-distance0-inversion-detour-population-0920", type="fcp-population",
            metadata={"layouts": LAYOUTS, "observers": OBSERVERS, "seeds": SEEDS,
                      "snapshots_per_seed": 3, "checkpoint_count": len(files)},
        )
        artifact.add_dir(str(POPULATION_ROOT))
        run.log_artifact(artifact, aliases=["latest", "final"])
        run.summary["population/checkpoint_count"] = len(files)
    finally:
        run.finish()
    state["population_archive"] = {
        "run_id": "popd0h20", "artifact": "fcp-distance0-inversion-detour-population-0920",
        "checkpoint_count": 432,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    POPULATION_ROOT.mkdir(parents=True, exist_ok=True)
    FCP_SAVES.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists():
        state = json.loads(STATUS_PATH.read_text())
        if state.get("campaign") != ROOT.name:
            raise RuntimeError("Existing status belongs to another campaign")
        for job in state["jobs"]:
            if job["status"] in {"running", "failed", "cancelled"}:
                job["status"] = "pending"
                job.pop("return_code", None)
        state.update(stage="running", active=[], resumed_at=time.time())
        state.pop("error", None)
    else:
        state = initial_state()
    save_state(state)
    jobs = {job["key"]: job for job in state["jobs"]}
    running = {}

    while True:
        failed = None
        for job_key, info in list(running.items()):
            code = info["process"].poll()
            if code is None:
                continue
            info["log"].close()
            job = jobs[job_key]
            job["finished_at"] = time.time()
            job["return_code"] = code
            job["status"] = "complete" if code == 0 else "failed"
            del running[job_key]
            if code != 0:
                retries = int(job.get("retry_count", 0))
                if retries < 2:
                    job.update(
                        retry_count=retries + 1,
                        status="pending",
                        ready_after=time.time() + 60,
                    )
                else:
                    failed = job
                    break

        if failed is not None:
            state["stage"] = "failed"
            state["error"] = f"Job failed: {failed['key']} return_code={failed['return_code']}"
            for info in running.values():
                info["process"].terminate()
                info["log"].close()
            save_state(state)
            raise RuntimeError(state["error"])

        pilot_eval = jobs[key("eval", PILOT_CELL[0], PILOT_CELL[1])]
        if pilot_eval["status"] == "complete" and not state["pilot_complete"]:
            state["pilot_complete"] = True
            state["pilot_completed_at"] = time.time()

        pending = [job for job in state["jobs"] if job["status"] == "pending"]
        if not pending and not running:
            break

        free_gpus = [gpu for gpu in GPUS if gpu not in {info["gpu"] for info in running.values()}]
        completed = {job["key"] for job in state["jobs"] if job["status"] == "complete"}
        def ready_now(job):
            if not set(job["dependencies"]).issubset(completed):
                return False
            if time.time() < float(job.get("ready_after", 0)):
                return False
            if job["dependencies"]:
                latest = max(
                    float(jobs[item].get("finished_at", 0))
                    for item in job["dependencies"]
                )
                wait_seconds = 90 if job["stage"] == "eval" else 5
                if time.time() - latest < wait_seconds:
                    return False
            return True

        ready = sorted(
            [job for job in pending if ready_now(job)],
            key=lambda job: (job["priority"], job["stage"], job["seed"] if job["seed"] is not None else 99),
        )
        for gpu, job in zip(free_gpus, ready):
            info = launch(job, gpu)
            job.update({
                "status": "running", "gpu": gpu, "pid": info["process"].pid,
                "started_at": time.time(), "log": info["log_path"], "command": info["command"],
            })
            running[job["key"]] = info

        state["active"] = [
            {"key": key_, "gpu": info["gpu"], "pid": info["process"].pid}
            for key_, info in running.items()
        ]
        state["stage"] = "running"
        save_state(state)
        # Dependency-ready jobs may be waiting for retry/artifact cooldown.
        dependency_ready = any(
            set(job["dependencies"]).issubset(completed) for job in pending
        )
        if not running and not ready and not dependency_ready:
            state["stage"] = "failed"
            state["error"] = "Dependency deadlock"
            save_state(state)
            raise RuntimeError(state["error"])
        time.sleep(5)

    archive_population(state)
    state["stage"] = "complete"
    state["completed_at"] = time.time()
    state["active"] = []
    save_state(state)
    print("CAMPAIGN_COMPLETE", ROOT, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CAMPAIGN_FAILED {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
