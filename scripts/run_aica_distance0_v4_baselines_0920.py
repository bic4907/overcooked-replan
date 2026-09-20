#!/usr/bin/env python3
"""Train and evaluate both-warning IPPO baselines on distance_0 v4."""

import hashlib
import json
import os
import subprocess
import time
from netrc import netrc
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
PYTHON = "/home/inchang/overcooked-replan-venv/bin/python"
ENTITY = "cilab-overcooked"
GPUS = ("6", "7")
SEEDS = tuple(range(6))
JOBS = {
    "IPPO-CNN": {
        "config": "ippo_cnn_transition_window_overcooked_v3",
        "slug": "ippo-cnn",
        "train_project": "overcooked-v3-ippo-cnn-75step-plate-0915_train",
        "eval_project": "overcooked-v3-ippo-cnn-75step-plate-0915_eval",
    },
    "IPPO-RNN": {
        "config": "ippo_rnn_transition_window_overcooked_v3",
        "slug": "ippo-rnn",
        "train_project": "overcooked-v3-ippo-rnn-observer-75step-plate-0915_train",
        "eval_project": "overcooked-v3-ippo-rnn-observer-75step-plate-0915_eval",
    },
}
STATUS = ROOT / "status.json"
LOGS = ROOT / "logs"


def run_id(algorithm, stage, seed=None):
    value = f"distance0-detour-v4-0920|{algorithm}|{stage}|{seed}"
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def make_jobs():
    jobs = []
    for algorithm in JOBS:
        for seed in SEEDS:
            jobs.append({"algorithm": algorithm, "stage": "train", "seed": seed,
                         "run_id": run_id(algorithm, "train", seed), "status": "pending"})
        jobs.append({"algorithm": algorithm, "stage": "eval", "seed": None,
                     "run_id": run_id(algorithm, "eval"), "status": "pending"})
    return jobs


def save(state):
    state["updated_at"] = time.time()
    state["counts"] = {
        status: sum(job["status"] == status for job in state["jobs"])
        for status in ("pending", "running", "complete", "failed")
    }
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True))
    temp.replace(STATUS)


def command(job, gpu):
    algorithm = job["algorithm"]
    info = JOBS[algorithm]
    slug = info["slug"]
    if job["stage"] == "train":
        seed = job["seed"]
        return [
            PYTHON, "baselines/IPPO/ippo_overcooked_v3.py",
            "--config-name", info["config"], "scenario=distance_0",
            "TRANSITION_OBSERVER=both", f"SEED={seed}",
            "TOTAL_TIMESTEPS=30000000", f"ENTITY={ENTITY}",
            f"PROJECT={info['train_project']}",
            f"SAVES_DIR={ROOT / 'checkpoints' / slug}",
            "EXPERIMENT_FOLDER=null", "CHECKPOINT_INTERVAL=0",
            "upload_final_checkpoint=true", "recording=disabled",
            "WANDB_TAGS=[0920,75-step,distance0-detour-v4,13x6,default-warning,observer-Both,aica]",
            f"WANDB_GROUP={slug}-distance_0-v4-Both",
            f"RUN_NAME={slug}-distance_0-v4-Both-s{seed}",
        ]
    return [
        PYTHON, "baselines/IPPO/eval_crossplay_overcooked_v3.py",
        f"{ENTITY}/{info['train_project']}", "--algorithms", algorithm,
        "--layout", "distance_0", "--transition-observer", "both",
        "--seeds", *map(str, SEEDS), "--episodes", "20", "--max-steps", "450",
        "--gpus", gpu, "--workers-per-gpu", "8",
        "--output-project", f"{ENTITY}/{info['eval_project']}",
        "--output-dir", str(ROOT / "evaluation" / slug / "greedy"),
        "--save-adaptation-traces", "--adaptation-horizon", "75",
        "--adaptation-window", "30", "--drop-baseline-window", "60",
        "--drop-horizon", "60", "--recovery-threshold", "0.9",
        "--recovery-persistence", "5",
        "--run-label", f"{slug}-greedy-distance_0-v4-Both-0920",
    ]


def launch(job, gpu, api_key):
    LOGS.mkdir(parents=True, exist_ok=True)
    label = f"{JOBS[job['algorithm']]['slug']}-{job['stage']}-{job['seed']}"
    log_path = LOGS / f"{label}.log"
    log = log_path.open("a")
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("WANDB_SWEEP_ID", None)
    for key in tuple(env):
        if "WANDB" in key and "SERVICE" in key:
            env.pop(key, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HYDRA_FULL_ERROR": "1",
        "WANDB_ENTITY": ENTITY,
        "WANDB_API_KEY": api_key,
        "WANDB_MODE": "online",
        "WANDB_RUN_ID": job["run_id"],
        "WANDB_RESUME": "allow",
        "PYTHONPATH": str(SOURCE),
    })
    proc = subprocess.Popen(command(job, gpu), cwd=SOURCE, env=env,
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    job.update(status="running", gpu=gpu, pid=proc.pid, started_at=time.time(),
               log=str(log_path), command=command(job, gpu))
    return proc, log


def eligible(job, jobs):
    if job["status"] != "pending":
        return False
    if job["stage"] == "train":
        return True
    return all(other["status"] == "complete" for other in jobs
               if other["algorithm"] == job["algorithm"]
               and other["stage"] == "train")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (SOURCE / "conf/scenario/distance_0.yaml").exists():
        raise RuntimeError("The immutable source snapshot is missing")
    auth = netrc().authenticators("api.wandb.ai")
    if not auth or not auth[2]:
        raise RuntimeError("W&B API credential is missing")
    if STATUS.exists():
        state = json.loads(STATUS.read_text())
        for job in state["jobs"]:
            if job["status"] in ("running", "failed"):
                job["status"] = "pending"
        state["resumed_at"] = time.time()
    else:
        state = {
            "campaign": "distance0-detour-v4-75step-baselines-0920",
            "source_branch": "codex/distance0-75step-baselines-0920",
            "layout_revision": "distance-inversion-detour-13x6-v4",
            "observer": "both", "gpus": GPUS, "seeds": SEEDS,
            "fcp": "reuse the completed v4 FCP train/eval after 0915 migration",
            "started_at": time.time(), "jobs": make_jobs(),
        }
    save(state)
    active = {}
    while True:
        for gpu, (job, proc, log) in tuple(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            job["return_code"] = code
            job["finished_at"] = time.time()
            if code == 0:
                job["status"] = "complete"
            elif job.get("retries", 0) < 2:
                job["retries"] = job.get("retries", 0) + 1
                job["status"] = "pending"
            else:
                job["status"] = "failed"
            del active[gpu]
            save(state)
        if any(job["status"] == "failed" for job in state["jobs"]):
            state["stage"] = "failed"
            save(state)
            raise RuntimeError("A job exhausted retries; inspect its log")
        if all(job["status"] == "complete" for job in state["jobs"]):
            state["stage"] = "complete"
            save(state)
            return
        for gpu in GPUS:
            if gpu in active:
                continue
            next_job = next((job for job in state["jobs"]
                            if eligible(job, state["jobs"])), None)
            if next_job is None:
                continue
            proc, log = launch(next_job, gpu, auth[2])
            active[gpu] = (next_job, proc, log)
            save(state)
        time.sleep(15)


if __name__ == "__main__":
    main()
