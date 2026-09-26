"""Link the ten FCP observer-a population snapshots to their exact W&B runs."""

from pathlib import Path

import wandb


ENTITY = "cilab-overcooked"
PROJECT = "overcooked-v3-fcp-observer-a-0921_population"
REVISION = "dual-handoff-recipe-priority-11x7-v1"
ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "campaigns/distance0-priority-0926-fcp-observer-a-full10/main/agent_0"
POPULATION = CAMPAIGN / "population"
SEEDS = range(10)


def matching_run(runs, seed):
    matches = [
        run for run in runs
        if run.config.get("CONDITION") == "distance_0"
        and run.config.get("LAYOUT_REVISION") == REVISION
        and run.config.get("SEED") == seed
        and (run.config.get("ENV_KWARGS") or {}).get("transition_observer") == "agent_0"
    ]
    if len(matches) != 1 or matches[0].state != "finished":
        raise RuntimeError(f"Expected one finished population run for seed {seed}: {matches}")
    return matches[0]


def snapshot_files(seed):
    paths = list(POPULATION.glob(f"distance_0_rnn_*_seed{seed}"))
    if len(paths) != 1 or not paths[0].is_dir():
        raise RuntimeError(f"Expected one snapshot directory for seed {seed}: {paths}")
    files = sorted(path for path in paths[0].iterdir() if path.is_file())
    checkpoints = [path for path in files if path.suffix == ".safetensors"]
    configs = [path for path in files if path.suffix == ".yaml"]
    if len(checkpoints) < 3 or len(configs) != 1:
        raise RuntimeError(f"Incomplete snapshots for seed {seed}: {files}")
    if not any(path.name.endswith("_vmap0.safetensors") for path in checkpoints):
        raise RuntimeError(f"Final population checkpoint missing for seed {seed}")
    return files


def main():
    for seed in SEEDS:
        if not (CAMPAIGN / f"state/population_distance_0_seed{seed}.done").is_file():
            raise RuntimeError(f"Population seed {seed} is incomplete")
    api = wandb.Api(timeout=120)
    runs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.CONDITION": "distance_0", "config.LAYOUT_REVISION": REVISION},
        per_page=100,
    ))
    for seed in SEEDS:
        public_run = matching_run(runs, seed)
        name = f"overcooked-v3-{public_run.id}-population-snapshots"
        existing = [
            artifact for artifact in public_run.logged_artifacts()
            if artifact.name.split(":", 1)[0] == name and artifact.state == "COMMITTED"
        ]
        if existing:
            print(f"already linked seed={seed} run={public_run.id}", flush=True)
            continue
        files = snapshot_files(seed)
        with wandb.init(entity=ENTITY, project=PROJECT, id=public_run.id, resume="allow") as active_run:
            artifact = wandb.Artifact(
                name,
                type="checkpoint",
                description="FCP observer-a population checkpoints at 10%, 50%, and 100% training.",
                metadata={
                    "run_id": public_run.id,
                    "layout": "distance_0",
                    "layout_revision": REVISION,
                    "transition_observer": "agent_0",
                    "seed": seed,
                    "checkpoint_format": "safetensors",
                },
            )
            for path in files:
                artifact.add_file(str(path), name=path.name)
            active_run.log_artifact(artifact, aliases=["final"])
            active_run.summary["population/artifact_name"] = name
            active_run.summary["population/snapshot_file_count"] = len(files)
        print(f"linked seed={seed} run={public_run.id} artifact={name}", flush=True)


if __name__ == "__main__":
    main()
