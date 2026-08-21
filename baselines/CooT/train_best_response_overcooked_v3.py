"""Train one Overcooked V3 best response for one explicit CooT partner job.

The response-job manifest keeps layout, partner identity, population type, skill
level, and checkpoint path in one record. Selecting a single JOB_INDEX avoids
the accidental Cartesian products that separate W&B sweep parameters create.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from jaxmarl._env import load_project_env

try:
    from baselines.FCP import fcp_overcooked_v3 as fcp
except ModuleNotFoundError as error:  # Direct execution from baselines/CooT.
    if error.name != "baselines":
        raise
    fcp_dir = Path(__file__).resolve().parents[1] / "FCP"
    sys.path.insert(0, str(fcp_dir))
    import fcp_overcooked_v3 as fcp


def _slug(value: object, *, field: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    if not slug:
        raise ValueError(f"Response job {field} must contain a path-safe value")
    return slug


def _manifest_jobs(payload: object) -> tuple[list[object], Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, Mapping):
        raise ValueError("Response job manifest must be a JSON object or list")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(
            "Response job manifest object must contain a non-empty 'jobs' list"
        )
    return jobs, payload


def _job_checkpoint(job: Mapping[str, Any]) -> object:
    checkpoint = job.get("partner_checkpoint") or job.get("checkpoint")
    partner = job.get("partner")
    if checkpoint is None and isinstance(partner, Mapping):
        checkpoint = partner.get("checkpoint")
    if checkpoint is None:
        raise ValueError(
            "Response job needs 'partner_checkpoint' (or partner.checkpoint)"
        )
    return checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_response_job(config: DictConfig) -> DictConfig:
    """Validate one manifest record, then inject it into a copied config."""
    raw_config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError("Resolved CooT BR config must be a mapping")

    stage = _slug(raw_config.get("RESPONSE_JOB_STAGE") or "exact", field="stage")
    manifest_value = raw_config.get("RESPONSE_JOB_MANIFEST")
    if not manifest_value:
        root_value = raw_config.get("RESPONSE_JOB_ROOT")
        if not root_value:
            roots = raw_config.get("RESPONSE_JOB_ROOTS")
            if not isinstance(roots, Mapping) or stage not in roots:
                raise ValueError(
                    f"No response-job root is configured for stage {stage!r}"
                )
            root_value = roots[stage]
        layout_hint = (raw_config.get("ENV_KWARGS") or {}).get("layout")
        if not layout_hint:
            raise ValueError(
                "A scenario layout is required when RESPONSE_JOB_MANIFEST is unset"
            )
        manifest_value = Path(str(root_value)) / f"{layout_hint}.json"
    manifest_path = Path(str(manifest_value)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Response job manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs, manifest_defaults = _manifest_jobs(payload)
    job_index = int(raw_config.get("JOB_INDEX", 0))
    if not 0 <= job_index < len(jobs):
        raise IndexError(
            f"JOB_INDEX {job_index} is outside response manifest range "
            f"[0, {len(jobs) - 1}]"
        )
    raw_job = jobs[job_index]
    if not isinstance(raw_job, Mapping):
        raise ValueError(f"Response job {job_index} must be a JSON object")
    job = dict(raw_job)

    partner_id = job.get("partner_id") or job.get("id")
    if partner_id is None:
        raise ValueError(f"Response job {job_index} needs 'partner_id'")
    population_type = job.get("population_type") or "unknown"
    skill = job.get("skill") or job.get("checkpoint_fraction") or "final"
    layout = job.get("layout") or manifest_defaults.get("layout")
    if not layout:
        raise ValueError(f"Response job {job_index} needs 'layout'")
    response_seed = int(job.get("response_seed", raw_config.get("SEED", 0)))
    if response_seed < 0:
        raise ValueError(
            f"Response job {job_index} has negative response_seed {response_seed}"
        )

    checkpoint = Path(str(_job_checkpoint(job))).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (manifest_path.parent / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Response job {job_index} partner checkpoint not found: {checkpoint}"
        )
    partner_sha256 = _sha256(checkpoint)

    partner_slug = _slug(partner_id, field="partner_id")
    skill_slug = _slug(skill, field="skill")
    layout_slug = _slug(layout, field="layout")
    population_split = str(job.get("split") or "train").lower()
    if population_split not in {"train", "validation", "test"}:
        raise ValueError(
            f"Response job {job_index} has invalid split {population_split!r}"
        )

    env_kwargs = dict(raw_config.get("ENV_KWARGS") or {})
    env_kwargs["layout"] = str(layout)
    fcp_config = dict(raw_config.get("FCP") or {})
    fcp_config.update(
        {
            "partner_checkpoint": str(checkpoint),
            "partner_id": str(partner_id),
            "partner_slug": partner_slug,
            "partner_skill": str(skill),
            "partner_skill_slug": skill_slug,
            "population_type": str(population_type),
            "population_split": population_split,
            "partner_sha256": partner_sha256,
            "population_size": 1,
        }
    )

    # Apply optional actor metadata from the same atomic job record. Any omitted
    # value retains the response config's architecture.
    actor_fields = {
        "architecture": "ARCHITECTURE",
        "activation": "ACTIVATION",
        "fc_dim_size": "FC_DIM_SIZE",
        "gru_hidden_dim": "GRU_HIDDEN_DIM",
    }
    partner_spec = job.get("partner")
    for source, target in actor_fields.items():
        value = job.get(source)
        if value is None and isinstance(partner_spec, Mapping):
            value = partner_spec.get(source)
        if value is not None:
            raw_config[target] = value

    raw_config["ENV_KWARGS"] = env_kwargs
    raw_config["FCP"] = fcp_config
    raw_config["SEED"] = response_seed
    raw_config["RESPONSE_JOB_STAGE"] = stage
    raw_config["RESPONSE_JOB_MANIFEST"] = str(manifest_path)
    raw_config["JOB_INDEX"] = job_index
    raw_config["EXPERIMENT_FOLDER"] = (
        f"coot_br_{stage}_{layout_slug}_{_slug(population_type, field='population_type')}_"
        f"{partner_slug}_{skill_slug}_job{job_index:04d}_p{partner_sha256[:12]}"
    )
    # The default Hydra scenario is only a bootstrap config. The manifest owns
    # the actual layout, so do not retain its experiment/group labels.
    raw_config["EXPERIMENT"] = "coot_response"
    raw_config["CONDITION"] = str(layout)
    raw_config["WANDB_GROUP"] = f"{stage}-{layout_slug}"
    raw_config["RUN_NAME"] = (
        f"coot-br-{stage}-{layout_slug}-"
        f"{_slug(population_type, field='population_type')}-{partner_slug}-"
        f"{skill_slug}-job{job_index:04d}-p{partner_sha256[:8]}-seed{response_seed}"
    )
    raw_config["WANDB_TAGS"] = list(
        dict.fromkeys([*(raw_config.get("WANDB_TAGS") or []), f"stage:{stage}"])
    )
    raw_config["RESOLVED_RESPONSE_JOB"] = {
        "manifest": str(manifest_path),
        "stage": stage,
        "job_index": job_index,
        "layout": str(layout),
        "partner_id": str(partner_id),
        "partner_slug": partner_slug,
        "population_type": str(population_type),
        "skill": str(skill),
        "skill_slug": skill_slug,
        "split": population_split,
        "response_seed": response_seed,
        "partner_checkpoint": str(checkpoint),
        "partner_sha256": partner_sha256,
        "source_job": job,
    }
    return OmegaConf.create(raw_config)


def _log_response_result_artifact(
    result_path: Path,
    final_checkpoint: Path,
    *,
    job: Mapping[str, Any],
    run_id: str,
) -> None:
    """Upload a relocatable result/checkpoint bundle before W&B finishes."""

    if wandb.run is None:
        return
    artifact = wandb.Artifact(
        f"coot-response-result-{_slug(run_id or 'local', field='run_id')}",
        type="coot-response-result",
        description=(
            "Completed atomic CooT partner-specific response job and its actual "
            "Overcooked V3 checkpoint."
        ),
        metadata={
            "run_id": run_id,
            "stage": str(job["stage"]),
            "layout": str(job["layout"]),
            "partner_id": str(job["partner_id"]),
            "population_type": str(job["population_type"]),
            "skill": str(job["skill"]),
            "partner_sha256": str(job["partner_sha256"]),
            "response_seed": int(job["response_seed"]),
        },
    )
    # Both files share the artifact root. The JSON deliberately stores only
    # final_checkpoint.name, so downloading this artifact anywhere preserves
    # the relative checkpoint reference consumed by scorer/manifest tools.
    artifact.add_file(str(result_path), name=result_path.name)
    artifact.add_file(str(final_checkpoint), name=final_checkpoint.name)
    wandb.run.log_artifact(artifact, aliases=["completed", str(job["skill"])])
    wandb.run.summary["response/result_artifact"] = artifact.name
    wandb.run.summary["response/result_file"] = result_path.name


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="coot_br_overcooked_v3",
)
def main(config: DictConfig) -> None:
    resolved_config = resolve_response_job(config)
    job = resolved_config["RESOLVED_RESPONSE_JOB"]
    print(
        f"[{fcp._timestamp()}] CooT response job "
        f"{job['job_index']}: {job['layout']} / {job['partner_id']} / "
        f"{job['skill']} ({job['population_type']})",
        flush=True,
    )
    try:
        training_result = fcp.run(resolved_config)
        final_checkpoints = [
            Path(path).resolve() for path in training_result["checkpoint_paths"]
        ]
        if len(final_checkpoints) != 1:
            raise RuntimeError(
                "CooT response jobs require exactly one final checkpoint; "
                f"got {len(final_checkpoints)}"
            )
        final_checkpoint = final_checkpoints[0]
        if not final_checkpoint.is_file():
            raise FileNotFoundError(
                f"CooT response checkpoint was not saved: {final_checkpoint}"
            )
        save_dir_value = training_result.get("save_dir")
        if not save_dir_value:
            raise RuntimeError("CooT response jobs require SAVES_DIR")
        save_dir = Path(save_dir_value).resolve()
        run_id = str(training_result.get("run_id") or "")
        run_slug = _slug(run_id or "local", field="run_id")
        result_path = save_dir / (
            f"response_job{int(job['job_index']):04d}_{job['partner_slug']}_"
            f"{job['skill_slug']}_seed{int(resolved_config['SEED'])}_{run_slug}.json"
        )
        portable_checkpoint = final_checkpoint.name
        result_payload = {
            "status": "completed",
            "method": "CooT-BR",
            "run_id": run_id,
            "response_job_manifest": str(job["manifest"]),
            "response_job_stage": str(job["stage"]),
            "job_index": int(job["job_index"]),
            "original_job": OmegaConf.to_container(job["source_job"], resolve=True),
            "resolved_job": {
                key: OmegaConf.to_container(value, resolve=True)
                if OmegaConf.is_config(value)
                else value
                for key, value in job.items()
                if key != "source_job"
            },
            "partner_checkpoint": str(job["partner_checkpoint"]),
            "partner_sha256": str(job["partner_sha256"]),
            "response_seed": int(job["response_seed"]),
            "best_response_checkpoint": portable_checkpoint,
            "best_response_checkpoints": [portable_checkpoint],
            # Keep BR PolicySpec metadata at the same level as the authoritative
            # checkpoint path so the downstream manifest builder never falls back
            # to its 128-unit defaults for 64-unit HSP/MEP response policies.
            "architecture": str(resolved_config["ARCHITECTURE"]),
            "activation": str(resolved_config["ACTIVATION"]),
            "fc_dim_size": int(resolved_config["FC_DIM_SIZE"]),
            "gru_hidden_dim": int(resolved_config["GRU_HIDDEN_DIM"]),
            "stochastic": True,
            "value_normalization": str(resolved_config["VALUE_NORMALIZATION"]),
            "porting_notes": [
                "The released adaptive BR runner enables ValueNorm; this V3 "
                "FCP-based port uses raw sparse-return targets because its "
                "TrainState has no running ValueNorm state."
            ],
            "best_response": {
                "checkpoint": portable_checkpoint,
                "architecture": str(resolved_config["ARCHITECTURE"]),
                "activation": str(resolved_config["ACTIVATION"]),
                "fc_dim_size": int(resolved_config["FC_DIM_SIZE"]),
                "gru_hidden_dim": int(resolved_config["GRU_HIDDEN_DIM"]),
                "stochastic": True,
            },
            "resolved_config_path": training_result.get("config_path"),
            "resolved_config": OmegaConf.to_container(resolved_config, resolve=True),
        }
        save_dir.mkdir(parents=True, exist_ok=True)
        with result_path.open("x", encoding="utf-8") as result_file:
            json.dump(result_payload, result_file, indent=2, sort_keys=True)
            result_file.write("\n")
        _log_response_result_artifact(
            result_path,
            final_checkpoint,
            job=job,
            run_id=run_id,
        )
        print(f"[{fcp._timestamp()}] Saved response result: {result_path}", flush=True)
    finally:
        if wandb.run is not None:
            wandb.finish()


def entrypoint() -> None:
    if load_project_env():
        print(f"[{fcp._timestamp()}] Loaded project .env")
    main()


if __name__ == "__main__":
    entrypoint()
