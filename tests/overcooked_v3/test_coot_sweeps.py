import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from baselines.CooT.build_population_manifest import parse_args as parse_builder_args
from baselines.CooT.eval_crossplay_overcooked_v3 import parse_args as parse_eval_args
from baselines.CooT.hsp_population import resolve_hsp_config
from baselines.CooT.preflight_sweep import REPO_ROOT, _compose, preflight
from baselines.CooT.train_best_response_overcooked_v3 import resolve_response_job
from baselines.IPPO.ippo_overcooked_v3 import _isolate_hsp_output
from jaxmarl._experiment import experiment_folder
from jaxmarl._wandb import require_sweep_target

COOT_SWEEPS = sorted((REPO_ROOT / "experiment" / "coot").glob("*.yaml"))


def _sweep(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def _parameter(sweep, name):
    specification = sweep["parameters"][name]
    return specification.get("value", specification.get("values"))


def test_every_coot_sweep_passes_static_preflight():
    assert COOT_SWEEPS
    for path in COOT_SWEEPS:
        messages = preflight(path, check_inputs=False)
        assert any(
            message.startswith("project: overcooked-v3-coot-") for message in messages
        )


def test_coot_projects_are_stage_isolated_from_fcp_and_self_play():
    coot_projects = set()
    for path in COOT_SWEEPS:
        sweep = _sweep(path)
        key = "PROJECT" if "PROJECT" in sweep["parameters"] else "project"
        coot_projects.add(str(_parameter(sweep, key)))

    other_projects = set()
    for path in (REPO_ROOT / "experiment").glob("*/*.yaml"):
        if path.parent.name == "coot":
            continue
        sweep = _sweep(path)
        for key in ("PROJECT", "project", "output-project"):
            if key not in sweep.get("parameters", {}):
                continue
            value = str(_parameter(sweep, key)).rsplit("/", 1)[-1]
            other_projects.add(value)

    assert coot_projects.isdisjoint(other_projects)
    assert "overcooked-v3-coot-smoke" in coot_projects


def test_response_sweeps_use_distinct_manifest_stages():
    expected = {
        "response_candidates.yaml": "candidates",
        "response_candidates_multi_recipe.yaml": "candidates",
        "response.yaml": "exact",
        "response_hsp_only.yaml": "hsp_only",
        "smoke_response.yaml": "smoke",
    }
    roots = _compose("coot_br_overcooked_v3", [])["RESPONSE_JOB_ROOTS"]
    assert len(set(roots.values())) == len(roots)
    for filename, stage in expected.items():
        sweep = _sweep(REPO_ROOT / "experiment" / "coot" / filename)
        assert _parameter(sweep, "RESPONSE_JOB_STAGE") == stage


def test_response_job_identity_includes_stage_job_and_partner_hash(tmp_path):
    partner = tmp_path / "partner.safetensors"
    partner.write_bytes(b"partner")
    manifest_dir = tmp_path / "jobs"
    manifest_dir.mkdir()
    manifest = manifest_dir / "split_0.json"
    manifest.write_text(
        json.dumps(
            {
                "layout": "split_0",
                "jobs": [
                    {
                        "layout": "split_0",
                        "partner_id": "hsp_0000",
                        "population_type": "hsp",
                        "skill": "final",
                        "response_seed": 1,
                        "partner_checkpoint": str(partner),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = _compose(
        "coot_br_overcooked_v3",
        ["scenario=split_0", "RESPONSE_JOB_STAGE=smoke"],
    )
    config["RESPONSE_JOB_MANIFEST"] = str(manifest)
    resolved = OmegaConf.to_container(
        resolve_response_job(OmegaConf.create(config)), resolve=True
    )

    assert resolved["RESOLVED_RESPONSE_JOB"]["stage"] == "smoke"
    assert (
        "coot_br_smoke_split_0_hsp_hsp_0000_final_job0000_p"
        in resolved["EXPERIMENT_FOLDER"]
    )
    assert resolved["RUN_NAME"].startswith("coot-br-smoke-split_0-hsp-hsp_0000")
    assert resolved["WANDB_GROUP"] == "smoke-split_0"


def test_hsp_candidates_have_distinct_local_directories():
    directories = []
    for candidate_id in (0, 1):
        config = _compose(
            "coot_population_overcooked_v3",
            [
                "scenario=split_0",
                "HSP.PROFILE=other",
                f"HSP.CANDIDATE_ID={candidate_id}",
            ],
        )
        candidate = resolve_hsp_config(config)
        _isolate_hsp_output(config, candidate)
        directories.append(experiment_folder(config))
    assert directories[0] != directories[1]
    assert "candidate0000" in directories[0]
    assert "candidate0001" in directories[1]


def test_population_smoke_is_two_tiny_updates_with_isolated_output():
    sweep = _sweep(REPO_ROOT / "experiment" / "coot" / "smoke_population.yaml")
    assert _parameter(sweep, "NUM_ENVS") == 1
    assert _parameter(sweep, "NUM_STEPS") == 1
    assert _parameter(sweep, "TOTAL_TIMESTEPS") == 2
    assert _parameter(sweep, "SAVES_DIR") == "saves/coot_smoke/population"
    assert _parameter(sweep, "PROJECT") == "overcooked-v3-coot-smoke"


def test_builder_and_eval_defaults_use_coot_namespaces(monkeypatch):
    builder = parse_builder_args(
        [
            "response-jobs",
            "--hsp-catalog",
            "catalog.json",
            "--output",
            "jobs.json",
        ]
    )
    assert builder.response_output_root == Path("saves/coot_responses")

    monkeypatch.delenv("COOT_CHECKPOINT_ROOT", raising=False)
    monkeypatch.delenv("COOT_CROSSPLAY_OUTPUT_ROOT", raising=False)
    evaluation = parse_eval_args(["--layout", "split_0"])
    assert evaluation.checkpoint_root == Path("saves/coot_train")
    assert evaluation.output_root == Path("saves/coot_eval/crossplay")
    assert evaluation.project == "overcooked-v3-coot-eval"
    assert evaluation.group == "coot-crossplay-matrix"


def test_sweep_target_guard_rejects_wrong_project_before_training():
    class FakeRun:
        entity = "cilab-overcooked"
        project = "overcooked-v3-fcp_train"

        def __init__(self):
            self.finish_calls = []

        def finish(self, **kwargs):
            self.finish_calls.append(kwargs)

    run = FakeRun()
    with pytest.raises(RuntimeError, match="W&B sweep target mismatch"):
        require_sweep_target(
            run,
            {
                "ENTITY": "cilab-overcooked",
                "PROJECT": "overcooked-v3-coot-smoke",
            },
            environ={"WANDB_SWEEP_ID": "sweep"},
        )
    assert run.finish_calls == [{"exit_code": 1}]
