import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.IPPO.eval_crossplay_overcooked_v3 import prepare_gpu_argv
from baselines.IPPO.eval_wandb_crossplay_matrix_overcooked_v3 import (
    _write_records_csv,
    add_run_outputs_to_artifact,
    build_algorithm_matrix,
    build_run_filters,
    discover_run_candidates,
    evaluation_run_name,
    is_self_play,
    match_algorithm,
    parse_args,
    recover_worker_records,
    resolve_run_paths,
    resolve_vmap_checkpoints,
    run_gpu_workers,
    select_matrix_views,
    shard_tasks,
    split_project_path,
    summarize_records,
    write_reproducibility_bundle,
)


class FakeArtifact:
    def __init__(self, name="checkpoint:v0", aliases=None):
        self.name = name
        self.type = "checkpoint"
        self.aliases = aliases or ["final"]


def test_user_entrypoint_defaults_to_one_visible_gpu_and_cpu_parent():
    argv = ["eval_crossplay_overcooked_v3.py", "entity/project"]
    environ = {"CUDA_VISIBLE_DEVICES": "3,5", "JAX_PLATFORMS": "rocm"}

    prepare_gpu_argv(argv, environ)

    assert argv[-2:] == ["--gpus", "3"]
    assert environ["JAX_PLATFORMS"] == "cpu"


def test_user_entrypoint_preserves_explicit_gpu_list():
    argv = ["eval_crossplay_overcooked_v3.py", "entity/project", "--gpus", "1", "2"]
    environ = {}

    prepare_gpu_argv(argv, environ)

    assert argv[-3:] == ["--gpus", "1", "2"]
    assert environ["JAX_PLATFORMS"] == "cpu"


def test_wandb_run_name_starts_with_xp():
    settings = SimpleNamespace(
        algorithms=["IPPO", "FCP"],
        layout="splitsig_0",
    )

    assert evaluation_run_name(settings) == "xp-ippo+fcp-splitsig_0"


def test_default_run_paths_keep_all_outputs_under_one_saves_directory():
    settings = SimpleNamespace(
        algorithms=["IPPO"],
        layout="splitsig_0",
        output_dir=None,
        artifact_dir=None,
    )

    output_dir, artifact_dir = resolve_run_paths(settings, "20260813-120000", 1234)

    assert output_dir == Path(
        "saves/crossplay/xp-ippo-splitsig_0-20260813-120000-p1234"
    )
    assert artifact_dir == output_dir / "artifacts"


def test_reproducibility_bundle_saves_command_config_and_executable_sources(
    tmp_path, monkeypatch
):
    settings = SimpleNamespace(
        algorithms=["IPPO"],
        layout="splitsig_0",
        output_dir=None,
        artifact_dir=None,
    )
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        "sys.argv", ["eval_crossplay_overcooked_v3.py", "entity/project"]
    )

    write_reproducibility_bundle(tmp_path, settings, "xp-ippo-splitsig_0", artifact_dir)

    assert (tmp_path / "command.txt").read_text(encoding="utf-8").strip() == (
        "eval_crossplay_overcooked_v3.py entity/project"
    )
    config = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert config["wandb_run_name"] == "xp-ippo-splitsig_0"
    assert config["artifact_dir"] == str(artifact_dir)
    assert (tmp_path / "source/eval_crossplay_overcooked_v3.py").is_file()
    assert (tmp_path / "source/eval_wandb_crossplay_matrix_overcooked_v3.py").is_file()
    assert (tmp_path / "source/ippo_seedwise_crossplay.yaml").is_file()


def test_result_artifact_excludes_downloaded_checkpoint_cache(tmp_path):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "policy.safetensors").write_bytes(b"checkpoint")
    wandb_dir = tmp_path / "wandb/latest-run"
    wandb_dir.mkdir(parents=True)
    (wandb_dir / "run.log").write_text("internal", encoding="utf-8")

    class ResultArtifact:
        def __init__(self):
            self.names = []

        def add_file(self, _path, name):
            self.names.append(name)

    result_artifact = ResultArtifact()
    add_run_outputs_to_artifact(result_artifact, tmp_path, artifact_dir)

    assert result_artifact.names == ["summary.json"]


def test_eval_cli_accepts_exactly_one_map_with_singular_or_legacy_flag():
    common = ["entity/project", "--algorithms", "IPPO"]

    parsed = parse_args([*common, "--layout", "splitsig_0"])
    assert parsed.layout == "splitsig_0"
    assert parsed.workers_per_gpu == 8
    assert parse_args([*common, "--layouts", "splitsig_0"]).layout == "splitsig_0"

    with pytest.raises(SystemExit):
        parse_args([*common, "--layouts", "splitsig_0", "splitsig_1"])


def test_pair_results_csv_records_the_map_in_the_first_column(tmp_path):
    csv_path = tmp_path / "pair_results.csv"

    _write_records_csv(
        csv_path,
        [{"layout": "splitsig_0", "pair_type": "XP", "mean_return": 3.0}],
    )

    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["map", "pair_type", "mean_return"]
        assert list(reader) == [
            {"map": "splitsig_0", "pair_type": "XP", "mean_return": "3.0"}
        ]


def fake_run(
    run_id,
    *,
    algorithm=None,
    tags=(),
    layout="splitsig_0",
    seed=0,
    created_at="2026-01-01T00:00:00Z",
    artifact=None,
):
    config = {"ENV_KWARGS": {"layout": layout}, "SEED": seed}
    if algorithm is not None:
        config["ALGORITHM"] = algorithm
    artifact = artifact or FakeArtifact()
    return SimpleNamespace(
        id=run_id,
        entity="entity",
        project="project",
        path=["entity", "project", run_id],
        config=config,
        tags=list(tags),
        created_at=created_at,
        logged_artifacts=lambda: [artifact],
    )


def test_split_project_path_accepts_qualified_and_bare_projects():
    assert split_project_path("other/train") == ("other", "train")
    assert split_project_path("train", "entity") == ("entity", "train")
    with pytest.raises(ValueError, match="provide --entity"):
        split_project_path("train")


def test_run_filters_push_layout_seed_and_state_to_wandb():
    assert build_run_filters(["splitsig_0"], [1, 2], "finished") == {
        "config.ENV_KWARGS.layout": {"$in": ["splitsig_0"]},
        "config.SEED": {"$in": [1, 2]},
        "state": "finished",
    }
    assert "state" not in build_run_filters(["splitsig_0"], run_state="all")


def test_shard_tasks_balances_pairs_without_dropping_ordered_tasks():
    tasks = [{"pair": index} for index in range(7)]

    shards = shard_tasks(tasks, 3)

    assert [[task["pair"] for task in shard] for shard in shards] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    assert sorted(task["pair"] for shard in shards for task in shard) == list(range(7))


def test_gpu_workers_start_multiple_instances_per_device_and_merge_results(
    tmp_path, monkeypatch
):
    launches = []

    class FakeProcess:
        def __init__(self, command, environment):
            self.command = command
            self.environment = environment

        def wait(self):
            return 0

    def fake_popen(command, env):
        task_path = command[command.index("--tasks") + 1]
        output_path = command[command.index("--output") + 1]
        tasks = json.loads(open(task_path, encoding="utf-8").read())
        records = [
            {"pair": task["pair"], "gpu": env["CUDA_VISIBLE_DEVICES"]} for task in tasks
        ]
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(records, stream)
        launches.append((command, env.copy()))
        return FakeProcess(command, env)

    monkeypatch.setattr(
        "baselines.IPPO.eval_wandb_crossplay_matrix_overcooked_v3.subprocess.Popen",
        fake_popen,
    )

    records, failures = run_gpu_workers(
        [{"pair": index} for index in range(5)],
        ["2", "5"],
        tmp_path,
        workers_per_gpu=2,
    )

    assert failures == []
    assert sorted(record["pair"] for record in records) == list(range(5))
    assert {record["gpu"] for record in records} == {"2", "5"}
    assert [env["CUDA_VISIBLE_DEVICES"] for _command, env in launches] == [
        "2",
        "2",
        "5",
        "5",
    ]
    assert all(env["JAX_PLATFORMS"] == "cuda" for _command, env in launches)
    assert [
        command[command.index("--worker-label") + 1] for command, _env in launches
    ] == [
        "gpu=2,instance=0",
        "gpu=2,instance=1",
        "gpu=5,instance=0",
        "gpu=5,instance=1",
    ]


def test_recover_worker_records_deduplicates_interrupted_results(tmp_path):
    base = {
        "layout": "splitsig_0",
        "agent_0_model_id": "a",
        "agent_1_model_id": "b",
        "episodes": 1,
        "max_steps": 2,
        "evaluation_seed": 0,
        "stochastic": False,
        "mean_return": 1.0,
    }
    recovered = {**base, "mean_return": 5.0}
    (tmp_path / "worker_0_results.json").write_text(
        json.dumps([recovered]), encoding="utf-8"
    )

    records = recover_worker_records(tmp_path, [base])

    assert records == [recovered]


def test_match_algorithm_uses_config_or_tags_case_insensitively():
    assert match_algorithm(fake_run("a", algorithm="IPPO"), ["ippo"]) == "ippo"
    assert match_algorithm(fake_run("b", tags=["FCP"]), ["IPPO", "fcp"]) == "fcp"
    assert match_algorithm(fake_run("c", tags=["other"]), ["IPPO"]) is None

    with pytest.raises(ValueError, match="multiple algorithm tags"):
        match_algorithm(fake_run("d", tags=["IPPO", "FCP"]), ["IPPO", "FCP"])


def test_discovery_keeps_latest_run_per_algorithm_layout_and_seed():
    old = fake_run("old", algorithm="IPPO", created_at="2026-01-01T00:00:00Z")
    new = fake_run("new", algorithm="IPPO", created_at="2026-02-01T00:00:00Z")
    other_seed = fake_run("seed1", algorithm="IPPO", seed=1)
    wrong_layout = fake_run("wrong", algorithm="IPPO", layout="splitsig_1")

    candidates = discover_run_candidates(
        [old, new, other_seed, wrong_layout],
        algorithms=["IPPO"],
        layouts=["splitsig_0"],
    )

    assert [candidate.run.id for candidate in candidates] == ["new", "seed1"]


def test_resolve_vmap_checkpoints_filters_intermediate_and_requested(tmp_path):
    (tmp_path / "policy_vmap0.safetensors").write_bytes(b"zero")
    (tmp_path / "policy_vmap1.safetensors").write_bytes(b"one")
    (tmp_path / "policy_vmap0_update000001.safetensors").write_bytes(b"old")

    assert [index for index, _path in resolve_vmap_checkpoints(tmp_path)] == [0, 1]
    assert [index for index, _path in resolve_vmap_checkpoints(tmp_path, [1])] == [1]
    with pytest.raises(FileNotFoundError, match="Missing"):
        resolve_vmap_checkpoints(tmp_path, [2])


def test_exact_model_identity_defines_self_play_and_summary_gap():
    records = [
        {"pair_type": "SP", "mean_return": 10.0},
        {"pair_type": "SP", "mean_return": 14.0},
        {"pair_type": "XP", "mean_return": 7.0},
        {"pair_type": "XP", "mean_return": 9.0},
    ]

    assert is_self_play("artifact:v0", "artifact:v0")
    assert not is_self_play("artifact:v0", "artifact:v1")
    assert summarize_records(records) == {
        "SP": 12.0,
        "XP": 8.0,
        "SP-XP_gap": 4.0,
        "SP_pairs": 2,
        "XP_pairs": 2,
    }


def test_algorithm_matrix_preserves_agent_order():
    records = [
        {
            "agent_0_algorithm": "IPPO",
            "agent_1_algorithm": "FCP",
            "mean_return": 3.0,
        },
        {
            "agent_0_algorithm": "FCP",
            "agent_1_algorithm": "IPPO",
            "mean_return": 8.0,
        },
    ]

    matrix = build_algorithm_matrix(records, ["IPPO", "FCP"])

    assert matrix[0, 1] == 3.0
    assert matrix[1, 0] == 8.0
    assert np.isnan(matrix[0, 0])


def test_matrix_views_hide_redundant_model_or_algorithm_panels():
    one_algorithm_models = [
        SimpleNamespace(algorithm="IPPO"),
        SimpleNamespace(algorithm="IPPO"),
    ]
    one_model_per_algorithm = [
        SimpleNamespace(algorithm="IPPO"),
        SimpleNamespace(algorithm="FCP"),
    ]
    multiple_models_for_an_algorithm = [
        *one_model_per_algorithm,
        SimpleNamespace(algorithm="IPPO"),
    ]

    assert select_matrix_views(one_algorithm_models, ["IPPO"]) == ("models",)
    assert select_matrix_views(one_model_per_algorithm, ["IPPO", "FCP"]) == (
        "algorithms",
    )
    assert select_matrix_views(multiple_models_for_an_algorithm, ["IPPO", "FCP"]) == (
        "models",
        "algorithms",
    )
