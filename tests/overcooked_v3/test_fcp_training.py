from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

from baselines.IPPO.ippo_overcooked_v3 import _checkpoint_update_steps
from baselines.FCP.fcp_overcooked_v3 import (
    _evenly_spaced,
    discover_population_checkpoints,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "conf"


def _population_config(population_dir):
    return {
        "ARCHITECTURE": "rnn",
        "ENV_KWARGS": {"layout": "split_0"},
        "FCP": {
            "population_dir": str(population_dir),
            "snapshots_per_policy": 3,
            "minimum_population_size": 2,
            "max_population_size": None,
        },
    }


def test_evenly_spaced_snapshot_selection_keeps_final_item():
    assert _evenly_spaced(["early", "middle", "late", "final"], 3) == [
        "early",
        "middle",
        "final",
    ]


def test_fractional_checkpoint_schedule_uses_final_save_for_100_percent():
    config = {
        "NUM_UPDATES": 457,
        "CHECKPOINT_FRACTIONS": [0.1, 0.5, 1.0],
    }

    assert _checkpoint_update_steps(config) == (46, 229)


def test_population_discovery_selects_matching_sp_snapshots(tmp_path):
    names = [
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0_update000010.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0_update000020.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0_update000030.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed1_vmap0.safetensors",
        "ippo_cnn_overcooked_v3_split_0_seed2_vmap0.safetensors",
        "ippo_rnn_overcooked_v3_outage_0_seed2_vmap0.safetensors",
        "fcp_rnn_overcooked_v3_split_0_seed2_vmap0.safetensors",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"checkpoint")

    selected = discover_population_checkpoints(_population_config(tmp_path))

    assert [path.name for path in selected] == [
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0_update000010.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0_update000020.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed0_vmap0.safetensors",
        "ippo_rnn_overcooked_v3_split_0_seed1_vmap0.safetensors",
    ]


def test_fcp_hydra_config_uses_rnn_and_default_v3_observation():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="fcp_overcooked_v3",
            overrides=["scenario=split_0"],
        )

    assert config.ALGORITHM == "FCP"
    assert config.ARCHITECTURE == "rnn"
    assert config.ENTITY == "cilab-overcooked"
    assert config.PROJECT == "overcooked-v3-fcp_train"
    assert config.ENV_KWARGS.layout == "split_0"
    assert config.FCP.population_dir == "saves/fcp_population"
    assert config.FCP.snapshots_per_policy == 3


def test_fcp_eval_sweep_runs_seedwise_fcp_only():
    evaluation = yaml.safe_load(
        (ROOT / "experiment/fcp/eval.yaml").read_text(encoding="utf-8")
    )

    assert evaluation["parameters"]["algorithms"]["value"] == "FCP"
    assert evaluation["parameters"]["layout"]["values"] == [
        "split_0",
        "split_1",
        "split_2",
        "outage_0",
        "outage_1",
        "outage_2",
        "recipe_switch_0",
        "recipe_switch_1",
        "recipe_switch_2",
        "distance_switch_0",
        "distance_switch_1",
        "distance_switch_2",
    ]
    assert evaluation["parameters"]["max-steps"]["value"] == 450
    assert (
        evaluation["parameters"]["output-project"]["value"]
        == "cilab-overcooked/overcooked-v3-fcp_eval"
    )
    assert "cilab-overcooked/overcooked-v3-fcp_train" in evaluation["command"]
    assert "--algorithms" not in evaluation["command"]
    assert evaluation["command"][-1] == "${args_no_equals}"


def test_fcp_switch_sweeps_use_three_population_and_six_training_seeds():
    population = yaml.safe_load(
        (ROOT / "experiment/fcp/population.yaml").read_text(encoding="utf-8")
    )
    training = yaml.safe_load(
        (ROOT / "experiment/fcp/train.yaml").read_text(encoding="utf-8")
    )
    expected_layouts = [
        "split_0",
        "split_1",
        "split_2",
        "outage_0",
        "outage_1",
        "outage_2",
        "recipe_switch_0",
        "recipe_switch_1",
        "recipe_switch_2",
        "distance_switch_0",
        "distance_switch_1",
        "distance_switch_2",
    ]

    assert population["parameters"]["scenario"]["values"] == expected_layouts
    assert population["parameters"]["SEED"]["values"] == [0, 1, 2]
    assert (
        population["parameters"]["PROJECT"]["value"]
        == "overcooked-v3-fcp-population"
    )
    assert population["parameters"]["CHECKPOINT_INTERVAL"]["value"] == 0
    assert population["parameters"]["CHECKPOINT_FRACTIONS"]["value"] == [
        0.1,
        0.5,
        1.0,
    ]
    assert training["parameters"]["scenario"]["values"] == expected_layouts
    assert training["parameters"]["SEED"]["values"] == [0, 1, 2, 3, 4, 5]
    assert training["parameters"]["PROJECT"]["value"] == "overcooked-v3-fcp_train"
