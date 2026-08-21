from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

from baselines.FCP.fcp_overcooked_v3 import (
    _evenly_spaced,
    discover_population_checkpoints,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "conf"


def _population_config(population_dir):
    return {
        "ARCHITECTURE": "rnn",
        "ENV_KWARGS": {"layout": "splitnosig_0"},
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


def test_population_discovery_selects_matching_sp_snapshots(tmp_path):
    names = [
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0_update000010.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0_update000020.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0_update000030.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed1_vmap0.safetensors",
        "ippo_cnn_overcooked_v3_splitnosig_0_seed2_vmap0.safetensors",
        "ippo_rnn_overcooked_v3_outagenosig_0_seed2_vmap0.safetensors",
        "fcp_rnn_overcooked_v3_splitnosig_0_seed2_vmap0.safetensors",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"checkpoint")

    selected = discover_population_checkpoints(_population_config(tmp_path))

    assert [path.name for path in selected] == [
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0_update000010.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0_update000020.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed0_vmap0.safetensors",
        "ippo_rnn_overcooked_v3_splitnosig_0_seed1_vmap0.safetensors",
    ]


def test_fcp_hydra_config_uses_rnn_and_default_v3_observation():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="fcp_overcooked_v3",
            overrides=["scenario=splitnosig_0"],
        )

    assert config.ALGORITHM == "FCP"
    assert config.ARCHITECTURE == "rnn"
    assert config.ENV_KWARGS.layout == "splitnosig_0"
    assert config.FCP.population_dir == "saves/fcp_population"
    assert config.FCP.snapshots_per_policy == 3


def test_fcp_eval_sweep_runs_seedwise_fcp_only():
    sweep = yaml.safe_load(
        (ROOT / "experiment/fcp/eval.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert sweep["parameters"]["algorithms"]["value"] == "FCP"
    assert "--algorithms" not in sweep["command"]
    assert sweep["command"][-1] == "${args_no_equals}"
