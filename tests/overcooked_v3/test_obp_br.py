"""The best response to a frozen human model: a population of one."""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from baselines.FCP.fcp_overcooked_v3 import discover_population_checkpoints
from baselines.OBP.eval_human_model import find_best_response
from jaxmarl._experiment import experiment_folder

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "conf"


def _compose(overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name="obp_br_overcooked_v3", overrides=overrides)
    return OmegaConf.to_container(config, resolve=True)


def _sweep(name):
    return yaml.safe_load((ROOT / "experiment/obp" / name).read_text(encoding="utf-8"))


def _run_count(sweep):
    count = 1
    for parameter in sweep["parameters"].values():
        if "values" in parameter:
            count *= len(parameter["values"])
    return count


def test_a_named_partner_is_a_population_of_one(tmp_path):
    """One partner is the point of a best response, not a shortfall."""
    checkpoint = tmp_path / "policy.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    selected = discover_population_checkpoints(
        {"FCP": {"population_checkpoints": [str(checkpoint)], "population_dir": None}}
    )
    assert selected == [checkpoint.resolve()]


def test_a_missing_partner_is_reported_rather_than_skipped(tmp_path):
    with pytest.raises(FileNotFoundError, match="do not exist"):
        discover_population_checkpoints(
            {"FCP": {"population_checkpoints": [str(tmp_path / "absent.safetensors")]}}
        )


def test_discovery_still_needs_a_directory_when_nothing_is_named():
    with pytest.raises(ValueError, match="population_dir"):
        discover_population_checkpoints({"FCP": {}})


def test_the_best_response_plays_the_environment_the_prior_was_trained_on():
    """The partner's weights only describe the same network on the same canvas."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        prior = OmegaConf.to_container(
            compose(
                config_name="ippo_overcooked_v3",
                overrides=["scenario=obp10", "ARCHITECTURE=cnn", "SEED=3"],
            ),
            resolve=True,
        )
    best_response = _compose(["SEED=3"])
    assert best_response["ENV_NAME"] == prior["ENV_NAME"]
    assert best_response["ENV_KWARGS"] == prior["ENV_KWARGS"]
    for key in ("ARCHITECTURE", "ACTIVATION", "FC_DIM_SIZE"):
        assert best_response[key] == prior[key], key


def test_each_arm_gets_its_own_checkpoint_folder():
    """The checkpoint filename carries layout, architecture and seed -- all
    shared by the three arms -- so the arm has to reach the folder name."""
    folders = {
        arm: experiment_folder(_compose([f"HUMAN_MODEL_ARM={arm}", "SEED=2"]))
        for arm in ("bc", "obp", "obp-frozen")
    }
    assert len(set(folders.values())) == 3
    for arm, folder in folders.items():
        assert arm in folder


def test_the_partner_checkpoint_follows_the_arm_and_the_seed():
    config = _compose(["HUMAN_MODEL_ARM=obp-frozen", "SEED=4"])
    assert config["FCP"]["population_checkpoints"] == [
        "saves/obp_bc/obp-frozen_seed4/policy.safetensors"
    ]


def test_the_final_best_response_is_played_not_an_intermediate(tmp_path):
    folder = tmp_path / "obp10_cnn_obp_br_obp_seed1"
    folder.mkdir()
    for name in (
        "fcp_cnn_overcooked_v3_multilayout_obp10_seed1_vmap0_update000010.safetensors",
        "fcp_cnn_overcooked_v3_multilayout_obp10_seed1_vmap0.safetensors",
    ):
        (folder / name).write_bytes(b"checkpoint")
    found = find_best_response(tmp_path, "obp", 1)
    assert found.name.endswith("seed1_vmap0.safetensors")
    assert "_update" not in found.name


def test_an_untrained_arm_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the BR sweep"):
        find_best_response(tmp_path, "obp", 1)


def test_the_br_sweeps_cover_three_arms_and_six_seeds():
    train = _sweep("br_train.yaml")
    assert train["parameters"]["HUMAN_MODEL_ARM"]["values"] == ["bc", "obp", "obp-frozen"]
    assert train["parameters"]["SEED"]["values"] == [0, 1, 2, 3, 4, 5]
    assert _run_count(train) == 18
    assert "obp_br_overcooked_v3" in train["command"]


def test_the_evaluation_human_is_fitted_on_the_side_the_arms_hold_out():
    sweep = _sweep("eval_human_train.yaml")
    assert sweep["parameters"]["split-side"]["value"] == "validation"
    assert sweep["parameters"]["arm"]["value"] == "bc"
    assert sweep["parameters"]["output-root"]["value"] == "saves/obp_eval_human"
    assert _run_count(sweep) == 6


def test_every_collaborative_agent_is_scored_against_the_same_human():
    sweep = _sweep("br_eval.yaml")
    assert sweep["parameters"]["partner-arm"]["values"] == [
        "prior",
        "bc",
        "obp",
        "obp-frozen",
    ]
    assert sweep["parameters"]["model-root"]["value"] == "saves/obp_eval_human"
    assert _run_count(sweep) == 24
