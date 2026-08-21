from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from baselines.IPPO.ippo_overcooked_v3 import (
    _prefixed_wandb_metrics,
    _resolve_wandb_mode,
)
from jaxmarl._env import load_project_env
from jaxmarl._experiment import experiment_folder

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "conf"
SCENARIO_FAMILIES = {
    "splitnosig": ("kitchen_split", False),
    "splitsig": ("kitchen_split", True),
    "outagenosig": ("resource_outage", False),
    "outagesig": ("resource_outage", True),
}
SCENARIOS = {
    f"{family}_{variant}": metadata
    for family, metadata in SCENARIO_FAMILIES.items()
    for variant in range(3)
}
SWEEP_SCENARIOS = list(SCENARIOS)


def test_default_training_config_preserves_dynamic_00():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="ippo_overcooked_v3")

    assert config.ENV_KWARGS.layout == "dynamic_00"
    assert config.get("LAYOUT_VARIANT") is None
    assert config.EXPERIMENT == "dynamic_map"
    assert config.SAVES_DIR == "saves"
    assert config.ENTITY == "cilab-overcooked"
    assert config.ENV_KWARGS.include_transition_countdown is True
    assert config.ENV_KWARGS.include_layout_change_mask is True
    assert config.ENV_KWARGS.include_signal_status is True
    assert config.ENV_KWARGS.transition_warning_steps == 20
    assert config.ENV_KWARGS.signal_activation_time == 10
    assert config.ENV_KWARGS.signal_activation_cost == 0.0
    assert config.get("WANDB_DIR") is None
    assert config.RECORD_FINAL_EPISODE is True
    assert config.RECORD_MAX_STEPS == 400
    assert config.RECORD_VIDEO_FPS == 10
    assert config.RECORD_VIDEO_QUALITY == 5
    assert config.upload_final_checkpoint is True


def test_recording_hydra_group_can_disable_final_video():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="ippo_overcooked_v3",
            overrides=["recording=disabled"],
        )

    assert config.RECORD_FINAL_EPISODE is False


def test_checkpoint_upload_boolean_can_disable_artifacts():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="ippo_overcooked_v3",
            overrides=["upload_final_checkpoint=false"],
        )

    assert config.upload_final_checkpoint is False


def test_hydra_scenario_group_composes_all_conditions():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        for scenario, (experiment, signal_enabled) in SCENARIOS.items():
            config = compose(
                config_name="ippo_overcooked_v3",
                overrides=[f"scenario={scenario}"],
            )
            assert config.ENV_NAME == "overcooked_v3"
            assert config.ENV_KWARGS.layout == scenario
            assert config.EXPERIMENT == experiment
            assert config.CONDITION == scenario
            assert config.SIGNAL_ENABLED is signal_enabled
            assert config.wandb_mode == "online"


def test_distance_switch_scenarios_use_fixed_positions_and_full_episode():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        for variant in range(3):
            scenario = f"distance_switch_{variant}"
            config = compose(
                config_name="ippo_overcooked_v3",
                overrides=[f"scenario={scenario}"],
            )
            assert config.ENV_KWARGS.layout == scenario
            assert config.ENV_KWARGS.max_steps == 450
            assert config.ENV_KWARGS.random_agent_positions is False
            assert config.EXPERIMENT == "distance_switch"
            assert config.CONDITION == scenario
            assert config.SIGNAL_ENABLED is False
            assert config.WANDB_GROUP == "distance_switch"


def test_dotenv_configures_wandb_but_not_hydra_saves_dir(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "WANDB_API_KEY=test-key-from-dotenv\n"
        "WANDB_ENTITY=test-entity-from-dotenv\n"
        "WANDB_PROJECT=test-project-from-dotenv\n"
        "WANDB_MODE=online\n"
        "SAVES_DIR=/tmp/test-saves-from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setenv("SAVES_DIR", "/tmp/test-saves-from-shell")
    monkeypatch.setenv("WANDB_MODE", "offline")

    assert load_project_env(dotenv_path)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="ippo_overcooked_v3")

    assert config.ENTITY == "test-entity-from-dotenv"
    assert config.PROJECT == "test-project-from-dotenv"
    assert config.wandb_mode == "offline"
    assert config.SAVES_DIR == "saves"
    assert config.get("WANDB_API_KEY") is None

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        overridden = compose(
            config_name="ippo_overcooked_v3",
            overrides=["wandb_mode=disabled"],
        )

    assert overridden.wandb_mode == "disabled"


def test_wandb_mode_defaults_offline_without_api_key():
    assert _resolve_wandb_mode({"wandb_mode": "online"}, {}) == "offline"


def test_wandb_mode_stays_online_with_api_key():
    environ = {"WANDB_API_KEY": "test-key"}
    assert _resolve_wandb_mode({"wandb_mode": "online"}, environ) == "online"


def test_explicit_offline_and_disabled_modes_are_preserved():
    assert _resolve_wandb_mode({"wandb_mode": "offline"}, {}) == "offline"
    assert _resolve_wandb_mode({"wandb_mode": "disabled"}, {}) == "disabled"


def test_wandb_sweep_covers_scenarios_and_seeds():
    sweep = OmegaConf.to_container(
        OmegaConf.load(
            ROOT / "experiment" / "sweeps" / "train_ippo.yaml"
        ),
        resolve=False,
    )

    assert sweep["method"] == "grid"
    assert sweep["metric"] == {
        "goal": "maximize",
        "name": "train/episode_return",
    }
    assert sweep["parameters"]["scenario"]["values"] == SWEEP_SCENARIOS
    assert "LAYOUT_VARIANT" not in sweep["parameters"]
    assert sweep["parameters"]["SEED"]["values"] == [0, 1, 2, 3, 4, 5]
    assert sweep["parameters"]["recording"]["value"] == "enabled"
    assert "${args_no_hyphens}" in sweep["command"]


def test_wandb_metrics_are_split_into_train_and_debug_namespaces():
    metrics = _prefixed_wandb_metrics(
        {
            "returned_episode_returns": 10.0,
            "actor_loss": 0.5,
            "env_step": 100,
            "layout_index": 1,
            "layout_change_events": 2,
            "transition_countdown": 0.25,
            "signal_steps_remaining": 7,
            "signal_active": 1.0,
            "signal_activation_events": 3,
        }
    )

    assert metrics == {
        "train/episode_return": 10.0,
        "train/actor_loss": 0.5,
        "train/env_step": 100,
        "debug/layout_index": 1,
        "debug/layout_change_events": 2,
        "debug/transition_countdown": 0.25,
        "debug/signal_steps_remaining": 7,
        "debug/signal_active": 1.0,
        "debug/signal_activation_events": 3,
    }


def test_experiment_folder_uses_only_key_experiment_parameters(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        hydra_config = compose(
            config_name="ippo_overcooked_v3",
            overrides=["scenario=splitsig_0", f"SAVES_DIR={tmp_path}"],
        )
    config = OmegaConf.to_container(hydra_config, resolve=True)

    assert experiment_folder(config) == "splitsig_0_cnn_seed0"

    fixed_parameter_change = deepcopy(config)
    fixed_parameter_change["LR"] = config["LR"] * 2
    fixed_parameter_change["NUM_ENVS"] = config["NUM_ENVS"] * 2
    fixed_parameter_change["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS"] * 2
    assert experiment_folder(fixed_parameter_change) == "splitsig_0_cnn_seed0"

    seed_change = deepcopy(config)
    seed_change["SEED"] = 1
    assert experiment_folder(seed_change) == "splitsig_0_cnn_seed1"

    architecture_change = deepcopy(config)
    architecture_change["ARCHITECTURE"] = "rnn"
    assert experiment_folder(architecture_change) == "splitsig_0_rnn_seed0"

    layout_change = deepcopy(config)
    layout_change["ENV_KWARGS"]["layout"] = "outagesig_0"
    assert experiment_folder(layout_change) == "outagesig_0_cnn_seed0"


def test_custom_experiment_name_is_safe_and_precedes_seed():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        hydra_config = compose(
            config_name="ippo_overcooked_v3",
            overrides=["scenario=outagenosig_0"],
        )
    config = OmegaConf.to_container(hydra_config, resolve=True)
    config["EXPERIMENT_FOLDER"] = "learning rate/sweep 01"

    folder = experiment_folder(config)
    changed = deepcopy(config)
    changed["SEED"] = 1

    assert folder == "outagenosig_0_cnn_learning-rate-sweep-01_seed0"
    assert "/" not in folder
    assert folder != experiment_folder(changed)

    config["EXPERIMENT_FOLDER"] = "양파 실험/01"
    assert experiment_folder(config) == "outagenosig_0_cnn_양파-실험-01_seed0"


def test_tracking_parameters_do_not_change_experiment_folder():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        hydra_config = compose(config_name="ippo_overcooked_v3")
    config = OmegaConf.to_container(hydra_config, resolve=True)
    changed = deepcopy(config)
    changed["PROJECT"] = "another-wandb-project"
    changed["wandb_mode"] = "offline"
    changed["SAVES_DIR"] = "/another/model/root"

    assert experiment_folder(config) == experiment_folder(changed)
