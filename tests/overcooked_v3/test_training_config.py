from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jaxmarl._env import load_project_env

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "conf"
SCENARIOS = {
    "split_no_sig": ("kitchen_split", False),
    "split_sig": ("kitchen_split", True),
    "outage_no_sig": ("resource_outage", False),
    "outage_sig": ("resource_outage", True),
}


def test_default_training_config_preserves_dynamic_00():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="ippo_overcooked_v3")

    assert config.ENV_KWARGS.layout == "dynamic_00"
    assert config.EXPERIMENT == "dynamic_map"


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
            assert config.WANDB_MODE == "disabled"


def test_dotenv_values_feed_hydra_without_overriding_shell(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "WANDB_API_KEY=test-key-from-dotenv\n"
        "WANDB_ENTITY=test-entity-from-dotenv\n"
        "WANDB_PROJECT=test-project-from-dotenv\n"
        "WANDB_MODE=online\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")

    assert load_project_env(dotenv_path)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="ippo_overcooked_v3")

    assert config.ENTITY == "test-entity-from-dotenv"
    assert config.PROJECT == "test-project-from-dotenv"
    assert config.WANDB_MODE == "offline"
    assert config.get("WANDB_API_KEY") is None

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        overridden = compose(
            config_name="ippo_overcooked_v3",
            overrides=["WANDB_MODE=disabled"],
        )

    assert overridden.WANDB_MODE == "disabled"


def test_wandb_sweep_covers_scenarios_and_seeds():
    sweep = OmegaConf.to_container(
        OmegaConf.load(ROOT / "sweeps" / "overcooked_v3_role_scenarios.yaml"),
        resolve=False,
    )

    assert sweep["method"] == "grid"
    assert sweep["run_cap"] == 20
    assert sweep["metric"] == {
        "goal": "maximize",
        "name": "returned_episode_returns",
    }
    assert set(sweep["parameters"]["scenario"]["values"]) == set(SCENARIOS)
    assert sweep["parameters"]["SEED"]["values"] == [0, 1, 2, 3, 4]
    assert "${args_no_hyphens}" in sweep["command"]
