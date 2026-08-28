from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from wandb.sdk.launch.sweeps.utils import create_sweep_command_args

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "conf"
EXPERIMENT_DIR = ROOT / "experiment" / "transition_window_observer"
FCP_EXPERIMENT_DIR = ROOT / "experiment" / "fcp_transition_window_observer"
OBSERVERS = ["none", "agent_0", "agent_1", "both"]
LAYOUTS = [
    f"{family}_{variant}"
    for family in ("split", "outage", "recipe_switch", "distance_switch")
    for variant in range(2)
]


def _load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_transition_observer_training_config_is_rnn_and_names_each_arm():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="ippo_rnn_transition_window_overcooked_v3",
            overrides=["scenario=outage_1", "TRANSITION_OBSERVER=agent_0", "SEED=3"],
        )

    assert config.ARCHITECTURE == "rnn"
    assert config.ENV_KWARGS.layout == "outage_1"
    assert config.ENV_KWARGS.transition_observer == "agent_0"
    assert config.EXPERIMENT_FOLDER == "transition-window-agent_0"
    assert config.WANDB_GROUP == "transition-window-outage_1"
    assert config.RUN_NAME == "ippo_rnn_outage_1_observer-agent_0_seed3"


def test_transition_observer_sweeps_cover_four_arms_and_eight_layouts():
    training = _load_yaml(EXPERIMENT_DIR / "train.yaml")
    evaluation = _load_yaml(EXPERIMENT_DIR / "eval.yaml")

    assert training["parameters"]["scenario"]["values"] == LAYOUTS
    assert training["parameters"]["TRANSITION_OBSERVER"]["values"] == OBSERVERS
    assert training["parameters"]["SEED"]["values"] == [0, 1, 2, 3, 4, 5]
    assert training["parameters"]["recording"]["value"] == "disabled"
    assert evaluation["parameters"]["layout"]["values"] == LAYOUTS
    assert evaluation["parameters"]["transition-observer"]["values"] == OBSERVERS
    assert training["command"][-1] == "${args_no_hyphens}"
    assert evaluation["command"][-1] == "${args_no_equals}"


def test_transition_observer_eval_sweep_renders_filter_flag():
    sweep = _load_yaml(EXPERIMENT_DIR / "eval.yaml")
    assigned = {
        name: {"value": values.get("value", values.get("values", [None])[0])}
        for name, values in sweep["parameters"].items()
    }

    rendered = create_sweep_command_args({"args": assigned})["args_no_equals"]

    assert "--transition-observer" in rendered
    assert rendered[rendered.index("--transition-observer") + 1] == "none"


def test_fcp_transition_observer_configs_keep_populations_separate():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        population = compose(
            config_name="fcp_population_transition_window_overcooked_v3",
            overrides=["scenario=split_1", "TRANSITION_OBSERVER=agent_0", "SEED=2"],
        )
        training = compose(
            config_name="fcp_transition_window_overcooked_v3",
            overrides=["scenario=split_1", "TRANSITION_OBSERVER=agent_0", "SEED=2"],
        )

    expected_root = "saves/fcp_transition_window_observer_3seed/population"
    assert population.SAVES_DIR == f"{expected_root}/agent_0"
    assert population.CHECKPOINT_FRACTIONS == [0.1, 0.5, 1.0]
    assert population.upload_final_checkpoint is False
    assert training.ENV_KWARGS.transition_observer == "agent_0"
    assert training.FCP.population_dir == f"{expected_root}/agent_0"
    assert training.RUN_NAME == "fcp_rnn_split_1_observer-agent_0_seed2"


def test_fcp_transition_observer_sweeps_cover_three_sequential_stages():
    population = _load_yaml(FCP_EXPERIMENT_DIR / "population_3seeds.yaml")
    training = _load_yaml(FCP_EXPERIMENT_DIR / "train_3seeds.yaml")
    evaluation = _load_yaml(FCP_EXPERIMENT_DIR / "eval_3seeds.yaml")

    assert population["parameters"]["scenario"]["values"] == LAYOUTS
    assert population["parameters"]["TRANSITION_OBSERVER"]["values"] == OBSERVERS
    assert population["parameters"]["SEED"]["values"] == [0, 1, 2]
    assert population["parameters"]["PROJECT"]["value"].endswith("_population")
    assert population["command"][-1] == "${args_no_hyphens}"

    assert training["parameters"]["scenario"]["values"] == LAYOUTS
    assert training["parameters"]["TRANSITION_OBSERVER"]["values"] == OBSERVERS
    assert training["parameters"]["SEED"]["values"] == [0, 1, 2]
    assert training["command"][-1] == "${args_no_hyphens}"

    assert evaluation["parameters"]["algorithms"]["value"] == "FCP"
    assert evaluation["parameters"]["layout"]["values"] == LAYOUTS
    assert evaluation["parameters"]["transition-observer"]["values"] == OBSERVERS
    assert evaluation["command"][4:8] == ["--seeds", "0", "1", "2"]
    assert evaluation["command"][-1] == "${args_no_equals}"
