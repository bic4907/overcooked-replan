from pathlib import Path

import yaml
from wandb.sdk.launch.sweeps.utils import create_sweep_command_args

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SWEEP = PROJECT_ROOT / "experiment/sweeps/train_ippo.yaml"
CROSSPLAY_SWEEP = PROJECT_ROOT / "experiment/sweeps/eval_ippo_seedwise.yaml"


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_crossplay_sweep_contains_all_eighty_training_maps():
    training = load_yaml(TRAIN_SWEEP)
    crossplay = load_yaml(CROSSPLAY_SWEEP)

    training_maps = training["parameters"]["scenario"]["values"]
    crossplay_maps = crossplay["parameters"]["layout"]["values"]

    expected_maps = [
        f"{family}_{variant}"
        for family in ("splitnosig", "splitsig", "outagenosig", "outagesig")
        for variant in range(20)
    ]

    assert training_maps == expected_maps
    assert crossplay_maps == training_maps
    assert len(set(crossplay_maps)) == 80


def test_crossplay_sweep_renders_argparse_compatible_flags():
    sweep = load_yaml(CROSSPLAY_SWEEP)
    assigned = {
        name: {"value": values.get("value", values.get("values", [None])[0])}
        for name, values in sweep["parameters"].items()
    }

    rendered = create_sweep_command_args({"args": assigned})["args_no_equals"]

    assert rendered[:4] == [
        "--algorithms",
        "IPPO",
        "--layout",
        "splitnosig_0",
    ]
    assert "--max-steps" in rendered
    assert rendered[rendered.index("--workers-per-gpu") + 1] == "8"
    assert "--vmap-indices" not in rendered
    assert sweep["command"][-1] == "${args_no_equals}"
