import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

EXPERIMENT_PARAMETER_KEYS = (
    "ENV_NAME",
    "ENV_KWARGS",
    "ARCHITECTURE",
    "LR",
    "ANNEAL_LR",
    "LR_WARMUP",
    "NUM_ENVS",
    "NUM_STEPS",
    "UPDATE_EPOCHS",
    "NUM_MINIBATCHES",
    "TOTAL_TIMESTEPS",
    "REW_SHAPING_HORIZON",
    "FC_DIM_SIZE",
    "GRU_HIDDEN_DIM",
    "CLIP_EPS",
    "ENT_COEF",
    "GAMMA",
    "GAE_LAMBDA",
    "SCALE_CLIP_EPS",
    "VF_COEF",
    "MAX_GRAD_NORM",
    "ACTIVATION",
    "SEED",
    "NUM_SEEDS",
)


def experiment_signature(config: Mapping[str, Any]) -> str:
    parameters = {key: config.get(key) for key in EXPERIMENT_PARAMETER_KEYS}
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _path_component(value: object) -> str:
    component = re.sub(r"[^\w.-]+", "-", str(value)).strip("-._")
    if not component:
        raise ValueError("EXPERIMENT_FOLDER must contain a path-safe character")
    return component


def experiment_folder(config: Mapping[str, Any]) -> str:
    label = config.get("EXPERIMENT_FOLDER")
    if not label:
        learning_rate = str(config["LR"]).replace(".", "p")
        total_steps = f"{float(config['TOTAL_TIMESTEPS']):g}".replace("+", "")
        label = (
            f"seed{config['SEED']}_lr{learning_rate}_envs{config['NUM_ENVS']}_"
            f"steps{config['NUM_STEPS']}_total{total_steps}"
        )
    return f"{_path_component(label)}_{experiment_signature(config)}"
