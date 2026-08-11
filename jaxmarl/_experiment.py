import re
from collections.abc import Mapping
from typing import Any


def _path_component(value: object) -> str:
    component = re.sub(r"[^\w.-]+", "-", str(value)).strip("-._")
    if not component:
        raise ValueError("EXPERIMENT_FOLDER must contain a path-safe character")
    return component


def experiment_folder(config: Mapping[str, Any]) -> str:
    layout = config["ENV_KWARGS"]["layout"]
    architecture = str(config["ARCHITECTURE"]).lower()
    context = f"{layout}_{architecture}_seed{config['SEED']}"
    prefix = config.get("EXPERIMENT_FOLDER")
    label = f"{prefix}_{context}" if prefix else context
    return _path_component(label)
