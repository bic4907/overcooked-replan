"""Checkpoint manifests and policy runners shared by CooT tools."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.wrappers.baselines import load_params

try:
    from baselines.IPPO.ippo_overcooked_v3 import (
        ActorCriticCNN,
        ActorCriticRNN,
        ScannedRNN,
    )
except ModuleNotFoundError as error:  # Direct execution from baselines/CooT.
    if error.name != "baselines":
        raise
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from baselines.IPPO.ippo_overcooked_v3 import (  # type: ignore[no-redef]
        ActorCriticCNN,
        ActorCriticRNN,
        ScannedRNN,
    )


@dataclass(frozen=True)
class PolicySpec:
    checkpoint: Path
    architecture: str = "rnn"
    activation: str = "relu"
    fc_dim_size: int = 128
    gru_hidden_dim: int = 128
    stochastic: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, base_dir: Path) -> "PolicySpec":
        if not values.get("checkpoint"):
            raise ValueError("Every policy manifest entry needs a checkpoint")
        checkpoint = Path(str(values["checkpoint"])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (base_dir / checkpoint).resolve()
        architecture = str(values.get("architecture", "rnn")).lower()
        if architecture not in {"cnn", "rnn"}:
            raise ValueError(f"Unsupported policy architecture: {architecture}")
        return cls(
            checkpoint=checkpoint,
            architecture=architecture,
            activation=str(values.get("activation", "relu")),
            fc_dim_size=int(values.get("fc_dim_size", 128)),
            gru_hidden_dim=int(values.get("gru_hidden_dim", 128)),
            stochastic=bool(values.get("stochastic", False)),
        )


def load_pair_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CooT pair manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("CooT pair manifest must contain a non-empty 'pairs' list")
    identifiers = [str(pair.get("id", index)) for index, pair in enumerate(pairs)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("CooT pair ids must be unique")
    return manifest, manifest_path


class CheckpointPolicy:
    """Small stateful wrapper around an IPPO/FCP CNN or RNN checkpoint."""

    def __init__(self, spec: PolicySpec, action_dim: int):
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {spec.checkpoint}")
        config = {
            "ACTIVATION": spec.activation,
            "FC_DIM_SIZE": spec.fc_dim_size,
            "GRU_HIDDEN_DIM": spec.gru_hidden_dim,
        }
        network_class: Any = (
            ActorCriticRNN if spec.architecture == "rnn" else ActorCriticCNN
        )
        self.network = network_class(action_dim, config=config)
        self.params = load_params(spec.checkpoint)
        self.spec = spec
        self.action_dim = int(action_dim)
        self.hidden = ScannedRNN.initialize_carry(1, spec.gru_hidden_dim)

        def _select(params, hidden, observation, done, key, stochastic):
            hidden, distribution, _ = cast(
                Any,
                self.network.apply(
                    params,
                    hidden,
                    (
                        observation[jnp.newaxis, jnp.newaxis, ...],
                        done.reshape(1, 1),
                    ),
                ),
            )
            logits = distribution.logits.squeeze((0, 1))
            action = jax.lax.cond(
                stochastic,
                lambda _: jax.random.categorical(key, logits),
                lambda _: jnp.argmax(logits),
                operand=None,
            )
            return hidden, action, jax.nn.softmax(logits)

        self._select = jax.jit(_select, static_argnums=(5,))

    def reset(self) -> None:
        self.hidden = ScannedRNN.initialize_carry(1, self.spec.gru_hidden_dim)

    def act(
        self, observation: np.ndarray | jax.Array, key: jax.Array
    ) -> tuple[int, np.ndarray]:
        self.hidden, action, probabilities = self._select(
            self.params,
            self.hidden,
            jnp.asarray(observation),
            jnp.asarray(False),
            key,
            self.spec.stochastic,
        )
        return int(action), np.asarray(jax.device_get(probabilities))


def policy_spec_dict(spec: PolicySpec) -> dict[str, Any]:
    return {
        "checkpoint": str(spec.checkpoint),
        "architecture": spec.architecture,
        "activation": spec.activation,
        "fc_dim_size": spec.fc_dim_size,
        "gru_hidden_dim": spec.gru_hidden_dim,
        "stochastic": spec.stochastic,
    }
