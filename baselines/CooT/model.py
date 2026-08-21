"""GPT-2-style Coordination Transformer used by the CooT baseline.

PORTING NOTE: the release instantiates Hugging Face's PyTorch GPT2Model. This
independent port uses Flax primitives already present in the repository while
preserving its tokenization, causal architecture, and configured dropouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class CooTConfig:
    """Architecture and sequence configuration for CooT.

    Defaults match the paper/supplementary Overcooked model: four GPT-2 blocks,
    two attention heads, 128 hidden units, five context episodes, and six query
    states. ``episode_horizon`` is intentionally supplied by the V3 dataset
    because the role-switch scenarios are longer than the paper's 200 steps.
    """

    observation_dim: int
    action_dim: int = 6
    episode_horizon: int = 450
    context_episodes: int = 5
    num_query_states: int = 6
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 2
    dropout_rate: float = 0.3
    attention_dropout_rate: float = 0.21
    layer_norm_epsilon: float = 1e-5

    @property
    def sequence_length(self) -> int:
        return self.context_episodes * self.episode_horizon + self.num_query_states

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CooTConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


def normalize_v3_observation(observation: jnp.ndarray) -> jnp.ndarray:
    """Normalize V3 observations without erasing its 0..1 transition signals.

    PORTING NOTE: the released CooT code divides legacy Overcooked's 0/255 grid
    tensor by 255. Overcooked V3 appends countdown and map/recipe-preview values
    that are already in 0..1 units. Scaling only magnitudes above one preserves
    the released preprocessing for legacy channels and the new V3 signals.
    """

    observation = observation.astype(jnp.float32)
    return jnp.where(jnp.abs(observation) > 1.0, observation / 255.0, observation)


class GPT2Block(nn.Module):
    """Pre-layer-normalized causal block matching Hugging Face GPT-2."""

    config: CooTConfig

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        attention_mask: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        residual = hidden
        hidden = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)(hidden)
        hidden = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.embedding_dim,
            out_features=self.config.embedding_dim,
            dropout_rate=self.config.attention_dropout_rate,
            deterministic=deterministic,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
            use_bias=True,
        )(hidden, mask=attention_mask)
        hidden = nn.Dropout(rate=self.config.dropout_rate)(
            hidden, deterministic=deterministic
        )
        hidden = residual + hidden

        residual = hidden
        hidden = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)(hidden)
        hidden = nn.Dense(
            4 * self.config.embedding_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        hidden = nn.gelu(hidden, approximate=True)
        hidden = nn.Dense(
            self.config.embedding_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        hidden = nn.Dropout(rate=self.config.dropout_rate)(
            hidden, deterministic=deterministic
        )
        return residual + hidden


class CooTTransformer(nn.Module):
    """Predict best-response actions from episode-aligned context and queries.

    Every context timestep is one ``(state, ego action, sparse reward)`` token.
    Query tokens contain only state; their action and reward fields are zero.
    This is the released supplementary representation rather than a conventional
    Decision Transformer return-to-go representation. The released runtime uses
    completed episodes; the V3 runtime may expose a right-aligned partial newest
    episode every N transitions without changing this model's tensor layout.
    """

    config: CooTConfig

    @nn.compact
    def __call__(
        self,
        context_states: jnp.ndarray,
        context_actions: jnp.ndarray,
        context_rewards: jnp.ndarray,
        query_states: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size = context_states.shape[0]
        context_states = normalize_v3_observation(context_states).reshape(
            batch_size, context_states.shape[1], -1
        )
        query_states = normalize_v3_observation(query_states).reshape(
            batch_size, query_states.shape[1], -1
        )

        if context_states.shape[-1] != self.config.observation_dim:
            raise ValueError(
                "context observation dimension does not match CooTConfig: "
                f"{context_states.shape[-1]} != {self.config.observation_dim}"
            )
        if query_states.shape[-1] != self.config.observation_dim:
            raise ValueError(
                "query observation dimension does not match CooTConfig: "
                f"{query_states.shape[-1]} != {self.config.observation_dim}"
            )

        context_actions = context_actions.astype(jnp.float32)
        context_rewards = context_rewards.astype(jnp.float32)
        if context_rewards.ndim == 2:
            context_rewards = context_rewards[..., None]

        query_actions = jnp.zeros(
            (batch_size, query_states.shape[1], self.config.action_dim),
            dtype=jnp.float32,
        )
        query_rewards = jnp.zeros(
            (batch_size, query_states.shape[1], 1), dtype=jnp.float32
        )
        context_tokens = jnp.concatenate(
            [context_states, context_actions, context_rewards], axis=-1
        )
        query_tokens = jnp.concatenate(
            [query_states, query_actions, query_rewards], axis=-1
        )
        tokens = jnp.concatenate([context_tokens, query_tokens], axis=1)

        if tokens.shape[1] > self.config.sequence_length:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds configured maximum "
                f"{self.config.sequence_length}"
            )

        hidden = nn.Dense(
            self.config.embedding_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
            name="transition_embedding",
        )(tokens)
        position_ids = jnp.arange(tokens.shape[1], dtype=jnp.int32)
        position_embedding = nn.Embed(
            num_embeddings=self.config.sequence_length,
            features=self.config.embedding_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="position_embedding",
        )(position_ids)
        hidden = hidden + position_embedding[None, :, :]
        hidden = nn.Dropout(rate=self.config.dropout_rate)(
            hidden, deterministic=deterministic
        )

        attention_mask = nn.make_causal_mask(
            jnp.ones((batch_size, tokens.shape[1]), dtype=jnp.bool_)
        )
        for layer_index in range(self.config.num_layers):
            hidden = GPT2Block(self.config, name=f"block_{layer_index}")(
                hidden,
                attention_mask,
                deterministic=deterministic,
            )
        hidden = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, name="final_layer_norm"
        )(hidden)
        logits = nn.Dense(
            self.config.action_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
            name="action_head",
        )(hidden)
        return logits[:, -query_states.shape[1] :, :]
