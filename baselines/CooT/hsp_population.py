"""CooT-compatible HSP utility catalog and Overcooked V3 reward routing.

This ports Table 5 of CooT and the reward enumeration rule from the released
ZSC-Eval scripts.  Candidate ids are zero-based and deterministic: axes are
enumerated in table order with ``itertools.product`` and candidates with more
than three non-zero *variable* axes are discarded.

PORTING NOTE: despite its name, the release's ``--share_policy`` flag uses
``store_false``.  Passing it selects the separated MAPPO runner, consistent
with the two separately saved actor checkpoints.  This V3 port instead keeps
the repository's existing shared IPPO network, assigns biased-agent utility to
logical agent 0 and sparse task reward to logical agent 1, and reproduces
``--random_index`` by swapping their physical seats per episode. A saved
checkpoint is consequently an explicit shared-policy approximation, not two
independently extractable HSP actors.
The V3 environment also has one active recipe at a time, so placement-quality
events use current-recipe prefix viability rather than the release's
multi-order value calculation.
The archive's extraction utilities choose intermediate skill checkpoints by
matching return targets, while the released CooT path does not identify one
portable target for every layout.  V3 therefore labels the 50%-of-updates
checkpoint as ``mid`` and records that proxy explicitly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, MutableMapping

import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.events import EVENT_INDEX, EVENT_NAMES


PORTING_NOTE = (
    "CooT/ZSC-Eval passes the store_false --share_policy flag and therefore "
    "uses separated MAPPO actors, consistent with its separate checkpoints. "
    "Overcooked V3 instead uses the existing "
    "shared IPPO parameters, logical agent_0's Table-5 utility, logical "
    "agent_1's sparse task reward, and per-episode physical-seat swapping. "
    "Mid/final files are shared-policy approximations. V3 "
    "placement quality is current-recipe prefix viability. The V3 mid file is "
    "the 50%-of-training checkpoint; legacy extraction used return-targeted "
    "skill checkpoints rather than a fixed update fraction."
)

SPARSE_REWARD_TARGET = "__sparse_reward__"
MAX_ACTIVE_BIAS_TERMS = 3


@dataclass(frozen=True)
class UtilityAxis:
    """One Table-5 reward choice, optionally shared by several events."""

    name: str
    targets: tuple[str, ...]
    values: tuple[float, ...]
    variable: bool = True


@dataclass(frozen=True)
class UtilityProfile:
    name: str
    axes: tuple[UtilityAxis, ...]


@dataclass(frozen=True)
class HSPCandidate:
    profile: str
    candidate_id: int
    event_weights: tuple[float, ...]
    sparse_reward_weight: float
    axis_values: tuple[tuple[str, float], ...]
    active_bias_terms: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "candidate_id": self.candidate_id,
            "event_names": list(EVENT_NAMES),
            "event_weight_vector": list(self.event_weights),
            "event_weights": dict(zip(EVENT_NAMES, self.event_weights)),
            "sparse_reward_weight": self.sparse_reward_weight,
            "axis_values": dict(self.axis_values),
            "active_bias_terms": list(self.active_bias_terms),
            "max_active_bias_terms": MAX_ACTIVE_BIAS_TERMS,
        }


# The three default shaping weights are fixed in the supplementary HSP script
# and do not count toward the <=3 variable-term filter.
_FIXED_EVENT_WEIGHTS = {
    "USEFUL_DISH_PICKUP": 3.0,
    "SOUP_PICKUP": 5.0,
    "PLACEMENT_IN_POT": 3.0,
}

UTILITY_PROFILES = {
    "bothway": UtilityProfile(
        name="bothway",
        axes=(
            UtilityAxis(
                "pickup_onions_from_dispenser",
                ("pickup_onion_from_O",),
                (-20.0, 0.0, 10.0),
            ),
            UtilityAxis(
                "pickup_dishes_from_dispenser",
                ("pickup_dish_from_D",),
                (0.0, 10.0),
            ),
            UtilityAxis(
                "pickup_onion_or_dish_from_counters",
                ("pickup_onion_from_X", "pickup_dish_from_X"),
                (-20.0, 0.0),
            ),
            UtilityAxis(
                "pickup_soup_from_counters",
                ("pickup_soup_from_X",),
                (-20.0, 0.0),
            ),
            UtilityAxis("stay", ("STAY",), (-0.1, 0.0, 0.1)),
            UtilityAxis(
                "order_reward_scale",
                (SPARSE_REWARD_TARGET,),
                (0.1, 1.0),
            ),
        ),
    ),
    "multi_recipe": UtilityProfile(
        name="multi_recipe",
        axes=(
            UtilityAxis("put_onions_into_pots", ("potting_onion",), (-20.0, 0.0)),
            UtilityAxis("put_tomatoes_into_pots", ("potting_tomato",), (-20.0, 0.0)),
            UtilityAxis(
                "deliver_two_ingredient_soup",
                ("deliver_size_two_order",),
                (-5.0, 0.0, 20.0),
            ),
            UtilityAxis(
                "deliver_three_ingredient_soup",
                ("deliver_size_three_order",),
                (-15.0, 0.0, 10.0),
            ),
            UtilityAxis("stay", ("STAY",), (-0.1, 0.0, 0.1)),
            UtilityAxis(
                "order_reward_scale",
                (SPARSE_REWARD_TARGET,),
                (1.0,),
                variable=False,
            ),
        ),
    ),
    "other": UtilityProfile(
        name="other",
        axes=(
            UtilityAxis(
                "pickup_onions_from_dispenser",
                ("pickup_onion_from_O",),
                (-20.0, 0.0, 10.0),
            ),
            UtilityAxis(
                "pickup_dishes_from_dispenser",
                ("pickup_dish_from_D",),
                (-20.0, 0.0, 10.0),
            ),
            UtilityAxis("deliver_soup", ("delivery",), (-20.0, 0.0)),
            UtilityAxis("stay", ("STAY",), (-0.1, 0.0, 0.1)),
            UtilityAxis(
                "order_reward_scale",
                (SPARSE_REWARD_TARGET,),
                (0.1, 1.0),
            ),
        ),
    ),
}

EXPECTED_CANDIDATE_COUNTS = {
    "bothway": 54,
    "multi_recipe": 72,
    "other": 52,
}


def normalize_profile_name(profile: str) -> str:
    normalized = str(profile).strip().lower().replace("-", "_")
    aliases = {
        "bothway_coord": "bothway",
        "multi": "multi_recipe",
        "multirecipe": "multi_recipe",
        "others": "other",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in UTILITY_PROFILES:
        raise ValueError(
            f"Unknown HSP utility profile {profile!r}; expected one of "
            f"{sorted(UTILITY_PROFILES)} or 'auto'"
        )
    return normalized


def profile_for_scenario(scenario: str) -> str:
    """Resolve the conservative automatic V3-to-Table-5 mapping.

    Recipe-switch layouts are the only V3 scenarios mapped to multi-recipe.
    Every other scenario maps to ``other``.  In particular, split layouts are
    not silently treated as the paper's Bothway Coordination layout.
    """

    scenario_name = str(scenario).rsplit("/", 1)[-1].lower()
    if scenario_name.startswith("recipe_switch"):
        return "multi_recipe"
    return "other"


@lru_cache(maxsize=None)
def candidate_catalog(profile: str) -> tuple[HSPCandidate, ...]:
    profile_name = normalize_profile_name(profile)
    definition = UTILITY_PROFILES[profile_name]
    candidates = []

    for values in itertools.product(*(axis.values for axis in definition.axes)):
        active_count = sum(
            axis.variable and value != 0.0
            for axis, value in zip(definition.axes, values)
        )
        if active_count > MAX_ACTIVE_BIAS_TERMS:
            continue

        event_weights = [0.0] * len(EVENT_NAMES)
        for event_name, weight in _FIXED_EVENT_WEIGHTS.items():
            event_weights[EVENT_INDEX[event_name]] = weight

        sparse_reward_weight = 0.0
        axis_values = []
        active_terms = []
        for axis, value in zip(definition.axes, values):
            value = float(value)
            axis_values.append((axis.name, value))
            if axis.variable and value != 0.0:
                active_terms.append(axis.name)
            for target in axis.targets:
                if target == SPARSE_REWARD_TARGET:
                    sparse_reward_weight = value
                else:
                    event_weights[EVENT_INDEX[target]] = value

        candidates.append(
            HSPCandidate(
                profile=profile_name,
                candidate_id=len(candidates),
                event_weights=tuple(event_weights),
                sparse_reward_weight=sparse_reward_weight,
                axis_values=tuple(axis_values),
                active_bias_terms=tuple(active_terms),
            )
        )

    catalog = tuple(candidates)
    expected_count = EXPECTED_CANDIDATE_COUNTS[profile_name]
    if len(catalog) != expected_count:
        raise RuntimeError(
            f"HSP {profile_name} catalog has {len(catalog)} candidates; "
            f"expected {expected_count}"
        )
    return catalog


def resolve_candidate(profile: str, candidate_id: int) -> HSPCandidate:
    profile_name = normalize_profile_name(profile)
    catalog = candidate_catalog(profile_name)
    index = int(candidate_id)
    if index < 0 or index >= len(catalog):
        raise ValueError(
            f"HSP candidate_id {index} is invalid for profile {profile_name!r}; "
            f"valid range is 0..{len(catalog) - 1}"
        )
    return catalog[index]


def hsp_enabled(config: Mapping[str, Any]) -> bool:
    hsp_config = config.get("HSP") or {}
    return bool(hsp_config.get("ENABLED", False))


def resolve_hsp_config(
    config: MutableMapping[str, Any],
) -> HSPCandidate | None:
    """Validate HSP config and attach JSON/W&B-safe resolved metadata."""

    if not hsp_enabled(config):
        return None
    hsp_config = config["HSP"]
    configured_limit = int(
        hsp_config.get("MAX_ACTIVE_BIAS_TERMS", MAX_ACTIVE_BIAS_TERMS)
    )
    if configured_limit != MAX_ACTIVE_BIAS_TERMS:
        raise ValueError(
            "This CooT-compatible catalog requires HSP.MAX_ACTIVE_BIAS_TERMS=3"
        )

    scenario = str(
        hsp_config.get("SCENARIO")
        or config.get("CONDITION")
        or config["ENV_KWARGS"]["layout"]
    )
    requested_profile = str(hsp_config.get("PROFILE", "auto"))
    if requested_profile.strip().lower() == "auto":
        profile = profile_for_scenario(scenario)
    else:
        profile = normalize_profile_name(requested_profile)

    candidate = resolve_candidate(profile, int(hsp_config["CANDIDATE_ID"]))
    hsp_config["RESOLVED_PROFILE"] = profile
    hsp_config["RESOLVED_CANDIDATE_COUNT"] = len(candidate_catalog(profile))
    hsp_config["RESOLVED_UTILITY"] = candidate.metadata()
    hsp_config["PORTING_NOTE"] = PORTING_NOTE
    return candidate


def apply_candidate_rewards(
    sparse_rewards: Mapping[str, Any],
    event_vectors: Mapping[str, Any],
    candidate: HSPCandidate,
    biased_agent_indices: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route hidden utility to the logical biased role and sparse to response.

    Returns the training reward dictionary and the event-only component for
    logging. ``biased_agent_indices`` stores the physical seat (0/1) occupied
    by logical HSP agent 0 in every vectorized environment. When omitted, the
    biased role occupies physical seat 0.
    """

    if "agent_0" not in sparse_rewards or "agent_1" not in sparse_rewards:
        raise ValueError("HSP population training requires agent_0 and agent_1")
    if "agent_0" not in event_vectors:
        raise ValueError("Overcooked V3 info is missing agent_0 event_vector")

    agent_names = ("agent_0", "agent_1")
    weights = jnp.asarray(candidate.event_weights, dtype=jnp.float32)
    stacked_sparse = jnp.stack([sparse_rewards[agent] for agent in agent_names])
    stacked_events = jnp.stack([event_vectors[agent] for agent in agent_names])
    event_utility = jnp.sum(stacked_events * weights, axis=-1)
    candidate_utility = event_utility + stacked_sparse * candidate.sparse_reward_weight
    if biased_agent_indices is None:
        biased_agent_indices = jnp.zeros_like(stacked_sparse[0], dtype=jnp.int32)
    biased_agent_indices = jnp.asarray(biased_agent_indices, dtype=jnp.int32)
    biased_mask = jnp.stack(
        (biased_agent_indices == 0, biased_agent_indices == 1), axis=0
    )
    training_reward_array = jnp.where(biased_mask, candidate_utility, stacked_sparse)
    event_component_array = jnp.where(biased_mask, event_utility, 0.0)

    training_rewards = dict(sparse_rewards)
    event_components = dict(sparse_rewards)
    for agent_index, agent in enumerate(agent_names):
        training_rewards[agent] = training_reward_array[agent_index]
        event_components[agent] = event_component_array[agent_index]
    return training_rewards, event_components


__all__ = [
    "EXPECTED_CANDIDATE_COUNTS",
    "HSPCandidate",
    "MAX_ACTIVE_BIAS_TERMS",
    "PORTING_NOTE",
    "UTILITY_PROFILES",
    "apply_candidate_rewards",
    "candidate_catalog",
    "hsp_enabled",
    "profile_for_scenario",
    "resolve_candidate",
    "resolve_hsp_config",
]
