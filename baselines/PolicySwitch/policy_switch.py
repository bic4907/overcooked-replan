"""Shared checkpoint and phase helpers for the Overcooked V3 policy switch."""

from pathlib import Path
from typing import Mapping, Sequence

from jaxmarl.environments.overcooked_v3 import (
    POLICY_SWITCH_BASE_LAYOUTS,
    phase_policy_sequence,
)
from jaxmarl.wrappers.baselines import load_params, save_params


ALGORITHM_NAME = "PolicySwitch-IPPO"


def policy_keys_for_layout(layout: str) -> tuple[str, ...]:
    sequence = phase_policy_sequence(layout)
    return tuple(f"policy_{index}" for index in range(max(sequence) + 1))


def policy_key_for_phase(layout: str, phase_index: int) -> str:
    """Return the combined-checkpoint key active in one dynamic phase."""
    sequence = phase_policy_sequence(layout)
    if not 0 <= int(phase_index) < len(sequence):
        raise ValueError(
            f"phase_index must be in [0, {len(sequence) - 1}] for {layout}"
        )
    return f"policy_{sequence[int(phase_index)]}"


def validate_combined_policy_params(
    params: Mapping, layout: str | None = None
) -> Mapping:
    """Validate a contiguous set of phase-policy parameter trees."""
    if not params:
        raise ValueError("Combined phase-policy checkpoint must not be empty")
    if layout is None:
        policy_indices = sorted(
            int(key.removeprefix("policy_"))
            for key in params
            if key.startswith("policy_") and key.removeprefix("policy_").isdigit()
        )
        expected_keys = tuple(f"policy_{index}" for index in range(len(params)))
        if policy_indices != list(range(len(params))):
            expected_keys = ()
    else:
        expected_keys = policy_keys_for_layout(validate_policy_switch_layout(layout))
    missing = [key for key in expected_keys if key not in params]
    extras = [key for key in params if key not in expected_keys]
    if missing or extras:
        raise ValueError(
            "Combined phase-policy checkpoint has invalid policy keys; "
            f"expected={expected_keys}, missing={missing}, extras={extras}"
        )
    return params


def save_combined_policy_params(policies: Sequence, filename) -> Path:
    """Store every independently trained phase policy in one safetensors file."""
    if not policies:
        raise ValueError("At least one phase policy is required")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_params(
        {f"policy_{index}": params for index, params in enumerate(policies)},
        path,
    )
    return path


def load_combined_policy_params(filename, layout: str | None = None):
    """Load a combined phase-policy checkpoint."""
    return validate_combined_policy_params(load_params(filename), layout=layout)


def validate_policy_switch_layout(layout: str) -> str:
    if layout not in POLICY_SWITCH_BASE_LAYOUTS:
        choices = ", ".join(POLICY_SWITCH_BASE_LAYOUTS)
        raise ValueError(f"layout must be one of: {choices}")
    phase_policy_sequence(layout)
    return layout


__all__ = [
    "ALGORITHM_NAME",
    "load_combined_policy_params",
    "policy_key_for_phase",
    "policy_keys_for_layout",
    "save_combined_policy_params",
    "validate_combined_policy_params",
    "validate_policy_switch_layout",
]
