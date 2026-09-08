"""Adaptation metrics for dynamic Overcooked V3 evaluation rollouts."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ADAPTATION_METRICS_VERSION = 1


@dataclass(frozen=True)
class AdaptationMetricConfig:
    """Window settings shared by every adaptation metric."""

    window: int = 20
    horizon: int = 40
    recovery_threshold: float = 0.9
    recovery_persistence: int = 5
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.window < 1:
            raise ValueError("adaptation window must be at least 1")
        if self.horizon < self.window:
            raise ValueError("adaptation horizon must be at least the window")
        if not 0 < self.recovery_threshold <= 1:
            raise ValueError("recovery threshold must be in (0, 1]")
        available_recovery_checks = self.horizon - self.window + 1
        if not 1 <= self.recovery_persistence <= available_recovery_checks:
            raise ValueError(
                "recovery persistence must fit within the adaptation horizon"
            )


@dataclass(frozen=True)
class EpisodeAdaptationTrace:
    """Team rewards and the phase active when each action was selected."""

    rewards: np.ndarray
    phase_indices: np.ndarray

    def __post_init__(self):
        rewards = np.asarray(self.rewards, dtype=float)
        phase_indices = np.asarray(self.phase_indices, dtype=int)
        if rewards.ndim != 1 or phase_indices.ndim != 1:
            raise ValueError("adaptation trace arrays must be one-dimensional")
        if rewards.shape != phase_indices.shape:
            raise ValueError("rewards and phase_indices must have the same length")
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "phase_indices", phase_indices)


@dataclass(frozen=True)
class TransitionAdaptationMetrics:
    """Metrics for one complete transition evaluation horizon."""

    boundary: int
    from_phase: int
    to_phase: int
    immediate_drop: float
    recovery_time: float
    recovered: bool | None
    adaptation_auc: float
    adaptation_auc_normalized: float


def add_adaptation_metric_args(parser):
    """Add shared adaptation-metric arguments to an evaluation parser."""
    parser.add_argument(
        "--adaptation-window",
        type=int,
        help=(
            "Steps used for pre/post reward rates and recovery smoothing. "
            "Defaults to min(30, half the resolved horizon)."
        ),
    )
    parser.add_argument(
        "--adaptation-horizon",
        type=int,
        help=(
            "Complete post-change horizon required for adaptation metrics. "
            "Defaults to the layout's shortest phase."
        ),
    )
    parser.add_argument(
        "--recovery-threshold",
        type=float,
        default=0.9,
        help="Fraction of the pre-change reward rate required for recovery.",
    )
    parser.add_argument(
        "--recovery-persistence",
        type=int,
        default=5,
        help="Consecutive smoothed steps that must meet the recovery threshold.",
    )
    return parser


def adaptation_config_from_args(
    args,
    layout: str | None = None,
) -> AdaptationMetricConfig:
    """Build a validated metric config from an argparse-like namespace."""
    horizon = getattr(args, "adaptation_horizon", None)
    if horizon is None:
        if layout is None:
            raise ValueError("layout is required for an automatic adaptation horizon")
        from jaxmarl.environments.overcooked_v3 import dynamic_layouts

        horizon = min(int(phase.steps) for phase in dynamic_layouts[layout].phases)

    window = getattr(args, "adaptation_window", None)
    if window is None:
        window = min(30, max(1, int(horizon) // 2))

    return AdaptationMetricConfig(
        window=int(window),
        horizon=int(horizon),
        recovery_threshold=float(getattr(args, "recovery_threshold", 0.9)),
        recovery_persistence=int(getattr(args, "recovery_persistence", 5)),
    )


def adaptation_config_dict(config: AdaptationMetricConfig) -> dict:
    """Return the serializable settings used by task caches and W&B configs."""
    return {
        "adaptation_metrics_version": ADAPTATION_METRICS_VERSION,
        "adaptation_window": config.window,
        "adaptation_horizon": config.horizon,
        "recovery_threshold": config.recovery_threshold,
        "recovery_persistence": config.recovery_persistence,
    }


def canonical_phase_mapping(layout: str, phase_count: int) -> tuple[int, ...]:
    """Map repeated physical phases (for example A-B-A) back to task IDs."""
    from jaxmarl.environments.overcooked_v3 import (
        POLICY_SWITCH_BASE_LAYOUTS,
        phase_policy_sequence,
    )

    if layout in POLICY_SWITCH_BASE_LAYOUTS:
        return tuple(int(index) for index in phase_policy_sequence(layout))
    return tuple(range(phase_count))


def _canonicalize_phases(
    phase_indices: np.ndarray,
    phase_mapping: Sequence[int] | None,
) -> np.ndarray:
    if phase_mapping is None:
        return phase_indices
    mapping = np.asarray(phase_mapping, dtype=int)
    if phase_indices.size and (
        np.min(phase_indices) < 0 or np.max(phase_indices) >= mapping.size
    ):
        raise ValueError("phase_mapping does not cover every trace phase index")
    return mapping[phase_indices]


def compute_transition_metrics(
    trace: EpisodeAdaptationTrace,
    config: AdaptationMetricConfig,
    phase_mapping: Sequence[int] | None = None,
) -> list[TransitionAdaptationMetrics]:
    """Compute metrics for every transition with a complete pre/post window."""
    rewards = trace.rewards
    phases = _canonicalize_phases(trace.phase_indices, phase_mapping)
    boundaries = np.flatnonzero(phases[1:] != phases[:-1]) + 1
    results = []

    for boundary_index, boundary in enumerate(boundaries):
        next_boundary = (
            int(boundaries[boundary_index + 1])
            if boundary_index + 1 < len(boundaries)
            else len(rewards)
        )
        boundary = int(boundary)
        if boundary < config.window:
            continue
        if boundary + config.horizon > next_boundary:
            continue

        pre_rewards = rewards[boundary - config.window : boundary]
        post_rewards = rewards[boundary : boundary + config.horizon]
        pre_rate = float(np.mean(pre_rewards))
        immediate_rate = float(np.mean(post_rewards[: config.window]))
        immediate_drop = pre_rate - immediate_rate

        cumulative_rewards = np.cumsum(post_rewards)
        adaptation_auc = float(np.mean(cumulative_rewards))
        ideal_auc = pre_rate * float(np.mean(np.arange(1, config.horizon + 1)))
        adaptation_auc_normalized = (
            adaptation_auc / ideal_auc if ideal_auc > config.epsilon else float("nan")
        )

        recovery_time = float("nan")
        recovered = None
        if pre_rate > config.epsilon:
            target = config.recovery_threshold * pre_rate
            qualifying_run = 0
            recovered = False
            recovery_time = float(config.horizon)
            for elapsed in range(config.window, config.horizon + 1):
                rolling_rate = float(
                    np.mean(post_rewards[elapsed - config.window : elapsed])
                )
                if rolling_rate >= target:
                    qualifying_run += 1
                    if qualifying_run >= config.recovery_persistence:
                        recovery_time = float(elapsed - config.recovery_persistence + 1)
                        recovered = True
                        break
                else:
                    qualifying_run = 0

        results.append(
            TransitionAdaptationMetrics(
                boundary=boundary,
                from_phase=int(phases[boundary - 1]),
                to_phase=int(phases[boundary]),
                immediate_drop=immediate_drop,
                recovery_time=recovery_time,
                recovered=recovered,
                adaptation_auc=adaptation_auc,
                adaptation_auc_normalized=adaptation_auc_normalized,
            )
        )

    return results


def _direction_name(from_phase: int, to_phase: int) -> str:
    if (from_phase, to_phase) == (0, 1):
        return "a_to_b"
    if (from_phase, to_phase) == (1, 0):
        return "b_to_a"
    return f"phase_{from_phase}_to_{to_phase}"


def _finite_mean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def summarize_transition_metrics(
    transitions: Sequence[TransitionAdaptationMetrics],
) -> dict:
    """Macro-average direction means so A->B and B->A have equal weight."""
    grouped = {}
    for transition in transitions:
        grouped.setdefault(
            _direction_name(transition.from_phase, transition.to_phase), []
        ).append(transition)

    summary = {
        "adaptation_transition_count": int(len(transitions)),
        "adaptation_direction_count": int(len(grouped)),
    }
    direction_values = {
        "immediate_drop": [],
        "recovery_time": [],
        "recovery_success_rate": [],
        "auc": [],
        "auc_normalized": [],
    }

    for direction, direction_transitions in grouped.items():
        recovered_values = [
            float(transition.recovered)
            for transition in direction_transitions
            if transition.recovered is not None
        ]
        values = {
            "immediate_drop": _finite_mean(
                transition.immediate_drop for transition in direction_transitions
            ),
            "recovery_time": _finite_mean(
                transition.recovery_time for transition in direction_transitions
            ),
            "recovery_success_rate": _finite_mean(recovered_values),
            "auc": _finite_mean(
                transition.adaptation_auc for transition in direction_transitions
            ),
            "auc_normalized": _finite_mean(
                transition.adaptation_auc_normalized
                for transition in direction_transitions
            ),
        }
        summary[f"adaptation_{direction}_transition_count"] = int(
            len(direction_transitions)
        )
        for metric, value in values.items():
            summary[f"adaptation_{direction}_{metric}"] = value
            if np.isfinite(value):
                direction_values[metric].append(value)

    for metric, values in direction_values.items():
        summary[f"adaptation_{metric}"] = _finite_mean(values)
    return summary


def summarize_adaptation_traces(
    traces: Sequence[EpisodeAdaptationTrace],
    config: AdaptationMetricConfig,
    phase_mapping: Sequence[int] | None = None,
) -> dict:
    """Compute and summarize all eligible transitions in multiple episodes."""
    transitions = [
        transition
        for trace in traces
        for transition in compute_transition_metrics(trace, config, phase_mapping)
    ]
    return summarize_transition_metrics(transitions)


_WANDB_METRIC_NAMES = {
    "immediate_drop": "immediate_drop",
    "recovery_time": "recovery_time_steps",
    "recovery_success_rate": "recovery_success_rate",
    "auc": "auc",
    "auc_normalized": "auc_normalized",
    "transition_count": "transition_count",
    "direction_count": "direction_count",
}


def adaptation_wandb_metrics(summary: dict, prefix: str = "adaptation") -> dict:
    """Convert flat result fields to stable hierarchical W&B metric names."""
    metrics = {}
    for metric, wandb_name in _WANDB_METRIC_NAMES.items():
        key = f"adaptation_{metric}"
        if key in summary:
            metrics[f"{prefix}/{wandb_name}"] = summary[key]
    return metrics


ADAPTATION_PAIR_METRIC_KEYS = tuple(
    f"adaptation_{suffix}"
    for suffix in (
        "immediate_drop",
        "recovery_time",
        "recovery_success_rate",
        "auc",
        "auc_normalized",
        "transition_count",
        "direction_count",
    )
)


def adaptation_result_metrics(summary: dict) -> dict:
    """Select aggregate fields exposed in pair result tables and artifacts."""
    return {key: summary[key] for key in ADAPTATION_PAIR_METRIC_KEYS if key in summary}
