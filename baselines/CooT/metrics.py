"""Adaptation and partner-switch metrics shared by CooT evaluation."""

from __future__ import annotations

import numpy as np


def rolling_mean(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    result = np.empty_like(values)
    for index in range(values.size):
        result[index] = values[max(0, index - window + 1) : index + 1].mean()
    return result


def adaptation_summary(
    returns,
    *,
    early_window: int = 5,
    late_window: int = 5,
    slope_episodes: int = 15,
    rolling_window: int = 3,
    target_fraction: float = 0.9,
) -> dict[str, float]:
    """Summarize how quickly performance changes across repeated episodes."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("adaptation_summary needs a non-empty 1D return array")
    early_count = min(values.size, max(1, int(early_window)))
    late_count = min(values.size, max(1, int(late_window)))
    initial = float(values[:early_count].mean())
    final = float(values[-late_count:].mean())
    gain = final - initial
    slope_count = min(values.size, max(2, int(slope_episodes)))
    slope = (
        float(np.polyfit(np.arange(slope_count), values[:slope_count], 1)[0])
        if slope_count >= 2
        else 0.0
    )
    auc = float(np.trapezoid(values, dx=1.0) / max(1, values.size - 1))
    target = initial + target_fraction * gain
    smoothed = rolling_mean(values, rolling_window)
    if gain >= 0:
        reached = np.flatnonzero(smoothed >= target)
    else:
        reached = np.flatnonzero(smoothed <= target)
    episodes_to_target = float(reached[0] + 1) if reached.size else float("nan")
    return {
        "initial_return": initial,
        "final_return": final,
        "absolute_gain": gain,
        "relative_gain": gain / max(abs(initial), 1.0),
        "early_slope": slope,
        "return_auc": auc,
        "target_return": target,
        "episodes_to_target": episodes_to_target,
    }


def recovery_episodes(
    returns,
    reference_return: float,
    *,
    target_fraction: float = 1.0,
    rolling_window: int = 1,
) -> float:
    """Episodes needed to recover a fraction of non-switching performance.

    The paper uses ``target_fraction=1`` and a one-episode window. More robust
    settings can be selected for noisy V3 policies without changing the curve.
    """

    values = rolling_mean(returns, rolling_window)
    target = float(reference_return) * float(target_fraction)
    reached = np.flatnonzero(values >= target)
    return float(reached[0] + 1) if reached.size else float("nan")


def jensen_shannon_divergence(left, right, epsilon: float = 1e-8) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = np.clip(left, epsilon, None)
    right = np.clip(right, epsilon, None)
    left /= left.sum()
    right /= right.sum()
    midpoint = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / midpoint))
        + 0.5 * np.sum(right * np.log(right / midpoint))
    )
