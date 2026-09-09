from types import SimpleNamespace

import numpy as np

from baselines.adaptation_metrics import (
    AdaptationMetricConfig,
    EpisodeAdaptationTrace,
    adaptation_config_from_args,
    adaptation_result_metrics,
    adaptation_wandb_metrics,
    canonical_phase_mapping,
    compute_transition_metrics,
    summarize_adaptation_traces,
)


def test_auto_parameters_follow_each_layouts_shortest_phase():
    args = SimpleNamespace(
        adaptation_window=None,
        adaptation_horizon=None,
        recovery_threshold=0.9,
        recovery_persistence=5,
    )

    distance = adaptation_config_from_args(args, "distance_switch_0")
    recipe = adaptation_config_from_args(args, "recipe_switch_0")
    split = adaptation_config_from_args(args, "split_0")

    assert (distance.window, distance.horizon) == (30, 150)
    assert (recipe.window, recipe.horizon) == (30, 150)
    assert (split.window, split.horizon) == (30, 150)


def test_role_scenario_final_phase_is_mapped_back_to_a():
    assert canonical_phase_mapping("recipe_switch_0", 3) == (0, 1, 0)
    assert canonical_phase_mapping("distance_switch_0", 3) == (0, 1, 0)


def test_metrics_macro_average_a_to_b_and_b_to_a():
    config = AdaptationMetricConfig(
        window=2,
        horizon=4,
        recovery_threshold=0.75,
        recovery_persistence=1,
    )
    trace = EpisodeAdaptationTrace(
        rewards=np.asarray([0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2]),
        phase_indices=np.asarray([0] * 4 + [1] * 4 + [2] * 4),
    )

    summary = summarize_adaptation_traces(
        [trace],
        config,
        phase_mapping=(0, 1, 0),
    )

    assert summary["adaptation_transition_count"] == 2
    assert summary["adaptation_direction_count"] == 2
    assert np.isclose(summary["adaptation_a_to_b_drop"], 0.55)
    assert np.isclose(summary["adaptation_b_to_a_drop"], 0.075)
    assert np.isclose(summary["adaptation_drop"], 0.3125)
    assert summary["adaptation_drop_valid_rate"] == 1.0
    assert summary["adaptation_a_to_b_immediate_drop"] == 2.0
    assert summary["adaptation_b_to_a_immediate_drop"] == 1.0
    assert summary["adaptation_immediate_drop"] == 1.5
    assert summary["adaptation_a_to_b_recovery_time"] == 4.0
    assert summary["adaptation_b_to_a_recovery_time"] == 3.0
    assert summary["adaptation_recovery_time"] == 3.5
    assert summary["adaptation_recovery_success_rate"] == 1.0
    assert summary["adaptation_a_to_b_auc"] == 1.5
    assert summary["adaptation_b_to_a_auc"] == 3.25
    assert summary["adaptation_auc"] == 2.375
    assert summary["adaptation_auc_normalized"] == 0.475


def test_incomplete_final_transition_is_excluded():
    config = AdaptationMetricConfig(
        window=2,
        horizon=4,
        recovery_threshold=0.9,
        recovery_persistence=1,
    )
    trace = EpisodeAdaptationTrace(
        rewards=np.ones(10),
        phase_indices=np.asarray([0] * 4 + [1] * 4 + [0] * 2),
    )

    transitions = compute_transition_metrics(trace, config)

    assert len(transitions) == 1
    assert transitions[0].boundary == 4


def test_zero_pre_change_rate_keeps_recovery_and_normalized_auc_undefined():
    config = AdaptationMetricConfig(
        window=2,
        horizon=4,
        recovery_threshold=0.9,
        recovery_persistence=1,
    )
    trace = EpisodeAdaptationTrace(
        rewards=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        phase_indices=np.asarray([0] * 4 + [1] * 4),
    )

    transition = compute_transition_metrics(trace, config)[0]

    assert transition.recovered is None
    assert np.isnan(transition.drop)
    assert np.isnan(transition.recovery_time)
    assert np.isnan(transition.adaptation_auc_normalized)


def test_wandb_names_only_include_direction_averaged_metrics():
    metrics = adaptation_wandb_metrics(
        {
            "adaptation_drop": 0.3125,
            "adaptation_drop_valid_rate": 1.0,
            "adaptation_immediate_drop": 1.5,
            "adaptation_recovery_time": 3.5,
            "adaptation_auc": 2.375,
            "adaptation_a_to_b_immediate_drop": 2.0,
            "adaptation_b_to_a_immediate_drop": 1.0,
        }
    )

    assert metrics == {
        "adaptation/drop": 0.3125,
        "adaptation/drop_valid_rate": 1.0,
        "adaptation/legacy_immediate_drop": 1.5,
        "adaptation/recovery_time_steps": 3.5,
        "adaptation/auc": 2.375,
    }


def test_result_table_fields_exclude_direction_details():
    fields = adaptation_result_metrics(
        {
            "adaptation_drop": 0.3125,
            "adaptation_drop_valid_rate": 1.0,
            "adaptation_immediate_drop": 1.5,
            "adaptation_recovery_time": 3.5,
            "adaptation_recovery_success_rate": 1.0,
            "adaptation_auc": 2.375,
            "adaptation_auc_normalized": 0.475,
            "adaptation_transition_count": 2,
            "adaptation_direction_count": 2,
            "adaptation_a_to_b_immediate_drop": 2.0,
            "adaptation_b_to_a_immediate_drop": 1.0,
        }
    )

    assert fields == {
        "adaptation_drop": 0.3125,
        "adaptation_drop_valid_rate": 1.0,
        "adaptation_immediate_drop": 1.5,
        "adaptation_recovery_time": 3.5,
        "adaptation_recovery_success_rate": 1.0,
        "adaptation_auc": 2.375,
        "adaptation_auc_normalized": 0.475,
        "adaptation_transition_count": 2,
        "adaptation_direction_count": 2,
    }
