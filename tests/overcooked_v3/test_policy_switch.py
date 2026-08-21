from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from baselines.IPPO.ippo_overcooked_v3 import ActorCriticCNN, ScannedRNN
from baselines.PolicySwitch.eval_overcooked_v3 import (
    evaluate_policy_switch_episode,
    prepare_policy_switch_runtime,
)
from baselines.PolicySwitch.policy_switch import (
    load_combined_policy_params,
    policy_key_for_phase,
    save_combined_policy_params,
)
from baselines.PolicySwitch.train_overcooked_v3 import _split_phase_gated_params
from jaxmarl.environments.overcooked_v3 import (
    POLICY_SWITCH_BASE_LAYOUTS,
    dynamic_layouts,
    phase_policy_layout_name,
    phase_policy_sequence,
)


ROOT = Path(__file__).resolve().parents[2]


def _run_config(layout):
    return {
        "ALGORITHM": "PolicySwitch-IPPO",
        "ARCHITECTURE": "cnn",
        "ACTIVATION": "relu",
        "FC_DIM_SIZE": 16,
        "GRU_HIDDEN_DIM": 16,
        "ENV_KWARGS": {
            "layout": layout,
            "include_transition_countdown": True,
            "include_layout_change_mask": True,
            "transition_warning_steps": 20,
        },
    }


def _initialized_params(runtime, config):
    key, reset_key, init_key = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, _ = runtime.env.reset(reset_key)
    obs_batch = jnp.stack([obs[agent] for agent in runtime.env.agents])
    network = ActorCriticCNN(
        runtime.env.action_space(runtime.env.agents[0]).n,
        config=config,
    )
    hidden = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    params = network.init(
        init_key,
        hidden,
        (
            obs_batch[jnp.newaxis, 0:1],
            jnp.zeros((1, 1), dtype=jnp.bool_),
        ),
    )
    return key, params


def test_static_policy_layouts_disable_all_dynamic_transitions():
    for base_layout in POLICY_SWITCH_BASE_LAYOUTS:
        base = dynamic_layouts[base_layout]
        sequence = phase_policy_sequence(base_layout)
        for policy_index in range(max(sequence) + 1):
            source_phase = base.phases[sequence.index(policy_index)]
            static = dynamic_layouts[
                phase_policy_layout_name(base_layout, policy_index)
            ]
            assert len(static.phases) == 1
            assert static.phases[0].steps > 450
            assert static.phases[0].recipe == source_phase.recipe
            assert static.phases[0].agent_positions == source_phase.agent_positions
            np.testing.assert_array_equal(
                static.phases[0].layout.static_objects,
                source_phase.layout.static_objects,
            )

    assert phase_policy_sequence("split_0") == (0, 1)
    assert phase_policy_sequence("outage_0") == (0, 1)
    assert phase_policy_sequence("recipe_switch_0") == (0, 1, 0)


def test_policy_switch_sweeps_target_all_role_scenarios_with_six_seeds():
    training = yaml.safe_load(
        (ROOT / "experiment/policy_switch/train.yaml").read_text(encoding="utf-8")
    )
    evaluation = yaml.safe_load(
        (ROOT / "experiment/policy_switch/eval.yaml").read_text(encoding="utf-8")
    )
    expected_layouts = [
        "split_0",
        "split_1",
        "split_2",
        "outage_0",
        "outage_1",
        "outage_2",
        "recipe_switch_0",
        "recipe_switch_1",
        "recipe_switch_2",
        "distance_switch_0",
        "distance_switch_1",
        "distance_switch_2",
    ]

    assert training["parameters"]["scenario"]["values"] == expected_layouts
    assert training["parameters"]["SEED"]["values"] == [0, 1, 2, 3, 4, 5]
    assert (
        training["parameters"]["PROJECT"]["value"]
        == "overcooked-v3-policyswitch_train"
    )
    assert training["metric"]["name"] == "train/episode_return"
    assert evaluation["parameters"]["layout"]["values"] == expected_layouts
    assert evaluation["parameters"]["max-steps"]["value"] == 450
    assert (
        evaluation["parameters"]["output-project"]["value"]
        == "cilab-overcooked/overcooked-v3-policyswitch_eval"
    )
    assert (
        "cilab-overcooked/overcooked-v3-policyswitch_train"
        in evaluation["command"]
    )


def test_phase_gated_params_split_into_existing_checkpoint_format():
    gated = {
        "params": {
            "policy_0": {"weight": jnp.asarray([1.0, 2.0])},
            "policy_1": {"weight": jnp.asarray([3.0, 4.0])},
        }
    }

    policies = _split_phase_gated_params(gated, policy_count=2)

    assert len(policies) == 2
    np.testing.assert_array_equal(policies[0]["params"]["weight"], [1, 2])
    np.testing.assert_array_equal(policies[1]["params"]["weight"], [3, 4])


def test_combined_safetensors_round_trip_with_arbitrary_policy_count(tmp_path):
    checkpoint = tmp_path / "combined_vmap0.safetensors"
    policies = [
        {"params": {"weight": jnp.asarray([1.0, 2.0])}},
        {"params": {"weight": jnp.asarray([3.0, 4.0])}},
        {"params": {"weight": jnp.asarray([5.0, 6.0])}},
    ]
    save_combined_policy_params(policies, checkpoint)

    loaded = load_combined_policy_params(checkpoint)
    np.testing.assert_array_equal(loaded["policy_0"]["params"]["weight"], [1, 2])
    np.testing.assert_array_equal(loaded["policy_1"]["params"]["weight"], [3, 4])
    np.testing.assert_array_equal(loaded["policy_2"]["params"]["weight"], [5, 6])


def test_eval_switches_policy_when_non_recipe_map_changes():
    layout = "split_0"
    config = _run_config(layout)
    args = SimpleNamespace(
        layout=layout,
        max_steps=205,
        stochastic=False,
    )
    runtime = prepare_policy_switch_runtime((config, config), args)
    key, params = _initialized_params(runtime, config)
    combined_params = (
        {"policy_0": params, "policy_1": params},
        {"policy_0": params, "policy_1": params},
    )

    _, length, _, _, policy_trace, _ = evaluate_policy_switch_episode(
        runtime,
        combined_params,
        key,
        record_trajectory=False,
    )

    assert length == 205
    assert policy_trace[:40] == ("policy_0",) * 40
    assert policy_trace[40:200] == ("policy_1",) * 160
    assert policy_trace[200:] == ("policy_0",) * 5
    assert policy_key_for_phase(layout, 0) == "policy_0"
    assert policy_key_for_phase(layout, 1) == "policy_1"
