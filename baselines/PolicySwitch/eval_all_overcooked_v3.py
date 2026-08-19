"""Discover PolicySwitch runs and evaluate a full seed-wise SP/XP matrix."""

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

try:
    from baselines.IPPO import eval_wandb_crossplay_matrix_overcooked_v3 as matrix
except ModuleNotFoundError as error:
    if error.name != "baselines":
        raise
    ippo_dir = Path(__file__).resolve().parents[1] / "IPPO"
    sys.path.insert(0, str(ippo_dir))
    import eval_wandb_crossplay_matrix_overcooked_v3 as matrix

try:
    from .eval_overcooked_v3 import (
        evaluate_policy_switch_episode,
        prepare_policy_switch_runtime,
    )
    from .policy_switch import load_combined_policy_params
except ImportError:  # Direct execution: python baselines/PolicySwitch/<script>.py
    from eval_overcooked_v3 import (
        evaluate_policy_switch_episode,
        prepare_policy_switch_runtime,
    )
    from policy_switch import load_combined_policy_params


def evaluate_pair_task(task, runtime_cache=None, params_cache=None):
    """Evaluate one ordered pair of combined phase-policy checkpoints."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    params_cache = {} if params_cache is None else params_cache
    pair_args = SimpleNamespace(
        layout=task["layout"],
        episodes=task["episodes"],
        max_steps=task["max_steps"],
        seed=task["evaluation_seed"],
        stochastic=task["stochastic"],
    )
    run_configs = (task["agent_0_config"], task["agent_1_config"])
    signature = matrix.evaluation_signature(run_configs, pair_args)
    runtime = runtime_cache.get(signature)
    if runtime is None:
        runtime = prepare_policy_switch_runtime(run_configs, pair_args)
        runtime_cache[signature] = runtime

    combined_params = []
    for checkpoint_path in (
        task["agent_0_checkpoint"],
        task["agent_1_checkpoint"],
    ):
        if checkpoint_path not in params_cache:
            params_cache[checkpoint_path] = load_combined_policy_params(
                checkpoint_path,
                layout=task["layout"],
            )
        combined_params.append(params_cache[checkpoint_path])

    key = jax.random.PRNGKey(int(pair_args.seed))
    returns = []
    lengths = []
    for _episode in range(int(pair_args.episodes)):
        episode_return, length, _, _, _, key = evaluate_policy_switch_episode(
            runtime,
            tuple(combined_params),
            key,
            record_trajectory=False,
        )
        returns.append(episode_return)
        lengths.append(length)

    record_keys = (
        "layout",
        "pair_type",
        "agent_0_model_id",
        "agent_1_model_id",
        "agent_0_label",
        "agent_1_label",
        "agent_0_algorithm",
        "agent_1_algorithm",
        "agent_0_seed",
        "agent_1_seed",
        "agent_0_run",
        "agent_1_run",
        "agent_0_vmap",
        "agent_1_vmap",
        "episodes",
        "max_steps",
        "evaluation_seed",
        "stochastic",
    )
    return {
        **{key: task[key] for key in record_keys},
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def evaluate_tasks_in_process(tasks, _gpu_ids, _output_dir, workers_per_gpu=1):
    """Keep the PolicySwitch adapter active when --gpus is supplied."""
    del workers_per_gpu
    runtime_cache = {}
    params_cache = {}
    return (
        [
            evaluate_pair_task(task, runtime_cache, params_cache)
            for task in tasks
        ],
        [],
    )


def main():
    matrix.evaluate_pair_task = evaluate_pair_task
    matrix.run_gpu_workers = evaluate_tasks_in_process
    matrix.main()


if __name__ == "__main__":
    main()
