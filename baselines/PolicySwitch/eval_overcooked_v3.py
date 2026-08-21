"""Evaluate combined phase policies on a dynamic Overcooked V3 layout."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import jaxmarl
from jaxmarl._env import load_project_env
from jaxmarl.environments.overcooked_v3 import POLICY_SWITCH_BASE_LAYOUTS
from jaxmarl.environments.overcooked_v3.common import OvercookedActionsEnum
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

try:
    from baselines.IPPO.eval_wandb_crossplay_overcooked_v3 import (
        _artifact_reference,
        _observation_config,
        _policy_config,
        qualify_run_path,
        resolve_vmap_checkpoint,
        select_final_artifact,
    )
    from baselines.IPPO.ippo_overcooked_v3 import (
        ActorCriticCNN,
        ActorCriticRNN,
        ScannedRNN,
    )
except ModuleNotFoundError as error:
    if error.name != "baselines":
        raise
    ippo_dir = Path(__file__).resolve().parents[1] / "IPPO"
    sys.path.insert(0, str(ippo_dir))
    from eval_wandb_crossplay_overcooked_v3 import (
        _artifact_reference,
        _observation_config,
        _policy_config,
        qualify_run_path,
        resolve_vmap_checkpoint,
        select_final_artifact,
    )
    from ippo_overcooked_v3 import ActorCriticCNN, ActorCriticRNN, ScannedRNN

try:
    from .policy_switch import (
        load_combined_policy_params,
        policy_key_for_phase,
        validate_policy_switch_layout,
    )
except ImportError:  # Direct execution: python baselines/PolicySwitch/<script>.py
    from policy_switch import (
        load_combined_policy_params,
        policy_key_for_phase,
        validate_policy_switch_layout,
    )


@dataclass
class PolicySwitchRuntime:
    layout: str
    env: object
    env_step: object
    policy: object
    hidden_sizes: tuple[int, int]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or two combined phase-policy checkpoints while "
            "switching policies with an Overcooked V3 dynamic layout."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        help="One checkpoint for SP or two checkpoints for XP.",
    )
    source.add_argument(
        "--run-ids",
        nargs="+",
        help="One W&B training run for SP or two runs for XP.",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=POLICY_SWITCH_BASE_LAYOUTS,
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--architecture", choices=("cnn", "rnn"), default="cnn")
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--fc-dim-size", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked")
    )
    parser.add_argument(
        "--source-project",
        default=os.getenv("WANDB_SOURCE_PROJECT", "overcooked-v3-role-coordination"),
    )
    parser.add_argument("--project", default="overcooked-v3-crossplay")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--artifact-alias", default="final")
    parser.add_argument(
        "--agent-vmap-indices",
        nargs="+",
        type=int,
        default=(0,),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("evaluation/overcooked_v3/policy_switch_artifacts"),
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-quality", type=int, choices=range(0, 11), default=5)
    return parser.parse_args(argv)


def _normalize_pair(values, label):
    values = tuple(values)
    if len(values) == 1:
        return values * 2
    if len(values) == 2:
        return values
    raise ValueError(f"{label} accepts one value for SP or two values for XP")


def prepare_policy_switch_runtime(run_configs, args):
    policy_configs = tuple(_policy_config(config) for config in run_configs)
    observation_configs = tuple(_observation_config(config) for config in run_configs)
    if observation_configs[0] != observation_configs[1]:
        raise ValueError(
            "The two checkpoints use incompatible observation configurations: "
            f"{observation_configs}"
        )

    layout = validate_policy_switch_layout(args.layout)
    env = jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=int(args.max_steps),
        random_agent_positions=False,
        **observation_configs[0],
    )
    networks = tuple(
        (ActorCriticRNN if config["ARCHITECTURE"] == "rnn" else ActorCriticCNN)(
            env.action_space(agent).n, config=config
        )
        for agent, config in zip(env.agents, policy_configs)
    )

    def select_action(params, hidden, obs, dones, action_key):
        action_keys = jax.random.split(action_key, env.num_agents)
        next_hidden = []
        actions = []
        for agent_index in range(env.num_agents):
            agent_hidden, distribution, _ = networks[agent_index].apply(
                params[agent_index],
                hidden[agent_index],
                (
                    obs[:, agent_index : agent_index + 1],
                    dones[:, agent_index : agent_index + 1],
                ),
            )
            action = (
                distribution.sample(seed=action_keys[agent_index])
                if args.stochastic
                else distribution.mode()
            )
            next_hidden.append(agent_hidden)
            actions.append(action)
        return tuple(next_hidden), jnp.concatenate(actions, axis=1)

    return PolicySwitchRuntime(
        layout=layout,
        env=env,
        env_step=jax.jit(env.step_env),
        policy=jax.jit(select_action),
        hidden_sizes=tuple(config["GRU_HIDDEN_DIM"] for config in policy_configs),
    )


def _initial_hidden(hidden_sizes):
    return tuple(
        ScannedRNN.initialize_carry(1, hidden_size) for hidden_size in hidden_sizes
    )


def evaluate_policy_switch_episode(
    runtime,
    combined_params,
    key,
    record_trajectory=True,
):
    key, reset_key = jax.random.split(key)
    obs, state = runtime.env.reset(reset_key)
    hidden = _initial_hidden(runtime.hidden_sizes)
    last_done = jnp.zeros((runtime.env.num_agents,), dtype=jnp.bool_)
    previous_policy_key = None

    states = [state] if record_trajectory else None
    captions = ["step=0 score=0 policy=0 actions=-/-"] if record_trajectory else None
    policy_trace = []
    episode_return = 0.0

    for step in range(runtime.env.max_steps):
        phase_index = int(state.layout_index)
        active_policy_key = policy_key_for_phase(runtime.layout, phase_index)
        if active_policy_key != previous_policy_key:
            hidden = _initial_hidden(runtime.hidden_sizes)
            previous_policy_key = active_policy_key
        policy_trace.append(active_policy_key)
        active_params = tuple(
            agent_params[active_policy_key] for agent_params in combined_params
        )

        key, action_key, step_key = jax.random.split(key, 3)
        obs_batch = jnp.stack([obs[agent] for agent in runtime.env.agents])
        hidden, action = runtime.policy(
            active_params,
            hidden,
            obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
            action_key,
        )
        action = action.squeeze(0)
        actions = {
            agent: action[index] for index, agent in enumerate(runtime.env.agents)
        }
        obs, state, reward, done, _ = runtime.env_step(step_key, state, actions)
        episode_return += float(reward[runtime.env.agents[0]])
        if record_trajectory:
            states.append(state)
            action_names = [
                OvercookedActionsEnum(int(action[index])).name
                for index in range(runtime.env.num_agents)
            ]
            captions.append(
                f"step={step + 1} score={episode_return:g} "
                f"policy={active_policy_key.removeprefix('policy_')} "
                f"actions={'/'.join(action_names)}"
            )
        last_done = jnp.asarray([done[agent] for agent in runtime.env.agents])
        if bool(done["__all__"]):
            return (
                episode_return,
                step + 1,
                states,
                captions,
                tuple(policy_trace),
                key,
            )

    return (
        episode_return,
        runtime.env.max_steps,
        states,
        captions,
        tuple(policy_trace),
        key,
    )


def evaluate_policy_switch(checkpoints, run_configs, args, record_trajectory=True):
    runtime = prepare_policy_switch_runtime(run_configs, args)
    combined_params = tuple(
        load_combined_policy_params(checkpoint, layout=args.layout)
        for checkpoint in checkpoints
    )
    key = jax.random.PRNGKey(int(args.seed))
    returns = []
    lengths = []
    first_states = None
    first_captions = None
    first_policy_trace = None
    for episode in range(int(args.episodes)):
        result = evaluate_policy_switch_episode(
            runtime,
            combined_params,
            key,
            record_trajectory=record_trajectory,
        )
        episode_return, length, states, captions, policy_trace, key = result
        returns.append(episode_return)
        lengths.append(length)
        if first_policy_trace is None:
            first_states = states
            first_captions = captions
            first_policy_trace = policy_trace
        print(
            f"episode={episode + 1} return={episode_return:.2f} length={length}",
            flush=True,
        )
    return {
        "layout": runtime.layout,
        "returns": np.asarray(returns),
        "lengths": np.asarray(lengths),
        "states": first_states,
        "captions": first_captions,
        "policy_trace": first_policy_trace,
        "env": runtime.env,
    }


def _local_run_config(args):
    return {
        "ALGORITHM": "PolicySwitch-IPPO",
        "ARCHITECTURE": args.architecture,
        "ACTIVATION": args.activation,
        "FC_DIM_SIZE": args.fc_dim_size,
        "GRU_HIDDEN_DIM": args.gru_hidden_dim,
        "ENV_KWARGS": {
            "layout": args.layout,
            "include_transition_countdown": True,
            "include_layout_change_mask": True,
            "transition_warning_steps": 20,
        },
    }


def _resolve_sources(args, evaluation_run):
    if args.checkpoints:
        checkpoints = _normalize_pair(args.checkpoints, "--checkpoints")
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        local_config = _local_run_config(args)
        return checkpoints, (local_config, local_config), ("local", "local")

    run_ids = _normalize_pair(args.run_ids, "--run-ids")
    vmap_indices = _normalize_pair(args.agent_vmap_indices, "--agent-vmap-indices")
    run_paths = tuple(
        qualify_run_path(run_id, args.entity, args.source_project) for run_id in run_ids
    )
    api = wandb.Api()
    source_runs = tuple(api.run(run_path) for run_path in run_paths)
    artifacts = tuple(
        select_final_artifact(source_run, args.artifact_alias)
        for source_run in source_runs
    )
    checkpoints = []
    for source_run, artifact, vmap_index in zip(source_runs, artifacts, vmap_indices):
        artifact_ref = _artifact_reference(artifact, source_run)
        used_artifact = evaluation_run.use_artifact(artifact_ref)
        download_root = (
            args.artifact_dir / source_run.id / artifact.name.replace(":", "-")
        )
        artifact_dir = Path(used_artifact.download(root=str(download_root)))
        checkpoints.append(resolve_vmap_checkpoint(artifact_dir, vmap_index))
    return tuple(checkpoints), tuple(dict(run.config) for run in source_runs), run_paths


def main(argv=None):
    load_project_env()
    args = parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be at least 1")
    validate_policy_switch_layout(args.layout)

    run_name = f"policy_switch_{args.layout}_seed{args.seed}"
    with wandb.init(
        entity=args.entity,
        project=args.project,
        mode=args.wandb_mode,
        name=run_name,
        group=args.layout,
        job_type="policy-switch-evaluation",
        config={
            "layout": args.layout,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "stochastic": args.stochastic,
        },
    ) as evaluation_run:
        checkpoints, run_configs, run_paths = _resolve_sources(args, evaluation_run)
        result = evaluate_policy_switch(checkpoints, run_configs, args)
        returns = result["returns"]
        lengths = result["lengths"]
        summary = {
            "eval/mean_return": float(np.mean(returns)),
            "eval/std_return": float(np.std(returns)),
            "eval/min_return": float(np.min(returns)),
            "eval/max_return": float(np.max(returns)),
            "eval/mean_episode_length": float(np.mean(lengths)),
            "debug/policy_switch_count": sum(
                left != right
                for left, right in zip(
                    result["policy_trace"], result["policy_trace"][1:]
                )
            ),
        }
        wandb.log(summary)
        evaluation_run.summary.update(summary)
        evaluation_run.summary["source/agent_0"] = str(run_paths[0])
        evaluation_run.summary["source/agent_1"] = str(run_paths[1])

        video_path = args.video or Path(
            f"evaluation/overcooked_v3/policy_switch/{run_name}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        env = result["env"]
        visualizer = OvercookedV3Visualizer(
            tile_size=24,
            seconds_per_step=1.0 / args.video_fps,
            transition_warning_steps=env.transition_warning_steps,
        )
        visualizer.save_video(
            result["states"],
            filename=str(video_path),
            agent_view_size=env.agent_view_size,
            captions=result["captions"],
            fps=args.video_fps,
            quality=args.video_quality,
        )
        wandb.log(
            {
                "visualization/policy_switch_episode": wandb.Video(
                    str(video_path),
                    format="mp4",
                    caption=(f"{run_paths[0]} x {run_paths[1]} on {args.layout}"),
                )
            }
        )
        print(f"Saved video: {video_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
