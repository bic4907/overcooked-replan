"""Cross-play two Overcooked V3 policies downloaded from W&B run artifacts."""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import jaxmarl
from jaxmarl._env import load_project_env
from jaxmarl.environments.overcooked_v3 import overcooked_v3_layouts
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from jaxmarl.wrappers.baselines import load_params

try:
    from .eval_ippo_overcooked_v3 import evaluate_episode
    from .ippo_overcooked_v3 import ActorCriticCNN, ActorCriticRNN, ScannedRNN
except ImportError:  # Direct execution: python baselines/IPPO/<script>.py
    from eval_ippo_overcooked_v3 import evaluate_episode
    from ippo_overcooked_v3 import ActorCriticCNN, ActorCriticRNN, ScannedRNN


ARTIFACT_TYPE = "checkpoint"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download final checkpoints from two W&B training runs, evaluate "
            "cross-play, and log the result as a separate W&B run."
        )
    )
    parser.add_argument(
        "--run-ids",
        nargs=2,
        required=True,
        metavar=("AGENT_0_RUN_ID", "AGENT_1_RUN_ID"),
        help=(
            "Two W&B run IDs. A value may be a bare run ID or a full "
            "entity/project/run_id path."
        ),
    )
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY"),
        help="Source and evaluation W&B entity for bare run IDs.",
    )
    parser.add_argument(
        "--source-project",
        default=os.getenv("WANDB_PROJECT", "overcooked-v3-role-coordination"),
        help="Training project containing bare --run-ids.",
    )
    parser.add_argument(
        "--project",
        default="overcooked-v3-crossplay",
        help="Project used for cross-play evaluation runs.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    parser.add_argument("--layout", choices=sorted(overcooked_v3_layouts))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using their modes.",
    )
    parser.add_argument(
        "--artifact-alias",
        default="final",
        help="Required alias on the checkpoint artifact (default: final).",
    )
    parser.add_argument(
        "--agent-vmap-indices",
        nargs=2,
        type=int,
        default=(0, 0),
        metavar=("AGENT_0_VMAP", "AGENT_1_VMAP"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("evaluation/overcooked_v3/wandb_artifacts"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Output MP4 path. Defaults under evaluation/overcooked_v3/crossplay.",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-quality", type=int, choices=range(0, 11), default=5)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help=(
            "Optional machine-readable summary path used by the live cross-play "
            "matrix viewer."
        ),
    )
    return parser.parse_args()


def qualify_run_path(run_id, entity, project):
    """Return entity/project/run_id while accepting already-qualified IDs."""
    components = run_id.strip("/").split("/")
    if len(components) == 3:
        return "/".join(components)
    if len(components) != 1:
        raise ValueError(
            f"Invalid W&B run ID '{run_id}'; expected run_id or entity/project/run_id"
        )
    if not entity or not project:
        raise ValueError("Bare W&B run IDs require --entity and --source-project")
    return f"{entity}/{project}/{components[0]}"


def select_final_artifact(source_run, alias="final"):
    """Select this run's checkpoint artifact carrying the requested alias."""
    checkpoints = [
        artifact
        for artifact in source_run.logged_artifacts()
        if getattr(artifact, "type", None) == ARTIFACT_TYPE
    ]
    matching = [
        artifact
        for artifact in checkpoints
        if alias in set(getattr(artifact, "aliases", []) or [])
    ]
    if not matching:
        available = [getattr(artifact, "name", "<unnamed>") for artifact in checkpoints]
        raise FileNotFoundError(
            f"Run {source_run.path} has no '{ARTIFACT_TYPE}' artifact with alias "
            f"'{alias}'. Available checkpoint artifacts: {available}. Train with "
            "upload_final_checkpoint=true."
        )
    if len(matching) > 1:
        names = [getattr(artifact, "name", "<unnamed>") for artifact in matching]
        raise RuntimeError(
            f"Run {source_run.path} has multiple checkpoint artifacts aliased "
            f"'{alias}': {names}"
        )
    return matching[0]


def resolve_vmap_checkpoint(artifact_dir, vmap_index):
    """Find one final safetensors checkpoint for a vmap index."""
    artifact_dir = Path(artifact_dir)
    pattern = re.compile(rf"_vmap{vmap_index}\.safetensors$")
    candidates = [
        path
        for path in artifact_dir.rglob("*.safetensors")
        if "_update" not in path.stem and pattern.search(path.name)
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one final vmap{vmap_index} checkpoint in {artifact_dir}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _artifact_reference(artifact, source_run):
    qualified_name = getattr(artifact, "qualified_name", None)
    if qualified_name:
        return qualified_name
    return f"{source_run.entity}/{source_run.project}/{artifact.name}"


def _policy_config(run_config):
    return {
        "ARCHITECTURE": str(run_config.get("ARCHITECTURE", "cnn")).lower(),
        "ACTIVATION": run_config.get("ACTIVATION", "relu"),
        "FC_DIM_SIZE": int(run_config.get("FC_DIM_SIZE", 128)),
        "GRU_HIDDEN_DIM": int(run_config.get("GRU_HIDDEN_DIM", 128)),
    }


def _observation_config(run_config):
    env_kwargs = dict(run_config.get("ENV_KWARGS") or {})
    return {
        "include_transition_countdown": bool(
            env_kwargs.get("include_transition_countdown", True)
        ),
        "include_layout_change_mask": bool(
            env_kwargs.get("include_layout_change_mask", True)
        ),
        "include_signal_status": bool(env_kwargs.get("include_signal_status", True)),
        "transition_warning_steps": int(env_kwargs.get("transition_warning_steps", 20)),
        "signal_activation_time": int(env_kwargs.get("signal_activation_time", 10)),
        "signal_activation_cost": float(env_kwargs.get("signal_activation_cost", 0.1)),
    }


def _source_layout(run_config):
    return (run_config.get("ENV_KWARGS") or {}).get("layout")


def _target_layout(run_configs, requested_layout):
    source_layouts = tuple(_source_layout(config) for config in run_configs)
    if requested_layout is not None:
        return requested_layout
    if source_layouts[0] != source_layouts[1] or source_layouts[0] is None:
        raise ValueError(
            "--layout is required when source runs were trained on different layouts"
        )
    return source_layouts[0]


def evaluation_signature(run_configs, args):
    """Return a hashable signature for sharing one compiled evaluation runtime."""
    return (
        _target_layout(run_configs, args.layout),
        int(args.max_steps),
        bool(args.stochastic),
        tuple(tuple(sorted(_policy_config(config).items())) for config in run_configs),
        tuple(
            tuple(sorted(_observation_config(config).items())) for config in run_configs
        ),
    )


@dataclass
class CrossplayRuntime:
    layout: str
    env: object
    env_step: object
    policy: object
    hidden_sizes: tuple[int, int]


def prepare_crossplay_runtime(run_configs, args):
    """Build and JIT the environment/policies shared by compatible model pairs."""
    policy_configs = tuple(_policy_config(config) for config in run_configs)
    observation_configs = tuple(_observation_config(config) for config in run_configs)
    if observation_configs[0] != observation_configs[1]:
        raise ValueError(
            "The two runs use incompatible observation configurations: "
            f"{observation_configs}"
        )

    layout = _target_layout(run_configs, args.layout)
    env = jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=args.max_steps,
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
            agent_hidden, pi, _ = networks[agent_index].apply(
                params[agent_index],
                hidden[agent_index],
                (
                    obs[:, agent_index : agent_index + 1],
                    dones[:, agent_index : agent_index + 1],
                ),
            )
            action = (
                pi.sample(seed=action_keys[agent_index])
                if args.stochastic
                else pi.mode()
            )
            next_hidden.append(agent_hidden)
            actions.append(action)
        return tuple(next_hidden), jnp.concatenate(actions, axis=1)

    return CrossplayRuntime(
        layout=layout,
        env=env,
        env_step=jax.jit(env.step_env),
        policy=jax.jit(select_action),
        hidden_sizes=tuple(config["GRU_HIDDEN_DIM"] for config in policy_configs),
    )


def write_metrics_json(path, run_paths, run_ids, layout, summary):
    """Atomically write one ordered-pair result for the matrix viewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_0_run": run_paths[0],
        "agent_1_run": run_paths[1],
        "agent_0_id": run_ids[0],
        "agent_1_id": run_ids[1],
        "layout": layout,
        **summary,
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def evaluate_crossplay(
    checkpoints,
    run_configs,
    args,
    runtime=None,
    params=None,
    episode_callback=None,
    record_trajectory=True,
):
    runtime = runtime or prepare_crossplay_runtime(run_configs, args)
    if params is None:
        params = tuple(load_params(checkpoint) for checkpoint in checkpoints)
    key = jax.random.PRNGKey(args.seed)
    returns = []
    lengths = []
    first_states = None
    first_captions = None

    for episode in range(args.episodes):
        episode_return, length, states, captions, key = evaluate_episode(
            runtime.policy,
            params,
            runtime.env_step,
            runtime.env,
            key,
            runtime.hidden_sizes,
            record_trajectory=record_trajectory,
        )
        returns.append(episode_return)
        lengths.append(length)
        if record_trajectory and first_states is None:
            first_states = states
            first_captions = captions
        if episode_callback is not None:
            episode_callback(
                {
                    "eval/episode": episode + 1,
                    "eval/episode_return": episode_return,
                    "eval/episode_length": length,
                }
            )
        if record_trajectory:
            print(f"episode={episode + 1} return={episode_return:.2f} length={length}")

    return {
        "layout": runtime.layout,
        "returns": np.asarray(returns),
        "lengths": np.asarray(lengths),
        "states": first_states,
        "captions": first_captions,
        "env": runtime.env,
    }


def main():
    load_project_env()
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be at least 1")

    run_paths = tuple(
        qualify_run_path(run_id, args.entity, args.source_project)
        for run_id in args.run_ids
    )
    api = wandb.Api()
    source_runs = tuple(api.run(run_path) for run_path in run_paths)
    artifacts = tuple(
        select_final_artifact(source_run, args.artifact_alias)
        for source_run in source_runs
    )
    run_configs = tuple(dict(source_run.config) for source_run in source_runs)
    source_layouts = tuple(_source_layout(config) for config in run_configs)
    target_layout = args.layout or (
        source_layouts[0] if source_layouts[0] == source_layouts[1] else "cross-layout"
    )
    short_ids = tuple(source_run.id for source_run in source_runs)
    run_name = f"{target_layout}_{short_ids[0]}-x-{short_ids[1]}"

    with wandb.init(
        entity=args.entity,
        project=args.project,
        mode=args.wandb_mode,
        name=run_name,
        group=target_layout,
        job_type="cross-play-evaluation",
        config={
            "agent_0_source_run": run_paths[0],
            "agent_1_source_run": run_paths[1],
            "agent_0_vmap_index": args.agent_vmap_indices[0],
            "agent_1_vmap_index": args.agent_vmap_indices[1],
            "layout": args.layout,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "stochastic": args.stochastic,
        },
    ) as evaluation_run:
        checkpoints = []
        for source_run, artifact, vmap_index in zip(
            source_runs, artifacts, args.agent_vmap_indices
        ):
            artifact_ref = _artifact_reference(artifact, source_run)
            used_artifact = evaluation_run.use_artifact(artifact_ref)
            download_root = (
                args.artifact_dir / source_run.id / artifact.name.replace(":", "-")
            )
            artifact_dir = Path(used_artifact.download(root=str(download_root)))
            checkpoint = resolve_vmap_checkpoint(artifact_dir, vmap_index)
            checkpoints.append(checkpoint)
            print(f"Using {source_run.id} checkpoint: {checkpoint}")

        result = evaluate_crossplay(
            tuple(checkpoints),
            run_configs,
            args,
            episode_callback=wandb.log,
        )
        returns = result["returns"]
        lengths = result["lengths"]
        summary = {
            "eval/mean_return": float(np.mean(returns)),
            "eval/std_return": float(np.std(returns)),
            "eval/min_return": float(np.min(returns)),
            "eval/max_return": float(np.max(returns)),
            "eval/mean_episode_length": float(np.mean(lengths)),
        }
        wandb.log(summary)
        evaluation_run.summary.update(summary)

        if args.metrics_json is not None:
            write_metrics_json(
                args.metrics_json,
                run_paths,
                short_ids,
                result["layout"],
                summary,
            )

        video_path = args.video or Path(
            f"evaluation/overcooked_v3/crossplay/{run_name}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        env = result["env"]
        visualizer = OvercookedV3Visualizer(
            tile_size=24,
            seconds_per_step=1.0 / args.video_fps,
            transition_warning_steps=env.transition_warning_steps,
            signal_activation_time=env.signal_activation_time,
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
                "visualization/crossplay_episode": wandb.Video(
                    str(video_path),
                    format="mp4",
                    caption=f"{run_paths[0]} x {run_paths[1]} on {result['layout']}",
                )
            }
        )
        print(
            f"mean_return={summary['eval/mean_return']:.2f} "
            f"std_return={summary['eval/std_return']:.2f} "
            f"mean_length={summary['eval/mean_episode_length']:.2f}"
        )
        print(f"Saved cross-play video: {video_path}")


if __name__ == "__main__":
    main()
