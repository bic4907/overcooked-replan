"""Evaluate CNN or RNN IPPO checkpoints on V1 Overcooked."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl
from ippo_overcooked import ActorCriticCNN, ActorCriticRNN, ScannedRNN
from jaxmarl.environments.overcooked import dynamic_layouts, overcooked_layouts
from jaxmarl.environments.overcooked.overcooked import OvercookedActions
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.wrappers.baselines import load_params


def parse_args(default_architecture="cnn"):
    parser = argparse.ArgumentParser(
        description="Evaluate an IPPO CNN or RNN checkpoint on V1 Overcooked."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Checkpoint to evaluate. If omitted, the most recently modified "
            "checkpoint for --layout is selected automatically."
        ),
    )
    parser.add_argument(
        "--agent-seeds",
        type=int,
        nargs=2,
        metavar=("AGENT_0_SEED", "AGENT_1_SEED"),
        help=(
            "Training seeds for the policies assigned to agent_0 and agent_1. "
            "For example, '0 0' evaluates a same-seed pair and '0 1' evaluates "
            "a cross-seed pair. Cannot be combined with --checkpoint."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/mnt/nas/overcooked-replan"),
        help=(
            "Model root used for automatic checkpoint selection "
            "(default: /mnt/nas/overcooked-replan)."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("cnn", "rnn"),
        default=default_architecture,
        help=f"Policy architecture (default: {default_architecture}).",
    )
    parser.add_argument(
        "--layout",
        choices=sorted(set(dynamic_layouts) | set(overcooked_layouts)),
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--fc-dim-size", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of selecting the highest-probability action.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        help="Optional path for an animation of the first evaluation episode.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Replay the first evaluation episode in a live window.",
    )
    parser.add_argument("--render-delay", type=float, default=0.2)
    return parser.parse_args()


def resolve_checkpoint(
    checkpoint,
    models_dir,
    layout,
    architecture,
    training_seed=None,
):
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    if layout in dynamic_layouts:
        experiment_name = f"overcooked_dynamic_{layout.removeprefix('dynamic_')}"
    else:
        experiment_name = f"overcooked_{layout}"
    checkpoint_prefix = f"ippo_{architecture}"
    checkpoint_dir = models_dir / "ippo_v1" / architecture / experiment_name
    seed_pattern = "*" if training_seed is None else str(training_seed)
    pattern = (
        f"{checkpoint_prefix}_{experiment_name}_seed{seed_pattern}_"
        "vmap*.safetensors"
    )
    candidates = [
        path
        for path in checkpoint_dir.glob(pattern)
        if "_update" not in path.stem
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} matching {pattern}. "
            "Pass --checkpoint explicitly or train this layout first."
        )

    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def evaluate_episode(policy, params, env_step, env, key, hidden_size):
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    hidden = ScannedRNN.initialize_carry(env.num_agents, hidden_size)
    last_done = jnp.zeros((env.num_agents,), dtype=jnp.bool_)

    state_seq = [state]
    captions = ["step=0 score=0 actions=-/-"]
    episode_return = 0.0

    for step in range(env.max_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        obs_batch = jnp.stack([obs[agent] for agent in env.agents])
        hidden, action = policy(
            params,
            hidden,
            obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
            action_key,
        )
        action = action.squeeze(0)
        actions = {agent: action[i] for i, agent in enumerate(env.agents)}

        obs, state, reward, done, info = env_step(step_key, state, actions)
        episode_return += float(reward["agent_0"])
        state_seq.append(state)
        action_names = [OvercookedActions(int(action[i])).name for i in range(2)]
        captions.append(
            f"step={step + 1} score={episode_return:g} "
            f"actions={action_names[0]}/{action_names[1]}"
        )
        last_done = jnp.asarray([done[agent] for agent in env.agents])

        if bool(done["__all__"]):
            return episode_return, step + 1, state_seq, captions, key

    return episode_return, env.max_steps, state_seq, captions, key


def main(default_architecture="cnn"):
    args = parse_args(default_architecture)
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.render_delay < 0:
        raise ValueError("--render-delay must be non-negative")
    if args.checkpoint is not None and args.agent_seeds is not None:
        raise ValueError("--checkpoint and --agent-seeds cannot be used together")

    if args.agent_seeds is None:
        checkpoint = resolve_checkpoint(
            args.checkpoint,
            args.models_dir,
            args.layout,
            args.architecture,
        )
        checkpoints = (checkpoint, checkpoint)
    else:
        checkpoints = tuple(
            resolve_checkpoint(
                None,
                args.models_dir,
                args.layout,
                args.architecture,
                training_seed=training_seed,
            )
            for training_seed in args.agent_seeds
        )
    for agent, checkpoint in zip(("agent_0", "agent_1"), checkpoints):
        print(f"Using {agent} checkpoint: {checkpoint}")

    if args.layout in dynamic_layouts:
        env = jaxmarl.make(
            "overcooked_dynamic",
            layout=args.layout,
            max_steps=args.max_steps,
            random_agent_positions=False,
        )
    else:
        env = jaxmarl.make(
            "overcooked",
            layout=overcooked_layouts[args.layout],
            max_steps=args.max_steps,
            random_agent_positions=False,
        )
    config = {
        "ACTIVATION": args.activation,
        "FC_DIM_SIZE": args.fc_dim_size,
        "GRU_HIDDEN_DIM": args.gru_hidden_dim,
    }
    network_class = ActorCriticRNN if args.architecture == "rnn" else ActorCriticCNN
    network = network_class(env.action_space(env.agents[0]).n, config=config)
    params = tuple(load_params(checkpoint) for checkpoint in checkpoints)

    def select_action(params, hidden, obs, dones, action_key):
        action_keys = jax.random.split(action_key, env.num_agents)
        next_hidden = []
        actions = []
        for agent_index in range(env.num_agents):
            agent_hidden, pi, _ = network.apply(
                params[agent_index],
                hidden[agent_index : agent_index + 1],
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
        return jnp.concatenate(next_hidden), jnp.concatenate(actions, axis=1)

    policy = jax.jit(select_action)
    env_step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(args.seed)
    returns = []
    lengths = []
    first_states = None
    first_captions = None

    for episode in range(args.episodes):
        episode_return, length, states, captions, key = evaluate_episode(
            policy,
            params,
            env_step,
            env,
            key,
            args.gru_hidden_dim,
        )
        returns.append(episode_return)
        lengths.append(length)
        if first_states is None:
            first_states = states
            first_captions = captions
        print(f"episode={episode + 1} return={episode_return:.2f} length={length}")

    print(
        f"mean_return={np.mean(returns):.2f} "
        f"std_return={np.std(returns):.2f} "
        f"mean_length={np.mean(lengths):.2f}"
    )

    if args.gif is not None:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        viz = OvercookedVisualizer()
        viz.animate(
            first_states,
            agent_view_size=env.agent_view_size,
            filename=str(args.gif),
            captions=first_captions,
        )
        print(f"Saved animation: {args.gif}")

    if args.render:
        viz = OvercookedVisualizer()
        window = viz._lazy_init_window()
        for state, caption in zip(first_states, first_captions):
            window.set_caption(caption)
            viz.render(env.agent_view_size, state, highlight=False)
            time.sleep(args.render_delay)
            if viz.window is not None and viz.window.closed:
                break
        if viz.window is not None and not viz.window.closed:
            viz.show(block=True)


if __name__ == "__main__":
    main()
