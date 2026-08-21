"""Seed-wise ordered SP/XP matrix evaluation for Overcooked V3 CooT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402

import jaxmarl  # noqa: E402
from jaxmarl._wandb import require_sweep_target  # noqa: E402
from jaxmarl.environments.overcooked_v3 import (  # noqa: E402
    overcooked_v3_layouts,
)

try:
    from .eval_overcooked_v3 import CooTController
except ImportError:  # Direct execution: python baselines/CooT/<script>.py
    from eval_overcooked_v3 import CooTController


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full ordered matrix of CooT training seeds and report "
            "the same SP, XP, and SP-XP_gap metrics as the existing FCP/IPPO eval."
        )
    )
    parser.add_argument(
        "--layout", required=True, choices=sorted(overcooked_v3_layouts)
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.getenv("COOT_CHECKPOINT_ROOT", "saves/coot_train")),
    )
    parser.add_argument(
        "--checkpoint-pattern",
        help=(
            "Optional glob with {layout} and {seed}. The default is the CooT "
            "best-checkpoint filename produced by train_overcooked_v3.py."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument(
        "--context-update-steps",
        type=int,
        default=20,
        help=(
            "Commit completed transitions to CooT's rolling context every N "
            "steps. Use the checkpoint horizon (normally 450) to recover the "
            "release update timing on full-length episodes."
        ),
    )
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument(
        "--stochastic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--random-agent-positions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--transition-countdown",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--layout-change-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--transition-warning-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.getenv("COOT_CROSSPLAY_OUTPUT_ROOT", "saves/coot_eval/crossplay")
        ),
        help="Root for the unique timestamped run directory when --output-dir is unset.",
    )
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked")
    )
    parser.add_argument(
        "--project", default=os.getenv("WANDB_PROJECT", "overcooked-v3-coot-eval")
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--group", default="coot-crossplay-matrix")
    return parser.parse_args(argv)


def _validate_args(args) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.context_update_steps < 1:
        raise ValueError("--context-update-steps must be positive")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain at least two unique values")
    if args.transition_warning_steps < 0:
        raise ValueError("--transition-warning-steps must be non-negative")


def _wandb_mode(raw_mode: str) -> Literal["online", "offline", "disabled"]:
    mode = raw_mode.lower()
    if mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        return "offline"
    return cast(Literal["online", "offline", "disabled"], mode)


def discover_checkpoints(
    root: Path,
    layout: str,
    seeds: list[int],
    pattern: str | None,
) -> dict[int, Path]:
    """Resolve the newest matching local checkpoint for every training seed."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CooT checkpoint root not found: {root}")
    checkpoints = {}
    for seed in seeds:
        seed_pattern = (
            pattern.format(layout=layout, seed=seed)
            if pattern
            else f"coot_overcooked_v3_{layout}_seed{seed}_best.safetensors"
        )
        candidates = list(root.rglob(seed_pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No CooT checkpoint matching {seed_pattern!r} under {root}"
            )
        checkpoint = max(
            candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))
        )
        sidecar = checkpoint.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"CooT checkpoint sidecar not found: {sidecar}")
        checkpoints[seed] = checkpoint
    if len(set(checkpoints.values())) != len(checkpoints):
        raise ValueError("Every training seed must resolve to a distinct checkpoint")
    return checkpoints


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identities(checkpoints: dict[int, Path]) -> dict[int, dict[str, str]]:
    """Hash weights and sidecars so resumed caches cannot use overwritten files."""

    return {
        seed: {
            "checkpoint_sha256": _sha256(checkpoint),
            "sidecar_sha256": _sha256(checkpoint.with_suffix(".json")),
        }
        for seed, checkpoint in checkpoints.items()
    }


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["layout"],
        record["agent_0_seed"],
        record["agent_1_seed"],
        record["agent_0_checkpoint"],
        record["agent_1_checkpoint"],
        record.get("agent_0_checkpoint_sha256"),
        record.get("agent_1_checkpoint_sha256"),
        record.get("agent_0_sidecar_sha256"),
        record.get("agent_1_sidecar_sha256"),
        record["episodes"],
        record["max_steps"],
        record.get("context_update_steps", record["max_steps"]),
        record["evaluation_seed"],
        record["stochastic"],
        record.get("random_agent_positions"),
        record.get("transition_countdown"),
        record.get("layout_change_mask"),
        record.get("transition_warning_steps"),
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid CooT pair cache: {path}") from error
    if not isinstance(payload, list):
        raise ValueError(f"CooT pair cache must be a list: {path}")
    return payload


def _write_reproducibility_bundle(
    output_dir: Path, args, checkpoints, checkpoint_identities
) -> None:
    command = shlex.join(sys.argv)
    (output_dir / "command.txt").write_text(f"{command}\n", encoding="utf-8")
    _atomic_write_json(
        output_dir / "run_config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "checkpoints": {seed: str(path) for seed, path in checkpoints.items()},
            "checkpoint_identities": checkpoint_identities,
        },
    )
    source_dir = output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    for source in (
        Path(__file__),
        Path(__file__).with_name("eval_overcooked_v3.py"),
        Path(__file__).with_name("model.py"),
    ):
        shutil.copy2(source, source_dir / source.name)


def run_pair(
    env,
    env_step,
    agent_0: CooTController,
    agent_1: CooTController,
    *,
    episodes: int,
    max_steps: int,
    evaluation_seed: int,
) -> tuple[list[float], list[int]]:
    """Run repeated episodes while preserving CooT context within one pair."""

    agent_0.reset_context()
    agent_1.reset_context()
    key = jax.random.PRNGKey(evaluation_seed)
    returns = []
    lengths = []

    for _episode in range(episodes):
        key, reset_key = jax.random.split(key)
        observations, state = env.reset(reset_key)
        agent_0.start_episode()
        agent_1.start_episode()
        episode_return = 0.0

        for step in range(max_steps):
            key, action_0_key, action_1_key, step_key = jax.random.split(key, 4)
            action_0, _probabilities_0, _context_jsd_0 = agent_0.act(
                observations["agent_0"], action_0_key
            )
            action_1, _probabilities_1, _context_jsd_1 = agent_1.act(
                observations["agent_1"], action_1_key
            )
            observations, state, rewards, dones, _info = env_step(
                step_key,
                state,
                {"agent_0": action_0, "agent_1": action_1},
            )
            reward_0 = float(rewards["agent_0"])
            reward_1 = float(rewards["agent_1"])
            agent_0.observe_reward(reward_0)
            agent_1.observe_reward(reward_1)
            episode_return += reward_0
            if bool(dones["__all__"]):
                break

        agent_0.finish_episode()
        agent_1.finish_episode()
        returns.append(episode_return)
        lengths.append(step + 1)

    return returns, lengths


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    self_play = np.asarray(
        [record["mean_return"] for record in records if record["pair_type"] == "SP"],
        dtype=np.float64,
    )
    cross_play = np.asarray(
        [record["mean_return"] for record in records if record["pair_type"] == "XP"],
        dtype=np.float64,
    )
    sp = float(np.mean(self_play)) if self_play.size else float("nan")
    xp = float(np.mean(cross_play)) if cross_play.size else float("nan")
    return {
        "SP": sp,
        "XP": xp,
        "SP-XP_gap": sp - xp,
        "SP_pairs": int(self_play.size),
        "XP_pairs": int(cross_play.size),
    }


def _matrix(records: list[dict[str, Any]], seeds: list[int]) -> np.ndarray:
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    matrix = np.full((len(seeds), len(seeds)), np.nan, dtype=np.float64)
    for record in records:
        matrix[
            seed_index[int(record["agent_0_seed"])],
            seed_index[int(record["agent_1_seed"])],
        ] = float(record["mean_return"])
    return matrix


def _save_heatmap(
    matrix: np.ndarray, seeds: list[int], layout: str, path: Path
) -> None:
    labels = [f"CooT|s{seed}" for seed in seeds]
    size = max(6.0, 0.7 * len(seeds) + 3.0)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set(
        title=f"{layout}: CooT seed-wise payoff matrix",
        xlabel="agent_1 policy",
        ylabel="agent_0 policy",
    )
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    for row in range(len(seeds)):
        for column in range(len(seeds)):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.1f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="mean episode return")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    columns = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    checkpoints = discover_checkpoints(
        args.checkpoint_root, args.layout, args.seeds, args.checkpoint_pattern
    )
    checkpoint_identities = _checkpoint_identities(checkpoints)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"xp-coot-{args.layout}"
    output_dir = args.output_dir or (
        args.output_root / f"{run_name}-{timestamp}-p{os.getpid()}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_reproducibility_bundle(output_dir, args, checkpoints, checkpoint_identities)

    mode = _wandb_mode(args.wandb_mode)
    target = {}
    if not os.environ.get("WANDB_SWEEP_ID"):
        target = {"entity": args.entity, "project": args.project}
    run = wandb.init(
        **target,
        config={
            **vars(args),
            "checkpoint_root": str(args.checkpoint_root),
            "output_dir": str(output_dir),
            "checkpoints": {seed: str(path) for seed, path in checkpoints.items()},
            "checkpoint_identities": checkpoint_identities,
        },
        name=run_name,
        group=args.group,
        tags=["CooT", "OvercookedV3", "SP-XP"],
        job_type="cross-play-matrix-evaluation",
        mode=mode,
    )
    if run is None:
        raise RuntimeError("W&B did not create a CooT cross-play run")
    require_sweep_target(
        run,
        {"ENTITY": args.entity, "PROJECT": args.project},
    )

    env = jaxmarl.make(
        "overcooked_v3",
        layout=args.layout,
        max_steps=args.max_steps,
        random_agent_positions=args.random_agent_positions,
        include_transition_countdown=args.transition_countdown,
        include_layout_change_mask=args.layout_change_mask,
        transition_warning_steps=args.transition_warning_steps,
    )
    env_step = jax.jit(env.step_env)

    # Separate banks are required on diagonal SP pairs: the two agents share
    # parameters but must never share their mutable query/context FIFO state.
    agent_0_controllers = {
        seed: CooTController(
            checkpoint,
            stochastic=args.stochastic,
            context_ablation_stride=0,
            context_update_steps=args.context_update_steps,
        )
        for seed, checkpoint in checkpoints.items()
    }
    agent_1_controllers = {
        seed: CooTController(
            checkpoint,
            stochastic=args.stochastic,
            context_ablation_stride=0,
            context_update_steps=args.context_update_steps,
        )
        for seed, checkpoint in checkpoints.items()
    }
    expected_model_config = None
    for controller in [*agent_0_controllers.values(), *agent_1_controllers.values()]:
        if expected_model_config is None:
            expected_model_config = controller.config
        elif controller.config != expected_model_config:
            raise ValueError("All CooT seed checkpoints must share one model config")
        if controller.config.episode_horizon != args.max_steps:
            raise ValueError(
                f"Checkpoint horizon {controller.config.episode_horizon} does not "
                f"match --max-steps={args.max_steps}"
            )

    cache_path = output_dir / "pair_cache.json"
    model_manifest = [
        {
            "model_id": str(checkpoints[seed]),
            "label": f"CooT|s{seed}",
            "algorithm": "CooT",
            "layout": args.layout,
            "training_seed": seed,
            "checkpoint": str(checkpoints[seed]),
            **checkpoint_identities[seed],
        }
        for seed in args.seeds
    ]
    _atomic_write_json(output_dir / "models.json", model_manifest)
    run.config.update({"selected_models": model_manifest}, allow_val_change=True)
    records = _load_records(cache_path)
    cached = {_record_key(record): record for record in records}
    total_pairs = len(args.seeds) ** 2
    pair_index = 0
    active_records = []

    for agent_0_seed in args.seeds:
        for agent_1_seed in args.seeds:
            pair_index += 1
            expected_key = (
                args.layout,
                agent_0_seed,
                agent_1_seed,
                str(checkpoints[agent_0_seed]),
                str(checkpoints[agent_1_seed]),
                checkpoint_identities[agent_0_seed]["checkpoint_sha256"],
                checkpoint_identities[agent_1_seed]["checkpoint_sha256"],
                checkpoint_identities[agent_0_seed]["sidecar_sha256"],
                checkpoint_identities[agent_1_seed]["sidecar_sha256"],
                args.episodes,
                args.max_steps,
                args.context_update_steps,
                args.evaluation_seed,
                args.stochastic,
                args.random_agent_positions,
                args.transition_countdown,
                args.layout_change_mask,
                args.transition_warning_steps,
            )
            if expected_key in cached:
                record = cached[expected_key]
            else:
                print(
                    f"[{pair_index}/{total_pairs}] CooT s{agent_0_seed} x "
                    f"s{agent_1_seed}",
                    flush=True,
                )
                returns, lengths = run_pair(
                    env,
                    env_step,
                    agent_0_controllers[agent_0_seed],
                    agent_1_controllers[agent_1_seed],
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    evaluation_seed=args.evaluation_seed,
                )
                record = {
                    "layout": args.layout,
                    "pair_type": (
                        "SP"
                        if checkpoints[agent_0_seed] == checkpoints[agent_1_seed]
                        else "XP"
                    ),
                    "agent_0_model_id": str(checkpoints[agent_0_seed]),
                    "agent_1_model_id": str(checkpoints[agent_1_seed]),
                    "agent_0_label": f"CooT|s{agent_0_seed}",
                    "agent_1_label": f"CooT|s{agent_1_seed}",
                    "agent_0_algorithm": "CooT",
                    "agent_1_algorithm": "CooT",
                    "agent_0_seed": agent_0_seed,
                    "agent_1_seed": agent_1_seed,
                    "agent_0_checkpoint": str(checkpoints[agent_0_seed]),
                    "agent_1_checkpoint": str(checkpoints[agent_1_seed]),
                    "agent_0_checkpoint_sha256": checkpoint_identities[agent_0_seed][
                        "checkpoint_sha256"
                    ],
                    "agent_1_checkpoint_sha256": checkpoint_identities[agent_1_seed][
                        "checkpoint_sha256"
                    ],
                    "agent_0_sidecar_sha256": checkpoint_identities[agent_0_seed][
                        "sidecar_sha256"
                    ],
                    "agent_1_sidecar_sha256": checkpoint_identities[agent_1_seed][
                        "sidecar_sha256"
                    ],
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "context_update_steps": args.context_update_steps,
                    "evaluation_seed": args.evaluation_seed,
                    "stochastic": args.stochastic,
                    "random_agent_positions": args.random_agent_positions,
                    "transition_countdown": args.transition_countdown,
                    "layout_change_mask": args.layout_change_mask,
                    "transition_warning_steps": args.transition_warning_steps,
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "mean_episode_length": float(np.mean(lengths)),
                    "episode_returns": returns,
                }
                records.append(record)
                cached[expected_key] = record
                _atomic_write_json(cache_path, records)
                wandb.log(
                    {
                        "progress/completed_pairs": pair_index,
                        "pair/mean_return": record["mean_return"],
                        "pair/is_self_play": int(record["pair_type"] == "SP"),
                    }
                )
            active_records.append(record)

    summary = summarize_records(active_records)
    expected_sp_pairs = len(args.seeds)
    expected_xp_pairs = len(args.seeds) * (len(args.seeds) - 1)
    if (
        summary["SP_pairs"] != expected_sp_pairs
        or summary["XP_pairs"] != expected_xp_pairs
    ):
        raise RuntimeError(
            "Incomplete CooT payoff matrix: expected "
            f"SP={expected_sp_pairs}, XP={expected_xp_pairs}; got "
            f"SP={summary['SP_pairs']}, XP={summary['XP_pairs']}"
        )
    payoff_matrix = _matrix(active_records, args.seeds)
    heatmap_path = output_dir / f"{args.layout}_coot_seed_matrix.png"
    _save_heatmap(payoff_matrix, args.seeds, args.layout, heatmap_path)
    _atomic_write_json(output_dir / "pair_results.json", active_records)
    _atomic_write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "pair_results.csv", active_records)

    labels = [f"CooT|s{seed}" for seed in args.seeds]
    table_data = [
        [label, *[float(value) for value in payoff_matrix[row]]]
        for row, label in enumerate(labels)
    ]
    pair_columns = list(active_records[0])
    wandb.log(
        {
            "SP": summary["SP"],
            "XP": summary["XP"],
            "SP-XP_gap": summary["SP-XP_gap"],
            "counts/SP_pairs": summary["SP_pairs"],
            "counts/XP_pairs": summary["XP_pairs"],
            "matrices/models": wandb.Image(str(heatmap_path)),
            "matrices/models_table": wandb.Table(
                columns=cast(Any, ["agent_0 \\ agent_1", *labels]),
                data=cast(Any, table_data),
            ),
            "results/pairs": wandb.Table(
                columns=cast(Any, pair_columns),
                data=cast(
                    Any,
                    [
                        [record[column] for column in pair_columns]
                        for record in active_records
                    ],
                ),
            ),
        }
    )
    run.summary.update(summary)
    run.summary["protocol/context_update_steps"] = args.context_update_steps
    run.summary["protocol/agent_0_role_is_release_ood"] = True
    artifact = wandb.Artifact(
        f"coot-crossplay-matrix-{run.id}",
        type="crossplay-evaluation",
        metadata=summary,
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact, aliases=["latest"])
    wandb.finish()
    print(
        f"CooT SP={summary['SP']:.2f} XP={summary['XP']:.2f} "
        f"gap={summary['SP-XP_gap']:.2f} | {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
