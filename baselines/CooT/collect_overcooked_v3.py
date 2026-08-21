"""Collect partner/best-response rollouts for Overcooked V3 CooT training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import jaxmarl

try:
    from .data import write_metadata, write_pair_shard
    from .runtime import (
        CheckpointPolicy,
        PolicySpec,
        load_pair_manifest,
        policy_spec_dict,
    )
except ImportError:  # Direct execution: python baselines/CooT/<script>.py
    from data import write_metadata, write_pair_shard
    from runtime import (
        CheckpointPolicy,
        PolicySpec,
        load_pair_manifest,
        policy_spec_dict,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Collect trajectories from explicit behavior-partner / best-response "
            "checkpoint pairs for CooT."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/coot"))
    parser.add_argument("--layout", help="Override the layout in the manifest.")
    parser.add_argument("--rollouts-per-pair", type=int, default=250)
    parser.add_argument("--validation-rollouts-per-pair", type=int, default=50)
    parser.add_argument(
        "--override-manifest-rollout-budgets",
        action="store_true",
        help=(
            "Testing-only: use the CLI rollout budgets even when a production "
            "manifest records paper budgets, and reallocate variants by weight."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--contexts-per-pair", type=int, default=125)
    parser.add_argument("--queries-per-context", type=int, default=70)
    parser.add_argument(
        "--random-agent-positions", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--transition-countdown", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--layout-change-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--transition-warning-steps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _allocate_variant_rollouts(weights: list[float], total: int) -> list[int]:
    """Allocate a fixed rollout budget with largest-remainder rounding."""

    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.ndim != 1 or not normalized.size or np.any(normalized <= 0):
        raise ValueError("Every rollout variant weight must be positive")
    normalized /= normalized.sum()
    exact = normalized * total
    counts = np.floor(exact).astype(np.int32)
    for index in np.argsort(-(exact - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return counts.tolist()


def _resolve_variant_rollouts(variants: list[dict], total: int) -> list[int]:
    """Prefer exact per-variant budgets, otherwise use weighted allocation."""

    explicit = [variant.get("num_rollouts") for variant in variants]
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError(
                "Either every rollout variant or no rollout variant must set "
                "num_rollouts"
            )
        counts = [int(value) for value in explicit]
        if any(count < 1 for count in counts):
            raise ValueError("Every explicit variant num_rollouts must be positive")
        if sum(counts) != total:
            raise ValueError(
                f"Variant num_rollouts sum to {sum(counts)}, expected pair total {total}"
            )
        return counts
    return _allocate_variant_rollouts(
        [float(variant.get("weight", 1.0)) for variant in variants], total
    )


def collect_pair(
    env,
    partner: CheckpointPolicy,
    best_response: CheckpointPolicy,
    *,
    rollouts: int,
    max_steps: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, jax.Array]:
    """Collect agent_1 observations/actions/rewards, as in the release."""

    step_env = jax.jit(env.step_env)
    observations = []
    actions = []
    rewards = []
    lengths = []

    for _episode in range(rollouts):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        partner.reset()
        best_response.reset()
        episode_observations = np.zeros(
            (max_steps, *env.observation_space("agent_1").shape), dtype=np.float16
        )
        episode_actions = np.zeros((max_steps,), dtype=np.uint8)
        episode_rewards = np.zeros((max_steps,), dtype=np.float32)
        episode_length = 0

        for step in range(max_steps):
            key, partner_key, response_key, step_key = jax.random.split(key, 4)
            partner_action, _ = partner.act(obs["agent_0"], partner_key)
            response_action, _ = best_response.act(obs["agent_1"], response_key)
            episode_observations[step] = np.asarray(
                jax.device_get(obs["agent_1"]), dtype=np.float16
            )
            episode_actions[step] = response_action
            obs, state, reward, done, _info = step_env(
                step_key,
                state,
                {"agent_0": partner_action, "agent_1": response_action},
            )
            episode_rewards[step] = float(reward["agent_1"])
            episode_length = step + 1
            if bool(done["__all__"]):
                break

        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        lengths.append(episode_length)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.asarray(lengths),
        key,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.rollouts_per_pair < 7:
        raise ValueError(
            "--rollouts-per-pair must be at least 7 so five context episodes "
            "and an independent query episode can be sampled"
        )
    if args.validation_rollouts_per_pair < 7:
        raise ValueError(
            "--validation-rollouts-per-pair must be at least 7 so five context "
            "episodes and an independent query episode can be sampled"
        )
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    manifest, manifest_path = load_pair_manifest(args.manifest)
    has_validation_pairs = any(
        str(pair.get("split", "train")).lower() == "validation"
        for pair in manifest["pairs"]
    )
    if not has_validation_pairs:
        validation_count = max(
            1, int(round(args.rollouts_per_pair * args.validation_fraction))
        )
        if min(validation_count, args.rollouts_per_pair - validation_count) <= 5:
            raise ValueError(
                "Without split=validation partner pairs, both trajectory holdouts "
                "need at least six rollouts for five contexts plus one query; "
                "increase --rollouts-per-pair or adjust --validation-fraction"
            )
    layout = args.layout or manifest.get("layout")
    if not layout:
        raise ValueError("Specify --layout or set 'layout' in the pair manifest")
    output_dir = (args.output_root / layout).expanduser().resolve()
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Dataset already exists: {output_dir}. Pass --overwrite to replace shards."
        )

    env = jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=args.max_steps,
        random_agent_positions=args.random_agent_positions,
        include_transition_countdown=args.transition_countdown,
        include_layout_change_mask=args.layout_change_mask,
        transition_warning_steps=args.transition_warning_steps,
    )
    action_dim = int(env.action_space("agent_0").n)
    observation_shape = tuple(env.observation_space("agent_1").shape)
    key = jax.random.PRNGKey(args.seed)
    pair_metadata = []

    for pair_index, pair in enumerate(manifest["pairs"]):
        if "partner" not in pair or "best_response" not in pair:
            raise ValueError(
                f"Manifest pair {pair_index} needs 'partner' and 'best_response'"
            )
        pair_id = str(pair.get("id", f"pair_{pair_index:03d}"))
        pair_split = str(pair.get("split", "train")).lower()
        if pair_split not in {"train", "validation"}:
            raise ValueError(
                f"Manifest pair {pair_id!r} has invalid split {pair_split!r}"
            )
        partner_spec = PolicySpec.from_mapping(
            pair["partner"], base_dir=manifest_path.parent
        )
        response_spec = PolicySpec.from_mapping(
            pair["best_response"], base_dir=manifest_path.parent
        )
        print(
            f"[{pair_index + 1}/{len(manifest['pairs'])}] collecting {pair_id}",
            flush=True,
        )
        # PORTING NOTE: the supplement contains the legacy ZSC-Eval HSP/MEP
        # builders, but not V3-compatible checkpoints. Explicit variants keep
        # each skill-level partner paired with the BR trained for that exact
        # checkpoint after the V3 population/response sweeps have run.
        variants = list(pair.get("rollout_variants") or [{}])
        if not all(isinstance(variant, dict) for variant in variants):
            raise ValueError(
                f"Manifest pair {pair_id!r} rollout_variants must be JSON objects"
            )
        default_rollout_count = (
            args.validation_rollouts_per_pair
            if pair_split == "validation"
            else args.rollouts_per_pair
        )
        pair_rollout_count = int(
            default_rollout_count
            if args.override_manifest_rollout_budgets
            else pair.get("num_rollouts", default_rollout_count)
        )
        if pair_rollout_count < 7:
            raise ValueError(
                f"Manifest pair {pair_id!r} needs at least seven rollouts; "
                f"got num_rollouts={pair_rollout_count}"
            )
        allocation_variants = variants
        if args.override_manifest_rollout_budgets:
            allocation_variants = [
                {key: value for key, value in variant.items() if key != "num_rollouts"}
                for variant in variants
            ]
        variant_counts = _resolve_variant_rollouts(
            allocation_variants, pair_rollout_count
        )
        observations = []
        actions = []
        rewards = []
        episode_lengths = []
        resolved_variants = []
        for variant, variant_count in zip(variants, variant_counts):
            if not variant_count:
                continue
            variant_partner = {
                **pair["partner"],
                **(variant.get("partner") or {}),
            }
            variant_response = {
                **pair["best_response"],
                **(variant.get("best_response") or {}),
            }
            variant_partner_spec = PolicySpec.from_mapping(
                variant_partner, base_dir=manifest_path.parent
            )
            variant_response_spec = PolicySpec.from_mapping(
                variant_response, base_dir=manifest_path.parent
            )
            variant_obs, variant_actions, variant_rewards, lengths, key = collect_pair(
                env,
                CheckpointPolicy(variant_partner_spec, action_dim),
                CheckpointPolicy(variant_response_spec, action_dim),
                rollouts=variant_count,
                max_steps=args.max_steps,
                key=key,
            )
            observations.append(variant_obs)
            actions.append(variant_actions)
            rewards.append(variant_rewards)
            episode_lengths.append(lengths)
            resolved_variants.append(
                {
                    "weight": float(variant.get("weight", 1.0)),
                    "num_trajectories": variant_count,
                    "partner": policy_spec_dict(variant_partner_spec),
                    "best_response": policy_spec_dict(variant_response_spec),
                }
            )
        shard = write_pair_shard(
            output_dir,
            pair_index,
            np.concatenate(observations),
            np.concatenate(actions),
            np.concatenate(rewards),
            np.concatenate(episode_lengths),
        )
        pair_metadata.append(
            {
                "id": pair_id,
                "split": pair_split,
                "population_type": pair.get("population_type"),
                "shard": shard.name,
                "num_trajectories": pair_rollout_count,
                "partner": policy_spec_dict(partner_spec),
                "best_response": policy_spec_dict(response_spec),
                "rollout_variants": resolved_variants,
                "reference_return": pair.get("reference_return"),
            }
        )

    metadata = {
        "method": "CooT",
        "source_paper": "arXiv:2506.23549",
        "layout": layout,
        "episode_horizon": args.max_steps,
        "observation_shape": observation_shape,
        "action_dim": action_dim,
        "rollouts_per_pair": args.rollouts_per_pair,
        "validation_rollouts_per_pair": args.validation_rollouts_per_pair,
        "contexts_per_pair": args.contexts_per_pair,
        "queries_per_context": args.queries_per_context,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "pair_manifest": str(manifest_path),
        "pairs": pair_metadata,
        "porting_notes": [
            "ZSC-Eval Overcooked was replaced by this repository's Overcooked V3.",
            "The supplement's legacy HSP/MEP pipeline is not checkpoint-compatible with V3; partner/BR lineage is therefore explicit in the manifest.",
            "Training HSP selection uses normalized event-count greedy L1 diversity; DPP is reserved for held-out evaluation population selection.",
            "Per-pair num_rollouts supports the paper's HSP 250 (30 mid, 220 final) and MEP 200 final-only budgets.",
            "A rollout variant must override both partner and BR when it represents a different HSP skill-level checkpoint.",
            "Raw trajectories are sharded once and context/query examples are sampled online to avoid duplicating V3 observations.",
            "V3 transition channels already in [0,1] remain unscaled; legacy 0/255 channels are divided by 255 in the model.",
        ],
    }
    write_metadata(output_dir, metadata)
    (output_dir / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "layout": layout,
                "pairs": pair_metadata,
                "switch_pairs": manifest.get("switch_pairs", []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved CooT dataset: {output_dir}")


if __name__ == "__main__":
    main()
