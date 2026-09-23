"""Section 6.3: does preparation by an informed agent cue an uninformed partner?

Three rollouts per (informed seed, partner seed, informed seat, episode),
all from the same reset:

    control    nobody sees the warning, partner stream k
    notice     only the informed seat sees the warning, partner stream k
    reference  nobody sees the warning, partner stream k'

Both policies sample their actions, so control and reference differ by
chance alone; the divergence of the uninformed partner's behaviour from
the reference under notice, over and above that chance level, is what the
informed agent's preparation induced. Behaviour in each 20-step window
before a boundary is a distribution over four completed subtasks --
ingredient acquisition, pot insertion, plate acquisition, soup
acquisition -- compared by Jensen-Shannon divergence. Drop and return are
scored on the same episodes.

    python experiment/notice/behavior_as_notice.py --layout split_0 --arm rnn

Writes one JSON line per (episode triple, boundary) to outputs/notice/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common import (  # noqa: E402  (sibling module)
    BASELINE_NAMES,
    BENCHMARK_LAYOUTS,
    EVENT_NAMES,
    FAMILY_OF,
    ROOT,
    WARNING_STEPS,
    build_env,
    change_boundaries,
    check_policy,
    dump_jsonl,
    fetch_policies,
    jensen_shannon,
    kitchen_of_phase,
    policy_controller,
    step_fn,
)
from baselines.adaptation_metrics import (  # noqa: E402
    AdaptationMetricConfig,
    EpisodeAdaptationTrace,
    compute_transition_metrics,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", default="split_0", choices=BENCHMARK_LAYOUTS)
    parser.add_argument("--arm", default="rnn", choices=("rnn", "fcp", "cnn"))
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=4, help="Episode keys per (pair, seat).")
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--greedy", action="store_true", help="Take the mode instead of sampling.")
    parser.add_argument("--self-play", action="store_true", help="Also play each seed with itself.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/notice"))
    return parser.parse_args(argv)


def make_player(env, controllers, steps):
    body = step_fn(env, controllers)

    def play(key, params):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        seats = tuple((controllers[s]["init"](state), params[s]) for s in range(env.num_agents))
        _, (rewards, events, states) = jax.lax.scan(
            body, (key, obs, state, jnp.zeros((), bool), seats), None, length=steps
        )
        return rewards, events, states.layout_index

    return jax.jit(jax.vmap(play, in_axes=(0, None)))


def window_counts(events, boundary, seat, width=WARNING_STEPS):
    """Subtask counts of one seat in the ``width`` steps before ``boundary``."""
    return np.asarray(events[boundary - width : boundary, seat, :]).sum(axis=0)


def transitions(rewards, layout_index, config, kitchens):
    """Adaptation metrics keyed by boundary step."""
    layout_index = np.asarray(layout_index)
    phases = np.concatenate([[0], layout_index[:-1]])  # phase when each action was chosen
    trace = EpisodeAdaptationTrace(np.asarray(rewards), phases)
    return {t.boundary: t for t in compute_transition_metrics(trace, config, kitchens)}


def main(argv=None):
    from jaxmarl._env import load_project_env

    load_project_env()
    args = parse_args(argv)
    family = args.layout.split("_")[0]
    policies = fetch_policies(args.arm, args.layout, args.seeds, artifact_dir=args.artifact_dir)
    config = policies[0][2]
    shape_params = policies[0][3]
    stochastic = not args.greedy

    envs = {
        observer: build_env(args.layout, observer, args.max_steps, config)
        for observer in ("none", "agent_0", "agent_1")
    }
    for _, _, cfg, params in policies:
        check_policy(envs["none"], cfg, params, 0)
    players = {}
    for observer, env in envs.items():
        controllers = [policy_controller(env, config, shape_params, s, stochastic) for s in range(2)]
        players[observer] = make_player(env, controllers, args.max_steps)

    horizon = int(envs["none"].phase_durations[0])
    metric_config = AdaptationMetricConfig(window=min(30, horizon // 2), horizon=horizon)
    kitchens = kitchen_of_phase(envs["none"])
    boundaries = change_boundaries(envs["none"], args.max_steps)

    pairs = [(i, j) for i in range(args.seeds) for j in range(args.seeds) if i != j or args.self_play]
    rows = []
    episodes = []
    started = time.time()
    for informed, partner in pairs:
        p_inf, p_par = policies[informed][3], policies[partner][3]
        for seat in (0, 1):
            seated = (p_inf, p_par) if seat == 0 else (p_par, p_inf)
            partner_seat = 1 - seat
            base = args.seed + 100_000 * informed + 10_000 * partner + 1_000 * seat
            keys = jnp.stack([jax.random.PRNGKey(base + e) for e in range(args.episodes)])
            ref_keys = jnp.stack([jax.random.PRNGKey(base + 500 + e) for e in range(args.episodes)])
            control = players["none"](keys, seated)
            notice = players[f"agent_{seat}"](keys, seated)
            reference = players["none"](ref_keys, seated)
            arms = dict(control=control, notice=notice, reference=reference)
            arms = {name: jax.tree_util.tree_map(np.asarray, value) for name, value in arms.items()}
            for e in range(args.episodes):
                metrics = {
                    name: transitions(value[0][e], value[2][e], metric_config, kitchens)
                    for name, value in arms.items()
                }
                returns = {name: float(value[0][e].sum()) for name, value in arms.items()}
                episodes.append(
                    dict(
                        layout=args.layout,
                        family=family,
                        scenario=FAMILY_OF[family],
                        arm=args.arm,
                        algorithm=BASELINE_NAMES[args.arm],
                        informed_seed=informed,
                        partner_seed=partner,
                        informed_seat=seat,
                        episode=e,
                        **{f"return_{name}": value for name, value in returns.items()},
                    )
                )
                for k, boundary in enumerate(boundaries):
                    counts = {
                        name: window_counts(value[1][e], boundary, partner_seat)
                        for name, value in arms.items()
                    }
                    informed_counts = {
                        name: window_counts(value[1][e], boundary, seat)
                        for name, value in arms.items()
                    }
                    eligible = {name: bool(counts[name].sum() > 0) for name in counts}
                    row = dict(
                        layout=args.layout,
                        family=family,
                        arm=args.arm,
                        informed_seed=informed,
                        partner_seed=partner,
                        informed_seat=seat,
                        episode=e,
                        transition=k,
                        boundary=boundary,
                        eligible_control=eligible["control"] and eligible["reference"],
                        eligible_notice=eligible["notice"] and eligible["reference"],
                        jsd_control=jensen_shannon(counts["control"], counts["reference"])
                        if eligible["control"] and eligible["reference"]
                        else None,
                        jsd_notice=jensen_shannon(counts["notice"], counts["reference"])
                        if eligible["notice"] and eligible["reference"]
                        else None,
                        jsd_control_vs_notice=jensen_shannon(counts["control"], counts["notice"])
                        if eligible["control"] and eligible["notice"]
                        else None,
                    )
                    for name in arms:
                        row[f"counts_{name}"] = counts[name].astype(int).tolist()
                        row[f"informed_counts_{name}"] = informed_counts[name].astype(int).tolist()
                        t = metrics[name].get(boundary)
                        row[f"drop_{name}"] = (
                            None if t is None or not np.isfinite(t.drop) else float(t.drop)
                        )
                        row[f"recovery_{name}"] = (
                            None
                            if t is None or not np.isfinite(t.recovery_time)
                            else float(t.recovery_time)
                        )
                        row[f"direction_{name}"] = None if t is None else f"{t.from_phase}->{t.to_phase}"
                        row[f"return_{name}"] = returns[name]
                    rows.append(row)
        done = [r for r in rows if r["informed_seed"] == informed and r["partner_seed"] == partner]
        jc = np.mean([r["jsd_control"] for r in done if r["jsd_control"] is not None])
        jn = np.mean([r["jsd_notice"] for r in done if r["jsd_notice"] is not None])
        ret = [e for e in episodes if e["informed_seed"] == informed and e["partner_seed"] == partner]
        print(
            f"  informed {informed} partner {partner}: JSD control {jc:.3f} notice {jn:.3f}  "
            f"return control {np.mean([e['return_control'] for e in ret]):.1f} "
            f"notice {np.mean([e['return_notice'] for e in ret]):.1f}"
            f"  ({time.time() - started:.0f}s)",
            flush=True,
        )

    out = Path(args.out_dir) / f"behavior_{args.layout}_{args.arm}.jsonl"
    dump_jsonl(out, rows)
    dump_jsonl(out.with_name(f"behavior_{args.layout}_{args.arm}.episodes.jsonl"), episodes)
    meta = dict(
        vars(args),
        boundaries=boundaries,
        horizon=horizon,
        event_names=EVENT_NAMES,
        runs=[(s, r) for s, r, _, _ in policies],
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rows)} window rows and {len(episodes)} episodes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
