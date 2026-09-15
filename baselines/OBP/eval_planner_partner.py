"""Play the simulated humans against every other agent we have trained.

The paper's H2 asks what a teammate trained against a human model is worth when
a real person sits down. We have no person in the loop, so the row of this
matrix is the planner-based stand-ins -- BR and H0/H1/H2 -- and the column is
everything that can hold the other side: the same stand-ins, and the IPPO CNN,
IPPO RNN and FCP policies already trained on this kitchen.

One run of this script is one cell. It plays the pair over the partner's six
training seeds, each seed in both seat orders, so the twelve games cover both
who-sits-where and which seed the partner came from; a pair of planners has no
training seed, so its twelve games are twelve episode seeds instead.

    python -m baselines.OBP.eval_planner_partner \\
        --layout split_0 --human h1 --partner fcp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# A sweep runs this file by path, not as a module, so the repository it lives in
# is not on the import path and ``baselines`` cannot be found.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: Dials of the paper's simulated humans, Table 1, whose columns run
#: (hltemp, lltemp, prob_wait): H0 (0, 0, 0.5), H1 (0.072, 0.286, 0.45),
#: H2 (0.070, 0.249, 0.04). H1 is the arm that waits, which is why the paper has
#: it scoring the least of the three. ``_local`` arms are the same three dials
#: fit to our own demonstrations instead and are read from --dials.
PAPER_ARMS = {
    "br": dict(prob_wait=0.0, lltemp=0.0, hltemp=0.0),
    "h0": dict(prob_wait=0.5, lltemp=0.0, hltemp=0.0),
    "h1": dict(prob_wait=0.45, lltemp=0.286, hltemp=0.072),
    "h2": dict(prob_wait=0.04, lltemp=0.249, hltemp=0.070),
}
LOCAL_ARMS = ("h0_local", "h1_local", "h2_local")
POLICY_ARMS = ("cnn", "rnn", "fcp")

#: Kitchens swept in the first round, whose policies live in the original
#: training projects; everything else was trained in the newmaps projects.
#: outage_2 was swept in both rounds and is taken from the newmaps sweep, which
#: is the one the seven new kitchens were evaluated under.
FIRST_ROUND_LAYOUTS = (
    "split_0",
    "split_1",
    "split_2",
    "outage_0",
    "outage_1",
    "distance_switch_0",
    "distance_switch_1",
    "distance_switch_2",
    "recipe_switch_0",
    "recipe_switch_1",
    "recipe_switch_2",
)
TRAINING_PROJECTS = {
    ("cnn", True): ("overcooked-v3-ippo_train", "overcooked-v3-newmaps-ippo_cnn-cd"),
    ("rnn", True): ("overcooked-v3-ippo-rnn_train", "overcooked-v3-newmaps-ippo_rnn-cd"),
    ("fcp", True): ("overcooked-v3-fcp_train", "overcooked-v3-newmaps-fcp-cd"),
}
#: How each arm names itself in a run config, in the two rounds.
ALGORITHMS = {
    "cnn": ("IPPO", "IPPO-cdon"),
    "rnn": ("IPPO", "IPPO-cdon"),
    "fcp": ("FCP", "FCP-cdon"),
}
ARCHITECTURES = {"cnn": "cnn", "rnn": "rnn", "fcp": "rnn"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", default="split_0")
    parser.add_argument(
        "--human",
        default="br",
        help="The simulated human: " + ", ".join(list(PAPER_ARMS) + list(LOCAL_ARMS)),
    )
    parser.add_argument(
        "--partner",
        default="br",
        help="The other seat: a simulated human, or one of " + ", ".join(POLICY_ARMS),
    )
    parser.add_argument(
        "--dials",
        default="outputs/human_fit/split_0.json",
        help="Dials fit to our demonstrations, for the _local arms.",
    )
    parser.add_argument("--seeds", type=int, default=6, help="Partner training seeds.")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Episodes per seed and seat order."
    )
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=0, help="Evaluation RNG seed.")
    parser.add_argument(
        "--seat-only",
        action="store_true",
        help="Let the simulated human plan for its own seat alone, expecting "
        "nothing of its partner, instead of planning the pair and using its "
        "own half of the plan.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the learned policy's action instead of taking its mode.",
    )
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked"))
    parser.add_argument("--artifact-alias", default="final")
    parser.add_argument(
        "--artifact-dir", default="artifacts", help="Where checkpoints are downloaded."
    )
    parser.add_argument("--project", default="overcooked-v3-obp-human-matrix")
    parser.add_argument(
        "--group",
        default=None,
        help="W&B group; by default the kitchen, so the matrix groups by it.",
    )
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--output", default=None, help="Also append one JSON line here.")
    return parser.parse_args(argv)


def arm_dials(name, dials_path):
    """The three dials of one simulated human."""
    if name in PAPER_ARMS:
        return dict(PAPER_ARMS[name])
    if name not in LOCAL_ARMS:
        raise ValueError(f"{name} is not a simulated human")
    payload = json.loads(Path(dials_path).read_text())
    key = name.split("_")[0].upper()
    if key not in payload:
        raise ValueError(
            f"{dials_path} has no {key}; fit it with baselines.OBP.fit_human "
            "(H2 is added by the prob_wait sweep)"
        )
    entry = payload[key]
    return dict(
        prob_wait=float(entry["prob_wait"]),
        lltemp=float(entry["lltemp"]),
        hltemp=float(entry["hltemp"]),
    )


def policy_runs(arm, layout, seeds, entity, alias, artifact_dir):
    """Every training run of one policy arm on one kitchen, newest per seed."""
    import wandb

    from baselines.IPPO.eval_wandb_crossplay_matrix_overcooked_v3 import (
        discover_run_candidates,
        resolve_vmap_checkpoints,
    )

    project = TRAINING_PROJECTS[(arm, True)][0 if layout in FIRST_ROUND_LAYOUTS else 1]
    api = wandb.Api()
    candidates = discover_run_candidates(
        api.runs(f"{entity}/{project}", per_page=500),
        ALGORITHMS[arm],
        [layout],
        artifact_alias=alias,
    )
    # FCP's project holds the self-play population it was built from as well,
    # and both rounds name their runs differently, so the architecture is what
    # separates an RNN arm from a CNN one.
    candidates = [
        candidate
        for candidate in candidates
        if str(candidate.config.get("ARCHITECTURE")) == ARCHITECTURES[arm]
    ]
    candidates = sorted(candidates, key=lambda item: (item.seed is None, item.seed))
    if len(candidates) < seeds:
        raise SystemExit(
            f"{arm} on {layout}: found {len(candidates)} runs in {project}, need {seeds}"
        )
    loaded = []
    for candidate in candidates[:seeds]:
        directory = (
            Path(artifact_dir)
            / candidate.run.id
            / candidate.artifact.name.replace(":", "-")
        )
        path = Path(candidate.artifact.download(root=str(directory)))
        _, checkpoint = resolve_vmap_checkpoints(path)[0]
        loaded.append((candidate.seed, candidate.run.id, candidate.config, checkpoint))
    return loaded


def build_env(layout, max_steps, run_config=None):
    import jaxmarl
    from baselines.IPPO.eval_wandb_crossplay_overcooked_v3 import _observation_config

    observation = (
        _observation_config(run_config)
        if run_config is not None
        else dict(
            include_transition_countdown=True,
            include_layout_change_mask=True,
            transition_observer="both",
            transition_warning_steps=20,
        )
    )
    return jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=max_steps,
        random_agent_positions=False,
        **observation,
    )


def make_policy(env, run_config, seat, stochastic):
    """A one-seat controller wrapping a trained network."""
    from baselines.IPPO.eval_wandb_crossplay_overcooked_v3 import _policy_config
    from baselines.IPPO.ippo_overcooked_v3 import (
        ActorCriticCNN,
        ActorCriticRNN,
        ScannedRNN,
    )
    from jaxmarl.wrappers.baselines import load_params

    config = _policy_config(run_config)
    network = (ActorCriticRNN if config["ARCHITECTURE"] == "rnn" else ActorCriticCNN)(
        env.action_space(env.agents[seat]).n, config=config
    )

    def act(params, hidden, observation, done, key):
        hidden, pi, _ = network.apply(
            params,
            hidden,
            (observation[jnp.newaxis, jnp.newaxis], done[jnp.newaxis, jnp.newaxis]),
        )
        action = pi.sample(seed=key) if stochastic else pi.mode()
        return hidden, action.reshape(-1)[0]

    return (
        jax.jit(act),
        lambda: ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"]),
        load_params,
    )


def play_episode(env, controllers, key, max_steps, step=None):
    """One game; each seat is served by whatever controller was given for it."""
    step = step or jax.jit(env.step_env)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    for controller in controllers:
        controller["reset"](state)
    done = jnp.zeros((), dtype=bool)
    total = 0.0
    for _ in range(max_steps):
        key, step_key, *action_keys = jax.random.split(key, 2 + env.num_agents)
        actions = {}
        for seat, agent in enumerate(env.agents):
            actions[agent] = controllers[seat]["act"](
                state, obs[agent], done, action_keys[seat]
            )
        obs, state, reward, dones, _ = step(step_key, state, actions)
        total += float(reward[env.agents[0]])
        done = jnp.asarray(dones[env.agents[0]])
        if bool(dones["__all__"]):
            break
    return total


def planner_controller(env, dials, seat, seat_only=False):
    """One seat played by the heuristic planner at the given dials.

    The planner plans for both seats and only its own action is used. It is
    still one independent agent -- it sends nothing to the other seat and reads
    only the state everyone sees -- but it assumes the other cook follows the
    same rules, which is what lets it leave the onion it expects to be taken and
    hold its ground in a corridor instead of both of them stepping aside. Paired
    with itself that reproduces the joint planner exactly (240 on split_0, every
    seed); planning for its own seat alone drops it to 163 and makes it swing
    between 80 and 220.

    ``seat_only`` keeps the other reading, where the cook expects nothing of its
    partner and is always the one to give way.
    """
    from baselines.OBP.planner import GreedyPlanner

    planner = GreedyPlanner(env, seat=seat if seat_only else None, **dials)
    actions = jax.jit(planner.actions)
    box = {}

    def reset(state):
        box["carry"] = planner.initial_carry(state)

    def act(state, observation, done, key):
        carry, chosen = actions(box["carry"], state, key)
        box["carry"] = carry
        return chosen[seat]

    return dict(reset=reset, act=act)


def policy_controller(env, run_config, checkpoint, seat, stochastic):
    act_fn, init_hidden, load = make_policy(env, run_config, seat, stochastic)
    params = load(str(checkpoint))
    box = {}

    def reset(state):
        box["hidden"] = init_hidden()

    def act(state, observation, done, key):
        hidden, action = act_fn(params, box["hidden"], observation, done, key)
        box["hidden"] = hidden
        return action

    return dict(reset=reset, act=act)


def main(argv=None):
    from jaxmarl._env import load_project_env

    load_project_env()
    args = parse_args(argv)
    human_dials = arm_dials(args.human, args.dials)
    partner_is_policy = args.partner in POLICY_ARMS
    partner_dials = None if partner_is_policy else arm_dials(args.partner, args.dials)

    if partner_is_policy:
        runs = policy_runs(
            args.partner,
            args.layout,
            args.seeds,
            args.entity,
            args.artifact_alias,
            args.artifact_dir,
        )
    else:
        runs = [(seed, None, None, None) for seed in range(args.seeds)]

    records = []
    environments = {}
    for index, (seed, run_id, run_config, checkpoint) in enumerate(runs):
        # One environment per observation configuration, so the compiled step
        # and the compiled planner are reused across the partner's seeds.
        signature = json.dumps(run_config, sort_keys=True, default=str)
        if signature not in environments:
            built = build_env(args.layout, args.max_steps, run_config)
            environments[signature] = (built, jax.jit(built.step_env), {})
        env, step, cache = environments[signature]
        for human_seat in range(env.num_agents):
            partner_seat = env.num_agents - 1 - human_seat
            controllers = [None, None]
            slot = ("human", human_seat)
            if slot not in cache:
                cache[slot] = planner_controller(
                    env, human_dials, human_seat, args.seat_only
                )
            controllers[human_seat] = cache[slot]
            if partner_is_policy:
                controllers[partner_seat] = policy_controller(
                    env, run_config, checkpoint, partner_seat, args.stochastic
                )
            else:
                slot = ("partner", partner_seat)
                if slot not in cache:
                    cache[slot] = planner_controller(
                        env, partner_dials, partner_seat, args.seat_only
                    )
                controllers[partner_seat] = cache[slot]
            for episode in range(args.episodes):
                key = jax.random.PRNGKey(
                    args.seed + 1000 * index + 100 * human_seat + episode
                )
                total = play_episode(env, controllers, key, args.max_steps, step)
                records.append(
                    dict(
                        layout=args.layout,
                        human=args.human,
                        partner=args.partner,
                        partner_seed=seed,
                        partner_run=run_id,
                        human_seat=human_seat,
                        episode=episode,
                        ret=total,
                    )
                )
                print(
                    f"  seed {seed} human_seat {human_seat} episode {episode}: {total:g}",
                    flush=True,
                )

    returns = np.array([record["ret"] for record in records])
    summary = dict(
        layout=args.layout,
        human=args.human,
        partner=args.partner,
        games=len(records),
        return_mean=float(returns.mean()),
        return_std=float(returns.std()),
        return_seat0=float(np.mean([r["ret"] for r in records if r["human_seat"] == 0])),
        return_seat1=float(np.mean([r["ret"] for r in records if r["human_seat"] == 1])),
        **{f"human_{key}": value for key, value in human_dials.items()},
    )
    print(json.dumps(summary, indent=1))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(dict(summary, records=records)) + "\n")

    if args.wandb_mode != "disabled":
        import wandb

        # One run per cell, grouped by kitchen and typed by partner, so the
        # project page already reads as the matrix: group by layout, split by
        # job type, and every panel is one column of it.
        with wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.group or args.layout,
            mode=args.wandb_mode,
            name=f"{args.human}-x-{args.partner}",
            job_type=args.partner,
            tags=[f"human:{args.human}", f"partner:{args.partner}", args.layout],
            config=dict(vars(args), **{f"dial_{k}": v for k, v in human_dials.items()}),
        ) as run:
            run.log(summary)
            # The same number under a key naming the column, so one chart can
            # hold every partner without grouping being set up by hand.
            run.log({f"partner/{args.partner}": summary["return_mean"]})
            table = wandb.Table(
                columns=list(records[0].keys()),
                data=[list(record.values()) for record in records],
            )
            # Not "games": that name is already the count in the summary, and
            # a table logged over it makes the count unreadable.
            run.log({"game_log": table})
            run.summary["return_mean"] = summary["return_mean"]
            run.summary["human"] = args.human
            run.summary["partner"] = args.partner
            run.summary["layout"] = args.layout
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
