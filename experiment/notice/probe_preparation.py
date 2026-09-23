"""Section 6.2: does advance notice change what the informed agent does?

Paired rollouts from the same pre-transition state: one keeps the informed
agent's warning, the other masks it. Everything else is shared -- the
kitchen, the policy and its recurrent history, the alert-blind heuristic
partner and its random stream -- so any difference in the twenty steps
before the boundary is the notice.

The probe states cross five partner positions, five task-progress
conditions and four inventories of the informed agent, one hundred per
kitchen. Each is played from every training seed of the baseline under
several partner streams, and scored on the scenario's endpoint (Table 1):

    Partition   crosses into the other future room before closure and is
                separated from the partner at the boundary
    Blackout    initiation: picks a fresh plate from a dispenser before it
                vanishes; completion: also leaves it on a counter that still
                holds a plate at the boundary
    Inversion   initiation: picks up a plate before the change; completion:
                plate pickups outnumber ingredient pickups in the twenty
                steps after it as well

    python experiment/notice/probe_preparation.py --layout split_0 --arm rnn

Writes one JSON line per (seed, stream, state, arm) to outputs/notice/.
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
    FAMILY_OF,
    H0_DIALS,
    ROOT,
    WARNING_STEPS,
    build_env,
    check_policy,
    dump_jsonl,
    fetch_policies,
    phase_regions,
    planner_controller,
    policy_controller,
    role_switch,
    step_fn,
)
from jaxmarl.environments.overcooked_v3.common import DIR_TO_VEC, DynamicObject, StaticObject
from jaxmarl.environments.overcooked_v3.settings import POT_COOK_TIME

ONION = int(DynamicObject.ingredient(0))
PLATE = int(DynamicObject.PLATE)
INVENTORIES = {"empty": 0, "onion": ONION, "plate": PLATE, "soup": 0}  # soup filled per recipe
PARTNER_QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", default="split_0", choices=BENCHMARK_LAYOUTS)
    parser.add_argument("--arm", default="rnn", choices=("rnn", "fcp", "cnn"))
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--streams", type=int, default=3, help="Partner random streams per state.")
    parser.add_argument("--informed-seat", type=int, default=0)
    parser.add_argument(
        "--probe-step", type=int, default=None,
        help="Episode step of the probe state; default first boundary minus the warning.",
    )
    parser.add_argument("--post", type=int, default=20, help="Steps kept after the boundary.")
    parser.add_argument(
        "--partner-knowledge", default="static", choices=("static", "schedule", "full"),
        help="What the fixed H0 partner knows. It is alert-blind by default, so the probe reads "
        "what the informed agent does beside a partner that is not preparing: static (a kitchen "
        "that never changes, the default), schedule (the phases but no countdown), full (everything).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed of the streams.")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/notice"))
    parser.add_argument("--no-sanity", action="store_true", help="Skip the self-play sanity score.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# the hundred states
# --------------------------------------------------------------------------


def probe_states(env, informed_seat, probe_step):
    """Stack of probe states plus a label per state.

    Positions: the informed agent at its phase-A start, the partner at one
    of five floor cells spread across the kitchen that stay floor in phase
    B. Task progress: pots empty / first pot two short of full / first pot
    cooking / first pot ready / first pot ready and the second cooking (or,
    with one pot, nearly done). Inventory of the informed agent: empty,
    onion, plate, soup.
    """
    _, base = env.reset(jax.random.PRNGKey(0))
    base = jax.tree_util.tree_map(np.asarray, base)
    static_a = np.asarray(env.phase_static_objects[0])
    static_b = np.asarray(env.phase_static_objects[1])
    height, width = static_a.shape
    partner_seat = 1 - informed_seat
    start = np.stack([base.agents.pos.x, base.agents.pos.y], axis=-1)  # (2, 2) x,y
    informed_xy = tuple(int(v) for v in start[informed_seat])

    floor = [
        (x, y)
        for x in range(width)
        for y in range(height)
        if static_a[y, x] == StaticObject.EMPTY
        and static_b[y, x] == StaticObject.EMPTY
        and (x, y) != informed_xy
    ]
    floor.sort()
    partner_cells = [floor[int(round(q * (len(floor) - 1)))] for q in PARTNER_QUANTILES]

    pots = sorted(
        [(x, y) for y in range(height) for x in range(width) if static_a[y, x] == StaticObject.POT],
        key=lambda c: (c[1], c[0]),
    )
    recipe = int(base.recipe)
    two_short = int(DynamicObject.ingredient(0)) * 2  # recipe is three onions
    cooked = recipe | int(DynamicObject.COOKED)
    half = POT_COOK_TIME // 2

    def progress(name):
        held = np.zeros((height, width), dtype=np.int32)
        timer = np.zeros((height, width), dtype=np.int32)
        p0 = pots[0]
        p1 = pots[1] if len(pots) > 1 else None
        if name == "empty":
            pass
        elif name == "partial":
            held[p0[1], p0[0]] = two_short
        elif name == "cooking":
            held[p0[1], p0[0]] = recipe
            timer[p0[1], p0[0]] = half
        elif name == "ready":
            held[p0[1], p0[0]] = cooked
        elif name == "ready+cooking":
            held[p0[1], p0[0]] = cooked
            if p1 is not None:
                held[p1[1], p1[0]] = recipe
                timer[p1[1], p1[0]] = half
            else:
                held[p0[1], p0[0]] = recipe
                timer[p0[1], p0[0]] = 3
        return held, timer

    inventories = dict(INVENTORIES)
    inventories["soup"] = recipe | int(DynamicObject.COOKED) | int(DynamicObject.PLATE)

    states, labels = [], []
    for pi, cell in enumerate(partner_cells):
        for progress_name in ("empty", "partial", "cooking", "ready", "ready+cooking"):
            held, timer = progress(progress_name)
            for inv_name, inv in inventories.items():
                xs = np.zeros(2, dtype=np.int32)
                ys = np.zeros(2, dtype=np.int32)
                xs[informed_seat], ys[informed_seat] = informed_xy
                xs[partner_seat], ys[partner_seat] = cell
                inventory = np.zeros(2, dtype=np.int32)
                inventory[informed_seat] = inv
                grid = np.array(base.grid)
                grid[:, :, 1] = held
                grid[:, :, 2] = timer
                state = base.replace(
                    agents=base.agents.replace(
                        pos=base.agents.pos.replace(x=xs, y=ys),
                        dir=np.zeros(2, dtype=np.int32),
                        inventory=inventory,
                    ),
                    grid=grid,
                    step=np.asarray(probe_step, dtype=np.int32),
                )
                states.append(state)
                labels.append(
                    dict(
                        partner_cell=pi,
                        partner_xy=list(cell),
                        progress=progress_name,
                        inventory=inv_name,
                    )
                )
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *states)
    stacked = jax.vmap(env._set_transition_awareness)(stacked)
    return stacked, labels


# --------------------------------------------------------------------------
# play
# --------------------------------------------------------------------------


def make_warmup(env, controllers, steps):
    body = step_fn(env, controllers)

    def warmup(key, params):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        seats = tuple((controllers[s]["init"](state), params[s]) for s in range(env.num_agents))
        carry, _ = jax.lax.scan(
            body, (key, obs, state, jnp.zeros((), bool), seats), None, length=steps
        )
        return carry[4]  # seats: hidden states / planner carries

    return jax.jit(warmup)


def make_probe(env, controllers, steps, informed_seat):
    body = step_fn(env, controllers)
    partner_seat = 1 - informed_seat

    def probe(key, params, hidden, state):
        obs = env.get_obs(state)
        seats = [None, None]
        seats[informed_seat] = (hidden, params[informed_seat])
        seats[partner_seat] = (controllers[partner_seat]["init"](state), params[partner_seat])
        carry = (key, obs, state, jnp.zeros((), bool), tuple(seats))
        _, (rewards, events, states) = jax.lax.scan(body, carry, None, length=steps)
        return rewards, events, states

    return jax.jit(jax.vmap(probe, in_axes=(None, None, None, 0)))


def make_selfplay(env, controllers, steps):
    body = step_fn(env, controllers)

    def play(key, params):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        seats = tuple((controllers[s]["init"](state), params[s]) for s in range(env.num_agents))
        _, (rewards, _, _) = jax.lax.scan(
            body, (key, obs, state, jnp.zeros((), bool), seats), None, length=steps
        )
        return jnp.sum(rewards)

    return jax.jit(play)


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


def prepend(initial, states):
    """(S, T+1, ...) trajectories with the probe state in front."""
    return jax.tree_util.tree_map(
        lambda a, b: np.concatenate([np.asarray(a)[:, None], np.asarray(b)], axis=1),
        initial,
        states,
    )


def faced_cells(x, y, d, height, width):
    vec = np.asarray(DIR_TO_VEC)[d]  # (..., 2)
    fx = np.clip(x + vec[..., 0], 0, width - 1)
    fy = np.clip(y + vec[..., 1], 0, height - 1)
    return fx, fy


def score_endpoints(env, family, traj, labels, informed_seat, pre, post):
    """One dict of endpoint flags per probe state."""
    partner_seat = 1 - informed_seat
    x = traj.agents.pos.x  # (S, T+1, 2)
    y = traj.agents.pos.y
    d = traj.agents.dir
    inv = traj.agents.inventory
    static = traj.grid[..., 0]  # (S, T+1, H, W)
    held = traj.grid[..., 1]
    S, T1 = inv.shape[:2]
    height, width = static.shape[2:]
    boundary = pre  # index of the state right after the boundary step

    fx, fy = faced_cells(x[:, :-1], y[:, :-1], d[:, :-1], height, width)  # faced when acting
    faced = static[
        np.arange(S)[:, None, None], np.arange(T1 - 1)[None, :, None], fy, fx
    ]  # (S, T, 2)
    inv0, inv1 = inv[:, :-1], inv[:, 1:]
    a = informed_seat
    is_ing = lambda v: ((v >> 2) != 0) & ((v & PLATE) == 0)  # noqa: E731
    plate_pick = (inv0[..., a] == 0) & (inv1[..., a] == PLATE)  # (S, T)
    fresh_plate_pick = plate_pick & (faced[..., a] == StaticObject.PLATE_PILE)
    onion_pick = (inv0[..., a] == 0) & is_ing(inv1[..., a])
    plate_place = (inv0[..., a] == PLATE) & (inv1[..., a] == 0) & (faced[..., a] == StaticObject.WALL)

    rows = []
    if family == "split":
        rooms = phase_regions(env, 1)
        for s in range(S):
            a_init = rooms[y[s, 0, a], x[s, 0, a]]
            a_end = rooms[y[s, boundary, a], x[s, boundary, a]]
            b_init = rooms[y[s, 0, partner_seat], x[s, 0, partner_seat]]
            b_end = rooms[y[s, boundary, partner_seat], x[s, boundary, partner_seat]]
            crossed = bool(a_end != a_init and a_end >= 0)
            separated = bool(a_end != b_end)
            rows.append(
                dict(
                    initiation=bool(crossed and separated),
                    completion=bool(crossed and separated),
                    crossed=crossed,
                    separated=separated,
                    partner_stayed=bool(b_end == b_init),
                    same_room_initially=bool(a_init == b_init),
                )
            )
    elif family == "outage":
        for s in range(S):
            picks = np.flatnonzero(fresh_plate_pick[s, :pre])
            acquired = picks.size > 0
            stored = False
            if acquired:
                for t in np.flatnonzero(plate_place[s, :pre]):
                    if t <= picks[0]:
                        continue
                    cx, cy = fx[s, t, a], fy[s, t, a]
                    if held[s, boundary, cy, cx] == PLATE:
                        stored = True
                        break
            rows.append(
                dict(
                    initiation=bool(acquired),
                    completion=bool(acquired and stored),
                    fresh_plate_picks=int(picks.size),
                    holds_plate_at_boundary=bool(inv[s, boundary, a] == PLATE),
                )
            )
    elif family == "distance":
        # The paper's endpoint reads the supplier's switch to plate work.
        # Which switch the informed seat faces is a matter of where it
        # stands, so it is looked up per state and the role-neutral
        # endpoint counts pickups of whichever item its new role fetches.
        for s in range(S):
            role_a, role_b = role_switch(env, (int(x[s, 0, a]), int(y[s, 0, a])))
            new_item_pick = plate_pick if role_b == "serve" else onion_pick
            old_item_pick = onion_pick if role_b == "serve" else plate_pick
            initiated = bool(plate_pick[s, :pre].any())
            post_plate = int(plate_pick[s, pre : pre + post].sum())
            post_onion = int(onion_pick[s, pre : pre + post].sum())
            role_initiated = bool(new_item_pick[s, :pre].any())
            post_new = int(new_item_pick[s, pre : pre + post].sum())
            post_old = int(old_item_pick[s, pre : pre + post].sum())
            rows.append(
                dict(
                    initiation=initiated,
                    completion=bool(initiated and post_plate > post_onion),
                    role_initiation=role_initiated,
                    role_completion=bool(role_initiated and post_new > post_old),
                    switch=f"{role_a}->{role_b}",
                    pre_plate_picks=int(plate_pick[s, :pre].sum()),
                    pre_onion_picks=int(onion_pick[s, :pre].sum()),
                    post_plate_picks=post_plate,
                    post_onion_picks=post_onion,
                )
            )
    else:
        raise ValueError(family)
    for row, label in zip(rows, labels):
        row.update(label)
    return rows


# --------------------------------------------------------------------------


def main(argv=None):
    from jaxmarl._env import load_project_env

    load_project_env()
    args = parse_args(argv)
    family = args.layout.split("_")[0]
    informed, partner = args.informed_seat, 1 - args.informed_seat
    policies = fetch_policies(args.arm, args.layout, args.seeds, artifact_dir=args.artifact_dir)
    config = policies[0][2]

    env_off = build_env(args.layout, "none", run_config=config)
    env_on = build_env(args.layout, f"agent_{informed}", run_config=config)
    env_both = build_env(args.layout, "both", run_config=config)
    first_boundary = int(env_off.phase_durations[0])
    probe_step = args.probe_step if args.probe_step is not None else first_boundary - WARNING_STEPS
    pre = first_boundary - probe_step
    steps = pre + args.post
    for _, _, cfg, params in policies:
        check_policy(env_off, cfg, params, informed)

    if not args.no_sanity:
        # A stale checkpoint of a same-size kitchen loads and scores zero.
        # FCP's Blackout best responses never fetch a plate and so score
        # zero with themselves while still cooking beside the heuristic
        # cook, so a seed passes on either partner.
        players = [policy_controller(env_both, config, policies[0][3], s, False) for s in range(2)]
        play = make_selfplay(env_both, players, env_both.max_steps)
        scores = [float(play(jax.random.PRNGKey(1), (p, p))) for _, _, _, p in policies]
        print(f"  greedy self-play sanity: {scores}", flush=True)
        with_cook = [None, None]
        with_cook[informed] = policy_controller(env_both, config, policies[0][3], informed, False)
        with_cook[partner] = planner_controller(env_both, H0_DIALS, partner, args.partner_knowledge)
        play = make_selfplay(env_both, with_cook, env_both.max_steps)
        cook_scores = [
            float(play(jax.random.PRNGKey(1), (p, None) if informed == 0 else (None, p)))
            for _, _, _, p in policies
        ]
        print(f"  greedy with-H0 sanity: {cook_scores}", flush=True)
        if max(scores) <= 0 and max(cook_scores) <= 0:
            raise SystemExit("every seed scores zero with itself and with H0: wrong kitchen?")

    stacked, labels = probe_states(env_off, informed, probe_step)
    initial = jax.tree_util.tree_map(np.asarray, stacked)
    n_states = len(labels)

    cooks = {}
    probes = {}
    for arm, env in (("OFF", env_off), ("ON", env_on)):
        controllers = [None, None]
        controllers[informed] = policy_controller(env, config, policies[0][3], informed, False)
        controllers[partner] = planner_controller(env, H0_DIALS, partner, args.partner_knowledge)
        cooks[arm] = controllers
        probes[arm] = make_probe(env, controllers, steps, informed)
    warmup = make_warmup(env_off, cooks["OFF"], probe_step)

    rows = []
    started = time.time()
    for seed, run_id, cfg, params in policies:
        for stream in range(args.streams):
            key = jax.random.PRNGKey(args.seed + 1000 * seed + stream)
            warm_key, probe_key = jax.random.split(key)
            seats = warmup(warm_key, (params, None) if informed == 0 else (None, params))
            hidden = seats[informed][0]
            for arm in ("OFF", "ON"):
                pair = (params, None) if informed == 0 else (None, params)
                rewards, events, states = probes[arm](probe_key, pair, hidden, stacked)
                traj = prepend(initial, states)
                scored = score_endpoints(env_off, family, traj, labels, informed, pre, args.post)
                for index, row in enumerate(scored):
                    rows.append(
                        dict(
                            layout=args.layout,
                            family=family,
                            scenario=FAMILY_OF[family],
                            arm=args.arm,
                            algorithm=BASELINE_NAMES[args.arm],
                            seed=seed,
                            run_id=run_id,
                            stream=stream,
                            state=index,
                            notice=arm,
                            reward=float(np.asarray(rewards)[index].sum()),
                            **row,
                        )
                    )
            done_off = np.mean(
                [
                    r["initiation"]
                    for r in rows
                    if r["seed"] == seed and r["stream"] == stream and r["notice"] == "OFF"
                ]
            )
            done_on = np.mean(
                [
                    r["initiation"]
                    for r in rows
                    if r["seed"] == seed and r["stream"] == stream and r["notice"] == "ON"
                ]
            )
            print(
                f"  seed {seed} stream {stream}: initiation OFF {done_off:.3f} ON {done_on:.3f}"
                f"  ({time.time() - started:.0f}s)",
                flush=True,
            )

    out = Path(args.out_dir) / f"probe_{args.layout}_{args.arm}_seat{informed}.jsonl"
    dump_jsonl(out, rows)
    meta = dict(
        vars(args),
        probe_step=probe_step,
        pre=pre,
        n_states=n_states,
        family=family,
        runs=[(s, r) for s, r, _, _ in policies],
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
