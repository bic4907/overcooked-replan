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
    Blackout    initiation: takes a fresh plate from a dispenser before the
                dispensers vanish -- an endpoint the quarter of probe states
                that begin holding a plate cannot satisfy by standing still;
                completion: sets that plate down again instead of carrying
                it through the boundary
    Inversion   initiation: takes from the station the change is about to
                put out of reach -- unreachable in the recipe-priority
                kitchen, six steps away instead of one in the wide one --
                at least once while it is still in reach; completion: still
                has what it took, in hand or in a pot, at the boundary

    python experiment/notice/probe_preparation.py --layout split_0 --arm rnn

Writes one JSON line per (seed, stream, state, arm) to outputs/notice/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common import (  # noqa: E402  (sibling module)
    BASELINE_NAMES,
    TRAINED_OBSERVER,
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
    station_distance,
    step_fn,
    walking_distance,
)
from jaxmarl.environments.overcooked_v3.common import DIR_TO_VEC, DynamicObject, StaticObject
from jaxmarl.environments.overcooked_v3.settings import POT_COOK_TIME

ONION = int(DynamicObject.ingredient(0))
PLATE = int(DynamicObject.PLATE)
INVENTORIES = {"empty": 0, "onion": ONION, "plate": PLATE, "soup": 0}  # soup filled per recipe
#: The jobs a cook can be standing at, in the order the partner cells cover them.
STATIONS = ("onion", "pot", "plate", "serve")

#: Where the informed agent stands in every probe state, chosen per kitchen
#: so that the job in front of it is the ingredient run. An endpoint only
#: means something when the agent would otherwise be doing something else:
#: from its own spawn the Blackout agent stands two steps from the plate
#: dispenser and the Inversion agent one step from the plates, so taking a
#: plate costs it nothing and the masked arm scores it anyway. Each cell
#: below is floor in both phases, one step from an ingredient pile, with
#: the plates three to nine steps away.
#:
#:   split_0     under the left room's onion pile, the pots it feeds behind it
#:   split_1     beside the left room's onion pile, same room as its two pots
#:   outage_0    in the big right room beside the onions, the plate nook across the floor
#:   outage_1    the right end of the lower corridor, the onions up its own side
#:   distance_0  in its own room's left lane, five steps from the onion the
#:               phase-A recipe favours and seven from the tomato phase B
#:               does, with the plates three away -- no cell of this kitchen
#:               touches a pile in both phases, because the change takes the
#:               piles out of the room altogether
#:   distance_1  under its own room's onion pile, the pot it feeds beside it
INFORMED_START = {
    "split_0": (2, 1),
    "split_1": (1, 4),
    "outage_0": (7, 1),
    "outage_1": (9, 3),
    "distance_0": (9, 2),
    "distance_1": (4, 3),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", default="split_0", choices=BENCHMARK_LAYOUTS)
    parser.add_argument("--arm", default="rnn", choices=("rnn", "rnn_a", "fcp", "cnn"))
    parser.add_argument("--seeds", type=int, default=10, help="Training seeds to use, at most.")
    parser.add_argument("--min-seeds", type=int, default=6, help="Fail below this many usable seeds.")
    parser.add_argument("--streams", type=int, default=3, help="Partner random streams per state.")
    parser.add_argument("--informed-seat", type=int, default=0)
    parser.add_argument(
        "--informed-xy", default=None,
        help="Where the informed agent stands, as x,y; by default the kitchen's designed cell.",
    )
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
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked"))
    parser.add_argument("--project", default="overcooked-v3-notice-probe")
    parser.add_argument("--group", default=None, help="W&B group; by default the kitchen.")
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# the hundred states
# --------------------------------------------------------------------------


def faced_stations(static, x, y):
    """Which jobs a cook standing on this cell can reach by facing a wall."""
    found = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < static.shape[1] and 0 <= ny < static.shape[0]:
            value = int(static[ny, nx])
            if value >= StaticObject.INGREDIENT_PILE_BASE:
                found.append("onion")
            elif value == StaticObject.POT:
                found.append("pot")
            elif value == StaticObject.PLATE_PILE:
                found.append("plate")
            elif value == StaticObject.GOAL:
                found.append("serve")
    return sorted(set(found), key=STATIONS.index)


def partner_cells(env, informed_xy, count=5):
    """Where the fixed partner stands: the floor, sampled evenly.

    Twenty steps before a boundary a cook is as often walking as standing
    at a station, and the informed agent is not supposed to know what its
    partner is about to do, so the candidates are every cell that is floor
    in both phases, taken at even quantiles of their sorted order. The
    station a cell happens to face is recorded, never selected on.

    The one exclusion is a configuration the kitchen cannot produce: where
    the floor is already divided before the change -- the Inversion pair,
    whose halves never join -- the two cooks are one per half for the whole
    episode, so the partner is drawn from its own half only.
    """
    static_a = np.asarray(env.phase_static_objects[0])
    static_b = np.asarray(env.phase_static_objects[1])
    rooms_now = phase_regions(env, 0)
    rooms_next = phase_regions(env, 1)
    informed_xy = tuple(int(v) for v in informed_xy)
    here = int(rooms_now[informed_xy[1], informed_xy[0]])
    divided = len({int(r) for r in np.unique(rooms_now) if r >= 0}) > 1

    floor = sorted(
        (x, y)
        for y in range(static_a.shape[0])
        for x in range(static_a.shape[1])
        if static_a[y, x] == StaticObject.EMPTY
        and static_b[y, x] == StaticObject.EMPTY
        and (x, y) != informed_xy
        and not (divided and int(rooms_now[y, x]) == here)
    )
    if len(floor) < count:
        raise SystemExit(f"only {len(floor)} cells available for the partner")
    quantiles = [index / (count - 1) for index in range(count)]
    picked = [floor[int(round(q * (len(floor) - 1)))] for q in quantiles]
    return [
        dict(
            xy=(x, y),
            stations=faced_stations(static_a, x, y),
            room=int(rooms_next[y, x]),
        )
        for x, y in picked
    ]


def probe_states(env, informed_seat, probe_step, informed_xy=None):
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
    if informed_xy is None:
        informed_xy = tuple(int(v) for v in start[informed_seat])
    informed_xy = tuple(int(v) for v in informed_xy)
    if static_a[informed_xy[1], informed_xy[0]] != StaticObject.EMPTY or (
        static_b[informed_xy[1], informed_xy[0]] != StaticObject.EMPTY
    ):
        raise SystemExit(f"the informed agent's cell {informed_xy} is not floor in both phases")

    partners = partner_cells(env, informed_xy)

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
    for pi, partner in enumerate(partners):
        cell = partner["xy"]
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
                        partner_station="+".join(partner["stations"]) or "floor",
                        partner_room=partner["room"],
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


# --------------------------------------------------------------------------
# how near the agent is to the job the next phase gives it
# --------------------------------------------------------------------------


def _spread(static, sources):
    """Steps over phase-A floor from a set of cells; inf where unreachable."""
    from collections import deque

    empty = static == StaticObject.EMPTY
    height, width = static.shape
    dist = np.full((height, width), np.inf)
    queue = deque()
    for x, y in sources:
        if empty[y, x] and dist[y, x] == np.inf:
            dist[y, x] = 0
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and empty[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((nx, ny))
    return dist


def approach_field(env, family, informed_xy):
    """Distance from every cell to what the informed agent needs after the change.

    A pickup is a rare event in twenty steps; walking towards the thing it
    will need is not, and it is the same preparation one stage earlier. The
    target is the scenario's own (Table 1 reads the act, this reads the
    approach):

        Partition   the floor that will belong to the other room
        Blackout    standing at the plate dispenser that is about to vanish
        Inversion   standing at the station the swapped role fetches from

    Distances are over the phase-A floor -- the kitchen the agent is
    walking in while the countdown runs -- and zero on a cell that already
    faces the station (or already lies in the other room).
    """
    static_a = np.asarray(env.phase_static_objects[0])
    static_b = np.asarray(env.phase_static_objects[1])
    height, width = static_a.shape

    if family == "split":
        rooms_next = phase_regions(env, 1)
        here = int(rooms_next[informed_xy[1], informed_xy[0]])
        sources = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if static_a[y, x] == StaticObject.EMPTY
            and int(rooms_next[y, x]) >= 0
            and int(rooms_next[y, x]) != here
        ]
        return _spread(static_a, sources), "other room"

    if family == "outage":
        piles = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if static_a[y, x] == StaticObject.PLATE_PILE
        ]
        vanishing = [(x, y) for x, y in piles if static_b[y, x] != StaticObject.PLATE_PILE]
        target = vanishing or piles
        name = "vanishing plate dispenser" if vanishing else "plate dispenser"
    else:
        _, role_b = role_switch(env, informed_xy)
        if role_b == "serve":
            target = [
                (x, y)
                for y in range(height)
                for x in range(width)
                if static_a[y, x] == StaticObject.PLATE_PILE
            ]
            name = "plate dispenser"
        else:
            target = [
                (x, y)
                for y in range(height)
                for x in range(width)
                if static_a[y, x] >= StaticObject.INGREDIENT_PILE_BASE
            ]
            name = "ingredient pile"

    sources = []
    for x, y in target:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                sources.append((nx, ny))
    return _spread(static_a, sources), name


def _recipe_counts(recipe):
    """How many of each ingredient a recipe asks for, by ingredient index."""
    counts, index = {}, 0
    recipe >>= 2
    while recipe > 0:
        counts[index] = recipe & 0x3
        recipe >>= 2
        index += 1
    return counts


def dispenser_shift(env, informed_xy):
    """What the recipe-priority Inversion does to this cook's ingredients.

    The 11x7 Inversion kitchen keeps its two rooms apart for the whole
    episode and moves the shared dispensers' one open side from one room to
    the other at every boundary, while the recipe's majority ingredient
    swaps with it. A cook therefore does not change role; it loses (or
    gains) the piles outright. Returns ``(kind, favoured)`` with kind
    ``'loses'``, ``'gains'`` or ``None`` on the older supply/serve kitchen,
    and ``favoured`` the ingredient phase B's recipe wants more of.
    """
    statics = [np.asarray(env.phase_static_objects[phase]) for phase in (0, 1)]
    codes = [int(c) for c in np.unique(statics[0]) if c >= StaticObject.INGREDIENT_PILE_BASE]
    reach = [
        np.isfinite(min((station_distance(static, informed_xy, code) for code in codes), default=np.inf))
        for static in statics
    ]
    if reach[0] and not reach[1]:
        kind = "loses"
    elif reach[1] and not reach[0]:
        kind = "gains"
    else:
        return None, None
    before = _recipe_counts(int(env.phase_recipes[0]))
    after = _recipe_counts(int(env.phase_recipes[1]))
    risen = [i for i in sorted(after) if after[i] > before.get(i, 0)]
    favoured = int(DynamicObject.ingredient(risen[0])) if risen else None
    return kind, favoured


def activating_source(env, informed_xy):
    """The ingredient supply the change is about to switch on for this cook.

    Both Inversion kitchens open an ingredient supply that the cook cannot
    use yet. The recipe-priority kitchen keeps the piles behind a wall that
    only its half of the map may open, and the boundary moves that opening
    to the other half; the wide kitchen swaps the pile and the serving
    window between the ends of the room, so a pile lights up where there
    was none. Either way the cook can see the countdown but not yet the
    supply.

    Returns ``None`` when the supply is not getting nearer -- the mirror
    cook, whose piles are going away -- and otherwise the pile codes, the
    walk it faces now, and the shortest walk it could be waiting with when
    the boundary lands. That last one is the endpoint's bar: the cell that
    opens onto a pile is often a wall until the change, so the mouth of it,
    not the cell itself, is as close as the cook can get in advance.
    """
    statics = [np.asarray(env.phase_static_objects[phase]) for phase in (0, 1)]
    codes = [int(c) for c in np.unique(statics[1]) if c >= StaticObject.INGREDIENT_PILE_BASE]
    if not codes:
        return None
    walk = [min(station_distance(static, informed_xy, code) for code in codes) for static in statics]
    if not (walk[1] < walk[0]):
        return None
    rooms_next = phase_regions(env, 1)
    home = int(rooms_next[informed_xy[1], informed_xy[0]])
    reach = np.full(statics[1].shape, np.inf)
    standable = (rooms_next == home) & (statics[0] == StaticObject.EMPTY)
    for y, x in np.argwhere(standable):
        reach[y, x] = min(station_distance(statics[1], (int(x), int(y)), code) for code in codes)
    best = float(np.min(reach[standable]))
    before = "unreachable" if not np.isfinite(walk[0]) else f"{walk[0]:.0f} steps"
    label = f"ingredient supply {before} -> {walk[1]:.0f} steps, {best:.0f} at best"
    return codes, reach, float(walk[0]), best, label


def handoff_counters(env):
    """Counters the two rooms share -- the only way an item crosses.

    A cook who is about to lose the ingredient piles can still leave what
    it has where the partner can pick it up, and a cook about to gain them
    can clear the counter first. Both are acts aimed at the other room, so
    the probe counts them separately from ordinary counter drops.
    """
    mask = np.zeros(np.asarray(env.phase_static_objects[0]).shape, dtype=bool)
    for phase in (0, 1):
        static = np.asarray(env.phase_static_objects[phase])
        rooms = phase_regions(env, phase)
        height, width = static.shape
        for y in range(height):
            for x in range(width):
                if static[y, x] != StaticObject.WALL:
                    continue
                touching = {
                    int(rooms[y + dy, x + dx])
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= x + dx < width and 0 <= y + dy < height and rooms[y + dy, x + dx] >= 0
                }
                if len(touching) > 1:
                    mask[y, x] = True
    return mask


def receding_station(env, informed_xy):
    """Every station the change puts further out of this cook's reach.

    Both Inversion kitchens take a station away, only to different degrees:
    the recipe-priority one moves the shared ingredient piles out of the
    room altogether -- both of them, and either will do -- while the wide
    one pushes its ingredient pile from one step to six. The preparation is
    the same either way, so the endpoint reads every station a pickup can
    come from whose walk gets longer, and scores taking from any of them.
    Returns their static codes and a label.
    """
    statics = [np.asarray(env.phase_static_objects[phase]) for phase in (0, 1)]
    present = sorted({int(c) for c in np.unique(statics[0])})
    codes, parts = [], []
    for code in present:
        if code != StaticObject.PLATE_PILE and code < StaticObject.INGREDIENT_PILE_BASE:
            continue  # nothing a cook can take is kept anywhere else
        before = station_distance(statics[0], informed_xy, code)
        after = station_distance(statics[1], informed_xy, code)
        if not np.isfinite(before) or after <= before:
            continue
        codes.append(code)
        name = (
            "plate dispenser" if code == StaticObject.PLATE_PILE
            else f"ingredient {code - int(StaticObject.INGREDIENT_PILE_BASE)}"
        )
        reach = "unreachable" if not np.isfinite(after) else f"{after:.0f} steps"
        parts.append(f"{name} {before:.0f} steps -> {reach}")
    if not codes:
        raise SystemExit(f"nothing recedes from {informed_xy}: no Inversion endpoint to read")
    return codes, "; ".join(parts)


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
    serve = (inv0[..., a] != 0) & (inv1[..., a] == 0) & (faced[..., a] == StaticObject.GOAL)
    shared = handoff_counters(env)
    faces_handoff = shared[fy[..., a], fx[..., a]] & (faced[..., a] == StaticObject.WALL)
    hand_over = (inv0[..., a] != 0) & (inv1[..., a] == 0) & faces_handoff
    hand_over_ingredient = hand_over & is_ing(inv0[..., a])
    take_across = (inv0[..., a] == 0) & (inv1[..., a] != 0) & faces_handoff

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
            taken = np.flatnonzero(fresh_plate_pick[s, :pre])
            acquired = taken.size > 0
            any_source = bool(plate_pick[s, :pre].any())
            holding = bool(inv[s, boundary, a] == PLATE)
            # Completion is putting the plate down again -- anywhere the
            # kitchen will keep it -- rather than still carrying it or
            # having spent it. A plate set aside survives the blackout;
            # one in the hand is one the cook cannot work with.
            stashed = bool(acquired and (plate_place[s, taken[0] + 1 : pre]).any())
            rows.append(
                dict(
                    initiation=acquired,
                    completion=stashed,
                    any_source_initiation=any_source,
                    fresh_plate_picks=int(fresh_plate_pick[s, :pre].sum()),
                    plate_picks=int(plate_pick[s, :pre].sum()),
                    plate_places=int(plate_place[s, :pre].sum()),
                    holds_plate_at_boundary=bool(holding),
                )
            )
    elif family == "distance" and activating_source(env, (int(x[0, 0, a]), int(y[0, 0, a]))) is not None:
        # Preparing for a supply that is about to switch on: be waiting
        # where it will open, then use it the moment it does. Standing
        # there buys nothing until the boundary, so it is an act only a
        # cook that knows the change is coming has a reason to perform.
        codes, reach, start_walk, best, label = activating_source(
            env, (int(x[0, 0, a]), int(y[0, 0, a]))
        )
        pile_pick = (inv0[..., a] == 0) & is_ing(inv1[..., a]) & (
            faced[..., a] >= StaticObject.INGREDIENT_PILE_BASE
        )
        at_boundary = reach[y[:, boundary, a], x[:, boundary, a]]
        for s in range(S):
            waiting = bool(at_boundary[s] <= best)
            took = bool(pile_pick[s, pre : pre + post].any())
            rows.append(
                dict(
                    initiation=waiting,
                    completion=bool(waiting and took),
                    closer=bool(at_boundary[s] < start_walk),
                    empty_handed=bool(inv[s, boundary, a] == 0),
                    takes_after=took,
                    post_pile_picks=int(pile_pick[s, pre : pre + post].sum()),
                    steps_to_the_source=float(at_boundary[s]) if np.isfinite(at_boundary[s]) else -1.0,
                    hands_over=int(hand_over[s, :pre].sum()),
                    takes_across=int(take_across[s, :pre].sum()),
                    switch=label,
                )
            )
    elif family == "distance":
        # The change takes a station away from this cook, outright in the
        # recipe-priority kitchen and by distance in the wide one. The
        # endpoint is the same in both: does it take from that station at
        # least once while the station is still within reach?
        codes, receding = receding_station(env, (int(x[0, 0, a]), int(y[0, 0, a])))
        shift, favoured = dispenser_shift(env, (int(x[0, 0, a]), int(y[0, 0, a])))
        take = (inv0[..., a] == 0) & (inv1[..., a] != 0) & np.isin(faced[..., a], codes)
        pot_insert = is_ing(inv0[..., a]) & (inv1[..., a] == 0) & (faced[..., a] == StaticObject.POT)
        soup_pick = (inv0[..., a] == PLATE) & ((inv1[..., a] & int(DynamicObject.COOKED)) != 0)
        counter_place = (inv0[..., a] != 0) & (inv1[..., a] == 0) & (faced[..., a] == StaticObject.WALL)
        # A pot reaching the full recipe is the act that commits the phase's
        # work; whether it happens before or after the change is the choice.
        recipe_now = int(env.phase_recipes[0])
        pots = np.argwhere(np.asarray(env.phase_static_objects[0]) == StaticObject.POT)
        filled = np.zeros((S, T1 - 1), dtype=bool)
        for py, px in pots:
            column = held[:, :, py, px]
            filled |= (column[:, :-1] != recipe_now) & (column[:, 1:] == recipe_now)
        favoured_take = take & (inv1[..., a] == favoured) if favoured is not None else None
        for s in range(S):
            held_at_boundary = int(inv[s, boundary, a])
            took = bool(take[s, :pre].any())
            kept = bool(held_at_boundary != 0 or pot_insert[s, :pre].any())
            row = dict(
                initiation=took,
                completion=bool(took and kept),
                receding=receding,
                pre_receding_picks=int(take[s, :pre].sum()),
                pre_onion_picks=int(onion_pick[s, :pre].sum()),
                pre_plate_picks=int(plate_pick[s, :pre].sum()),
                pot_inserts=int(pot_insert[s, :pre].sum()),
                pot_completions=int(filled[s, :pre].sum()),
                pre_soup_picks=int(soup_pick[s, :pre].sum()),
                serves=int(serve[s, :pre].sum()),
                counter_places=int(counter_place[s, :pre].sum()),
                hands_over=int(hand_over[s, :pre].sum()),
                hands_over_ingredient=int(hand_over_ingredient[s, :pre].sum()),
                takes_across=int(take_across[s, :pre].sum()),
                holds_at_boundary=bool(held_at_boundary != 0),
            )
            if favoured_take is not None:
                # Only the recipe-priority kitchen swaps its majority
                # ingredient, so only there is there a right one to take.
                favoured_took = bool(favoured_take[s, :pre].any())
                row.update(
                    priority_initiation=favoured_took,
                    priority_completion=bool(
                        favoured_took and (held_at_boundary == favoured or pot_insert[s, :pre].any())
                    ),
                    switch=f"{shift} the piles",
                )
            else:
                role_a, role_b = role_switch(env, (int(x[s, 0, a]), int(y[s, 0, a])))
                row["switch"] = f"{role_a}->{role_b}"
            rows.append(row)
    else:
        raise ValueError(family)

    # The approach reading of the same preparation: every scenario also asks
    # whether the agent closed the gap to what the next phase needs.
    field, _ = approach_field(env, family, (int(x[0, 0, a]), int(y[0, 0, a])))
    near = field[y[:, : boundary + 1, a], x[:, : boundary + 1, a]]  # (S, pre+1)
    start, at_boundary = near[:, 0], near[:, boundary]
    closest = near.min(axis=1)
    for row, d0, dB, dmin in zip(rows, start, at_boundary, closest):
        gap = d0 - dB
        row.update(
            approach=bool(dB < d0),
            approach_any=bool(dmin < d0),
            arrived=bool(dB == 0),
            approach_gap=float(gap) if np.isfinite(gap) else 0.0,
            target_start=float(d0) if np.isfinite(d0) else -1.0,
            target_boundary=float(dB) if np.isfinite(dB) else -1.0,
        )

    for row, label in zip(rows, labels):
        row.update(label)
    return rows


# --------------------------------------------------------------------------


def log_run(args, rows, meta, path):
    """One W&B run per cell, so the sweep page is the table of contents."""
    import wandb

    from summarize_probe import ENDPOINTS, summarize

    endpoints = list(ENDPOINTS) + sorted(
        {"role_initiation", "role_completion", "partner_stayed"} & set(rows[0])
    )
    summary = {}
    for endpoint in endpoints:
        stats = summarize(rows, endpoint)
        for key in ("OFF", "ON", "delta", "p_raw"):
            summary[f"{endpoint}/{key}"] = stats[key]
        summary[f"{endpoint}/pairs"] = stats["n_pairs"]
    with wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group or args.layout,
        job_type=args.arm,
        name=f"probe-{args.layout}-{args.arm}-seat{args.informed_seat}",
        mode=args.wandb_mode,
        tags=[args.layout, args.arm, f"seat{args.informed_seat}", rows[0]["family"]],
        config=meta,
    ) as run:
        run.log(summary)
        for key, value in summary.items():
            run.summary[key] = value
        table = wandb.Table(
            columns=["seed", "endpoint", "masked", "notice"],
            data=[
                [seed, endpoint, off, on]
                for endpoint in endpoints
                for seed, (off, on) in enumerate(summarize(rows, endpoint)["seed_rates"])
            ],
        )
        run.log({"seed_rates": table})
        artifact = wandb.Artifact(f"probe-{args.layout}-{args.arm}-seat{args.informed_seat}", type="probe")
        artifact.add_file(str(path))
        artifact.add_file(str(Path(path).with_suffix(".meta.json")))
        run.log_artifact(artifact)


def main(argv=None):
    from jaxmarl._env import load_project_env

    load_project_env()
    args = parse_args(argv)
    family = args.layout.split("_")[0]
    informed, partner = args.informed_seat, 1 - args.informed_seat
    trained_observer = TRAINED_OBSERVER[args.arm]
    if trained_observer != "both" and trained_observer != f"agent_{informed}":
        raise SystemExit(
            f"{args.arm} trained with the warning on {trained_observer}; masking seat {informed} "
            "would intervene on a channel it never learned to read"
        )
    policies = fetch_policies(args.arm, args.layout, args.seeds, args.min_seeds, artifact_dir=args.artifact_dir)
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

    informed_xy = (
        tuple(int(v) for v in args.informed_xy.split(","))
        if args.informed_xy
        else INFORMED_START.get(args.layout)
    )
    stacked, labels = probe_states(env_off, informed, probe_step, informed_xy)
    print(
        f"  informed agent at {informed_xy}; partner at "
        + ", ".join(
            f"{tuple(label['partner_xy'])} ({label['partner_station']})" for label in labels[::20]
        ),
        flush=True,
    )
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
        informed_xy=list(informed_xy) if informed_xy else None,
        partner_cells=[list(label["partner_xy"]) for label in labels[::20]],
        partner_stations=[label["partner_station"] for label in labels[::20]],
        runs=[(s, r) for s, r, _, _ in policies],
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rows)} rows to {out}")
    if args.wandb_mode != "disabled":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        log_run(args, rows, meta, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
