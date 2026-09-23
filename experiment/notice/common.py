"""Shared pieces of the advance-notice experiments (paper Sections 6.2, 6.3).

Both experiments pair rollouts that differ only in whether one agent's
transition warning is visible. The environment already gates the warning
per seat (``transition_observer``), so a pair is two environments built on
the same kitchen with a different observer, stepped from the same state
with the same keys.

What lives here: fetching the six training seeds of a recurrent baseline
from W&B, the observer-gated environments, controllers for a learned seat
and for the fixed alert-blind heuristic cook, and the subtask events read
off a state trajectory (ingredient pickup, pot insertion, plate pickup,
soup pickup) that both experiments score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.overcooked_v3.common import (  # noqa: E402
    DIR_TO_VEC,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.dynamic_layouts import (  # noqa: E402
    DynamicLayout,
    DynamicLayoutPhase,
    dynamic_layouts,
)
from jaxmarl.environments.overcooked_v3.utils import compute_enclosed_spaces  # noqa: E402

BENCHMARK_LAYOUTS = ("split_0", "split_1", "outage_0", "outage_1", "distance_0", "distance_1")
#: Internal kitchen family -> the paper's scenario name.
FAMILY_OF = {"split": "Partition", "outage": "Blackout", "distance": "Inversion"}
#: Internal arm -> the paper's baseline name.
BASELINE_NAMES = {"rnn": "IPPO-RNN", "fcp": "FCP", "cnn": "IPPO-0"}
#: The benchmark's own training projects come first. The 0915 project's
#: distance_0 recurrent checkpoints are of the kitchen before it was widened
#: to 13x6, so that one kitchen falls through to the 0920 retraining -- the
#: same split the paper's alert-group figure makes with its OVERRIDES.
TRAINING_PROJECTS = {
    "cnn": ("overcooked-v3-ippo-cnn-75step-plate-0915_train",),
    "rnn": ("overcooked-v3-ippo-rnn-75step-plate-0915_train", "overcooked-v3-ippo-rnn-0920_train"),
    "fcp": ("overcooked-v3-fcp-75step-plate-0915_train",),
}
ALGORITHMS = {"cnn": ("IPPO-CNN", "IPPO"), "rnn": ("IPPO-RNN", "IPPO"), "fcp": ("FCP",)}
ARCHITECTURES = {"cnn": "cnn", "rnn": "rnn", "fcp": "rnn"}
ENTITY = "cilab-overcooked"

EVENT_NAMES = ("ingredient_acquire", "pot_insert", "plate_acquire", "soup_acquire")
#: Steps of warning before a boundary, the benchmark default.
WARNING_STEPS = 20


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------


def _run_layout(run):
    config = run.config
    return (config.get("ENV_KWARGS") or {}).get("layout") or config.get("layout")


def _run_algorithm(run):
    config = run.config
    return str(config.get("ALGORITHM") or config.get("algorithm") or "")


def _run_seed(run):
    config = run.config
    seed = config.get("SEED")
    return config.get("seed") if seed is None else seed


def _negated(created_at):
    """Sort key putting the newest run first within one project."""
    return tuple(-ord(c) for c in created_at)


def _named_final_checkpoint(run, artifact_dir):
    """The checkpoint a run uploaded under 'latest' rather than 'final'.

    The 0920 FCP retraining aliased its final checkpoint only 'latest'; the
    artifact's own name still says what it is.
    """
    from baselines.IPPO.eval_wandb_crossplay_matrix_overcooked_v3 import resolve_vmap_checkpoints

    named = [a for a in run.logged_artifacts() if a.type == "checkpoint" and "final" in a.name]
    if len(named) != 1:
        raise FileNotFoundError(f"{run.path}: {len(named)} final-named checkpoint artifacts")
    artifact = named[0]
    directory = Path(artifact_dir) / run.id / artifact.name.replace(":", "-")
    path = Path(artifact.download(root=str(directory)))
    return run, resolve_vmap_checkpoints(path)[0][1]


def fetch_policies(arm, layout, seeds=6, entity=ENTITY, alias="final", artifact_dir="artifacts"):
    """One training run per seed of an arm, on the kitchen as it is now.

    Returns ``[(seed, run_id, run_config, params)]`` in seed order. The
    benchmark project is preferred and the retraining used only where the
    benchmark has no checkpoint of the current kitchen; within a project the
    newest run wins, since these projects hold both the first sweep and the
    re-sweeps after a kitchen was redesigned. A checkpoint whose
    convolutional trunk was built for another grid size is skipped. A stale
    checkpoint of a same-size kitchen would pass this and is caught by the
    caller's self-play sanity score.
    """
    import wandb

    from baselines.planner.eval_planner_partner import _checkpoint_of
    from jaxmarl.wrappers.baselines import load_params

    api = wandb.Api()
    height, width = np.asarray(dynamic_layouts[layout].phases[0].layout.static_objects).shape
    expected_in = height * width * 32
    candidates = {}
    for order, project in enumerate(TRAINING_PROJECTS[arm]):
        for run in api.runs(f"{entity}/{project}", per_page=500):
            if run.state != "finished" or _run_layout(run) != layout:
                continue
            if _run_algorithm(run) not in ALGORITHMS[arm]:
                continue
            arch = run.config.get("ARCHITECTURE")
            if arch is not None and str(arch).lower() != ARCHITECTURES[arm]:
                continue
            seed = _run_seed(run)
            if seed is None or seed >= seeds:
                continue
            candidates.setdefault(seed, []).append((order, run))
    loaded = []
    for seed in range(seeds):
        chosen = None
        ordered = sorted(
            candidates.get(seed, []), key=lambda item: (item[0], _negated(item[1].created_at))
        )
        for _, run in ordered:
            try:
                source, checkpoint = _checkpoint_of(run, alias, artifact_dir, entity)
            except FileNotFoundError:
                source, checkpoint = _named_final_checkpoint(run, artifact_dir)
            params = load_params(str(checkpoint))
            dense_in = int(params["params"]["CNN_0"]["Dense_0"]["kernel"].shape[0])
            if dense_in != expected_in:
                print(
                    f"  {arm} {layout} seed {seed}: {run.project}/{run.id} skipped, trunk for "
                    f"{dense_in // 32} cells, kitchen has {height * width}",
                    flush=True,
                )
                continue
            chosen = (seed, run.id, dict(source.config), params)
            print(
                f"  {arm} {layout} seed {seed}: {run.project}/{run.id} ({run.created_at[:10]})",
                flush=True,
            )
            break
        if chosen is None:
            raise SystemExit(f"{arm} on {layout}: no checkpoint of seed {seed} fits the current kitchen")
        loaded.append(chosen)
    return loaded


def build_env(layout, observer, max_steps=450, run_config=None, warning_steps=WARNING_STEPS):
    """The kitchen with one seat's warning gated by ``observer``.

    The observation flags come from the training run so the policy sees the
    layout it was trained on; only the observer is overridden.
    """
    import jaxmarl
    from baselines.IPPO.eval_wandb_crossplay_overcooked_v3 import _observation_config

    observation = (
        _observation_config(run_config)
        if run_config is not None
        else dict(
            include_transition_countdown=True,
            include_layout_change_mask=True,
            transition_warning_steps=warning_steps,
        )
    )
    observation["transition_observer"] = observer
    return jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=max_steps,
        random_agent_positions=False,
        **observation,
    )


def static_env(env, phase=0, max_steps=450):
    """A kitchen frozen in one phase of ``env``'s layout.

    The alert-blind heuristic cook plans on this: it neither sees a
    countdown nor knows that another kitchen is coming, so nothing it does
    before a boundary can be preparation.
    """
    import jaxmarl

    source = env.dynamic_layout.phases[phase]
    frozen = DynamicLayout(
        phases=(
            DynamicLayoutPhase(
                layout=source.layout,
                agent_positions=source.agent_positions,
                steps=10**6,
                name=f"{source.name}-frozen",
                recipe=source.recipe,
            ),
        )
    )
    return jaxmarl.make(
        "overcooked_v3",
        layout=frozen,
        max_steps=max_steps,
        random_agent_positions=False,
        include_transition_countdown=env.include_transition_countdown,
        include_layout_change_mask=env.include_layout_change_mask,
        transition_observer="none",
        transition_warning_steps=env.transition_warning_steps,
    )


def policy_config(run_config, params):
    """Network hyper-parameters, checked against the checkpoint itself.

    The recurrent 0915 project holds reference runs whose config names the
    layout and the seed but no architecture; the shared helper reads that
    off the algorithm name, and the weights have the last word.
    """
    from baselines.IPPO.eval_wandb_crossplay_overcooked_v3 import (
        _policy_config,
        assert_architecture_matches,
    )

    config = _policy_config(run_config)
    leaves = params["params"]
    config["ARCHITECTURE"] = "rnn" if "ScannedRNN_0" in leaves else "cnn"
    assert_architecture_matches(config, params)
    if "LayerNorm_0" in leaves:
        config["GRU_HIDDEN_DIM"] = int(leaves["LayerNorm_0"]["scale"].shape[0])
    config["FC_DIM_SIZE"] = int(leaves["Dense_0"]["kernel"].shape[1])
    return config


def make_policy(env, run_config, seat, stochastic, params):
    """``(act, init_hidden)`` for a trained network in ``seat``."""
    from baselines.IPPO.ippo_overcooked_v3 import ActorCriticCNN, ActorCriticRNN, ScannedRNN

    config = policy_config(run_config, params)
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

    return act, (lambda: ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"]))


def check_policy(env, run_config, params, seat=0):
    """Fail loudly when a checkpoint was trained on a different kitchen.

    A checkpoint of another kitchen of the same size loads and plays
    silently for zero; one of another size fails only when applied. Both
    are caught by applying the network once to a real observation and, for
    the same-size case, by the caller's self-play sanity score.
    """
    act, init = make_policy(env, run_config, seat, False, params)
    obs, state = env.reset(jax.random.PRNGKey(0))
    act(params, init(), obs[env.agents[seat]], jnp.zeros((), bool), jax.random.PRNGKey(0))


# --------------------------------------------------------------------------
# controllers
# --------------------------------------------------------------------------


def policy_controller(env, run_config, params, seat, stochastic):
    """``params`` here only shapes the network; the weights are passed at play."""
    act, init_hidden = make_policy(env, run_config, seat, stochastic, params)
    return dict(
        init=lambda state: init_hidden(),
        act=lambda params, hidden, state, observation, done, key: act(
            params, hidden, observation, done, key
        ),
        params=params,
    )


BLIND_COUNTDOWN = 10**6


def blind_state(state):
    """What an alert-blind cook is shown: the same kitchen, no countdown."""
    return state.replace(
        layout_index=jnp.zeros_like(state.layout_index),
        steps_until_layout_change=jnp.full_like(state.steps_until_layout_change, BLIND_COUNTDOWN),
        layout_change_mask=jnp.zeros_like(state.layout_change_mask),
    )


def planner_controller(env, dials, seat, knowledge="static"):
    """The heuristic cook H0 in ``seat``.

    H0 is the fixed reference partner of both arms -- same rule, same
    information, same random stream -- so the only thing that differs
    within a pair is the learned agent's own access to the warning. It is
    held alert-blind: the question the probe asks is what the informed
    agent does beside a partner that is not preparing, and a partner that
    prepares on its own answers it for the pair.

    ``'static'`` plans on a frozen copy of the kitchen it started in, so it
    neither counts down nor knows another kitchen is coming; ``'schedule'``
    keeps the phases and hides only the countdown; ``'full'`` sees
    everything and is kept for contrast.
    """
    from baselines.planner.planner import GreedyPlanner

    if knowledge == "static":
        planner = GreedyPlanner(static_env(env, 0, env.max_steps), **dials)
        seen = blind_state
    elif knowledge == "schedule":
        planner = GreedyPlanner(env, **dials)
        seen = lambda state: state.replace(  # noqa: E731
            steps_until_layout_change=jnp.full_like(
                state.steps_until_layout_change, BLIND_COUNTDOWN
            ),
            layout_change_mask=jnp.zeros_like(state.layout_change_mask),
        )
    elif knowledge == "full":
        planner = GreedyPlanner(env, **dials)
        seen = lambda state: state  # noqa: E731
    else:
        raise ValueError("knowledge is 'static', 'schedule' or 'full'")

    def act(params, carry, state, observation, done, key):
        carry, chosen = planner.actions(carry, seen(state), key)
        return carry, chosen[seat]

    return dict(init=lambda state: planner.initial_carry(seen(state)), act=act, params=None)


H0_DIALS = dict(prob_wait=0.5, lltemp=0.0, hltemp=0.0)


# --------------------------------------------------------------------------
# one step of play, shared by both experiments
# --------------------------------------------------------------------------


def step_fn(env, controllers):
    """``carry -> carry, (reward, events, state)`` for one joint step.

    ``events[agent, k]`` flags the k-th subtask of :data:`EVENT_NAMES`
    completed by that agent on this step, read off its inventory before and
    after the step and the cell it faced when acting.
    """
    agents = env.agents

    def body(carry, _):
        key, obs, state, done, seats = carry
        key, step_key, *action_keys = jax.random.split(key, 2 + env.num_agents)
        taken = []
        actions = {}
        for seat, agent in enumerate(agents):
            seat_state, params = seats[seat]
            seat_state, action = controllers[seat]["act"](
                params, seat_state, state, obs[agent], done, action_keys[seat]
            )
            taken.append((seat_state, params))
            actions[agent] = action
        before = state
        obs, state, reward, dones, _ = env.step_env(step_key, state, actions)
        events = subtask_events(before, state)
        done = dones[agents[0]]
        return (key, obs, state, done, tuple(taken)), (reward[agents[0]], events, state)

    return body


def faced_static(state):
    """Static object in front of each agent, clipped to the grid."""
    fwd = state.agents.pos.move(state.agents.dir)
    x = jnp.clip(fwd.x, 0, state.grid.shape[1] - 1)
    y = jnp.clip(fwd.y, 0, state.grid.shape[0] - 1)
    return state.grid[y, x, 0]


def subtask_events(before, after):
    """(num_agents, 4) flags of the subtasks completed on one step."""
    inv0 = before.agents.inventory
    inv1 = after.agents.inventory
    faced = faced_static(before)
    is_ing = DynamicObject.is_ingredient
    ingredient_acquire = (inv0 == DynamicObject.EMPTY) & is_ing(inv1)
    pot_insert = is_ing(inv0) & (inv1 == DynamicObject.EMPTY) & (faced == StaticObject.POT)
    plate_acquire = (inv0 == DynamicObject.EMPTY) & (inv1 == DynamicObject.PLATE)
    soup_acquire = (inv0 == DynamicObject.PLATE) & ((inv1 & DynamicObject.COOKED) != 0)
    return jnp.stack([ingredient_acquire, pot_insert, plate_acquire, soup_acquire], axis=-1)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def jensen_shannon(p, q):
    """Base-2 Jensen-Shannon divergence of two count vectors, in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))


def kitchen_of_phase(env):
    """Phase index -> kitchen id, so an A-B-A-B-A-A schedule has boundaries
    only where the kitchen actually changes (its last A->A step still shows
    a countdown, with nothing behind it)."""
    statics = np.asarray(env.phase_static_objects)
    positions = [tuple(p.agent_positions) for p in env.dynamic_layout.phases]
    recipes = [p.recipe for p in env.dynamic_layout.phases]
    ids, reps = [], []
    for i in range(statics.shape[0]):
        match = next(
            (
                k
                for k, r in enumerate(reps)
                if np.array_equal(statics[r], statics[i])
                and positions[r] == positions[i]
                and recipes[r] == recipes[i]
            ),
            None,
        )
        if match is None:
            reps.append(i)
            match = len(reps) - 1
        ids.append(match)
    return tuple(ids)


def change_boundaries(env, max_steps):
    """Episode steps at which the kitchen changes, with a full horizon after."""
    ids = kitchen_of_phase(env)
    ends = np.cumsum(np.asarray(env.phase_durations))
    horizon = int(env.phase_durations[0])
    return [
        int(ends[i])
        for i in range(len(ids) - 1)
        if ids[i] != ids[i + 1] and ends[i] + horizon <= max_steps
    ]


def sign_flip_p(deltas):
    """Two-sided exact sign-flip test on paired seed differences."""
    deltas = np.asarray(deltas, dtype=float)
    n = deltas.size
    observed = abs(deltas.sum())
    count = 0
    for bits in range(2**n):
        signs = np.array([1 if bits >> i & 1 else -1 for i in range(n)])
        if abs(float(np.sum(signs * deltas))) >= observed - 1e-12:
            count += 1
    return count / 2**n


def seed_bootstrap_ci(values, draws=10_000, seed=0, level=0.95):
    """Percentile CI of the mean over seeds, resampling seeds."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.shape[0], size=(draws, values.shape[0]))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return lo, hi


# --------------------------------------------------------------------------
# kitchen geometry
# --------------------------------------------------------------------------


def walking_distance(static, start):
    """BFS steps over floor cells from ``start`` (x, y); inf where unreachable."""
    from collections import deque

    empty = static == StaticObject.EMPTY
    height, width = static.shape
    dist = np.full((height, width), np.inf)
    dist[start[1], start[0]] = 0
    queue = deque([tuple(start)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and empty[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((nx, ny))
    return dist


def station_distance(static, start, code):
    """Steps from ``start`` to a cell facing the nearest station of ``code``."""
    dist = walking_distance(static, start)
    best = np.inf
    for y, x in zip(*np.where(static == code)):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < static.shape[1] and 0 <= ny < static.shape[0]:
                best = min(best, dist[ny, nx] + 1)
    return best


def role_switch(env, seat_xy):
    """Which role an Inversion swap hands the agent standing at ``seat_xy``.

    A seat is the supplier when its ingredient pile is nearer than its
    serving window and the server otherwise; the swap reverses that. Returns
    ``(role_in_A, role_in_B)`` with roles ``'supply'`` / ``'serve'``.
    """
    roles = []
    for phase in (0, 1):
        static = np.asarray(env.phase_static_objects[phase])
        onion = min(
            station_distance(static, seat_xy, code)
            for code in np.unique(static)
            if code >= StaticObject.INGREDIENT_PILE_BASE
        )
        goal = station_distance(static, seat_xy, StaticObject.GOAL)
        roles.append("supply" if onion < goal else "serve")
    return tuple(roles)


def phase_regions(env, phase):
    """Room id of every cell in one phase's kitchen (-1 off the floor)."""
    empty = np.asarray(env.phase_static_objects[phase]) == StaticObject.EMPTY
    return np.asarray(compute_enclosed_spaces(jnp.asarray(empty)))


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
