"""Kitchens for the shift-design diagnostic: the designed shift, and drawn ones.

The diagnostic asks what an arbitrary mid-episode change buys, next to a change
that was designed. Both sides run in the V3 environment on one schedule, so the
kitchen is the only thing that varies.

Two families of base kitchen:

* the six V3 benchmark layouts, which already carry a designed phase B.
* the five original Overcooked layouts, which carry no phase B at all. Their
  grids use the same symbols the V3 parser reads (``O`` is already an onion
  pile there), so they need no translation -- only a phase B, which is drawn.

Three kinds of edit, one per designed family, each taking the size that family
takes and fixing nothing else:

``block``
    one walkable cell becomes a wall, as Kitchen Split closes one cell. Agent
    spawn cells are excluded, because a policy cannot start inside a wall and a
    crash is not a finding.
``swap``
    two pairs of stations exchange places, as Distance Switch swaps two pairs.
    The two stations in a pair are always of different kinds, since exchanging
    like for like changes nothing.
``remove``
    half of the map's supply, rounded down, becomes wall. Only dispensers count
    and only dispensers are taken -- plate piles and ingredient piles -- so the
    pot and the delivery square always survive. Resource Outage removes a whole
    class of dispenser and leaves the rest of the kitchen standing; this takes
    that shape and scales it to the map.

What is edited and where is drawn uniformly; how much is edited is not. That is
the whole of the claim these kitchens support: the edits were placed
arbitrarily and their magnitude was comparable across maps.

Nothing is rejected. A remove draw that takes the last plate pile leaves phase
B unable to serve, which is what Resource Outage does by design: the
alternating episode still scores in its A phases, and a kitchen that cannot
finish a dish on its own is the mechanism rather than a defect.

Every layout keeps the benchmark's phase durations and therefore fires the same
boundaries and the same alert. Only the kitchens in those phases differ:

``rs_<base>_a``            single map: phase A only               (R_A)
``rs_<base>_b``            single map: the designed phase B only  (R_B)
``rs_v1_<map>_a``          single map: the original kitchen only  (R_A)
``rs_v1_<map>_<kind>``     it alternating with a drawn kitchen    (R_AB)
``rs_v1_<map>_<kind>_b``   single map: the drawn kitchen only     (R_B)

The drawn kitchens are frozen in ``shift_diagnostic_data`` rather than
regenerated from a seed, so editing the sampler cannot move the experiment
underneath a checkpoint.
"""

import dataclasses
import hashlib
import math

import numpy as np

from jaxmarl.environments.overcooked_v3.common import StaticObject

V3_BASE_LAYOUTS = (
    "split_0",
    "split_1",
    "outage_0",
    "outage_1",
    "distance_0",
    "distance_1",
)
# The handcrafted kitchens of the V2 paper, already in the V3 registry with
# their recipes, so they need no translation either. Grounded Coordination
# Simple is left out: its one difference from Test-Time Simple is the button
# indicator, which V3 carries as a blocker, and a blocker is a wall to both the
# mover and the observation. The two are the same kitchen here.
V2_BASE_LAYOUTS = (
    "grounded_coord_ring",
    "test_time_simple",
    "test_time_wide",
    "demo_cook_simple",
    "demo_cook_wide",
)
V1_BASE_LAYOUTS = (
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "forced_coord",
    "counter_circuit",
)
EDIT_KINDS = ("block", "swap", "remove")
SWAP_PAIRS = 2

# The recipe indicator is left alone: it is what the kitchen tells the team,
# not part of the kitchen's supply or geometry. The V1 kitchens carry none, so
# excluding it changes nothing there.
_STATION_OBJECTS = (
    StaticObject.GOAL,
    StaticObject.POT,
    StaticObject.PLATE_PILE,
)
# A remove takes supply only: what a dish is drawn from, never where it is
# cooked or served.
_DISPENSER_OBJECTS = (StaticObject.PLATE_PILE,)


# Rerolls, by edit kind. A kind is redrawn by bumping its number here, which
# leaves every other kind's draw untouched: each kind is seeded on its own.
# remove was redrawn once, when the rule changed from taking any station to
# taking half the dispensers.
_REROLL = {"remove": 2}


def _draw_seed(base_layout, kind):
    """A stable seed, so a layout name always means the same kitchen."""
    reroll = _REROLL.get(kind, 0)
    # A kind that has never been redrawn keeps its original key, so bumping one
    # kind cannot disturb the others.
    key = f"{base_layout}|{kind}".encode()
    if reroll:
        key = f"{base_layout}|{kind}|{reroll}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def original_layout_grid(name):
    """Rebuild an original Overcooked layout as a grid the V3 parser reads."""
    from jaxmarl.environments.overcooked.layouts import overcooked_layouts

    layout = overcooked_layouts[name]
    height, width = int(layout["height"]), int(layout["width"])
    cells = [" "] * (height * width)
    # Stations are walls too in the original encoding, so they are written last.
    for key, symbol in (
        ("wall_idx", "W"),
        ("goal_idx", "X"),
        ("plate_pile_idx", "B"),
        ("onion_pile_idx", "O"),
        ("pot_idx", "P"),
        ("agent_idx", "A"),
    ):
        for index in np.asarray(layout[key]).tolist():
            cells[int(index)] = symbol
    rows = ["".join(cells[row * width : (row + 1) * width]) for row in range(height)]
    return "\n".join(rows)


def walkable_cells(static_objects, agent_positions):
    """Cells a block may close: empty floor an agent does not start on."""
    static_objects = np.asarray(static_objects)
    mask = static_objects == StaticObject.EMPTY
    # agent_positions are (x, y); the grid is indexed [y, x].
    for x, y in agent_positions:
        mask[int(y), int(x)] = False
    return np.argwhere(mask)


def station_cells(static_objects):
    """Every station on the map, as (row, column) pairs."""
    static_objects = np.asarray(static_objects)
    mask = np.isin(static_objects, [int(obj) for obj in _STATION_OBJECTS])
    mask |= static_objects >= StaticObject.INGREDIENT_PILE_BASE
    return np.argwhere(mask)


def dispenser_cells(static_objects):
    """Plate piles and ingredient piles: the cells a remove may take."""
    static_objects = np.asarray(static_objects)
    mask = np.isin(static_objects, [int(obj) for obj in _DISPENSER_OBJECTS])
    mask |= static_objects >= StaticObject.INGREDIENT_PILE_BASE
    return np.argwhere(mask)


# An outage never takes more than this, however many dispensers a kitchen has.
# Half of a kitchen with twelve piles is not the same edit as half of one with
# three; the cap keeps the magnitude comparable across families. Every V1
# kitchen takes two or fewer, so the cap does not touch them.
REMOVAL_CAP = 4


def removal_count(dispenser_total):
    """How many dispensers a remove takes: half, rounded down, capped."""
    return min(max(dispenser_total // 2, 1), REMOVAL_CAP)


def apply_edit(static_objects, agent_positions, kind, rng):
    """Draw one edit of the given kind and return the edited grid."""
    edited = np.array(static_objects, copy=True)
    if kind == "block":
        candidates = walkable_cells(edited, agent_positions)
        row, column = candidates[rng.choice(len(candidates))]
        edited[row, column] = int(StaticObject.WALL)
        return edited
    if kind == "remove":
        dispensers = dispenser_cells(edited)
        count = removal_count(len(dispensers))
        for row, column in dispensers[
            rng.choice(len(dispensers), size=count, replace=False)
        ]:
            edited[row, column] = int(StaticObject.WALL)
        return edited
    stations = station_cells(edited)
    if kind != "swap":
        raise ValueError(f"Unknown edit kind: {kind!r}")
    # Two pairs, each of two different station kinds; a like-for-like exchange
    # would leave the grid untouched.
    remaining = list(map(tuple, stations))
    for _ in range(SWAP_PAIRS):
        pairs = [
            (first, second)
            for index, first in enumerate(remaining)
            for second in remaining[index + 1 :]
            if edited[first] != edited[second]
        ]
        if not pairs:
            break
        first, second = pairs[rng.choice(len(pairs))]
        edited[first], edited[second] = edited[second], edited[first]
        remaining.remove(first)
        remaining.remove(second)
    return edited


def draw_edit(static_objects, agent_positions, base_layout, kind):
    """The frozen draw for one cell, regenerated from its stable seed."""
    rng = np.random.default_rng(_draw_seed(base_layout, kind))
    return apply_edit(static_objects, agent_positions, kind, rng)


def _with_static_objects(phase, static_objects, phase_class):
    layout = dataclasses.replace(phase.layout, static_objects=static_objects)
    return phase_class(
        layout=layout,
        agent_positions=phase.agent_positions,
        steps=phase.steps,
        name=phase.name,
        recipe=phase.recipe,
    )


def register_diagnostic_layouts():
    """Register every cell of the diagnostic. Returns the names registered."""
    from jaxmarl.environments.overcooked_v3.dynamic_layouts import (
        DynamicLayout,
        DynamicLayoutPhase,
        dynamic_layouts,
        phase_policy_sequence,
    )
    from jaxmarl.environments.overcooked_v3.layouts import (
        overcooked_v3_base_layouts,
    )
    from jaxmarl.environments.overcooked_v3.shift_diagnostic_data import (
        DRAWN_KITCHENS,
        ORIGINAL_KITCHENS,
        V2_DRAWN_KITCHENS,
        V2_ORIGINAL_KITCHENS,
    )

    registered = []

    def register(name, phases):
        dynamic_layouts[name] = DynamicLayout(tuple(phases))
        registered.append(name)

    def single_map(phases, static_kitchen):
        """One kitchen for the whole episode: the schedule without the shift."""
        return [
            _with_static_objects(phase, static_kitchen, DynamicLayoutPhase)
            for phase in phases
        ]

    # Beside the designed shift, the two single-map conditions it is read against.
    for base_layout in V3_BASE_LAYOUTS:
        if base_layout not in dynamic_layouts:
            continue
        phases = dynamic_layouts[base_layout].phases
        sequence = phase_policy_sequence(base_layout)
        static_a = np.asarray(phases[sequence.index(0)].layout.static_objects)
        static_b = np.asarray(phases[sequence.index(1)].layout.static_objects)
        register(f"rs_{base_layout}_a", single_map(phases, static_a))
        register(f"rs_{base_layout}_b", single_map(phases, static_b))

    # The drawn shifts, on kitchens that never had a designed one. They borrow
    # the benchmark's schedule so every cell fires the same boundaries.
    schedule = dynamic_layouts[V3_BASE_LAYOUTS[0]]
    durations = [phase.steps for phase in schedule.phases]
    sequence = phase_policy_sequence(V3_BASE_LAYOUTS[0])
    for base_layout in V1_BASE_LAYOUTS:

        def parse(grid):
            return [
                DynamicLayoutPhase.from_grid(grid, steps, base_layout)
                for steps in durations
            ]

        phases_a = parse(ORIGINAL_KITCHENS[base_layout])
        register(f"rs_v1_{base_layout}_a", phases_a)
        for kind in EDIT_KINDS:
            key = (base_layout, kind)
            if key not in DRAWN_KITCHENS:
                continue
            phases_b = parse(DRAWN_KITCHENS[key])
            register(
                f"rs_v1_{base_layout}_{kind}",
                [
                    phases_a[index] if policy_index == 0 else phases_b[index]
                    for index, policy_index in enumerate(sequence)
                ],
            )
            register(f"rs_v1_{base_layout}_{kind}_b", phases_b)

    # The V2 kitchens, on the same schedule. They carry their own recipes, so
    # the phase is built with them rather than with the default: what V2 asks
    # of a team is which dish the indicator names, and that has to survive.
    for base_layout in V2_BASE_LAYOUTS:
        recipes = overcooked_v3_base_layouts[base_layout].possible_recipes

        def parse(grid, recipes=recipes, base_layout=base_layout):
            return [
                DynamicLayoutPhase.from_grid(
                    grid, steps, base_layout, possible_recipes=recipes
                )
                for steps in durations
            ]

        phases_a = parse(V2_ORIGINAL_KITCHENS[base_layout])
        register(f"rs_v2_{base_layout}_a", phases_a)
        for kind in EDIT_KINDS:
            key = (base_layout, kind)
            if key not in V2_DRAWN_KITCHENS:
                continue
            phases_b = parse(V2_DRAWN_KITCHENS[key])
            register(
                f"rs_v2_{base_layout}_{kind}",
                [
                    phases_a[index] if policy_index == 0 else phases_b[index]
                    for index, policy_index in enumerate(sequence)
                ],
            )
            register(f"rs_v2_{base_layout}_{kind}_b", phases_b)

    return registered
