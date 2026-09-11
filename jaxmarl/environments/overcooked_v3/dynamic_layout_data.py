# Overcooked V3 layouts. Both ``0`` and ``O`` denote an onion pile.

# Role-formation scenarios -------------------------------------------------
#
# The recipe display stays at a separate fixed cell. A non-storage blocker
# separates the dynamic doorway or handoff counters from the rest of the map.
#
# Kitchen Split starts with one open central doorway. After 150 steps, that
# doorway becomes a handoff counter and traps agents in their chosen bays until
# step 300. The left bay has onions and pots, while the right bay has
# plates and serving, so agents must occupy different sides and divide labor.
#
# Resource Outage permanently separates two otherwise complete kitchens. Each
# bay owns at least one pot, plate pile, serving station, and onion pile in the normal
# phase. When the right onion pile disappears, the left agent must trade off
# local cooking against supplying onions through the shared center counters.
#
# Each paper category keeps its selected ``0`` layout. The Split, Outage, and
# Distance Switch families additionally expose a deliberately redesigned ``1``
# candidate with the same scenario mechanics and a different route geometry.


_ROLE_PHASE_STEPS = 150
_FINAL_PHASE_STEPS = 1000


def _role_grid(
    resources,
    agent_positions,
    blocker_row,
    door=None,
    recipe_row=None,
    counters=(),
    width=11,
    height=7,
):
    """Build one role-scenario phase from explicit design constraints."""
    center_x = width // 2
    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "
        rows[y][center_x] = "W"

    rows[blocker_row][center_x] = "N"
    if recipe_row is not None:
        rows[recipe_row][center_x] = "R"
    if door is not None:
        door_row, is_open = door
        rows[door_row][center_x] = " " if is_open else "W"

    for x, y in counters:
        if rows[y][x] != " ":
            raise ValueError(f"Role-layout counter collision at {(x, y)}")
        rows[y][x] = "W"
    for symbol, (x, y) in resources:
        if rows[y][x] not in {"W", " "}:
            raise ValueError(f"Role-layout resource collision at {(x, y)}")
        rows[y][x] = symbol
    for x, y in agent_positions:
        if rows[y][x] != " ":
            raise ValueError(f"Role-layout agent collision at {(x, y)}")
        rows[y][x] = "A"
    return "\n" + "\n".join("".join(row) for row in rows) + "\n"


def _build_split_workload(spec, width=11, recipe_row=0, counters=()):
    door_row, blocker_row, agents, left_resources, right_resources = spec
    resources = [*left_resources, *right_resources]
    open_grid = _role_grid(
        resources,
        agents,
        blocker_row,
        door=(door_row, True),
        recipe_row=recipe_row,
        counters=counters,
        width=width,
    )
    closed_grid = _role_grid(
        resources,
        agents,
        blocker_row,
        door=(door_row, False),
        recipe_row=recipe_row,
        counters=counters,
        width=width,
    )
    return [
        [open_grid, _ROLE_PHASE_STEPS],
        [closed_grid, _ROLE_PHASE_STEPS],
        [open_grid, _FINAL_PHASE_STEPS],
    ]


def _compact_outage_grid(
    resources,
    agent_positions,
    blocker_row,
    notches=(),
):
    """Build a compact 5x7 kitchen with permanently separated movement bays."""
    width, height, center_x = 7, 5, 3
    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "
        rows[y][center_x] = "W"

    rows[0][center_x] = "R"
    rows[blocker_row][center_x] = "N"
    for x, y in notches:
        rows[y][x] = "W"
        rows[y][width - 1 - x] = "W"
    for symbol, (x, y) in resources:
        if rows[y][x] not in {"W", " "}:
            raise ValueError(f"Compact outage resource collision at {(x, y)}")
        rows[y][x] = symbol
    for x, y in agent_positions:
        if rows[y][x] != " ":
            raise ValueError(f"Compact outage agent collision at {(x, y)}")
        rows[y][x] = "A"
    return "\n" + "\n".join("".join(row) for row in rows) + "\n"


def _build_compact_outage_variant(spec):
    blocker_row, agents, left_resources, notches = spec
    right_resources = tuple((symbol, (6 - x, y)) for symbol, (x, y) in left_resources)
    normal_grid = _compact_outage_grid(
        left_resources + right_resources,
        agents,
        blocker_row,
        notches,
    )
    outage_right_resources = [
        (("W" if symbol == "0" else symbol), position)
        for symbol, position in right_resources
    ]
    outage_grid = _compact_outage_grid(
        [*left_resources, *outage_right_resources],
        agents,
        blocker_row,
        notches,
    )
    return [
        [normal_grid, _ROLE_PHASE_STEPS],
        [outage_grid, _ROLE_PHASE_STEPS],
        [normal_grid, _FINAL_PHASE_STEPS],
    ]


def _rotated_take(positions, count, offset):
    """Take unique positions from a cyclically rotated placement palette."""
    if count > len(positions):
        raise ValueError(
            f"Requested {count} resources for {len(positions)} placement slots"
        )
    split = offset % len(positions)
    rotated = positions[split:] + positions[:split]
    return rotated[:count]


# Candidate source layouts retain the 7x9 split topology. The observer-positive
# source is registered as ``split_0``; a hand-designed route variant is added as
# ``split_1`` below. The workload tuple is
# (onion piles, pots, plate piles, serving stations). Resources remain assigned
# to their role-specific bay, while placement and starting positions vary.
_SPLIT_WORKLOADS = (
    (1, 1, 1, 1),
    (1, 2, 1, 1),
    (2, 1, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
    (2, 2, 1, 1),
    (1, 3, 1, 1),
    (2, 1, 2, 1),
    (1, 2, 2, 1),
    (2, 2, 2, 1),
    (2, 3, 1, 1),
    (1, 3, 2, 1),
    (2, 2, 1, 2),
    (1, 2, 2, 2),
    (2, 3, 2, 1),
    (2, 2, 2, 2),
    (2, 3, 1, 2),
    (2, 3, 2, 2),
    (3, 3, 2, 2),
)
_SPLIT_LEFT_BOUNDARY = (
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 6),
    (2, 6),
    (3, 6),
)
_SPLIT_RIGHT_BOUNDARY = tuple((8 - x, y) for x, y in _SPLIT_LEFT_BOUNDARY)
_SPLIT_DOOR_BLOCKER_ROWS = (
    (4, 2),
    (3, 1),
    (2, 4),
    (5, 3),
    (1, 4),
    (4, 1),
    (3, 5),
    (2, 5),
    (5, 2),
    (1, 3),
    (4, 2),
    (3, 1),
    (2, 4),
    (5, 3),
    (1, 4),
    (4, 1),
    (3, 5),
    (2, 5),
    (5, 2),
)
_SPLIT_AGENT_STARTS = (
    ((2, 4), (6, 4)),
    ((1, 2), (7, 2)),
    ((3, 3), (5, 3)),
    ((2, 1), (6, 5)),
    ((1, 4), (7, 1)),
    ((3, 5), (5, 2)),
)


def _build_split_catalog_variant(variant_index):
    onions, pots, plates, goals = _SPLIT_WORKLOADS[variant_index - 1]
    door_row, blocker_row = _SPLIT_DOOR_BLOCKER_ROWS[variant_index - 1]
    agents = _SPLIT_AGENT_STARTS[(variant_index - 1) % len(_SPLIT_AGENT_STARTS)]
    left_symbols = ("0",) * onions + ("P",) * pots
    right_symbols = ("B",) * plates + ("X",) * goals
    left_positions = _rotated_take(
        _SPLIT_LEFT_BOUNDARY,
        len(left_symbols),
        2 * variant_index,
    )
    right_positions = _rotated_take(
        _SPLIT_RIGHT_BOUNDARY,
        len(right_symbols),
        3 * variant_index + 1,
    )
    spec = (
        door_row,
        blocker_row,
        agents,
        tuple(zip(left_symbols, left_positions)),
        tuple(zip(right_symbols, right_positions)),
    )
    return _build_split_workload(spec, width=9)


# Outage candidate sources keep the compact 5x7, permanently separated two-bay
# topology. The observer-positive source becomes ``outage_0`` and the new
# route variant becomes ``outage_1``. Each side starts with an identical
# complete kitchen. All right-side
# onion piles disappear during outage, and the two center handoff counters stay
# available. Anchors keep an onion-to-handoff and handoff-to-pot route short.
_OUTAGE_WORKLOADS = (
    (1, 1, 1, 1),
    (1, 2, 1, 1),
    (2, 1, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
    (2, 2, 1, 1),
    (1, 2, 2, 1),
    (2, 1, 2, 1),
    (2, 2, 2, 1),
    (1, 3, 1, 1),
    (2, 3, 1, 1),
    (1, 3, 2, 1),
    (2, 2, 1, 2),
    (1, 2, 2, 2),
    (2, 2, 2, 1),
    (1, 3, 1, 2),
    (2, 1, 2, 2),
    (2, 2, 1, 2),
    (2, 2, 2, 1),
)
_OUTAGE_BOUNDARY = (
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 4),
)
_OUTAGE_ONION_POT_ANCHORS = (
    ((2, 0), (1, 0)),
    ((0, 1), (0, 2)),
    ((0, 2), (0, 1)),
    ((1, 0), (2, 0)),
)
_OUTAGE_AGENT_STARTS = (
    ((2, 2), (4, 2)),
    ((1, 1), (5, 3)),
    ((2, 3), (4, 1)),
    ((1, 2), (5, 2)),
)


def _build_outage_catalog_variant(variant_index):
    onions, pots, plates, goals = _OUTAGE_WORKLOADS[variant_index - 1]
    onion_anchor, pot_anchor = _OUTAGE_ONION_POT_ANCHORS[
        (variant_index - 1) % len(_OUTAGE_ONION_POT_ANCHORS)
    ]
    occupied = {onion_anchor, pot_anchor}
    available = tuple(
        position for position in _OUTAGE_BOUNDARY if position not in occupied
    )
    remaining_symbols = (
        ("0",) * (onions - 1) + ("P",) * (pots - 1) + ("B",) * plates + ("X",) * goals
    )
    remaining_positions = _rotated_take(
        available,
        len(remaining_symbols),
        variant_index,
    )
    left_resources = (
        ("0", onion_anchor),
        ("P", pot_anchor),
        *tuple(zip(remaining_symbols, remaining_positions)),
    )
    spec = (
        3,
        _OUTAGE_AGENT_STARTS[(variant_index - 1) % len(_OUTAGE_AGENT_STARTS)],
        left_resources,
        (),
    )
    return _build_compact_outage_variant(spec)


def _register_role_catalog():
    # Keep only the observer-positive paper layouts as public Easy tag 0.
    split_sources = (14,)
    outage_sources = (4,)
    for new_index, (split_source, outage_source) in enumerate(
        zip(split_sources, outage_sources)
    ):
        globals()[f"split_{new_index}"] = _build_split_catalog_variant(split_source)
        globals()[f"outage_{new_index}"] = _build_outage_catalog_variant(outage_source)

    # Candidate 1 places both agents in mirrored branches two moves from the
    # centered doorway. Their shortest routes share the same first tile, so one
    # must yield before an agent can cross to the right bay.
    split_candidate = (
        3,
        1,
        ((3, 2), (3, 4)),
        (("0", (2, 0)), ("P", (0, 2)), ("P", (1, 6))),
        (("B", (6, 0)), ("B", (8, 4)), ("X", (8, 1)), ("X", (7, 6))),
    )
    globals()["split_1"] = _build_split_workload(
        split_candidate,
        width=9,
        counters=((2, 2), (2, 4), (6, 2), (6, 4)),
    )

    # Candidate 1 keeps the two disconnected Outage bays, but places the two
    # handoff counters next to each other above the blocker.  The surviving
    # onion pile, handoff, and right pot form a one-move relay during outage;
    # mirrored lower notches prevent the relay from degenerating into a wide
    # open loop.  This preserves the 5x7 concept while making the transition
    # demand an immediate local-cook-versus-supplier decision.
    outage_candidate = (
        3,
        ((1, 2), (5, 2)),
        (
            ("0", (1, 0)),
            ("P", (2, 0)),
            ("B", (0, 1)),
            ("B", (0, 2)),
            ("X", (2, 4)),
        ),
        ((1, 3),),
    )
    globals()["outage_1"] = _build_compact_outage_variant(outage_candidate)


_register_role_catalog()


# Mixed Recipe Relay ------------------------------------------------------
#
# Both agents stay in permanently separated bays. The left bay owns onion
# piles and serving stations, while the right bay owns tomato and plate piles.
# Both sides have a pot, and exactly two shared counter cells are embedded in
# the otherwise non-storage center divider. The map never changes; only the
# active recipe follows a deterministic A -> B -> A schedule.


def _recipe_switch_grid(spec):
    """Build one separated mixed-recipe kitchen from an explicit layout spec."""
    width = spec["width"]
    height = spec["height"]
    center_x = width // 2
    if width % 2 != 1:
        raise ValueError("Recipe-switch layouts must have an odd width")

    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "
        rows[y][center_x] = "N"

    rows[0][center_x] = "R"
    rows[height - 1][center_x] = "N"
    if len(set(spec["handoff_rows"])) != 2:
        raise ValueError("Recipe-switch layouts require exactly two handoffs")
    for y in spec["handoff_rows"]:
        if not 0 < y < height - 1:
            raise ValueError("Recipe-switch handoffs must be inside the map")
        rows[y][center_x] = "W"

    for x, y in spec.get("notches", ()):
        if x == center_x:
            raise ValueError("Recipe-switch notches cannot alter the divider")
        rows[y][x] = "W"

    resources = (*spec["left_resources"], *spec["right_resources"])
    for symbol, (x, y) in resources:
        if rows[y][x] not in {"W", " "}:
            raise ValueError(f"Recipe-switch resource collision at {(x, y)}")
        if symbol == "0" and x >= center_x:
            raise ValueError("Onion piles must stay in the left bay")
        if symbol == "1" and x <= center_x:
            raise ValueError("Tomato piles must stay in the right bay")
        rows[y][x] = symbol

    for x, y in spec["agent_positions"]:
        if rows[y][x] != " ":
            raise ValueError(f"Recipe-switch agent collision at {(x, y)}")
        rows[y][x] = "A"

    return "\n" + "\n".join("".join(row) for row in rows) + "\n"


# Retain the selected previous tag 2 (catalog 7) as public Easy tag 0.
_RECIPE_SWITCH_SPECS = (
    {
        "width": 7,
        "height": 5,
        "handoff_rows": (1, 3),
        "agent_positions": ((1, 2), (5, 2)),
        "left_resources": (
            ("0", (1, 0)),
            ("P", (0, 1)),
            ("P", (0, 3)),
            ("X", (2, 4)),
        ),
        "right_resources": (
            ("1", (5, 0)),
            ("P", (6, 1)),
            ("P", (6, 3)),
            ("B", (4, 4)),
        ),
    },
)

_RECIPE_ONION_MAJOR = [0, 0, 1]
_RECIPE_TOMATO_MAJOR = [0, 1, 1]
_RECIPE_SWITCH_TIMINGS = (
    (150, 150),
)
_RECIPE_SWITCH_ONION_MAJOR_FIRST = (False,)


def _register_recipe_switch_catalog():
    for variant_index, (spec, timings, onion_major_first) in enumerate(
        zip(
            _RECIPE_SWITCH_SPECS,
            _RECIPE_SWITCH_TIMINGS,
            _RECIPE_SWITCH_ONION_MAJOR_FIRST,
        )
    ):
        grid = _recipe_switch_grid(spec)
        recipe_a, recipe_b = (
            (_RECIPE_ONION_MAJOR, _RECIPE_TOMATO_MAJOR)
            if onion_major_first
            else (_RECIPE_TOMATO_MAJOR, _RECIPE_ONION_MAJOR)
        )
        first_phase_steps, second_phase_steps = timings
        globals()[f"recipe_switch_{variant_index}"] = [
            [grid, first_phase_steps, recipe_a],
            [grid, second_phase_steps, recipe_b],
            # Training episodes stop at step 450. A long final duration avoids
            # displaying a countdown for an unused wraparound transition.
            [grid, _FINAL_PHASE_STEPS, recipe_a],
        ]


_register_recipe_switch_catalog()


# Distance-Driven Role Switch --------------------------------------------
#
# The recipe stays fixed at the standard three-onion dish. These layouts follow
# Overcooked-AI's asymmetric_advantages design: a central pot bar separates the
# two agents, while each side still has direct access to an onion pile, pot,
# plate pile, and serving station. The role split is therefore a comparative
# cost advantage rather than an access restriction.
#
# Phase A gives agent 0 the short pot-to-plate-to-serving loop and agent 1 the
# short onion-to-pot loop. Phase B reverses those loop costs without moving
# pots, plates, agents, or walkable floor. The canonical layout swaps endpoint
# types in place, while the route candidate moves them onto previously unused
# counters. Phase C returns to the original assignment.


def _distance_switch_role_resources(spec, roles_swapped):
    """Return the role-dependent onion and serving placements for one phase."""
    phase_key = (
        "phase_b_role_resources" if roles_swapped else "phase_a_role_resources"
    )
    if phase_key in spec:
        return spec[phase_key]

    left_far, left_near = spec["left_role_slots"]
    right_near, right_far = spec["right_role_slots"]
    if roles_swapped:
        return (
            (left_far, "X"),
            (left_near, "0"),
            (right_near, "X"),
            (right_far, "0"),
        )
    return (
        (left_far, "0"),
        (left_near, "X"),
        (right_near, "0"),
        (right_far, "X"),
    )


def _distance_switch_grid(spec, roles_swapped=False):
    """Build one phase of an asymmetric-advantages-style kitchen."""
    width = spec["width"]
    height = spec["height"]
    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "

    for x, y in spec["divider"]:
        if rows[y][x] != " ":
            raise ValueError(f"Distance-switch divider collision at {(x, y)}")
        rows[y][x] = "W"

    for x, y in spec.get("counters", ()):
        if rows[y][x] != " ":
            raise ValueError(f"Distance-switch counter collision at {(x, y)}")
        rows[y][x] = "W"

    for (x, y), symbol in (
        *((position, "P") for position in spec["pot_positions"]),
        *((position, "B") for position in spec["plate_positions"]),
    ):
        if rows[y][x] != "W":
            raise ValueError(
                f"Distance-switch fixed station needs a counter at {(x, y)}"
            )
        rows[y][x] = symbol

    for (x, y), symbol in _distance_switch_role_resources(spec, roles_swapped):
        if rows[y][x] != "W":
            raise ValueError(
                f"Distance-switch role station needs a counter at {(x, y)}"
            )
        rows[y][x] = symbol

    for x, y in spec["agent_positions"]:
        if rows[y][x] != " ":
            raise ValueError(f"Distance-switch agent collision at {(x, y)}")
        rows[y][x] = "A"
    return "\n" + "\n".join("".join(row) for row in rows) + "\n"


def _distance_switch_floor_distances(rows, start):
    floor = {
        (x, y)
        for y, row in enumerate(rows)
        for x, cell in enumerate(row)
        if cell in {" ", "A"}
    }
    frontier = [(start, 0)]
    distances = {start: 0}
    while frontier:
        (x, y), distance = frontier.pop(0)
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in floor and neighbor not in distances:
                distances[neighbor] = distance + 1
                frontier.append((neighbor, distance + 1))
    return distances


def _distance_switch_interaction_floors(position, distances):
    x, y = position
    return {
        candidate
        for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if candidate in distances
    }


def _distance_switch_route_distance(reachable, starts, goals):
    goal_floors = _distance_switch_interaction_floors(goals, reachable)
    start_floors = _distance_switch_interaction_floors(starts, reachable)
    if not start_floors or not goal_floors:
        raise ValueError(
            f"Distance-switch route endpoint is not interactable: {starts} -> {goals}"
        )
    frontier = [(position, 0) for position in start_floors]
    visited = {position for position, _ in frontier}
    while frontier:
        position, distance = frontier.pop(0)
        if position in goal_floors:
            return distance
        x, y = position
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in reachable and neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, distance + 1))
    raise ValueError(f"No distance-switch route from {starts} to {goals}")


def _validate_distance_switch_spec(spec):
    """Enforce local resource access and a reversible task-loop advantage."""
    phase_a = _distance_switch_grid(spec, roles_swapped=False)
    phase_b = _distance_switch_grid(spec, roles_swapped=True)
    rows_a = [row for row in phase_a.splitlines() if row]
    rows_b = [row for row in phase_b.splitlines() if row]
    if len(rows_a) != len(rows_b) or any(
        len(a) != len(b) for a, b in zip(rows_a, rows_b)
    ):
        raise ValueError("Distance-switch phases must keep the same shape")
    if any(
        (cell_a in {" ", "A"}) != (cell_b in {" ", "A"})
        for row_a, row_b in zip(rows_a, rows_b)
        for cell_a, cell_b in zip(row_a, row_b)
    ):
        raise ValueError("Distance-switch phases must keep the same walkable floor")

    agents = spec["agent_positions"]
    reachable = tuple(
        _distance_switch_floor_distances(rows_a, agent) for agent in agents
    )
    if agents[1] in reachable[0]:
        raise ValueError("Distance-switch agents should occupy separate work regions")

    minimum_advantage = spec.get("minimum_advantage", 3)
    pots = spec["pot_positions"]
    plates = spec["plate_positions"]
    local_pots_by_agent = []
    local_plates_by_agent = []
    for agent_index, distances in enumerate(reachable):
        local_pots = [
            position
            for position in pots
            if _distance_switch_interaction_floors(position, distances)
        ]
        local_plates = [
            position
            for position in plates
            if _distance_switch_interaction_floors(position, distances)
        ]
        if not local_pots or not local_plates:
            raise ValueError(f"Agent {agent_index} needs local access to pot and plate")
        local_pots_by_agent.append(local_pots)
        local_plates_by_agent.append(local_plates)

    def route(agent_index, starts, goals):
        return _distance_switch_route_distance(reachable[agent_index], starts, goals)

    def input_cost(agent_index, onion):
        return min(
            route(agent_index, onion, pot) for pot in local_pots_by_agent[agent_index]
        )

    def serving_cost(agent_index, goal):
        return min(
            route(agent_index, pot, plate) + route(agent_index, plate, goal)
            for pot in local_pots_by_agent[agent_index]
            for plate in local_plates_by_agent[agent_index]
        )

    def local_role_station(rows, agent_index, symbol):
        positions = [
            (x, y)
            for y, row in enumerate(rows)
            for x, cell in enumerate(row)
            if cell == symbol
            and _distance_switch_interaction_floors(
                (x, y), reachable[agent_index]
            )
        ]
        if len(positions) != 1:
            raise ValueError(
                f"Agent {agent_index} needs exactly one local {symbol!r} station"
            )
        return positions[0]

    phase_a_onions = tuple(
        local_role_station(rows_a, agent_index, "0") for agent_index in range(2)
    )
    phase_a_goals = tuple(
        local_role_station(rows_a, agent_index, "X") for agent_index in range(2)
    )
    phase_b_onions = tuple(
        local_role_station(rows_b, agent_index, "0") for agent_index in range(2)
    )
    phase_b_goals = tuple(
        local_role_station(rows_b, agent_index, "X") for agent_index in range(2)
    )
    phase_a_input = (
        input_cost(0, phase_a_onions[0]),
        input_cost(1, phase_a_onions[1]),
    )
    phase_a_serve = (
        serving_cost(0, phase_a_goals[0]),
        serving_cost(1, phase_a_goals[1]),
    )
    phase_b_input = (
        input_cost(0, phase_b_onions[0]),
        input_cost(1, phase_b_onions[1]),
    )
    phase_b_serve = (
        serving_cost(0, phase_b_goals[0]),
        serving_cost(1, phase_b_goals[1]),
    )

    if phase_a_input[1] + minimum_advantage > phase_a_input[0]:
        raise ValueError("Phase A needs an agent 1 onion-input advantage")
    if phase_a_serve[0] + minimum_advantage > phase_a_serve[1]:
        raise ValueError("Phase A needs an agent 0 serving advantage")
    if phase_b_input[0] + minimum_advantage > phase_b_input[1]:
        raise ValueError("Phase B needs an agent 0 onion-input advantage")
    if phase_b_serve[1] + minimum_advantage > phase_b_serve[0]:
        raise ValueError("Phase B needs an agent 1 serving advantage")


def _vertical_distance_switch_spec(width, height, extra_counters=()):
    """Build a vertical-pot-bar asymmetric-advantages variant."""
    center_x = width // 2
    bottom_y = height - 1
    return {
        "width": width,
        "height": height,
        "left_role_slots": ((0, 1), (center_x - 1, 1)),
        "right_role_slots": ((center_x + 1, 1), (width - 1, 1)),
        "pot_positions": ((center_x, 2), (center_x, height - 2)),
        "plate_positions": ((center_x - 1, bottom_y), (center_x + 1, bottom_y)),
        "agent_positions": ((center_x - 2, height - 2), (center_x + 1, height - 2)),
        "divider": tuple((center_x, y) for y in range(1, height - 1)),
        "counters": (
            (center_x - 2, 1),
            (center_x - 1, 1),
            (center_x + 1, 1),
            (center_x + 2, 1),
            *extra_counters,
        ),
        "minimum_advantage": 3,
    }


# Retain the canonical Overcooked-AI asymmetric_advantages map as Easy tag 0.
_DISTANCE_SWITCH_SPECS = (
    _vertical_distance_switch_spec(9, 5),
    {
        "width": 9,
        "height": 6,
        "pot_positions": ((4, 2), (4, 4)),
        "plate_positions": ((3, 5), (5, 5)),
        "agent_positions": ((2, 4), (6, 4)),
        "divider": tuple((4, y) for y in range(1, 5)),
        "counters": ((2, 2), (6, 2)),
        "phase_a_role_resources": (
            ((0, 2), "0"),
            ((2, 5), "X"),
            ((6, 2), "0"),
            ((8, 1), "X"),
        ),
        "phase_b_role_resources": (
            ((2, 2), "0"),
            ((0, 1), "X"),
            ((8, 2), "0"),
            ((6, 5), "X"),
        ),
        "minimum_advantage": 3,
    },
)


def _register_distance_switch_catalog():
    for variant_index, spec in enumerate(_DISTANCE_SWITCH_SPECS):
        _validate_distance_switch_spec(spec)
        phase_a = _distance_switch_grid(spec, roles_swapped=False)
        phase_b = _distance_switch_grid(spec, roles_swapped=True)
        globals()[f"distance_switch_{variant_index}"] = [
            [phase_a, _ROLE_PHASE_STEPS],
            [phase_b, _ROLE_PHASE_STEPS],
            [phase_a, _FINAL_PHASE_STEPS],
        ]


_register_distance_switch_catalog()


# Selected 11x7 pillars map, published as split_wide. Stations occupy
# opposite outer walls; N obstacles block movement and cannot store objects.
split_wide = _build_split_workload(
    (
        3, 1, ((3, 3), (7, 3)),
        (("0", (0, 2)), ("P", (0, 3)), ("P", (0, 4)), ("N", (2, 2))),
        (("B", (10, 2)), ("B", (10, 4)), ("X", (10, 1)), ("X", (10, 5)),
         ("N", (8, 4))),
    ),
    width=11,
)


def _build_shared_room_outage(source, missing_resource="0"):
    """Open the divider and suspend every dispenser of one resource in B.

    Existing boundary counters provide shared stockpiling space. Inventory,
    stored objects on unchanged counters, and pot contents survive the outage.
    """
    if missing_resource not in {"0", "B"}:
        raise ValueError("Shared-room outage must remove onions or plates")
    rows = [list(row) for row in source[0][0].strip("\n").splitlines()]
    center_x = len(rows[0]) // 2
    for row in rows[1:-1]:
        row[center_x] = " "
    normal = "\n" + "\n".join("".join(row) for row in rows) + "\n"
    removed_symbols = {"0", "O"} if missing_resource == "0" else {"B"}
    suspended = "".join("W" if cell in removed_symbols else cell for cell in normal)
    return [
        [normal, _ROLE_PHASE_STEPS],
        [suspended, _ROLE_PHASE_STEPS],
        [normal, _FINAL_PHASE_STEPS],
    ]


# Shared-room tag 2 retains the compact footprint. Wide uses an 11x6 room
# with staggered two-cell non-storage obstacles and an open central aisle.
# The default removes all onion piles; plate variants use identical geometry.
_SHARED_WIDE_OUTAGE_SOURCE = [["""
WWWP0R0PWWW
X         X
W ANN   A W
W     NN  W
W         W
WWWBWWWBWWW
""", _ROLE_PHASE_STEPS]]

outage_2 = _build_shared_room_outage(outage_0)
outage_wide_2 = _build_shared_room_outage(_SHARED_WIDE_OUTAGE_SOURCE)
outage_2_plate = _build_shared_room_outage(outage_0, missing_resource="B")
outage_wide_2_plate = _build_shared_room_outage(
    _SHARED_WIDE_OUTAGE_SOURCE, missing_resource="B"
)


# Short aliases for the first selected layouts.
split = split_0
outage = outage_0
