# Overcooked V3 layouts. Both ``0`` and ``O`` denote an onion pile.

# Role-formation scenarios -------------------------------------------------
#
# The recipe display stays at a separate fixed cell in both conditions. The
# Sig/NoSig pairs differ only at the signal cell: ``L`` is V2's interactable
# button indicator, while ``S`` is a blank non-storage blocker with no button.
# Both block movement and neither stores objects, so the comparison isolates
# signaling capability.
#
# Kitchen Split starts with one open central doorway. After 40 steps, that
# doorway becomes a handoff counter and traps agents in their chosen bays until
# the next cycle. The left bay has onions and pots, while the right bay has
# plates and serving, so agents must occupy different sides and divide labor.
#
# Resource Outage permanently separates two otherwise complete kitchens. Each
# bay owns at least one pot, plate pile, serving station, and onion pile in the normal
# phase. When the right onion pile disappears, the left agent must trade off
# local cooking against supplying onions through the shared center counters.
#
# Each role category exposes the three layouts selected from the cross-play
# report. Sig/NoSig use matched geometry at each index.


def _role_grid(
    resources,
    agent_positions,
    signal_row,
    signal_enabled,
    door=None,
    recipe_row=None,
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

    rows[signal_row][center_x] = "L" if signal_enabled else "S"
    if recipe_row is not None:
        rows[recipe_row][center_x] = "R"
    if door is not None:
        door_row, is_open = door
        rows[door_row][center_x] = " " if is_open else "W"

    for symbol, (x, y) in resources:
        if rows[y][x] not in {"W", " "}:
            raise ValueError(f"Role-layout resource collision at {(x, y)}")
        rows[y][x] = symbol
    for x, y in agent_positions:
        if rows[y][x] != " ":
            raise ValueError(f"Role-layout agent collision at {(x, y)}")
        rows[y][x] = "A"
    return "\n" + "\n".join("".join(row) for row in rows) + "\n"


def _build_split_workload(spec, signal_enabled, width=11, recipe_row=0):
    door_row, signal_row, agents, left_resources, right_resources = spec
    resources = [*left_resources, *right_resources]
    open_grid = _role_grid(
        resources,
        agents,
        signal_row,
        signal_enabled,
        door=(door_row, True),
        recipe_row=recipe_row,
        width=width,
    )
    closed_grid = _role_grid(
        resources,
        agents,
        signal_row,
        signal_enabled,
        door=(door_row, False),
        recipe_row=recipe_row,
        width=width,
    )
    return [[open_grid, 40], [closed_grid, 160]]


def _compact_outage_grid(
    resources,
    agent_positions,
    signal_row,
    signal_enabled,
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
    rows[signal_row][center_x] = "L" if signal_enabled else "S"
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


def _build_compact_outage_variant(spec, signal_enabled):
    signal_row, agents, left_resources, notches = spec
    right_resources = tuple((symbol, (6 - x, y)) for symbol, (x, y) in left_resources)
    normal_grid = _compact_outage_grid(
        left_resources + right_resources,
        agents,
        signal_row,
        signal_enabled,
        notches,
    )
    outage_right_resources = [
        (("W" if symbol == "0" else symbol), position)
        for symbol, position in right_resources
    ]
    outage_grid = _compact_outage_grid(
        [*left_resources, *outage_right_resources],
        agents,
        signal_row,
        signal_enabled,
        notches,
    )
    return [[normal_grid, 40], [outage_grid, 160]]


def _rotated_take(positions, count, offset):
    """Take unique positions from a cyclically rotated placement palette."""
    if count > len(positions):
        raise ValueError(
            f"Requested {count} resources for {len(positions)} placement slots"
        )
    split = offset % len(positions)
    rotated = positions[split:] + positions[:split]
    return rotated[:count]


# Candidate source layouts retain the 7x9 split topology. Only the three
# cross-play-selected candidates are registered below. The workload tuple is
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
_SPLIT_DOOR_SIGNAL_ROWS = (
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


def _build_split_catalog_variant(variant_index, signal_enabled):
    onions, pots, plates, goals = _SPLIT_WORKLOADS[variant_index - 1]
    door_row, signal_row = _SPLIT_DOOR_SIGNAL_ROWS[variant_index - 1]
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
        signal_row,
        agents,
        tuple(zip(left_symbols, left_positions)),
        tuple(zip(right_symbols, right_positions)),
    )
    return _build_split_workload(spec, signal_enabled, width=9)


# Outage candidate sources keep the compact 5x7, permanently separated two-bay
# topology. Only the three selected candidates are registered. Each side starts
# with an identical complete kitchen. All right-side
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


def _build_outage_catalog_variant(variant_index, signal_enabled):
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
    return _build_compact_outage_variant(spec, signal_enabled)


def _register_role_catalog():
    # Ranked candidates from the 2026-08-17 cross-play report. Reindexing the
    # selected source layouts keeps the public scenario names compact.
    split_sources = (9, 19, 14)
    outage_sources = (4, 12, 8)
    for new_index, (split_source, outage_source) in enumerate(
        zip(split_sources, outage_sources)
    ):
        globals()[f"splitnosig_{new_index}"] = _build_split_catalog_variant(
            split_source, False
        )
        globals()[f"splitsig_{new_index}"] = _build_split_catalog_variant(
            split_source, True
        )
        globals()[f"outagenosig_{new_index}"] = _build_outage_catalog_variant(
            outage_source, False
        )
        globals()[f"outagesig_{new_index}"] = _build_outage_catalog_variant(
            outage_source, True
        )


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
        rows[y][center_x] = "S"

    rows[0][center_x] = "R"
    rows[height - 1][center_x] = "S"
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


# Selected from the original ten-map catalog and reindexed as:
# new 0 <- old 4, new 1 <- old 5, new 2 <- old 7.
_RECIPE_SWITCH_SPECS = (
    {
        "width": 9,
        "height": 5,
        "handoff_rows": (2, 3),
        "agent_positions": ((3, 2), (5, 2)),
        "left_resources": (("0", (0, 1)), ("P", (2, 0)), ("X", (3, 4))),
        "right_resources": (
            ("1", (8, 1)),
            ("P", (6, 0)),
            ("P", (8, 3)),
            ("B", (5, 4)),
        ),
        "notches": ((1, 2), (7, 2)),
    },
    {
        "width": 7,
        "height": 5,
        "handoff_rows": (1, 3),
        "agent_positions": ((2, 2), (4, 2)),
        "left_resources": (("0", (0, 2)), ("P", (2, 0)), ("X", (1, 4))),
        "right_resources": (("1", (6, 2)), ("P", (4, 0)), ("B", (5, 4))),
    },
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
    (165, 135),
    (150, 150),
    (180, 120),
)
_RECIPE_SWITCH_ONION_MAJOR_FIRST = (True, False, False)


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
            [grid, 1000, recipe_a],
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
# short onion-to-pot loop. Phase B swaps only the onion and serving endpoints on
# each side, reversing those loop costs without moving pots, plates, counters,
# agents, or walkable floor. Phase C returns to the original assignment.


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

    left_far, left_near = spec["left_role_slots"]
    right_near, right_far = spec["right_role_slots"]
    role_resources = (
        ((left_far, "X"), (left_near, "0"), (right_near, "X"), (right_far, "0"))
        if roles_swapped
        else ((left_far, "0"), (left_near, "X"), (right_near, "0"), (right_far, "X"))
    )
    for (x, y), symbol in role_resources:
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
    left_far, left_near = spec["left_role_slots"]
    right_near, right_far = spec["right_role_slots"]

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

    phase_a_input = (
        input_cost(0, left_far),
        input_cost(1, right_near),
    )
    phase_a_serve = (
        serving_cost(0, left_near),
        serving_cost(1, right_far),
    )
    phase_b_input = (
        input_cost(0, left_near),
        input_cost(1, right_far),
    )
    phase_b_serve = (
        serving_cost(0, left_far),
        serving_cost(1, right_near),
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


_DISTANCE_SWITCH_SPECS = (
    # Canonical Overcooked-AI asymmetric_advantages.
    _vertical_distance_switch_spec(9, 5),
    # Wider canonical corridor.
    _vertical_distance_switch_spec(11, 5),
    # Symmetric counter islands leave inner and outer bypass lanes.
    _vertical_distance_switch_spec(
        11,
        7,
        ((2, 3), (3, 3), (7, 3), (8, 3)),
    ),
    # Offset vertical pillars create different left/right doglegs.
    _vertical_distance_switch_spec(
        9,
        7,
        ((2, 2), (2, 3), (6, 3), (6, 4)),
    ),
    # Mirrored stair counters produce a zigzag route to the outer endpoints.
    _vertical_distance_switch_spec(
        11,
        7,
        ((2, 2), (3, 2), (3, 3), (3, 4), (7, 2), (7, 3), (7, 4), (8, 4)),
    ),
    # Long asymmetric shelves emphasize inner versus outer lanes.
    _vertical_distance_switch_spec(
        13,
        7,
        ((2, 3), (3, 3), (4, 3), (4, 4), (8, 2), (8, 3), (9, 3), (10, 3)),
    ),
    # Staggered islands make the two work regions visually non-isomorphic.
    _vertical_distance_switch_spec(
        11,
        8,
        ((2, 2), (2, 3), (3, 3), (3, 5), (7, 2), (8, 2), (8, 3), (7, 5)),
    ),
    # Large counter blocks form narrow outer and inner circulation lanes.
    _vertical_distance_switch_spec(
        13,
        8,
        (
            (2, 2),
            (3, 2),
            (4, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (8, 2),
            (9, 2),
            (10, 2),
            (8, 3),
            (9, 3),
            (10, 3),
            (8, 4),
            (9, 4),
            (10, 4),
        ),
    ),
    # Tall hooked counters create U-shaped detours on both sides.
    _vertical_distance_switch_spec(
        11,
        9,
        (
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 5),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (7, 5),
        ),
    ),
    # Rotated asymmetric-advantages: agent 0 works above the pot bar and
    # agent 1 below it. Endpoint swapping still reverses the same two roles.
    {
        "width": 9,
        "height": 9,
        "left_role_slots": ((0, 1), (4, 2)),
        "right_role_slots": ((4, 6), (8, 7)),
        "pot_positions": ((3, 4), (5, 4)),
        "plate_positions": ((4, 0), (4, 8)),
        "agent_positions": ((3, 2), (5, 6)),
        "divider": tuple((x, 4) for x in range(1, 8)),
        "counters": ((4, 2), (4, 6)),
        "minimum_advantage": 3,
    },
)


def _register_distance_switch_catalog():
    for variant_index, spec in enumerate(_DISTANCE_SWITCH_SPECS):
        _validate_distance_switch_spec(spec)
        phase_a = _distance_switch_grid(spec, roles_swapped=False)
        phase_b = _distance_switch_grid(spec, roles_swapped=True)
        globals()[f"distance_switch_{variant_index}"] = [
            [phase_a, 150],
            [phase_b, 150],
            [phase_a, 1000],
        ]


_register_distance_switch_catalog()


# Backward-compatible aliases for existing commands and checkpoints.
split_no_sig = splitnosig_0
split_sig = splitsig_0
outage_no_sig = outagenosig_0
outage_sig = outagesig_0

dynamic_00 = [
    [
        """
WWWWWWW
0 AWA X
W  W  W
B     P
WWWWWWW
""",
        100,
    ],
    [
        """
WWWWWWW
0 A A X
W  W  W
B  W  P
WWWWWWW
""",
        100,
    ],
]

dynamic_01 = [
    [
        """
WWWOWWW
W     W
XAWWWAB
W WWW W
W     W
WWWPWWW
""",
        100,
    ],
    [
        """
WWWWWWW
W WWW W
XAWOWAB
W     W
W     W
WWWPWWW
""",
        100,
    ],
]

dynamic_02 = [
    [
        """
WWWWW
0AWAX
W W W
B W P
WWWWW
""",
        100,
    ],
    [
        """
WWWWW
0A AX
WWWWW
B   P
WWWWW
""",
        100,
    ],
]

dynamic_03 = [
    [
        """
WWWWW
0A AW
W   W
B   P
WWXWW
""",
        100,
    ],
    [
        """
WWWWW
WA AO
W   W
B   P
WWXWW
""",
        100,
    ],
]

dynamic_04 = [
    [
        """
WWWOOWWW
W      W
XAWWWWAB
W      W
WWWPPWWW
""",
        100,
    ],
    [
        """
WWWPPWWW
W      W
BAWWWWAX
W      W
WWWOOWWW
""",
        100,
    ],
]


dynamic_05 = [
    [
        """
WWWWWOW
WWW A P
X     B
WWW A P
WWWWWOW
""",
        100,
    ],
    [
        """
WWWWWOW
WWW A P
X W   B
WWW A P
WWWWWOW
""",
        100,
    ],
]

dynamic_06 = [
    [
        """
WWPWW
W   W
BA AO
W   W
WWXWW
""",
        100,
    ],
    [
        """
WWOWW
W   W
PA AX
W   W
WWBWW
""",
        100,
    ],
    [
        """
WWXWW
W   W
OA AB
W   W
WWPWW
""",
        100,
    ],
    [
        """
WWBWW
W   W
XA AP
W   W
WWOWW
""",
        100,
    ],
]


dynamic_07 = [
    [
        """
WWWWW
WA AX
W W P
W   B
WWOWW
""",
        50,
    ],
    [
        """
WWWWW
WA AX
O W P
W   B
WWWWW
""",
        50,
    ],
    [
        """
WWOWW
WA AX
W W P
W   B
WWWWW
""",
        50,
    ],
]

dynamic_08 = [
    [
        """
WWWPWWW
WW   WW
WW A WW
WBWWWOW
W  A  W
W     W
WWWXWWW
""",
        100,
    ],
    [
        """
WWWPWWW
W    WW
W  A WW
WBWWWOW
WW A  W
WW    W
WWWXWWW
""",
        100,
    ],
    [
        """
WWWPWWW
W     W
W  A  W
WBWWWOW
WW A WW
WW   WW
WWWXWWW
""",
        100,
    ],
    [
        """
WWWPWWW
WW    W
WW A  W
WBWWWOW
W  A WW
W    WW
WWWXWWW
""",
        100,
    ],
]


dynamic_09 = [
    [
        """
WWWWW
0AWAX
W W W
B W P
WWWWW
""",
        100,
    ],
    [
        """
WWWWW
XAWAO
W W W
P W B
WWWWW
""",
        100,
    ],
]

dynamic_10 = [
    [
        """
WWWWWWWWWWW
0A  B     W
W   W     W
XA  P     W
WWWWWWWWWWW
""",
        100,
    ],
    [
        """
WWWWWWWWWWW
0A        B
W         W
XA        P
WWWWWWWWWWW
""",
        100,
    ],
]

dynamic_11 = [
    [
        """
WWWWWWWWWWWW
B          O
BAWWWPPWWWAO
B    WW    O
WWWWXWWXWWWW
""",
        50,
    ],
    [
        """
WWWWXWWXWWWW
B    WW    O
BAWWWPPWWWAO
B          O
WWWWWWWWWWWW
""",
        50,
    ],
]


dynamic_12 = [
    [
        """
WOWWWXW
0A    X
W WWW W
W WWW W
W WWW W
B  W AP
WBWWWPW
""",
        20,
    ],
    [
        """
WOWWWXW
0A    X
W WWW W
WWWWW W
W WWW W
B    AP
WBWWWPW
""",
        20,
    ],
    [
        """
WOWWWXW
0A W  X
W WWW W
W WWW W
W WWW W
B    AP
WBWWWPW
""",
        20,
    ],
    [
        """
WOWWWXW
0A    X
W WWW W
W WWWWW
W WWW W
B    AP
WBWWWPW
""",
        20,
    ],
]


dynamic_13 = [
    [
        """
WWWWWWW
0 AWA X
W  W  W
W  WWWW
W     W
B     P
WWWWWWW
""",
        10,
    ],
    [
        """
WWWWWWW
0 AWA X
W  W  W
W  W  W
W  W  W
B  W  P
WWWWWWW
""",
        10,
    ],
    [
        """
WWWWWWW
0 AWA X
W  W  W
WWWW  W
W     W
B     P
WWWWWWW
""",
        10,
    ],
    [
        """
WWWWWWW
0 AWA X
W  W  W
W  W  W
W  W  W
B  W  P
WWWWWWW
""",
        10,
    ],
]


dynamic_14 = [
    [
        """
WXWOWBWPW
WA  W W W
W W   W W
W W W  AW
WXWOWBWPW
""",
        10,
    ],
    [
        """
WXWOWBWPW
WAW W   W
W   W W W
W W   WAW
WXWOWBWPW
""",
        10,
    ],
    [
        """
WXWOWBWPW
WAW   W W
W W W   W
W   W WAW
WXWOWBWPW
""",
        10,
    ],
]
