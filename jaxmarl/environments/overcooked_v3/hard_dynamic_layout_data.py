"""Larger, multi-change role scenarios for the Overcooked V3 hard mode.

The hard layouts preserve the causal manipulation of each original family while
increasing navigation cost and the number of static cells affected at a phase
boundary.  Every episode still follows the same A -> B -> A schedule used by
the selected role-scenario catalog.
"""

from collections import Counter

_ROLE_PHASE_STEPS = 150
_FINAL_PHASE_STEPS = 1000


def _partitioned_rows(width, height, divider_storage_rows=(), counters=()):
    """Return two walkable bays separated by a non-storage center divider."""
    if width < 9 or width % 2 != 1:
        raise ValueError("Hard role layouts require an odd width of at least 9")
    if height < 7:
        raise ValueError("Hard role layouts require a height of at least 7")

    center_x = width // 2
    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "
        rows[y][center_x] = "N"

    rows[0][center_x] = "R"
    rows[height - 1][center_x] = "N"
    for y in divider_storage_rows:
        if not 0 < y < height - 1:
            raise ValueError("Divider storage rows must be inside the map")
        rows[y][center_x] = "W"

    for x, y in counters:
        if x == center_x:
            raise ValueError("Interior counters cannot overwrite the divider")
        if rows[y][x] != " ":
            raise ValueError(f"Hard-layout counter collision at {(x, y)}")
        rows[y][x] = "W"
    return rows


def _place_resources(rows, resources):
    width = len(rows[0])
    height = len(rows)
    for symbol, (x, y) in resources:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Hard-layout resource outside map at {(x, y)}")
        if rows[y][x] not in {"W", " "}:
            raise ValueError(f"Hard-layout resource collision at {(x, y)}")
        rows[y][x] = symbol


def _place_agents(rows, agent_positions):
    for x, y in agent_positions:
        if rows[y][x] != " ":
            raise ValueError(f"Hard-layout agent collision at {(x, y)}")
        rows[y][x] = "A"


def _grid_string(rows):
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Hard-layout rows must be rectangular")
    return "\n" + "\n".join("".join(row) for row in rows) + "\n"


# Variants 5..19 deliberately mix wide, tall, and balanced kitchens instead of
# growing along one fixed size sequence.  Their observations remain 9x9 because
# every hard scenario config uses agent_view_size=4.
_GENERATED_HARD_DIMENSIONS = (
    (23, 17),
    (27, 13),
    (25, 19),
    (29, 15),
    (23, 21),
    (31, 13),
    (27, 19),
    (33, 15),
    (25, 23),
    (31, 17),
    (35, 15),
    (29, 21),
    (33, 19),
    (37, 15),
    (35, 23),
)


def _generated_dimensions(variant):
    index = variant - 5
    if not 0 <= index < len(_GENERATED_HARD_DIMENSIONS):
        raise ValueError(f"Generated hard variant outside 5..19: {variant}")
    return _GENERATED_HARD_DIMENSIONS[index]


def _rotated(values, offset):
    values = tuple(values)
    if not values:
        return values
    offset %= len(values)
    return values[offset:] + values[:offset]


def _bay_border_slots(width, height, side, variant):
    """Return varied boundary stations whose interior interaction lane is clear."""
    center_x = width // 2
    left_top = tuple((x, 0) for x in range(1, center_x))
    left_side = tuple((0, y) for y in range(2, height - 2))
    left_bottom = tuple((x, height - 1) for x in range(center_x - 1, 0, -1))
    slots = (
        left_top[::2]
        + left_side[1::2]
        + left_bottom[::2]
        + left_top[1::2]
        + left_side[::2]
        + left_bottom[1::2]
    )
    slots = _rotated(slots, variant * 3 + (1 if side == "right" else 0))
    if side == "right":
        slots = tuple((width - 1 - x, y) for x, y in slots)
    return slots


def _generated_bay_resources(width, height, side, counts, variant):
    """Interleave station types around one bay's boundary."""
    symbols = tuple(counts)
    symbols = _rotated(symbols, variant + (2 if side == "right" else 0))
    sequence = []
    for occurrence in range(max(counts.values())):
        sequence.extend(
            symbol for symbol in symbols if occurrence < counts[symbol]
        )
    slots = _bay_border_slots(width, height, side, variant)
    if len(sequence) > len(slots):
        raise ValueError("Generated hard layout has too many boundary stations")
    return tuple(zip(sequence, slots))


def _generated_row_order(height, variant):
    low = 1
    high = height - 2
    rows = []
    while low <= high:
        rows.append(low)
        if low != high:
            rows.append(high)
        low += 1
        high -= 1
    return _rotated(rows, variant * 2 + 1)


def _generated_counter_islands(width, height, variant, reserved=()):
    """Create varied islands while preserving station and divider access lanes."""
    center_x = width // 2
    reserved = set(reserved)
    counters = set()
    band_rows = tuple(range(3, height - 3, 3)) or (height // 2,)
    connector_rows = {height // 2 - 1, height // 2 + 1}
    lane_count = max(1, (center_x - 3) // 5)
    usable_span = max(1, center_x - 5)

    for side_index in range(2):
        for band_index, base_y in enumerate(band_rows):
            for lane_index in range(lane_count):
                evenly_spaced_x = 2 + (
                    (lane_index + 1) * usable_span // (lane_count + 1)
                )
                jitter = (variant + band_index * 2 + side_index) % 3 - 1
                base_x = min(
                    center_x - 3,
                    max(3, evenly_spaced_x + jitter),
                )
                motif = (
                    variant + band_index * 2 + lane_index * 3 + side_index
                ) % 5
                if motif == 0:  # horizontal slalom bar
                    cells = (
                        (base_x - 1, base_y),
                        (base_x, base_y),
                        (base_x + 1, base_y),
                    )
                elif motif == 1:  # vertical picket
                    cells = (
                        (base_x, base_y - 1),
                        (base_x, base_y),
                        (base_x, base_y + 1),
                    )
                elif motif == 2:  # staggered L island
                    cells = (
                        (base_x, base_y),
                        (base_x + 1, base_y),
                        (base_x, base_y + 1),
                    )
                elif motif == 3:  # sparse asymmetric island
                    cells = ((base_x, base_y),)
                else:  # dense two-row chicane
                    cells = (
                        (base_x - 1, base_y),
                        (base_x, base_y),
                        (base_x, base_y + 1),
                        (base_x + 1, base_y + 1),
                    )

                for x, y in cells:
                    # x=1 and x=center-1 form permanent interaction spines;
                    # y=1/height-2 and the connector rows form escape loops.
                    if not (2 <= x <= center_x - 2 and 2 <= y <= height - 3):
                        continue
                    if y in connector_rows:
                        continue
                    if side_index:
                        x = width - 1 - x
                    position = (x, y)
                    if position not in reserved:
                        counters.add(position)

    return tuple(sorted(counters, key=lambda position: (position[1], position[0])))


def _generated_split_spec(variant):
    width, height = _generated_dimensions(variant)
    center_x = width // 2
    middle_y = height // 2
    upper_offset = 1 + variant % 3
    lower_offset = 1 + (variant + 1) % 3
    outer_offset = 4 + variant % 3
    if variant % 2:
        agent_positions = (
            (center_x + 2, middle_y - upper_offset),
            (center_x + outer_offset, middle_y + lower_offset),
        )
    else:
        agent_positions = (
            (center_x - 2, middle_y - upper_offset),
            (center_x - outer_offset, middle_y + lower_offset),
        )
    handoff_count = 4 + variant % 4
    handoff_rows = tuple(sorted(_generated_row_order(height, variant)[:handoff_count]))
    return {
        "width": width,
        "height": height,
        "handoff_rows": handoff_rows,
        "agent_positions": agent_positions,
        "counters": _generated_counter_islands(
            width, height, variant, reserved=agent_positions
        ),
        "left_resources": _generated_bay_resources(
            width,
            height,
            "left",
            {"0": 3 + variant % 3, "P": 3 + (variant + 1) % 3},
            variant,
        ),
        "right_resources": _generated_bay_resources(
            width,
            height,
            "right",
            {"B": 3 + (variant + 2) % 3, "X": 3 + variant % 4},
            variant,
        ),
    }


def _generated_outage_spec(variant):
    width, height = _generated_dimensions(variant)
    center_x = width // 2
    middle_y = height // 2
    vertical_offset = variant % 3 - 1
    horizontal_offset = 2 + variant % 2
    agent_positions = (
        (center_x - horizontal_offset, middle_y + vertical_offset),
        (center_x + horizontal_offset, middle_y - vertical_offset),
    )
    handoff_count = min(3 + variant % 4, (height - 2) // 2)
    row_order = _generated_row_order(height, variant)
    normal_handoffs = tuple(sorted(row_order[:handoff_count]))
    outage_handoffs = tuple(sorted(row_order[handoff_count : 2 * handoff_count]))
    if len(normal_handoffs) != len(outage_handoffs):
        raise ValueError("Hard outage handoff banks must have equal capacity")
    common_counts = {
        "0": 3 + variant % 3,
        "P": 3 + (variant + 1) % 2,
        "B": 3 + (variant + 2) % 3,
        "X": 2 + variant % 3,
    }
    return {
        "width": width,
        "height": height,
        "normal_handoffs": normal_handoffs,
        "outage_handoffs": outage_handoffs,
        "agent_positions": agent_positions,
        "counters": _generated_counter_islands(
            width, height, variant + 11, reserved=agent_positions
        ),
        "left_resources": _generated_bay_resources(
            width, height, "left", common_counts, variant + 1
        ),
        "right_resources": _generated_bay_resources(
            width, height, "right", common_counts, variant + 7
        ),
    }


def _generated_recipe_spec(variant):
    width, height = _generated_dimensions(variant)
    center_x = width // 2
    middle_y = height // 2
    vertical_offset = 1 + variant % 3
    agent_positions = (
        (center_x - 2, middle_y - vertical_offset),
        (center_x + 2, middle_y + vertical_offset),
    )
    handoff_count = 4 + (variant + 1) % 5
    handoffs = tuple(sorted(_generated_row_order(height, variant + 4)[:handoff_count]))
    left_resources = _generated_bay_resources(
        width,
        height,
        "left",
        {
            "0": 3 + variant % 3,
            "2": 2 + (variant + 1) % 3,
            "P": 3 + (variant + 2) % 3,
            "X": 3 + variant % 2,
        },
        variant + 3,
    )
    right_resources = _generated_bay_resources(
        width,
        height,
        "right",
        {
            "1": 3 + (variant + 1) % 3,
            "2": 2 + (variant + 2) % 3,
            "P": 3 + variant % 3,
            "B": 3 + (variant + 1) % 2,
        },
        variant + 9,
    )
    return {
        "width": width,
        "height": height,
        "handoffs": handoffs,
        "agent_positions": agent_positions,
        "counters": _generated_counter_islands(
            width, height, variant + 23, reserved=agent_positions
        ),
        "resources": (*left_resources, *right_resources),
    }


def _generated_distance_spec(variant):
    width, height = _generated_dimensions(variant)
    center_x = width // 2
    middle_y = height // 2
    left_direction = 1 if variant % 2 else -1
    right_direction = 1 if (variant // 2) % 2 else -1

    def near_anchors(pot_x, direction):
        return (
            (pot_x, middle_y - 3 * direction),
            (pot_x, middle_y + 2 * direction),
            (pot_x, middle_y + 4 * direction),
        )

    def far_anchors(outer_x, direction):
        if direction == 1:
            rows = (1, height - 4, height - 2)
        else:
            rows = (height - 2, 3, 1)
        return tuple((outer_x, y) for y in rows)

    pots = ((center_x - 2, middle_y), (center_x + 2, middle_y))
    left_near = near_anchors(pots[0][0], left_direction)
    right_near = near_anchors(pots[1][0], right_direction)
    left_far = far_anchors(0, left_direction)
    right_far = far_anchors(width - 1, right_direction)
    agent_x = max(3, center_x // 2)
    agent_offset = variant % 3 - 1
    agent_positions = (
        (agent_x, middle_y + agent_offset),
        (width - 1 - agent_x, middle_y - agent_offset),
    )
    reserved = {
        *left_near,
        *right_near,
        *left_far,
        *right_far,
        *pots,
        *agent_positions,
    }
    counters = set(
        _generated_counter_islands(
            width, height, variant + 37, reserved=reserved
        )
    )
    counters.update(pots)
    return {
        "width": width,
        "height": height,
        "agent_positions": agent_positions,
        "counters": tuple(
            sorted(counters, key=lambda position: (position[1], position[0]))
        ),
        "pots": pots,
        "phase_a": (
            ("0", left_far[0]),
            ("B", left_near[1]),
            ("X", left_near[2]),
            ("0", right_near[0]),
            ("B", right_far[1]),
            ("X", right_far[2]),
        ),
        "phase_b": (
            ("0", left_near[0]),
            ("B", left_far[1]),
            ("X", left_far[2]),
            ("0", right_far[0]),
            ("B", right_near[1]),
            ("X", right_near[2]),
        ),
    }


# Hard Role Formation -----------------------------------------------------
#
# Both agents begin in the same bay.  Every interior crossover closes at once,
# forcing one agent to commit to the complementary-resource bay before the
# transition.  Counter islands add route choice without changing the core
# access-loss manipulation.


_HARD_SPLIT_SPECS = (
    {
        "width": 13,
        "height": 9,
        "handoff_rows": (2, 4, 6),
        "agent_positions": ((3, 4), (4, 5)),
        "counters": (
            (2, 2),
            (3, 2),
            (2, 6),
            (3, 6),
            (9, 2),
            (10, 2),
            (9, 6),
            (10, 6),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 6)),
            ("P", (4, 0)),
            ("P", (0, 2)),
            ("P", (3, 8)),
        ),
        "right_resources": (
            ("B", (8, 0)),
            ("B", (12, 2)),
            ("B", (10, 8)),
            ("X", (12, 4)),
            ("X", (12, 6)),
        ),
    },
    {
        "width": 15,
        "height": 11,
        "handoff_rows": (2, 5, 8),
        "agent_positions": ((10, 4), (11, 7)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 4),
            (2, 6),
            (3, 6),
            (4, 8),
            (10, 6),
            (11, 2),
            (12, 2),
            (11, 6),
            (12, 6),
            (11, 8),
            (12, 8),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 3)),
            ("0", (0, 9)),
            ("P", (5, 0)),
            ("P", (0, 1)),
            ("P", (4, 10)),
        ),
        "right_resources": (
            ("B", (9, 0)),
            ("B", (14, 1)),
            ("B", (14, 7)),
            ("B", (11, 10)),
            ("X", (14, 3)),
            ("X", (13, 10)),
        ),
    },
    {
        "width": 17,
        "height": 11,
        "handoff_rows": (2, 5, 8),
        "agent_positions": ((3, 5), (6, 7)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 4),
            (6, 4),
            (2, 7),
            (3, 7),
            (4, 7),
            (6, 8),
            (12, 2),
            (13, 2),
            (14, 2),
            (10, 4),
            (11, 4),
            (12, 7),
            (13, 7),
            (14, 7),
            (10, 8),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (2, 10)),
            ("P", (6, 0)),
            ("P", (0, 1)),
            ("P", (6, 10)),
        ),
        "right_resources": (
            ("B", (10, 0)),
            ("B", (16, 2)),
            ("B", (16, 7)),
            ("B", (14, 10)),
            ("X", (16, 5)),
            ("X", (10, 10)),
            ("X", (15, 0)),
        ),
    },
    {
        "width": 19,
        "height": 13,
        "handoff_rows": (2, 4, 7, 10),
        "agent_positions": ((12, 4), (16, 8)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (6, 3),
            (7, 3),
            (2, 6),
            (3, 6),
            (5, 8),
            (6, 8),
            (7, 8),
            (2, 10),
            (3, 10),
            (14, 2),
            (15, 2),
            (16, 2),
            (11, 3),
            (12, 3),
            (15, 6),
            (16, 6),
            (11, 8),
            (12, 8),
            (13, 8),
            (15, 10),
            (16, 10),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 10)),
            ("0", (4, 12)),
            ("P", (7, 0)),
            ("P", (0, 1)),
            ("P", (0, 7)),
            ("P", (7, 12)),
        ),
        "right_resources": (
            ("B", (11, 0)),
            ("B", (18, 1)),
            ("B", (18, 7)),
            ("B", (14, 12)),
            ("X", (18, 4)),
            ("X", (18, 10)),
            ("X", (11, 12)),
            ("X", (17, 0)),
        ),
    },
    {
        "width": 21,
        "height": 15,
        "handoff_rows": (2, 5, 8, 11, 13),
        "agent_positions": ((5, 6), (8, 12)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (6, 4),
            (7, 4),
            (8, 4),
            (2, 8),
            (3, 8),
            (4, 8),
            (6, 10),
            (7, 10),
            (8, 10),
            (3, 12),
            (4, 12),
            (16, 2),
            (17, 2),
            (18, 2),
            (12, 4),
            (13, 4),
            (14, 4),
            (16, 8),
            (17, 8),
            (18, 8),
            (12, 10),
            (13, 10),
            (14, 10),
            (16, 12),
            (17, 12),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 10)),
            ("0", (5, 14)),
            ("P", (8, 0)),
            ("P", (0, 1)),
            ("P", (0, 7)),
            ("P", (8, 14)),
        ),
        "right_resources": (
            ("B", (12, 0)),
            ("B", (20, 2)),
            ("B", (20, 7)),
            ("B", (20, 12)),
            ("B", (16, 14)),
            ("X", (20, 4)),
            ("X", (20, 10)),
            ("X", (11, 14)),
            ("X", (19, 0)),
        ),
    },
)

_HARD_SPLIT_SPECS = (
    *_HARD_SPLIT_SPECS,
    *(_generated_split_spec(variant) for variant in range(5, 20)),
)


def _hard_split_grid(spec, closed):
    rows = _partitioned_rows(
        spec["width"],
        spec["height"],
        divider_storage_rows=spec["handoff_rows"] if closed else (),
        counters=spec["counters"],
    )
    if not closed:
        center_x = spec["width"] // 2
        for y in range(1, spec["height"] - 1):
            rows[y][center_x] = " "
    _place_resources(rows, (*spec["left_resources"], *spec["right_resources"]))
    _place_agents(rows, spec["agent_positions"])
    return _grid_string(rows)


def _register_hard_split_catalog():
    for variant_index, spec in enumerate(_HARD_SPLIT_SPECS):
        phase_a = _hard_split_grid(spec, closed=False)
        phase_b = _hard_split_grid(spec, closed=True)
        globals()[f"split_hard_{variant_index}"] = [
            [phase_a, _ROLE_PHASE_STEPS],
            [phase_b, _ROLE_PHASE_STEPS],
            [phase_a, _FINAL_PHASE_STEPS],
        ]


_register_hard_split_catalog()


# Hard Resource Reallocation ---------------------------------------------
#
# Every right-side onion source and every left-side serving station fails while
# the usable handoff bank relocates.  The agents remain in disconnected movement
# regions, so the left agent must reroute supply instead of completing dishes
# independently.


_HARD_OUTAGE_SPECS = (
    {
        "width": 13,
        "height": 9,
        "normal_handoffs": (2, 3),
        "outage_handoffs": (5, 6),
        "agent_positions": ((4, 4), (8, 4)),
        "counters": (
            (2, 2),
            (3, 2),
            (3, 5),
            (2, 6),
            (9, 2),
            (10, 2),
            (9, 5),
            (10, 6),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 6)),
            ("P", (4, 0)),
            ("P", (0, 2)),
            ("B", (2, 8)),
            ("B", (4, 8)),
            ("X", (0, 4)),
        ),
        "right_resources": (
            ("0", (11, 0)),
            ("0", (12, 6)),
            ("P", (8, 0)),
            ("P", (12, 2)),
            ("B", (10, 8)),
            ("B", (8, 8)),
            ("X", (12, 4)),
        ),
    },
    {
        "width": 15,
        "height": 11,
        "normal_handoffs": (1, 5, 9),
        "outage_handoffs": (2, 6, 8),
        "agent_positions": ((5, 4), (9, 4)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 5),
            (2, 7),
            (3, 7),
            (4, 7),
            (9, 5),
            (10, 2),
            (11, 2),
            (12, 2),
            (10, 7),
            (11, 7),
            (12, 7),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 5)),
            ("0", (1, 10)),
            ("P", (5, 0)),
            ("P", (0, 2)),
            ("P", (5, 10)),
            ("B", (3, 0)),
            ("B", (0, 7)),
            ("B", (3, 10)),
            ("X", (0, 9)),
            ("X", (6, 10)),
        ),
        "right_resources": (
            ("0", (13, 0)),
            ("0", (14, 5)),
            ("0", (13, 10)),
            ("P", (9, 0)),
            ("P", (14, 2)),
            ("P", (9, 10)),
            ("B", (11, 0)),
            ("B", (14, 7)),
            ("B", (11, 10)),
            ("X", (14, 9)),
            ("X", (8, 10)),
        ),
    },
    {
        "width": 17,
        "height": 11,
        "normal_handoffs": (1, 4, 8),
        "outage_handoffs": (2, 6, 9),
        "agent_positions": ((6, 5), (10, 5)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 4),
            (6, 4),
            (2, 7),
            (3, 7),
            (4, 7),
            (6, 8),
            (12, 2),
            (13, 2),
            (14, 2),
            (10, 4),
            (11, 4),
            (12, 7),
            (13, 7),
            (14, 7),
            (10, 8),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("0", (2, 10)),
            ("P", (6, 0)),
            ("P", (0, 1)),
            ("P", (6, 10)),
            ("B", (3, 0)),
            ("B", (0, 6)),
            ("B", (4, 10)),
            ("X", (0, 3)),
            ("X", (0, 8)),
        ),
        "right_resources": (
            ("0", (15, 0)),
            ("0", (16, 3)),
            ("0", (16, 8)),
            ("0", (14, 10)),
            ("P", (10, 0)),
            ("P", (16, 1)),
            ("P", (10, 10)),
            ("B", (13, 0)),
            ("B", (16, 6)),
            ("B", (12, 10)),
            ("X", (16, 4)),
            ("X", (16, 9)),
        ),
    },
    {
        "width": 19,
        "height": 13,
        "normal_handoffs": (1, 3, 7, 11),
        "outage_handoffs": (2, 5, 8, 10),
        "agent_positions": ((7, 6), (11, 6)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (6, 3),
            (7, 3),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 8),
            (6, 8),
            (7, 8),
            (2, 10),
            (3, 10),
            (4, 10),
            (14, 2),
            (15, 2),
            (16, 2),
            (11, 3),
            (12, 3),
            (14, 6),
            (15, 6),
            (16, 6),
            (11, 8),
            (12, 8),
            (13, 8),
            (14, 10),
            (15, 10),
            (16, 10),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("0", (3, 12)),
            ("P", (7, 0)),
            ("P", (0, 1)),
            ("P", (0, 7)),
            ("P", (6, 12)),
            ("B", (4, 0)),
            ("B", (0, 6)),
            ("B", (0, 11)),
            ("B", (5, 12)),
            ("X", (0, 3)),
            ("X", (0, 8)),
            ("X", (8, 12)),
        ),
        "right_resources": (
            ("0", (17, 0)),
            ("0", (18, 3)),
            ("0", (18, 6)),
            ("0", (18, 10)),
            ("0", (15, 12)),
            ("P", (11, 0)),
            ("P", (18, 1)),
            ("P", (18, 7)),
            ("P", (11, 12)),
            ("B", (14, 0)),
            ("B", (18, 5)),
            ("B", (18, 11)),
            ("B", (13, 12)),
            ("X", (18, 8)),
            ("X", (10, 12)),
            ("X", (17, 12)),
        ),
    },
    {
        "width": 21,
        "height": 15,
        "normal_handoffs": (1, 4, 7, 11, 13),
        "outage_handoffs": (2, 5, 8, 10, 12),
        "agent_positions": ((8, 7), (12, 7)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (6, 4),
            (7, 4),
            (8, 4),
            (2, 8),
            (3, 8),
            (4, 8),
            (6, 10),
            (7, 10),
            (8, 10),
            (3, 12),
            (4, 12),
            (16, 2),
            (17, 2),
            (18, 2),
            (12, 4),
            (13, 4),
            (14, 4),
            (16, 8),
            (17, 8),
            (18, 8),
            (12, 10),
            (13, 10),
            (14, 10),
            (16, 12),
            (17, 12),
        ),
        "left_resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("0", (0, 13)),
            ("0", (4, 14)),
            ("P", (8, 0)),
            ("P", (0, 1)),
            ("P", (0, 7)),
            ("P", (8, 14)),
            ("B", (4, 0)),
            ("B", (0, 6)),
            ("B", (0, 11)),
            ("B", (6, 14)),
            ("X", (0, 3)),
            ("X", (0, 8)),
            ("X", (0, 12)),
            ("X", (9, 14)),
        ),
        "right_resources": (
            ("0", (19, 0)),
            ("0", (20, 3)),
            ("0", (20, 6)),
            ("0", (20, 10)),
            ("0", (20, 13)),
            ("0", (16, 14)),
            ("P", (12, 0)),
            ("P", (20, 1)),
            ("P", (20, 7)),
            ("P", (12, 14)),
            ("B", (16, 0)),
            ("B", (20, 5)),
            ("B", (20, 11)),
            ("B", (14, 14)),
            ("X", (20, 8)),
            ("X", (20, 12)),
            ("X", (11, 14)),
            ("X", (18, 14)),
        ),
    },
)

_HARD_OUTAGE_SPECS = (
    *_HARD_OUTAGE_SPECS,
    *(_generated_outage_spec(variant) for variant in range(5, 20)),
)


def _hard_outage_grid(spec, outage):
    handoffs = spec["outage_handoffs"] if outage else spec["normal_handoffs"]
    rows = _partitioned_rows(
        spec["width"],
        spec["height"],
        divider_storage_rows=handoffs,
        counters=spec["counters"],
    )
    left_resources = tuple(
        (("N" if symbol == "X" and outage else symbol), position)
        for symbol, position in spec["left_resources"]
    )
    right_resources = tuple(
        (("N" if symbol == "0" and outage else symbol), position)
        for symbol, position in spec["right_resources"]
    )
    _place_resources(rows, (*left_resources, *right_resources))
    _place_agents(rows, spec["agent_positions"])
    return _grid_string(rows)


def _register_hard_outage_catalog():
    for variant_index, spec in enumerate(_HARD_OUTAGE_SPECS):
        phase_a = _hard_outage_grid(spec, outage=False)
        phase_b = _hard_outage_grid(spec, outage=True)
        globals()[f"outage_hard_{variant_index}"] = [
            [phase_a, _ROLE_PHASE_STEPS],
            [phase_b, _ROLE_PHASE_STEPS],
            [phase_a, _FINAL_PHASE_STEPS],
        ]


_register_hard_outage_catalog()


# Hard Goal Rebinding -----------------------------------------------------
#
# Geometry stays fixed so goal rebinding remains isolated from access loss.
# A third ingredient and larger, counter-obstructed bays increase the number of
# dependencies that must be rebound when the dominant ingredient changes.


_HARD_RECIPE_SPECS = (
    {
        "width": 13,
        "height": 9,
        "handoffs": (2, 4, 6),
        "agent_positions": ((4, 4), (8, 4)),
        "counters": (
            (2, 2),
            (3, 2),
            (2, 6),
            (3, 6),
            (9, 2),
            (10, 2),
            (9, 6),
            (10, 6),
        ),
        "resources": (
            ("0", (1, 0)),
            ("0", (0, 5)),
            ("2", (3, 8)),
            ("P", (4, 0)),
            ("P", (0, 2)),
            ("X", (0, 7)),
            ("X", (5, 8)),
            ("1", (11, 0)),
            ("1", (12, 5)),
            ("2", (9, 8)),
            ("P", (8, 0)),
            ("P", (12, 2)),
            ("B", (12, 7)),
            ("B", (7, 8)),
        ),
    },
    {
        "width": 15,
        "height": 11,
        "handoffs": (1, 3, 6, 9),
        "agent_positions": ((5, 5), (9, 5)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (3, 6),
            (2, 6),
            (4, 8),
            (10, 2),
            (11, 2),
            (12, 2),
            (11, 6),
            (10, 6),
            (10, 8),
        ),
        "resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("2", (3, 10)),
            ("P", (5, 0)),
            ("P", (0, 1)),
            ("X", (0, 7)),
            ("X", (6, 10)),
            ("1", (13, 0)),
            ("1", (14, 4)),
            ("1", (14, 9)),
            ("2", (11, 10)),
            ("P", (9, 0)),
            ("P", (14, 1)),
            ("B", (14, 7)),
            ("B", (8, 10)),
        ),
    },
    {
        "width": 17,
        "height": 11,
        "handoffs": (2, 5, 8),
        "agent_positions": ((6, 5), (10, 5)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 4),
            (6, 4),
            (2, 7),
            (3, 7),
            (4, 7),
            (6, 8),
            (12, 2),
            (13, 2),
            (14, 2),
            (10, 4),
            (11, 4),
            (12, 7),
            (13, 7),
            (14, 7),
            (10, 8),
        ),
        "resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("2", (3, 10)),
            ("2", (4, 0)),
            ("P", (6, 0)),
            ("P", (0, 1)),
            ("P", (6, 10)),
            ("X", (0, 7)),
            ("X", (7, 10)),
            ("1", (15, 0)),
            ("1", (16, 4)),
            ("1", (16, 9)),
            ("2", (13, 10)),
            ("2", (12, 0)),
            ("P", (10, 0)),
            ("P", (16, 1)),
            ("P", (10, 10)),
            ("B", (16, 7)),
            ("B", (9, 10)),
        ),
    },
    {
        "width": 19,
        "height": 13,
        "handoffs": (2, 5, 8, 11),
        "agent_positions": ((7, 6), (11, 6)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (6, 3),
            (7, 3),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 8),
            (6, 8),
            (7, 8),
            (2, 10),
            (3, 10),
            (4, 10),
            (14, 2),
            (15, 2),
            (16, 2),
            (11, 3),
            (12, 3),
            (14, 6),
            (15, 6),
            (16, 6),
            (11, 8),
            (12, 8),
            (13, 8),
            (14, 10),
            (15, 10),
            (16, 10),
        ),
        "resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("0", (3, 12)),
            ("2", (5, 0)),
            ("2", (6, 12)),
            ("P", (7, 0)),
            ("P", (0, 1)),
            ("P", (7, 12)),
            ("X", (0, 7)),
            ("X", (0, 11)),
            ("X", (8, 12)),
            ("1", (17, 0)),
            ("1", (18, 4)),
            ("1", (18, 9)),
            ("1", (15, 12)),
            ("2", (13, 0)),
            ("2", (12, 12)),
            ("P", (11, 0)),
            ("P", (18, 1)),
            ("P", (11, 12)),
            ("B", (18, 7)),
            ("B", (18, 11)),
            ("B", (10, 12)),
        ),
    },
    {
        "width": 21,
        "height": 15,
        "handoffs": (2, 5, 8, 11, 13),
        "agent_positions": ((8, 7), (12, 7)),
        "counters": (
            (2, 2),
            (3, 2),
            (4, 2),
            (6, 4),
            (7, 4),
            (8, 4),
            (2, 8),
            (3, 8),
            (4, 8),
            (6, 10),
            (7, 10),
            (8, 10),
            (3, 12),
            (4, 12),
            (16, 2),
            (17, 2),
            (18, 2),
            (12, 4),
            (13, 4),
            (14, 4),
            (16, 8),
            (17, 8),
            (18, 8),
            (12, 10),
            (13, 10),
            (14, 10),
            (16, 12),
            (17, 12),
        ),
        "resources": (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("0", (0, 9)),
            ("0", (0, 13)),
            ("0", (4, 14)),
            ("2", (5, 0)),
            ("2", (0, 6)),
            ("2", (6, 14)),
            ("P", (8, 0)),
            ("P", (0, 1)),
            ("P", (0, 7)),
            ("P", (8, 14)),
            ("X", (0, 3)),
            ("X", (0, 11)),
            ("X", (0, 12)),
            ("X", (9, 14)),
            ("1", (19, 0)),
            ("1", (20, 4)),
            ("1", (20, 9)),
            ("1", (20, 13)),
            ("1", (16, 14)),
            ("2", (15, 0)),
            ("2", (20, 6)),
            ("2", (14, 14)),
            ("P", (12, 0)),
            ("P", (20, 1)),
            ("P", (20, 7)),
            ("P", (12, 14)),
            ("B", (20, 3)),
            ("B", (20, 11)),
            ("B", (20, 12)),
            ("B", (11, 14)),
        ),
    },
)

_HARD_RECIPE_SPECS = (
    *_HARD_RECIPE_SPECS,
    *(_generated_recipe_spec(variant) for variant in range(5, 20)),
)


_HARD_RECIPE_A = [0, 0, 2]
_HARD_RECIPE_B = [1, 1, 2]


def _hard_recipe_grid(spec):
    rows = _partitioned_rows(
        spec["width"],
        spec["height"],
        divider_storage_rows=spec["handoffs"],
        counters=spec["counters"],
    )
    _place_resources(rows, spec["resources"])
    _place_agents(rows, spec["agent_positions"])
    return _grid_string(rows)


def _register_hard_recipe_catalog():
    for variant_index, spec in enumerate(_HARD_RECIPE_SPECS):
        grid = _hard_recipe_grid(spec)
        globals()[f"recipe_switch_hard_{variant_index}"] = [
            [grid, _ROLE_PHASE_STEPS, _HARD_RECIPE_A],
            [grid, _ROLE_PHASE_STEPS, _HARD_RECIPE_B],
            [grid, _FINAL_PHASE_STEPS, _HARD_RECIPE_A],
        ]


_register_hard_recipe_catalog()


# Hard Distance-Driven Role Reassignment ---------------------------------
#
# Pots stay fixed so an in-flight dish is not destroyed.  Onion, plate, and
# serving stations all move to previously inactive non-storage anchors.  Phase B
# therefore cannot be solved by learning the original pairwise station swap.


_HARD_DISTANCE_SPECS = (
    {
        "width": 13,
        "height": 9,
        "agent_positions": ((3, 4), (9, 4)),
        "counters": (
            (2, 2),
            (2, 3),
            (2, 5),
            (10, 2),
            (10, 3),
            (10, 5),
            (5, 2),
            (5, 4),
            (5, 5),
            (5, 6),
            (7, 2),
            (7, 4),
            (7, 5),
            (7, 6),
        ),
        "pots": ((5, 4), (7, 4)),
        "phase_a": (
            ("0", (0, 1)),
            ("B", (5, 6)),
            ("X", (4, 8)),
            ("0", (7, 2)),
            ("B", (7, 6)),
            ("X", (12, 1)),
        ),
        "phase_b": (
            ("0", (5, 2)),
            ("B", (0, 5)),
            ("X", (0, 7)),
            ("0", (12, 7)),
            ("B", (7, 5)),
            ("X", (7, 7)),
        ),
    },
    {
        "width": 15,
        "height": 11,
        "agent_positions": ((4, 5), (10, 5)),
        "counters": (
            (3, 2),
            (3, 3),
            (3, 7),
            (3, 8),
            (11, 2),
            (11, 3),
            (11, 7),
            (11, 8),
            (6, 3),
            (6, 5),
            (6, 7),
            (6, 8),
            (8, 3),
            (8, 5),
            (8, 7),
            (8, 8),
        ),
        "pots": ((6, 5), (8, 5)),
        "phase_a": (
            ("0", (0, 1)),
            ("B", (6, 8)),
            ("X", (5, 10)),
            ("0", (8, 3)),
            ("B", (8, 8)),
            ("X", (14, 1)),
        ),
        "phase_b": (
            ("0", (6, 3)),
            ("B", (0, 7)),
            ("X", (0, 9)),
            ("0", (14, 9)),
            ("B", (8, 7)),
            ("X", (8, 9)),
        ),
    },
    {
        "width": 17,
        "height": 11,
        "agent_positions": ((4, 5), (12, 5)),
        "counters": (
            (3, 2),
            (3, 3),
            (3, 7),
            (3, 8),
            (7, 3),
            (7, 5),
            (7, 7),
            (7, 9),
            (9, 3),
            (9, 5),
            (9, 7),
            (9, 9),
            (13, 2),
            (13, 3),
            (13, 7),
            (13, 8),
        ),
        "pots": ((7, 5), (9, 5)),
        "phase_a": (
            ("0", (0, 1)),
            ("B", (7, 7)),
            ("X", (7, 9)),
            ("0", (9, 3)),
            ("B", (16, 8)),
            ("X", (16, 1)),
        ),
        "phase_b": (
            ("0", (7, 3)),
            ("B", (0, 8)),
            ("X", (0, 2)),
            ("0", (16, 3)),
            ("B", (9, 7)),
            ("X", (9, 9)),
        ),
    },
    {
        "width": 19,
        "height": 13,
        "agent_positions": ((4, 6), (14, 6)),
        "counters": (
            (3, 2),
            (3, 3),
            (3, 9),
            (3, 10),
            (7, 2),
            (7, 4),
            (7, 8),
            (7, 10),
            (8, 3),
            (8, 6),
            (8, 9),
            (8, 11),
            (10, 3),
            (10, 6),
            (10, 9),
            (10, 11),
            (11, 2),
            (11, 4),
            (11, 8),
            (11, 10),
            (15, 2),
            (15, 3),
            (15, 9),
            (15, 10),
        ),
        "pots": ((8, 6), (10, 6)),
        "phase_a": (
            ("0", (0, 1)),
            ("B", (8, 9)),
            ("X", (8, 11)),
            ("0", (10, 3)),
            ("B", (18, 10)),
            ("X", (18, 1)),
        ),
        "phase_b": (
            ("0", (8, 3)),
            ("B", (0, 10)),
            ("X", (0, 2)),
            ("0", (18, 11)),
            ("B", (10, 9)),
            ("X", (10, 11)),
        ),
    },
    {
        "width": 21,
        "height": 15,
        "agent_positions": ((5, 7), (15, 7)),
        "counters": (
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 10),
            (3, 11),
            (3, 12),
            (7, 2),
            (7, 4),
            (7, 6),
            (7, 8),
            (7, 10),
            (7, 12),
            (8, 5),
            (8, 9),
            (9, 3),
            (9, 7),
            (9, 10),
            (9, 13),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 10),
            (17, 11),
            (17, 12),
            (13, 2),
            (13, 4),
            (13, 6),
            (13, 8),
            (13, 10),
            (13, 12),
            (12, 5),
            (12, 9),
            (11, 3),
            (11, 7),
            (11, 10),
            (11, 13),
        ),
        "pots": ((9, 7), (11, 7)),
        "phase_a": (
            ("0", (0, 1)),
            ("B", (9, 10)),
            ("X", (9, 13)),
            ("0", (11, 3)),
            ("B", (20, 12)),
            ("X", (20, 1)),
        ),
        "phase_b": (
            ("0", (9, 3)),
            ("B", (0, 12)),
            ("X", (0, 2)),
            ("0", (20, 13)),
            ("B", (11, 10)),
            ("X", (11, 13)),
        ),
    },
)

_HARD_DISTANCE_SPECS = (
    *_HARD_DISTANCE_SPECS,
    *(_generated_distance_spec(variant) for variant in range(5, 20)),
)


def _validate_hard_distance_spec(spec):
    expected = Counter({"0": 2, "B": 2, "X": 2})
    for phase_name in ("phase_a", "phase_b"):
        resources = spec[phase_name]
        if Counter(symbol for symbol, _ in resources) != expected:
            raise ValueError(
                f"Hard distance {phase_name} must expose two local copies of "
                "onion, plate, and serving stations"
            )
        positions = [position for _, position in resources]
        if len(set(positions)) != len(positions):
            raise ValueError(f"Hard distance {phase_name} reuses a station cell")

    old_positions = {position for _, position in spec["phase_a"]}
    new_positions = {position for _, position in spec["phase_b"]}
    if old_positions & new_positions:
        raise ValueError(
            "Hard distance stations must relocate to previously unused counters"
        )


def _hard_distance_grid(spec, phase_name):
    rows = _partitioned_rows(
        spec["width"],
        spec["height"],
        counters=spec["counters"],
    )
    _place_resources(rows, tuple(("P", position) for position in spec["pots"]))
    station_positions = {
        position
        for candidate_phase in ("phase_a", "phase_b")
        for _symbol, position in spec[candidate_phase]
    }
    for x, y in station_positions:
        if rows[y][x] not in {"W", " "}:
            raise ValueError(f"Hard distance inactive-station collision at {(x, y)}")
        rows[y][x] = "N"
    for symbol, (x, y) in spec[phase_name]:
        if rows[y][x] != "N":
            raise ValueError(f"Hard distance active-station collision at {(x, y)}")
        rows[y][x] = symbol
    _place_agents(rows, spec["agent_positions"])
    return _grid_string(rows)


def _register_hard_distance_catalog():
    for variant_index, spec in enumerate(_HARD_DISTANCE_SPECS):
        _validate_hard_distance_spec(spec)
        phase_a = _hard_distance_grid(spec, "phase_a")
        phase_b = _hard_distance_grid(spec, "phase_b")
        globals()[f"distance_switch_hard_{variant_index}"] = [
            [phase_a, _ROLE_PHASE_STEPS],
            [phase_b, _ROLE_PHASE_STEPS],
            [phase_a, _FINAL_PHASE_STEPS],
        ]


_register_hard_distance_catalog()
