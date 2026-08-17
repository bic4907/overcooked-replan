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
        ("0",) * (onions - 1)
        + ("P",) * (pots - 1)
        + ("B",) * plates
        + ("X",) * goals
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
        _OUTAGE_AGENT_STARTS[
            (variant_index - 1) % len(_OUTAGE_AGENT_STARTS)
        ],
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
