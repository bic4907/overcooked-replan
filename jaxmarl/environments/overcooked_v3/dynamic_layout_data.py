# Overcooked V3 layouts. Both ``0`` and ``O`` denote an onion pile.

# Role-formation scenarios -------------------------------------------------
#
# The Sig/NoSig pairs differ only at one shared cell: ``L`` is V2's
# interactable button indicator, while ``R`` is a non-interactable recipe
# display. Both block movement and neither stores objects, so the comparison
# changes signaling capability without adding a spare handoff counter.
#
# Kitchen Split starts with one open central doorway. After 40 steps, that
# doorway becomes a handoff counter and traps agents in their chosen bays until
# the next cycle. The left bay has onions and two pots, while the right bay has
# plates and serving, so agents must occupy different sides and divide labor.
#
# Resource Outage permanently separates two otherwise complete kitchens. Each
# bay owns a pot, plate pile, serving station, and onion pile in the normal
# phase. When the right onion pile disappears, the left agent must trade off
# local cooking against supplying onions through the shared center counters.
#
# Each role category exposes one selected canonical layout with index ``0``.

splitnosig_0 = [
    [
        """
WWWWWWWWWWW
W0   W  B W
W    R    W
W A     A W
WP   W    W
WP   W  X W
WWWWWWWWWWW
""",
        40,
    ],
    [
        """
WWWWWWWWWWW
W0   W  B W
W    R    W
W A  W  A W
WP   W    W
WP   W  X W
WWWWWWWWWWW
""",
        160,
    ],
]

splitsig_0 = [
    [
        """
WWWWWWWWWWW
W0   W  B W
W    L    W
W A     A W
WP   W    W
WP   W  X W
WWWWWWWWWWW
""",
        40,
    ],
    [
        """
WWWWWWWWWWW
W0   W  B W
W    L    W
W A  W  A W
WP   W    W
WP   W  X W
WWWWWWWWWWW
""",
        160,
    ],
]

outagenosig_0 = [
    [
        """
WWWWWWWWWWW
W0   W   0W
WP   W   PW
W  A W A  W
WB   R   BW
WX   W   XW
WWWWWWWWWWW
""",
        100,
    ],
    [
        """
WWWWWWWWWWW
W0   W   WW
WP   W   PW
W  A W A  W
WB   R   BW
WX   W   XW
WWWWWWWWWWW
""",
        100,
    ],
]

outagesig_0 = [
    [
        """
WWWWWWWWWWW
W0   W   0W
WP   W   PW
W  A W A  W
WB   L   BW
WX   W   XW
WWWWWWWWWWW
""",
        100,
    ],
    [
        """
WWWWWWWWWWW
W0   W   WW
WP   W   PW
W  A W A  W
WB   L   BW
WX   W   XW
WWWWWWWWWWW
""",
        100,
    ],
]


def _role_grid(resources, agent_positions, signal_row, signal_enabled, door=None):
    """Build one 7x11 role-scenario phase from explicit design constraints."""
    width, height, center_x = 11, 7, 5
    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "
        rows[y][center_x] = "W"

    rows[signal_row][center_x] = "L" if signal_enabled else "R"
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


# door row, signal row, agents, left (onion/pot/pot), right (plate/goal)
_SPLIT_VARIANT_SPECS = (
    (3, 2, ((2, 3), (8, 3)), ((1, 0), (3, 0), (0, 4)), ((8, 0), (10, 4))),
    (2, 4, ((3, 2), (7, 2)), ((0, 1), (0, 3), (2, 6)), ((9, 0), (10, 5))),
    (4, 2, ((2, 4), (8, 4)), ((4, 0), (0, 2), (3, 6)), ((6, 0), (10, 3))),
    (3, 5, ((3, 3), (7, 3)), ((1, 6), (0, 1), (4, 0)), ((9, 6), (10, 1))),
    (2, 5, ((2, 2), (8, 2)), ((2, 0), (0, 4), (4, 6)), ((7, 0), (10, 4))),
    (4, 1, ((3, 4), (7, 4)), ((0, 5), (1, 0), (3, 0)), ((10, 5), (8, 0))),
    (3, 1, ((2, 3), (8, 3)), ((3, 6), (0, 2), (2, 0)), ((7, 6), (10, 2))),
    (2, 4, ((3, 2), (7, 2)), ((4, 6), (0, 3), (1, 6)), ((6, 6), (10, 3))),
    (4, 2, ((2, 4), (8, 4)), ((0, 1), (4, 0), (2, 6)), ((10, 1), (9, 6))),
)


# signal row, agents, left complete-kitchen positions in onion/pot/plate/goal order.
# The right kitchen mirrors every resource so pre-outage capabilities match.
_OUTAGE_VARIANT_SPECS = (
    (4, ((2, 3), (8, 3)), ((1, 0), (0, 2), (3, 6), (0, 5))),
    (2, ((3, 3), (7, 3)), ((0, 1), (2, 0), (0, 4), (4, 6))),
    (4, ((2, 3), (8, 3)), ((4, 0), (0, 2), (1, 6), (0, 5))),
    (2, ((3, 3), (7, 3)), ((0, 5), (3, 0), (0, 2), (2, 6))),
    (5, ((2, 2), (8, 2)), ((1, 6), (0, 1), (4, 0), (0, 4))),
    (1, ((3, 4), (7, 4)), ((0, 3), (2, 6), (0, 5), (4, 0))),
    (4, ((2, 3), (8, 3)), ((3, 0), (0, 1), (4, 6), (0, 5))),
    (2, ((3, 3), (7, 3)), ((0, 4), (1, 0), (0, 1), (3, 6))),
    (5, ((2, 2), (8, 2)), ((2, 0), (0, 5), (4, 6), (0, 1))),
)


def _build_split_variant(spec, signal_enabled):
    door_row, signal_row, agents, left_positions, right_positions = spec
    resources = list(zip(("0", "P", "P"), left_positions))
    resources.extend(zip(("B", "X"), right_positions))
    open_grid = _role_grid(
        resources,
        agents,
        signal_row,
        signal_enabled,
        door=(door_row, True),
    )
    closed_grid = _role_grid(
        resources,
        agents,
        signal_row,
        signal_enabled,
        door=(door_row, False),
    )
    return [[open_grid, 40], [closed_grid, 160]]


def _build_outage_variant(spec, signal_enabled):
    signal_row, agents, left_positions = spec
    right_positions = tuple((10 - x, y) for x, y in left_positions)
    left_resources = list(zip(("0", "P", "B", "X"), left_positions))
    right_resources = list(zip(("0", "P", "B", "X"), right_positions))
    normal_grid = _role_grid(
        left_resources + right_resources,
        agents,
        signal_row,
        signal_enabled,
    )
    outage_right_resources = [("W", right_positions[0]), *right_resources[1:]]
    outage_grid = _role_grid(
        left_resources + outage_right_resources,
        agents,
        signal_row,
        signal_enabled,
    )
    return [[normal_grid, 100], [outage_grid, 100]]


def _register_role_variants():
    for variant_index, spec in enumerate(_SPLIT_VARIANT_SPECS, start=1):
        globals()[f"splitnosig_{variant_index}"] = _build_split_variant(spec, False)
        globals()[f"splitsig_{variant_index}"] = _build_split_variant(spec, True)
    for variant_index, spec in enumerate(_OUTAGE_VARIANT_SPECS, start=1):
        globals()[f"outagenosig_{variant_index}"] = _build_outage_variant(spec, False)
        globals()[f"outagesig_{variant_index}"] = _build_outage_variant(spec, True)


# Workload-controlled Split maps. Counts increase across the five variants:
# (onions, pots, plates, goals) =
# (1,1,1,1), (1,2,1,1), (2,2,1,1), (1,3,2,1), (2,3,2,2).
_SPLIT_WORKLOAD_SPECS = (
    (
        3,
        2,
        ((2, 3), (8, 3)),
        (("0", (1, 0)), ("P", (3, 0))),
        (("B", (7, 0)), ("X", (9, 0))),
    ),
    (
        2,
        4,
        ((3, 2), (7, 2)),
        (("0", (1, 0)), ("P", (3, 0)), ("P", (0, 4))),
        (("B", (7, 0)), ("X", (10, 4))),
    ),
    (
        4,
        2,
        ((2, 4), (8, 4)),
        (("0", (1, 0)), ("0", (0, 2)), ("P", (3, 0)), ("P", (2, 6))),
        (("B", (7, 0)), ("X", (10, 4))),
    ),
    (
        3,
        5,
        ((3, 3), (7, 3)),
        (
            ("0", (1, 0)),
            ("P", (3, 0)),
            ("P", (0, 2)),
            ("P", (2, 6)),
        ),
        (("B", (7, 0)), ("B", (10, 2)), ("X", (9, 6))),
    ),
    (
        2,
        5,
        ((2, 2), (8, 2)),
        (
            ("0", (1, 0)),
            ("0", (0, 4)),
            ("P", (3, 0)),
            ("P", (0, 2)),
            ("P", (2, 6)),
        ),
        (
            ("B", (7, 0)),
            ("B", (10, 2)),
            ("X", (9, 6)),
            ("X", (10, 4)),
        ),
    ),
)


def _build_split_workload(spec, signal_enabled):
    door_row, signal_row, agents, left_resources, right_resources = spec
    resources = [*left_resources, *right_resources]
    open_grid = _role_grid(
        resources,
        agents,
        signal_row,
        signal_enabled,
        door=(door_row, True),
    )
    closed_grid = _role_grid(
        resources,
        agents,
        signal_row,
        signal_enabled,
        door=(door_row, False),
    )
    return [[open_grid, 40], [closed_grid, 160]]


# Keep the former index 1 design and publish it as the only Split layout.
splitnosig_0 = _build_split_workload(_SPLIT_WORKLOAD_SPECS[1], False)
splitsig_0 = _build_split_workload(_SPLIT_WORKLOAD_SPECS[1], True)


def _compact_outage_grid(
    resources,
    agent_positions,
    signal_row,
    signal_enabled,
    notches=(),
):
    """Build a compact 6x9 kitchen with permanently separated movement bays."""
    width, height, center_x = 9, 6, 4
    rows = [["W"] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            rows[y][x] = " "
        rows[y][center_x] = "W"

    rows[signal_row][center_x] = "L" if signal_enabled else "R"
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


# The compact variants reduce an onion-to-handoff-to-pot transfer to only a few
# movement steps. A 40-step normal phase permits at most a short warm-up, while
# the 160-step outage makes sustained right-kitchen production depend on supply
# from the left cook. Counts match the Split workload progression.
_COMPACT_OUTAGE_SPECS = (
    (
        3,
        ((2, 2), (6, 2)),
        (("0", (3, 0)), ("P", (1, 0)), ("B", (1, 5)), ("X", (3, 5))),
        (),
    ),
    (
        4,
        ((2, 3), (6, 3)),
        (
            ("0", (3, 0)),
            ("P", (1, 0)),
            ("P", (0, 3)),
            ("B", (1, 5)),
            ("X", (3, 5)),
        ),
        (),
    ),
    (
        1,
        ((2, 3), (6, 3)),
        (
            ("0", (3, 0)),
            ("0", (0, 2)),
            ("P", (2, 0)),
            ("P", (0, 4)),
            ("B", (1, 5)),
            ("X", (3, 5)),
        ),
        (),
    ),
    (
        4,
        ((2, 3), (6, 3)),
        (
            ("0", (3, 0)),
            ("P", (1, 0)),
            ("P", (2, 0)),
            ("P", (0, 3)),
            ("B", (1, 5)),
            ("B", (2, 5)),
            ("X", (3, 5)),
        ),
        (),
    ),
    (
        1,
        ((2, 3), (6, 3)),
        (
            ("0", (3, 0)),
            ("0", (0, 1)),
            ("P", (1, 0)),
            ("P", (2, 0)),
            ("P", (0, 3)),
            ("B", (1, 5)),
            ("B", (2, 5)),
            ("X", (3, 5)),
            ("X", (0, 4)),
        ),
        (),
    ),
)


def _build_compact_outage_variant(spec, signal_enabled):
    signal_row, agents, left_resources, notches = spec
    right_resources = tuple((symbol, (8 - x, y)) for symbol, (x, y) in left_resources)
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


# Keep the former index 0 design as the only Outage layout.
outagenosig_0 = _build_compact_outage_variant(_COMPACT_OUTAGE_SPECS[0], False)
outagesig_0 = _build_compact_outage_variant(_COMPACT_OUTAGE_SPECS[0], True)

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
