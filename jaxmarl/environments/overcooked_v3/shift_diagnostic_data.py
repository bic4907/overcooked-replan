"""Frozen kitchens for the shift-design diagnostic.

Phase B of every drawn cell, written out as a grid rather than regenerated
from a seed, so changing the sampler cannot silently change the experiment and
a layout name means one kitchen for good. ORIGINAL_KITCHENS holds phase A, the
five original Overcooked layouts, for the same reason.

Each kitchen carries one edit of one kind, at the size that kind always takes:
one blocked cell, two swapped station pairs, or half the dispensers rounded
down. What was edited and where was drawn uniformly, with one exception noted
below.

cramped_room's remove was set by hand rather than drawn: the draw kept taking
an onion pile, and with every kitchen keeping its plates the remove edits never
reached the state Resource Outage is about -- a kitchen that can cook but
cannot serve. One kitchen now loses its last plate pile, and it is the smallest
one, where a single pot, a single plate pile and a single delivery square make
that loss unambiguous. The other four removes are as drawn.
"""

ORIGINAL_KITCHENS = {
    "cramped_room": """
WWPWW
OA AO
W   W
WBWXW
""",
    "asymm_advantages": """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
""",
    "coord_ring": """
WWWPW
W A P
BAW W
O   W
WOXWW
""",
    "forced_coord": """
WWWPW
O WAP
OAW W
B W W
WWWXW
""",
    "counter_circuit": """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
""",
}

DRAWN_KITCHENS = {
    ("cramped_room", "block"): """
WWPWW
OAWAO
W   W
WBWXW
""",
    ("cramped_room", "swap"): """
WWPWW
XA AB
W   W
WOWOW
""",
    ("cramped_room", "remove"): """
WWPWW
OA AO
W   W
WWWXW
""",
    ("asymm_advantages", "block"): """
WWWWWWWWW
O WXWOW X
W   P  WW
W A PA  W
WWWBWBWWW
""",
    ("asymm_advantages", "swap"): """
WWWWWWWWW
P WPWOW X
W   O   W
W A XA  W
WWWBWBWWW
""",
    ("asymm_advantages", "remove"): """
WWWWWWWWW
W WXWOW X
W   P   W
W A PA  W
WWWBWWWWW
""",
    ("coord_ring", "block"): """
WWWPW
W A P
BAWWW
O   W
WOXWW
""",
    ("coord_ring", "swap"): """
WWWOW
W A P
BAW W
P   W
WXOWW
""",
    ("coord_ring", "remove"): """
WWWPW
W A P
BAW W
W   W
WOXWW
""",
    ("forced_coord", "block"): """
WWWPW
O WAP
OAW W
B WWW
WWWXW
""",
    ("forced_coord", "swap"): """
WWWOW
O WAX
PAW W
B W W
WWWPW
""",
    ("forced_coord", "remove"): """
WWWPW
O WAP
WAW W
B W W
WWWXW
""",
    ("counter_circuit", "block"): """
WWWPPWWW
W A    W
B WWWWWX
W     AW
WWWOOWWW
""",
    ("counter_circuit", "swap"): """
WWWPOWWW
W A    W
O WWWW X
W     AW
WWWBPWWW
""",
    ("counter_circuit", "remove"): """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWWOWWW
""",
}


# The handcrafted kitchens of the V2 paper and their drawn phase B, under the
# same rule the V1 kitchens were drawn with: half the dispensers, rounded down,
# taken uniformly, capped at four. These kitchens carry few piles, so most of
# these draws leave phase B unable to serve -- which is what Resource Outage
# does by design. The alternating episode still scores in its A phases.
#
# Grounded Coordination Simple is absent: its one difference from Test-Time
# Simple is a button indicator, which V3 carries as a blocker, and a blocker is
# a wall to both the mover and the observation.
V2_ORIGINAL_KITCHENS = {
    "grounded_coord_ring": """
WWW2R2WWW
W       W
W WWNWW W
2 0   B 2
RAXAP X R
2 1   B 2
W WWNWW W
W       W
WWW2R2WWW
""",
    "test_time_simple": """
WW2WWWWW
W  WB  0
R AWPA X
W  WB  1
WW2WWWWW
""",
    "test_time_wide": """
WWXBWW
0 A  0
1    1
WPWPWW
3 A  3
W    W
WWRWWW
""",
    "demo_cook_simple": """
WWWWWR2W0WW
0      W  B
W     APA X
1      W  B
WWWWWR2W1WW
""",
    "demo_cook_wide": """
WWWWBXBWWWW
WWW0 A 1WWW
WWWWWPWWWWW
W    A    W
0  W3R3W  0
W1WWWWWWW1W
""",
}

V2_DRAWN_KITCHENS = {
    ("grounded_coord_ring", "block"): """
WWW2R2WWW
W       W
W WWNWW W
2 0   B 2
RAXAP X R
2 1   B 2
W WWNWW W
W     W W
WWW2R2WWW
""",
    ("grounded_coord_ring", "swap"): """
WWW2RBWWW
W       W
W WWNWW W
2 0   2 2
RAXA2 X R
2 1   B 2
W WWNWW W
W       W
WWWPR2WWW
""",
    ("grounded_coord_ring", "remove"): """
WWW2R2WWW
W       W
W WWNWW W
2 0   B W
RAXAP X R
W 1   W 2
W WWNWW W
W       W
WWWWR2WWW
""",
    ("test_time_simple", "block"): """
WW2WWWWW
W  WBW 0
R AWPA X
W  WB  1
WW2WWWWW
""",
    ("test_time_simple", "swap"): """
WWPWWWWW
W  WB  B
R AW2A X
W  W0  1
WW2WWWWW
""",
    ("test_time_simple", "remove"): """
WW2WWWWW
W  WW  0
R AWPA X
W  WW  W
WW2WWWWW
""",
    ("test_time_wide", "block"): """
WWXBWW
0 A  0
1    1
WPWPWW
3WA  3
W    W
WWRWWW
""",
    ("test_time_wide", "swap"): """
WWXBWW
3 A  1
1    0
WPWPWW
3 A  0
W    W
WWRWWW
""",
    ("test_time_wide", "remove"): """
WWXWWW
0 A  W
W    1
WPWPWW
3 A  3
W    W
WWRWWW
""",
    ("demo_cook_simple", "block"): """
WWWWWR2W0WW
0      W  B
W  W  APA X
1      W  B
WWWWWR2W1WW
""",
    ("demo_cook_simple", "swap"): """
WWWWWRPW0WW
0      W  X
W     A2A B
1      W  B
WWWWWR2W1WW
""",
    ("demo_cook_simple", "remove"): """
WWWWWR2WWWW
W      W  B
W     APA X
W      W  B
WWWWWRWW1WW
""",
    ("demo_cook_wide", "block"): """
WWWWBXBWWWW
WWW0 A 1WWW
WWWWWPWWWWW
W W  A    W
0  W3R3W  0
W1WWWWWWW1W
""",
    ("demo_cook_wide", "swap"): """
WWWWBPBWWWW
WWW0 A 1WWW
WWWWWXWWWWW
W    A    W
1  W3R3W  0
W1WWWWWWW0W
""",
    ("demo_cook_wide", "remove"): """
WWWWWXWWWWW
WWW0 A 1WWW
WWWWWPWWWWW
W    A    W
W  W3R3W  W
W1WWWWWWW1W
""",
}
