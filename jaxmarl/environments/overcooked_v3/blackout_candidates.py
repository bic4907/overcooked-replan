"""Candidate Blackout kitchens 2-9. Grids only -- nothing here is registered.

Symbols, as everywhere else in Overcooked V3:

    ' '  floor            W  counter (objects may be put down on it)
    A    a cook's start   N  blocker (a wall nothing can be put on)
    0    onion pile       B  plate pile
    P    pot              X  serving window        R  recipe indicator

Each kitchen is a pair of whole grids. A phase does not toggle tiles: the
engine swaps one grid for the other every 75 steps and carries the cooks,
what they are holding and what is in the pots straight across. So the two
grids of a pair may differ in anything at all -- walls, piles, pots, doors --
as long as they agree on the things ``DynamicLayout`` checks:

    the same width and height, exactly two cooks each, whose starts stand on
    floor, the same number of ingredient kinds, and the same recipe list.

Here the second grid of each pair only takes something away, because that is
what this family is for, but nothing stops a pair from rearranging the room.

``ORDER`` says which grid of the pair runs first; the episode alternates from
there and holds the sixth phase to the end.

Edit the grids. Every row of one grid must be the same width, the border must
carry no floor, and every pile, pot and window needs a floor cell beside it or
nobody can reach it.
"""

# -- 1. The serving window disappears --------------------------------------
# Cooking carries on with nowhere to put the food, so the pair banks soups in
# the pots, on the plates in their hands and on the counters, then empties the
# bank when the window comes back. With the window open they have to settle
# who cooks and who runs the food out.
#
# These two run the window-less grid FIRST, so the episode ends with the
# window back; otherwise the last phase cannot score what it cooked.

BLACKOUT_2_OPEN = """
WW0W0WR
B     B
W W W W
W A A W
WWPWPWW
"""
BLACKOUT_2_SHUT = """
WW0W0WR
B     B
W W W W
X A A X
WWPWPWW
"""

BLACKOUT_3_OPEN = """
WW0W0WR
B     B
W W W W
W A A W
WWPWPWW
"""
BLACKOUT_3_SHUT = """
WW0W0WR
W     W
W W W W
X A A X
WWPWPWW
"""

# -- 2. Plates worth stacking (blackout_0 is the selected one) --------------
# Each plate pile sits in a nook whose counters are all within a step, so a
# stockpile costs three or four steps a plate instead of a round trip, and the
# blackout is decided by whether one was built. Both nooks have two ways out,
# so a cook fetching plates cannot be shut in by the other.

BLACKOUT_4_PLATES = """
WWWWWWWWW
WWWW    O
WB W    W
W A   A O
WWPWXWPWW
"""
BLACKOUT_4_NONE = """
WWWWWWWWW
WWWW    O
WW W    W
W A   A O
WWPWXWPWW
"""

BLACKOUT_5_PLATES = """
WWWWXWWWW
WB W W BW
W  A A  W
WWP0W0PWW
"""
BLACKOUT_5_NONE = """
WWWWXWWWW
WW W W WW
W  A A  W
WWP0W0PWW
"""

# -- 3. Two rooms, one supply each -----------------------------------------
# The dividing wall is the only way between them: cooks cannot pass, objects
# can, by being put down on it. Blackout_6 takes the right cook's onions away,
# so the left one has to supply both rooms through the wall. Blackout_7 gives
# both rooms everything except plates, which only the right room can reach,
# and then takes the plates away.

BLACKOUT_6_BOTH = """
WW0WWW0WW
W   W   W
B A W A W
W   W   W
WPWXWXWPW
"""
BLACKOUT_6_ONE = """
WWWWWW0WW
W   W   W
B A W A W
W   W   W
WPWXWXWPW
"""

BLACKOUT_7_PLATES = """
WW0WWW0WW
W   W   W
B A W A W
W   W   W
WPWXWXWPW
"""
BLACKOUT_7_NONE = """
WW0WWW0WW
W   W   W
W A W A W
W   W   W
WPWXWXWPW
"""

# -- 4. Plates expensive before they are gone (blackout_1 selected) ---------
# Blackout_1 puts the pile at the end of a one-wide dead end, so one cook does
# the fetching and the other cannot help until the countdown makes it worth
# taking turns. Blackout_9 splits the kitchen into a cooking floor and a
# serving corridor joined only at the two ends, with the plates down in the
# corridor, where the pile also divides the corridor into a half per window.

BLACKOUT_1_PLATES = """
WWP0R0PWW
WA     AW
WXWW WWXW
WWWW WWWW
WB     BW
WWWWWWWWW
"""
BLACKOUT_1_NONE = """
WWP0R0PWW
WA     AW
WXWW WWXW
WWWW WWWW
WW     WW
WWWWWWWWW
"""

BLACKOUT_0_PLATES = """
WWWWWXWWWWW
P A     A O
W WWWRWWW W
W         W
WWWWWBWWWWW
"""
BLACKOUT_0_NONE = """
WWWWWXWWWWW
P A     A O
W WWWRWWW W
W         W
WWWWWWWWWWW
"""


#: The pair of grids each kitchen alternates between. blackout_0 and
#: blackout_1 are the two the benchmark keeps; the rest stay registered as
#: the candidates they were chosen from.
PAIRS = {
    "blackout_0": (BLACKOUT_0_PLATES, BLACKOUT_0_NONE),
    "blackout_1": (BLACKOUT_1_PLATES, BLACKOUT_1_NONE),
    "blackout_2": (BLACKOUT_2_OPEN, BLACKOUT_2_SHUT),
    "blackout_3": (BLACKOUT_3_OPEN, BLACKOUT_3_SHUT),
    "blackout_5": (BLACKOUT_5_PLATES, BLACKOUT_5_NONE),
    "blackout_6": (BLACKOUT_6_BOTH, BLACKOUT_6_ONE),
    "blackout_7": (BLACKOUT_7_PLATES, BLACKOUT_7_NONE),
    "blackout_4": (BLACKOUT_4_PLATES, BLACKOUT_4_NONE),
}

#: Which grid of the pair the episode opens on: 0 is the full kitchen, 1 the
#: blacked-out one. The two that lose their serving window open blacked out,
#: so the last phase is one the pair can actually score in.
ORDER = {
    "blackout_0": 0,
    "blackout_1": 0,
    "blackout_2": 0,
    "blackout_3": 0,
    "blackout_5": 0,
    "blackout_6": 0,
    "blackout_7": 0,
    "blackout_4": 0,
}

PHASE_STEPS = 75
FINAL_STEPS = 1000
