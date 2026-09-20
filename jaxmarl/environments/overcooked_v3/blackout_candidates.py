"""The two Blackout kitchens. Grids only -- nothing here is registered.

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

Here the second grid of each pair only takes the plates away, because that is
what this family is for, but nothing stops a pair from rearranging the room.

``ORDER`` says which grid of the pair runs first; the episode alternates from
there and holds the sixth phase to the end.

Edit the grids. Every row of one grid must be the same width, the border must
carry no floor, and every pile, pot and window needs a floor cell beside it or
nobody can reach it.
"""

# Both kitchens ask the same question -- what a pair does when the plates stop
# coming -- of two floors that make fetching a plate expensive in different
# ways, so that stocking up before the countdown is worth the detour it costs.

# blackout_0 splits the floor in two lengthways. Cooking happens along the top,
# where the single pot sits at one end and the onions at the other, so a soup
# is already a walk; the plates are down in the lower corridor, which the top
# only reaches round its two ends. The pile stands in the middle of that
# corridor and divides it, giving each half its own way back up.
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

# blackout_1 keeps everything but the plates on one open row: two pots, two
# onion piles and the recipe above it, two serving windows below. The plates
# are in a corridor under that, reached by a single one-wide passage down the
# middle, so only one cook can be fetching at a time and the other cannot help
# until it comes back up.
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


#: The pair of grids each kitchen alternates between.
PAIRS = {
    "blackout_0": (BLACKOUT_0_PLATES, BLACKOUT_0_NONE),
    "blackout_1": (BLACKOUT_1_PLATES, BLACKOUT_1_NONE),
}

#: Which grid of the pair the episode opens on: 0 is the full kitchen, 1 the
#: blacked-out one. Both of these open full and end blacked out, the order the
#: rest of the family uses. A kitchen that lost its serving window rather than
#: its plates would want the other order, so that its last phase is one the
#: pair can score in.
ORDER = {
    "blackout_0": 0,
    "blackout_1": 0,
}

PHASE_STEPS = 75
FINAL_STEPS = 1000
