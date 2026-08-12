# Overcooked V3 layouts. Both ``0`` and ``O`` denote an onion pile.

# Role-formation scenarios -------------------------------------------------
#
# The Sig/NoSig pairs differ only at one shared cell: ``L`` is V2's
# interactable button indicator, while ``R`` is a non-interactable recipe
# display. Both block movement and neither stores objects, so the comparison
# changes signaling capability without adding a spare handoff counter.
#
# Kitchen Split keeps two passages open between the bays. The complete cooking
# workload (two pots, plates, and serving) alternates sides, while onions remain
# available in both bays. The agent in the inactive bay therefore has a reason
# to carry a resource across a passage and assist the active bay.
#
# Resource Outage permanently separates two otherwise complete kitchens. Each
# bay owns a pot, plate pile, serving station, and onion pile in the normal
# phase. When the right onion pile disappears, the left agent must trade off
# local cooking against supplying onions through the shared counter at (5, 2).

split_no_sig = [
    [
        """
WWWWWWWWWWWWW
W0    W    0W
WP          W
WP  A R A   W
WB          W
WX    W     W
WWWWWWWWWWWWW
""",
        100,
    ],
    [
        """
WWWWWWWWWWWWW
W0    W    0W
W          PW
W   A R A  PW
W          BW
W     W    XW
WWWWWWWWWWWWW
""",
        100,
    ],
]

split_sig = [
    [
        """
WWWWWWWWWWWWW
W0    W    0W
WP          W
WP  A L A   W
WB          W
WX    W     W
WWWWWWWWWWWWW
""",
        100,
    ],
    [
        """
WWWWWWWWWWWWW
W0    W    0W
W          PW
W   A L A  PW
W          BW
W     W    XW
WWWWWWWWWWWWW
""",
        100,
    ],
]

outage_no_sig = [
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

outage_sig = [
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
