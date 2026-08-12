# Overcooked V3

`overcooked_v3` starts from the `main` branch's `overcooked_v2` implementation
and adds cyclic layout changes. The existing `overcooked` and `overcooked_v2`
environments remain unchanged from `main`.

```python
from jaxmarl import make

env = make("overcooked_v3", layout="dynamic_00")
```

Layout definitions live in `dynamic_layout_data.py` and are validated by
`dynamic_layouts.py`. At phase boundaries, V3 updates the static map, clears
objects on changed cells, and relocates agents that would otherwise be covered
by a newly blocked tile.

The default V3 observation adds three channels to V2's 30-channel grid encoding.
The first added channel is a public signal timer located on the signal tile. It
decreases from `1.0` to `0.1` over 10 observed steps after a press. Each press
costs `0.1` team reward by default. The remaining channels contain a spatially
constant transition countdown and a binary map-change mask. Both stay at zero
until `transition_warning_steps` (20 by default) before a transition. During
that window, the countdown decreases from `1.0` to `0.05`, and the mask marks
cells whose static object differs in the next phase. Disable all three added
features when loading a legacy 30-channel policy.

## Role-coordination scenarios

`split_no_sig` and `split_sig` contain connected left and right kitchen bays.
The full cooking workload (two pots, plates, and serving) alternates between
the bays every 100 steps, while both onion piles remain available. This gives
the agent in the inactive bay a concrete reason to cross over and assist.

`outage_no_sig` and `outage_sig` have disconnected movement regions and one
shared handoff counter. Both bays are complete kitchens with a pot, plates,
serving, and onions. During the outage, the right onion pile becomes a wall, so
the left cook must trade off local production against supplying the right cook
through the handoff counter. Sig layouts replace one inert recipe display with
the activatable public signal; it cannot be used as extra object storage in
either condition.
