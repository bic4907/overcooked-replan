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

`splitnosig_0` through `splitnosig_19` and their matching `splitsig` layouts use
7×9 maps. They open one central doorway
for 40 steps, then turn it into a handoff counter for 160 steps. The
left bay contains onions and pots, while the right bay contains plates and
serving. Agents must choose opposite sides before the wall closes, then
coordinate cook–server work through the counter.

`outagenosig_0` through `outagenosig_19` and their matching `outagesig` layouts
have compact 5×7 maps with
disconnected movement regions and shared center counters. Both bays are
complete kitchens with pots, plates, serving, and onions. After a 40-step
normal phase, every right onion pile becomes a wall for 160 steps, so the left cook must trade off local
production against supplying the right cook through a handoff counter. Both
conditions keep a separate fixed recipe display. At the signal cell, Sig uses
an activatable public button while NoSig uses a blank non-storage blocker with
no button; neither can be used as extra object storage. The
onion-to-handoff route requires no movement and the handoff-to-right-pot route
requires at most one move,
making cross-kitchen supply competitive with continuing local production. The
center column remains blocked in every phase: agents cannot cross bays and can
exchange onions only through shared handoff counters.
The signal occupies the bottom center tile, leaving two adjacent counters above
it where the left cook can preload onions.
Outage uses a two-onion recipe, so its pots begin cooking as soon as the second
onion is added. Split retains the standard three-onion recipe, and both
scenarios retain the standard 20-step cooking timer. Index `_0` is the
unchanged canonical design; indices `_1` through `_19` vary resource counts,
perimeter placement, and agent starts. Equal Sig/NoSig indices differ only at
the signal cell.

The selected Split layout assigns two onions and two pots to the left and one
plate pile and one goal to the right. The selected Outage layout mirrors one of
each resource in both bays during its normal phase and removes the right onion
during outage.
