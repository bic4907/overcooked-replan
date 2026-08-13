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

`splitnosig_0` ... `splitnosig_4` and their `splitsig_*` pairs open one central
doorway for 40 steps, then turn it into a handoff counter for 160 steps. The
left bay contains onions and pots, while the right bay contains plates and
serving. Agents must choose opposite sides before the wall closes, then
coordinate cook–server work through the counter.

`outagenosig_0` ... `outagenosig_4` and their `outagesig_*` pairs have compact
6×9 layouts with
disconnected movement regions and shared center counters. Both bays are
complete kitchens with pots, plates, serving, and onions. After a 40-step
normal phase, every right onion pile becomes a wall for 160 steps, so the left cook must trade off local
production against supplying the right cook through a handoff counter. Sig
layouts replace one inert recipe display with the activatable public signal;
it cannot be used as extra object storage in either condition. The
onion-to-handoff and handoff-to-right-pot routes each require at most two moves,
making cross-kitchen supply competitive with continuing local production. The
center column remains blocked in every phase: agents cannot cross bays and can
exchange onions only through shared handoff counters.

Across indices 0 through 4, per-bay onion/pot/plate/goal counts are
`1/1/1/1`, `1/2/1/1`, `2/2/1/1`, `1/3/2/1`, and `2/3/2/2`. Split assigns
onions/pots to the left and plates/goals to the right. Outage mirrors all
resources during its normal phase and removes every right onion during outage.
