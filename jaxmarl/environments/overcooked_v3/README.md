# Overcooked V3

`overcooked_v3` starts from the `main` branch's `overcooked_v2` implementation
and adds cyclic layout changes. The existing `overcooked` and `overcooked_v2`
environments remain unchanged from `main`.

```python
from jaxmarl import make

env = make("overcooked_v3", layout="split_0")
```

Layout definitions live in `dynamic_layout_data.py` and are validated by
`dynamic_layouts.py`. At phase boundaries, V3 updates the static map, clears
objects on changed cells, and relocates agents that would otherwise be covered
by a newly blocked tile.

The signal-free V3 grid encoding has 29 channels. The default observation adds
a spatially constant transition countdown and a binary map-change mask, for 31
channels total. Both stay at zero
until `transition_warning_steps` (20 by default) before a transition. During
that window, the countdown decreases from `1.0` to `0.05`, and the mask marks
cells whose static object differs in the next phase. Disable both transition
features for the 29-channel encoding.

## Role-coordination scenarios

The three `*_wide` variants expand the selected `_0` maps to 13 columns,
adding one row for Outage and Distance Switch: `split_wide` is 13×7, `outage_wide` is 13×6,
and `distance_switch_wide` is 13×6 (width×height). They retain resource counts,
recipes, and transition timing. See the [design notes and phase images](../../../docs/overcooked_v3/wide_maps.md).

`split_0` uses a 7×9 map. `split_narrow` compresses the same workload and
doorway cycle into a 6×7 map. It opens one central doorway
for 150 steps, then turns it into a handoff counter for 150 steps. The
left bay contains onions and pots, while the right bay contains plates and
serving. Agents must choose opposite sides before the wall closes, then
coordinate cook–server work through the counter.

`outage_0` has a compact 5×7 map with
disconnected movement regions and shared center counters. Both bays are
complete kitchens with pots, plates, serving, and onions. After a 150-step
normal phase, every right onion pile becomes a wall for 150 steps, so the left cook must trade off local
production against supplying the right cook through a handoff counter. Both
conditions keep a separate fixed recipe display and a generic non-storage
blocker in the center column. The
onion-to-handoff route requires no movement and the handoff-to-right-pot route
requires at most one move,
making cross-kitchen supply competitive with continuing local production. The
center column remains blocked in every phase: agents cannot cross bays and can
exchange onions only through shared handoff counters.
In selected `outage_0`, the blocker occupies the bottom center tile, leaving two
adjacent counters above it where the left cook can preload onions. Candidate
`outage_1` retains adjacent handoffs but changes the resource locations and adds
mirrored lower notches. The surviving onion-to-handoff and handoff-to-pot relay
therefore stays short while the left cook must choose between local production
and supplying the right bay as soon as the outage begins.
Outage uses a two-onion recipe, so its pots begin cooking as soon as the second
onion is added. Split retains the standard three-onion recipe, and both
scenarios retain the standard 20-step cooking timer. Each family exposes one
selected Easy layout under `_0`; Split and Outage also expose redesigned `_1`
candidates. The existing observer W&B runs recorded the current `outage_0`
geometry under the historical `outage_1` name, so new candidate runs must also
be filtered by their `LAYOUT_REVISION`.

`recipe_switch_0` is a Mixed Recipe Relay layout.
The center divider permanently separates an onion/serving bay from a
tomato/plate bay and leaves exactly two shared storage counters. Both bays have
at least one pot. The retained former catalog variant `_7` is reindexed as
`_0`. It is 7×5 and starts tomato-majority. The lower-cost cooking
side changes with the active mixed recipe. Every 450-step episode follows a
deterministic A → B → A schedule. Layout geometry never changes. A dish
that had already started cooking before a switch remains deliverable, but a new
pot can start only when its contents match the current recipe. Recipe Relay
adds two next-recipe preview channels to the standard V3 observation.

`distance_switch_0` and `distance_switch_1` are Distance-Driven Role Switch
layouts based on `asymm_advantages`. All four selected Easy role scenarios use the
same 450-step A → B → A schedule, with changes at steps 150 and 300.
The standard three-onion recipe is
fixed, and each agent's separate work region contains access to an onion pile,
central pot, plate pile, and serving station. Pots and plates remain fixed.
Canonical `_0` exchanges endpoint types in place; 9×6 candidate `_1` relocates
all onion and serving stations onto positions that were inactive counters in
the other phase. Both reverse which agent has the short onion-input loop and
which has the short serving loop.
Because no recipe is scheduled, this layout uses the standard 31-channel V3
observation rather than next-recipe preview channels.
