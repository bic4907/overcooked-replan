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

`split_0` and `split_1` use 7×9 maps. They open one central doorway
for 150 steps, then turn it into a handoff counter for 150 steps. The
left bay contains onions and pots, while the right bay contains plates and
serving. Agents must choose opposite sides before the wall closes, then
coordinate cook–server work through the counter.

`outage_0` and `outage_1` have compact 5×7 maps with
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
The blocker occupies the bottom center tile, leaving two adjacent counters above
it where the left cook can preload onions.
Outage uses a two-onion recipe, so its pots begin cooking as soon as the second
onion is added. Split retains the standard three-onion recipe, and both
scenarios retain the standard 20-step cooking timer. The two layouts per
category were ranked by mean absolute XP-SP gap in the 2026-08-22 baseline
report and tagged `_0` and `_1`. Split uses previous tags `_2`, `_0`; Outage
uses previous tags `_1`, `_0`.

`recipe_switch_0` and `recipe_switch_1` are Mixed Recipe Relay layouts.
The center divider permanently separates an onion/serving bay from a
tomato/plate bay and leaves exactly two shared storage counters. Both bays have
at least one pot. The retained former catalog variants `_7` and `_5` are
reindexed as `_0` and `_1`. Both are 7×5 and start tomato-majority. The lower-cost cooking
side changes with the active mixed recipe. Every 450-step episode follows a
deterministic A → B → A schedule. Layout geometry never changes. A dish
that had already started cooking before a switch remains deliverable, but a new
pot can start only when its contents match the current recipe. Recipe Relay
adds two next-recipe preview channels to the standard V3 observation.

`distance_switch_0` and `distance_switch_1` are Distance-Driven Role
Switch layouts based on `asymm_advantages`. All eight role scenarios use the
same 450-step A → B → A schedule, with changes at steps 150 and 300.
The standard three-onion recipe is
fixed, and each agent's separate work region contains access to an onion pile,
central pot, plate pile, and serving station. Pots and plates remain fixed;
only the onion and serving endpoints exchange at steps 150 and 300. This
reverses which agent has the short onion-input loop and which has the short
serving loop. The retained `_0` and `_1` tags are respectively the canonical
9×5 map and its wider 11×5 version, without changing the comparative-cost
objective.
Because no recipe is scheduled, these layouts use the standard 31-channel V3
observation rather than next-recipe preview channels.
