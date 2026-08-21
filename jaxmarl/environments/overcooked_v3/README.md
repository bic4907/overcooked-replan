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
decreases from `1.0` to `0.1` over 10 observed steps after a press and is
visible to both agents. Pressing the button has no reward cost. The remaining
channels contain a spatially
constant transition countdown and a binary map-change mask. Both stay at zero
until `transition_warning_steps` (20 by default) before a transition. During
that window, the countdown decreases from `1.0` to `0.05`, and the mask marks
cells whose static object differs in the next phase. Disable all three added
features when loading a legacy 30-channel policy.

## Role-coordination scenarios

`splitnosig_0` through `splitnosig_2` and their matching `splitsig` layouts use
7×9 maps. They open one central doorway
for 40 steps, then turn it into a handoff counter for 160 steps. The
left bay contains onions and pots, while the right bay contains plates and
serving. Agents must choose opposite sides before the wall closes, then
coordinate cook–server work through the counter.

`outagenosig_0` through `outagenosig_2` and their matching `outagesig` layouts
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
scenarios retain the standard 20-step cooking timer. The three layouts per
category were selected from the cross-play catalog and reindexed as `_0`, `_1`,
and `_2`. Equal Sig/NoSig indices differ only at the signal cell.

`recipe_switch_0` through `recipe_switch_2` are Mixed Recipe Relay layouts.
The center divider permanently separates an onion/serving bay from a
tomato/plate bay and leaves exactly two shared storage counters. Both bays have
at least one pot. The retained former variants `_4`, `_5`, and `_7` are
reindexed as `_0`, `_1`, and `_2`. New `_0` is 9×5 and starts onion-majority;
new `_1` and `_2` are 7×5 and start tomato-majority. The lower-cost cooking
side changes with the active mixed recipe. Every 450-step episode follows a
deterministic A → B → A schedule. Layout geometry never changes. A dish
that had already started cooking before a switch remains deliverable, but a new
pot can start only when its contents match the current recipe. Recipe Relay
adds two next-recipe preview channels to the standard V3 observation.

`distance_switch_0` through `distance_switch_2` are Distance-Driven Role
Switch layouts based on `asymm_advantages`. The standard three-onion recipe is
fixed, and each agent's separate work region contains access to an onion pile,
central pot, plate pile, and serving station. Pots and plates remain fixed;
only the onion and serving endpoints exchange at steps 150 and 300. This
reverses which agent has the short onion-input loop and which has the short
serving loop. The retained former variants `_0`, `_1`, and `_6` are reindexed
as `_0`, `_1`, and `_2`. They are respectively the canonical 9×5 map, its
wider 11×5 version, and the 11×8 staggered-islands version, without changing
the comparative-cost objective.
Because no recipe is scheduled, these layouts use the standard 33-channel V3
observation rather than next-recipe preview channels.
