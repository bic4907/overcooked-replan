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

The selected benchmark exposes two layouts per family: `split_0`, `split_1`,
`outage_0`, `outage_1`, `distance_0`, and `distance_1`. Here `_0` selects the
retained diagonal narrow layout for Split and Outage, while `_1` selects their
base layouts and the retained wide Distance layout. Descriptive selection-time
names remain available as aliases.

`split_0` is the selected 7×6 diagonal narrow layout. `split_1` is the retained
9×7 base layout. Both open a central doorway for 150 steps, turn it into
a handoff counter for 150 steps, and restore it for the final 150 steps. The
left bay contains onions and pots, while the right bay contains plates and
serving stations.

`outage_0` and `outage_1` are 7×5 shared-room layouts. The first uses the
selected compact diagonal arrangement. Both remove every onion dispenser in
phase B. Stored objects, held inventory, and pot contents persist until the
onions recover at step 300. Outage uses a two-onion recipe; Split retains the
standard three-onion recipe and 20-step cooking timer.

The descriptive aliases `split`, `split_narrow_diagonal`, `outage`, and
`outage_narrow_diagonal` refer to the corresponding numbered layouts. Other
selection-time candidates remain available under their descriptive or explicit
legacy names for reproducing earlier runs.

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

`distance_0` and `distance_1` are Distance-Driven Role Switch layouts based on
`asymm_advantages`. The first is the canonical 9×5 layout and the second is the
selected 13×6 wide layout. Both reverse which agent has the short onion-input
and serving loops during the middle phase. The former `distance_switch` and
`distance_switch_wide` names remain compatible aliases.
Because no recipe is scheduled, this layout uses the standard 31-channel V3
observation rather than next-recipe preview channels.
