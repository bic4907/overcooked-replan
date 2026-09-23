# The scripted cook

`planner.py` is the paper's scripted human (Appendix B.2): every step it
re-plans the next high-level task, takes the first action of that plan, and
three dials — `prob_wait`, `lltemp`, `hltemp` — set how far from optimal it
plays. With every dial at zero it is the optimal greedy player, which is the
reference the human model is read against (called BR here); `prob_wait=0.5`
is H0.

## What it adds to the paper's model

The paper's kitchen never asks two questions this benchmark does: who does
which job when the kitchen changes under the pair, and what to do about a
partner that will not move. The rules below answer them. Each is written in
terms a kitchen can be measured by — distances, cooking time, how many rooms
the floor has — never in terms of a particular kitchen's coordinates, and
each was kept only because it paid on the six 75-step kitchens without
costing another one.

**Dividing the work**

- *Loader and plate cook.* With two pots reachable, the cook nearer the
  ingredient pile loads them and the other keeps to plates and serving. The
  job does not change hands until the other cook is nearer by a clear margin
  (`role_hysteresis`), because re-deciding it every step has both of them
  carrying ingredients and both waiting with plates.
- *Long hauls.* One pot is a full-time job too when the round trip to the
  pile outlasts the cooking time (`long_haul_specialist`): a second loader in
  that corridor is in the first one's way more than it is at the pile.
- *Comparative advantage.* Where the kitchen is divided in every phase, who
  delivers is decided by what each cook loses by walking the soup out rather
  than by who is nearer the window (`advantage_roles`).
- *A partner that has stopped.* With the other cook's hands unchanged for
  `partner_idle_steps`, or no pot taking or giving up anything for
  `stall_steps`, there is nothing to divide and this cook does the whole job.

**Preparing for the change**

- *Crossing before the door shuts.* Whichever cook can reach the far room
  soonest goes, and it leaves when the walk needs it to rather than on a
  fixed warning. A cook standing in the doorway belongs to no room of the
  phase ahead; it counts as sharing, or the door shuts with both of them on
  one side and the far end scores nothing for a phase.
- *Stocking against a blackout.* From the start of a phase whose successor
  has no plate pile, the plate cook keeps two plates a pot out on the
  counters (`plate_stock`).
- *Plating what is already cooked.* A soup that is done and has nobody
  bringing it a plate is worth a plate however many are already out, and
  worth taking back one this cook had set down for the other.

**Getting out of each other's way**

- Cells the other cook stands in, or is about to step into, are expensive but
  not impassable, so the one route through a corridor stays usable.
- Blocked, a cook holds its plan for as long as the other cook's record of
  giving way warrants, then steps aside on a coin (`yield_style`,
  `hold_max`).
- Blocked for `stubborn_steps`, or no closer to its goal for
  `no_progress_steps`, it treats the other cook as a wall and re-plans --
  which is what keeps a partner that never gives way from costing the
  episode. The second test is the one that fires in a one-wide corridor,
  where a cook can turn on the spot one step and step back the next without
  ever being blocked twice running.
- A cook with nothing to do walks to where its own next job will start
  rather than standing where it stopped.
- *A soup nobody comes for.* After `neglect_steps` in a finished pot or on a
  counter, it is this cook's to plate and serve, whatever its usual job.

## Measuring

    python experiment/planner_bench.py '{}'                    # six kitchens, six seeds
    python experiment/planner_bench.py '{"plate_stock": 3}'    # one option changed
    python experiment/planner_pair.py '{}' '{"yield_style": "never"}' split_0

`planner_bench` prints each kitchen's mean, its change against the recorded
baseline and the worst of those changes, so a rule that buys one kitchen with
another is as visible as the mean. `planner_pair` puts a differently
configured planner in the other seat, which is how the collision rules are
measured against a partner that never gives way and one that always does.

Options that were tried and cost more than they earned are listed, with what
they measured, in the module docstring of `planner.py`. Read those numbers
before turning one back on.
