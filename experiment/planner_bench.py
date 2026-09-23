"""The planner on the six 75-step kitchens, one line a kitchen.

Prints each kitchen's six-seed mean, the change against the recorded
baseline and the worst of those changes, so a rule that buys one kitchen
with another shows up in the same glance as the mean.

    python experiment/planner_bench.py '{}'
    python experiment/planner_bench.py '{"stage_while_cooking": true}'
    python experiment/planner_bench.py '{}' --maps outage_0 outage_1 --seeds 3
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, numpy as np, jaxmarl
from baselines.planner.planner import GreedyPlanner
from baselines.planner.eval_planner import play

BASE = {"split_0": 240.0, "split_1": 240.0, "outage_0": 336.7, "outage_1": 160.0,
        "distance_0": 260.0, "distance_1": 400.0}
XP = {"split_0": 225, "split_1": 167, "outage_0": 234, "outage_1": 83,
      "distance_0": 117, "distance_1": 387}

args = sys.argv[1:]
kwargs = json.loads(args[0]) if args and not args[0].startswith("--") else {}
rest = args[1:] if args and not args[0].startswith("--") else args
seeds = 6
maps = list(BASE)
if "--seeds" in rest:
    seeds = int(rest[rest.index("--seeds") + 1])
if "--maps" in rest:
    start = rest.index("--maps") + 1
    named = []
    for value in rest[start:]:
        if value.startswith("--"):
            break
        named.append(value)
    maps = named

means, deltas = {}, []
for layout in maps:
    env = jaxmarl.make("overcooked_v3", layout=layout, max_steps=450, random_agent_positions=False,
                       include_transition_countdown=True, include_layout_change_mask=True,
                       transition_observer="both", transition_warning_steps=20)
    planner = GreedyPlanner(env, **kwargs)
    scores = [play(env, planner, i, 450) for i in range(seeds)]
    m = float(np.mean(scores)); means[layout] = m
    d = m - BASE[layout]; deltas.append(d)
    flag = "" if m >= XP[layout] else f"  (XP {XP[layout]}, short {m - XP[layout]:+.0f})"
    print(f"  {layout:11s} {m:6.1f}  {d:+6.1f}{flag}   {scores}", flush=True)
print(f"  MEAN {np.mean(list(means.values())):6.1f}  {np.mean(deltas):+6.1f}   worst {min(deltas):+.1f}")
