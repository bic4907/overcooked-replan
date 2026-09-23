"""Two differently configured planners, one to a seat.

What the planner does about a partner it has never met is measured by
giving the other seat a planner with its collision rule fixed: one that
never gives way, or one that always does.

    python experiment/planner_pair.py '{}' '{"yield_style": "never"}' split_0
    python experiment/planner_pair.py '{"yield_style": "always"}' '{}' outage_1
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np, jaxmarl
from baselines.planner.planner import GreedyPlanner
ka, kb = json.loads(sys.argv[1]), json.loads(sys.argv[2]); layouts = sys.argv[3:]
for layout in layouts:
    env = jaxmarl.make("overcooked_v3", layout=layout, max_steps=450, random_agent_positions=False,
                       include_transition_countdown=True, include_layout_change_mask=True,
                       transition_observer="both", transition_warning_steps=20)
    pa, pb = GreedyPlanner(env, **ka), GreedyPlanner(env, **kb)
    acta, actb = jax.jit(pa.actions), jax.jit(pb.actions); step = jax.jit(env.step_env)
    scores = []; stuck_total = []
    for seed in range(6):
        key = jax.random.PRNGKey(seed); _, s = env.reset(key); ca, cb = pa.initial_carry(s), pb.initial_carry(s)
        total = 0.0; stuck = 0
        for t in range(450):
            key, k1, k2, k3 = jax.random.split(key, 4)
            ca, aa = acta(ca, s, k1); cb, ab = actb(cb, s, k2)
            actions = {env.agents[0]: aa[0], env.agents[1]: ab[1]}
            _, s, r, d, _ = step(k3, s, actions); total += float(r["agent_0"])
            stuck += int(ca.streak[0] > 0) + int(cb.streak[1] > 0)
        scores.append(total); stuck_total.append(stuck)
    print(f"  {layout}: mean {np.mean(scores):.1f} {scores}  blocked-steps/ep {np.mean(stuck_total):.0f}", flush=True)
