"""Paired Notice / Masked trajectories from the Section 6.2 probe, for drawing.

The probe itself (experiment/notice/probe_preparation.py) keeps only the
endpoint of each rollout. This replays chosen probe states with the same
policy, the same alert-blind H0 partner, the same random stream and the same
recurrent warm-up, and keeps where both cooks stand at every step, so the
twenty steps before the boundary can be drawn.

    python results/explore_notice_traces.py --layout split_0 --seed 0 --stream 0 --state 22

Writes outputs/notice/traces/<layout>_s<seed>_r<stream>_p<state>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment/notice"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import probe_preparation as pp  # noqa: E402
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer  # noqa: E402
from common import (  # noqa: E402
    H0_DIALS, build_env, fetch_policies, planner_controller, policy_controller,
)

OUT = ROOT / "outputs/notice/traces"
RENDER_PX = 128


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", default="split_0")
    parser.add_argument("--arm", default="rnn")
    parser.add_argument("--seed", type=int, default=0, help="Training seed of the informed policy.")
    parser.add_argument("--stream", type=int, default=0, help="Partner random stream.")
    parser.add_argument("--state", type=int, nargs="+", default=[0], help="Probe state indices.")
    parser.add_argument("--artifact-dir", default=str(ROOT / "experiment/notice/artifacts"))
    return parser.parse_args(argv)


def main(argv=None):
    from jaxmarl._env import load_project_env

    load_project_env()
    args = parse_args(argv)
    informed, partner = 0, 1
    family = args.layout.split("_")[0]
    policies = fetch_policies(args.arm, args.layout, 10, 6, artifact_dir=args.artifact_dir)
    config = policies[0][2]
    chosen = next(p for p in policies if p[0] == args.seed)
    params = chosen[3]

    env_off = build_env(args.layout, "none", run_config=config)
    env_on = build_env(args.layout, f"agent_{informed}", run_config=config)
    boundary = int(env_off.phase_durations[0])
    probe_step = boundary - pp.WARNING_STEPS
    pre, post = boundary - probe_step, 20
    steps = pre + post

    stacked, labels = pp.probe_states(env_off, informed, probe_step, pp.INFORMED_START.get(args.layout))
    initial = jax.tree_util.tree_map(np.asarray, stacked)
    cooks, probes = {}, {}
    for arm, env in (("OFF", env_off), ("ON", env_on)):
        controllers = [None, None]
        controllers[informed] = policy_controller(env, config, params, informed, False)
        controllers[partner] = planner_controller(env, H0_DIALS, partner, "static")
        cooks[arm] = controllers
        probes[arm] = pp.make_probe(env, controllers, steps, informed)
    warmup = pp.make_warmup(env_off, cooks["OFF"], probe_step)

    # Exactly the key schedule of probe_preparation.main (base seed 0).
    key = jax.random.PRNGKey(0 + 1000 * args.seed + args.stream)
    warm_key, probe_key = jax.random.split(key)
    hidden = warmup(warm_key, (params, None))[informed][0]

    trajectories, scores = {}, {}
    for arm in ("OFF", "ON"):
        _, _, states = probes[arm](probe_key, (params, None), hidden, stacked)
        traj = pp.prepend(initial, states)
        trajectories[arm] = traj
        scores[arm] = pp.score_endpoints(env_off, family, traj, labels, informed, pre, post)

    OUT.mkdir(parents=True, exist_ok=True)
    for index in args.state:
        record = dict(
            layout=args.layout, arm=args.arm, seed=args.seed, stream=args.stream, state=index,
            pre=pre, post=post, probe_step=probe_step, label=labels[index],
            static_a=np.asarray(env_off.phase_static_objects[0]).tolist(),
            static_b=np.asarray(env_off.phase_static_objects[1]).tolist(),
        )
        frames = {}
        for arm in ("OFF", "ON"):
            traj = trajectories[arm]
            pos = traj.agents.pos
            record[arm] = dict(
                xy=np.stack([np.asarray(pos.x)[index], np.asarray(pos.y)[index]], axis=-1).tolist(),
                dir=np.asarray(traj.agents.dir)[index].tolist(),
                inventory=np.asarray(traj.agents.inventory)[index].tolist(),
                initiation=bool(scores[arm][index]["initiation"]),
                # What sits on every cell and each pot's cooking timer, step by
                # step, so a figure can say when a pot starts and finishes.
                items=np.asarray(traj.grid)[index, : pre + 1, :, :, 1].tolist(),
                timer=np.asarray(traj.grid)[index, : pre + 1, :, :, 2].tolist(),
            )
            # The kitchen as the boundary arrives, rendered, so a figure can
            # show where the twenty steps left things.
            final = jax.tree_util.tree_map(lambda a: a[index, pre], traj)
            viewer = OvercookedV3Visualizer(tile_size=RENDER_PX)
            frames[arm] = np.asarray(viewer._render_frame(final))
            # And with the informed cook alone, so a figure can draw the
            # partner itself -- faded, say -- over the kitchen.
            solo = final.replace(agents=jax.tree_util.tree_map(
                lambda a: a[informed: informed + 1], final.agents))
            frames[f"{arm}_solo"] = np.asarray(viewer._render_frame(solo))
        path = OUT / f"{args.layout}_s{args.seed}_r{args.stream}_p{index}.json"
        path.write_text(json.dumps(record, default=float))
        np.savez_compressed(path.with_suffix(".npz"), **frames)
        print(f"state {index}: initiation OFF {record['OFF']['initiation']} "
              f"ON {record['ON']['initiation']}  -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
