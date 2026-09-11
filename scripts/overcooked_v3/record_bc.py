"""Record a full BC-versus-BC episode using the checkpoint's environment config."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from baselines.BC.policy import load_policy
from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/bc/split_0_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logits_fn, config = load_policy(args.model)
    env = OvercookedV3(**config["env_kwargs"])
    key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed))
    obs, state = env.reset(reset_key)
    step = jax.jit(env.step_env)
    states, scores, actions_log = [state], [0.0], []
    total = 0.0
    for t in range(config["env_kwargs"]["max_steps"]):
        key, action_key, step_key = jax.random.split(key, 3)
        logits = logits_fn(jnp.stack([obs[a] for a in env.agents]))
        sampled = jax.random.categorical(action_key, logits)
        actions = {a: sampled[i] for i, a in enumerate(env.agents)}
        obs, state, rewards, dones, _ = step(step_key, state, actions)
        total += float(rewards[env.agents[0]])
        states.append(state)
        scores.append(total)
        actions_log.append(np.asarray(sampled).tolist())
        if (t + 1) % 50 == 0:
            print(f"Rollout {t + 1}: score={total:g}", flush=True)
        if bool(dones["__all__"]):
            break
    visualizer = OvercookedV3Visualizer(
        tile_size=64,
        subdivs=1,
        seconds_per_step=0.2,
        transition_warning_steps=env.transition_warning_steps,
    )
    font = ImageFont.load_default(size=18)
    thumbnails = []
    selected = {0, 149, 150, 299, 300, len(states) - 1}
    with imageio.get_writer(
        str(args.output),
        format="FFMPEG",
        fps=5,
        codec="libx264",
        quality=7,
        macro_block_size=2,
        output_params=["-r", "60", "-movflags", "+faststart"],
    ) as writer:
        for i, (frame_state, score) in enumerate(zip(states, scores)):
            board = Image.fromarray(visualizer.render_sequence([frame_state])[0])
            frame = Image.new(
                "RGB", (max(640, board.width), board.height + 100), "#111827"
            )
            frame.paste(board, ((frame.width - board.width) // 2, 100))
            draw = ImageDraw.Draw(frame)
            draw.text(
                (18, 12),
                f"BC vs BC | {config['env_kwargs']['layout']} | sampled | seed {args.seed}",
                font=font,
                fill="white",
            )
            draw.text(
                (18, 40),
                f"Step {i:03d}/{len(states) - 1}    Score {score:g}    Time {i / 5:.1f}s",
                font=font,
                fill="white",
            )
            caption = visualizer.caption_with_countdown(
                frame_state, "Red: BC     Blue: BC"
            )
            draw.text((18, 68), caption, font=font, fill="#fbbf24")
            writer.append_data(np.asarray(frame))
            if i in selected:
                thumbnails.append(frame.copy())
            if i % 100 == 0:
                print(f"Rendered {i}/{len(states) - 1}", flush=True)
    sheet = Image.new("RGB", (thumbnails[0].width * 3, thumbnails[0].height * 2))
    for i, frame in enumerate(thumbnails):
        sheet.paste(frame, ((i % 3) * frame.width, (i // 3) * frame.height))
    sheet.save(args.output.with_suffix(".png"))
    report = {
        "seed": args.seed,
        "steps": len(actions_log),
        "team_return": total,
        "policy": str(args.model),
        "policy_sha256": hashlib.sha256(
            (args.model / "policy.msgpack").read_bytes()
        ).hexdigest(),
        "action_selection": "independent categorical samples from shared BC logits",
        "env_kwargs": config["env_kwargs"],
        "environment_hz": 5,
        "video_fps": 60,
        "actions": actions_log,
        "scores": scores,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output.resolve()} | score={total:g}", flush=True)


if __name__ == "__main__":
    main()
