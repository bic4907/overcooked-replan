"""Render observer-study or wide layouts with the native V3 visualizer."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import Direction, Position
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


LAYOUT_ROWS = (
    ("split_0", "split_1"),
    ("outage_0", "outage_1"),
    ("distance_0", "distance_1"),
)
WIDE_LAYOUT_ROWS = (
    ("split_wide",),
    ("outage_wide",),
    ("distance_switch_wide",),
)
PHASES = (("Phase A", 0), ("Phase B", 1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", action="store_true", help="Render the three wide maps")
    parser.add_argument("--layouts", nargs="+", help="Render named layouts, one per row")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    mac_name = "Arial Bold.ttf" if bold else "Arial.ttf"
    for candidate in (name, f"/System/Library/Fonts/Supplemental/{mac_name}"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _phase_state(env, initial_state, phase_index):
    phase_positions = env.phase_agent_positions[phase_index]
    phase_step = jnp.sum(env.phase_durations[:phase_index], dtype=jnp.int32)
    recipe = jnp.where(
        env.phase_has_recipe[phase_index],
        env.phase_recipes[phase_index],
        initial_state.recipe,
    )
    grid = initial_state.grid.at[:, :, 0].set(
        env.phase_static_objects[phase_index]
    )
    grid = grid.at[:, :, 1:].set(0)
    agents = initial_state.agents.replace(
        pos=Position(x=phase_positions[:, 0], y=phase_positions[:, 1]),
        dir=jnp.full_like(initial_state.agents.dir, Direction.UP),
        inventory=jnp.zeros_like(initial_state.agents.inventory),
    )
    state = initial_state.replace(
        agents=agents,
        grid=grid,
        step=phase_step,
        done=jnp.array(False),
        recipe=recipe,
        previous_recipe=recipe,
        next_recipe=recipe,
        legacy_recipe_deliveries_remaining=jnp.array(0, dtype=jnp.int32),
        layout_index=jnp.array(phase_index, dtype=jnp.int32),
    )
    return env._set_transition_awareness(state)


def _render_layout(layout_name, tile_size, seed):
    env = make(
        "overcooked_v3",
        layout=layout_name,
        max_steps=450,
        random_agent_positions=False,
    )
    _, initial_state = env.reset(jax.random.PRNGKey(seed))
    visualizer = OvercookedV3Visualizer(tile_size=tile_size)
    return {
        label: Image.fromarray(
            np.asarray(
                visualizer._render_frame(
                    _phase_state(env, initial_state, phase_index)
                )
            )
        )
        for label, phase_index in PHASES
    }


def _pair_sheet(layout_name, frames):
    padding = 28
    frame_gap = 24
    title_font = _load_font(42, bold=True)
    phase_font = _load_font(36, bold=True)
    title_height = 58
    phase_height = 50
    frame_widths = [frames[label].width for label, _ in PHASES]
    frame_height = max(frames[label].height for label, _ in PHASES)
    width = 2 * padding + sum(frame_widths) + frame_gap
    height = 2 * padding + title_height + phase_height + frame_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    title_box = draw.textbbox((0, 0), layout_name, font=title_font)
    title_width = title_box[2] - title_box[0]
    draw.text(
        ((width - title_width) / 2, padding),
        layout_name,
        fill="black",
        font=title_font,
    )

    x = padding
    frame_y = padding + title_height + phase_height
    for label, _ in PHASES:
        frame = frames[label]
        label_box = draw.textbbox((0, 0), label, font=phase_font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (x + (frame.width - label_width) / 2, padding + title_height),
            label,
            fill="black",
            font=phase_font,
        )
        sheet.paste(frame, (x, frame_y + (frame_height - frame.height) // 2))
        x += frame.width + frame_gap
    return sheet


def _contact_sheet(pair_sheets, layout_rows=LAYOUT_ROWS):
    outer_padding = 32
    column_gap = 36
    row_gap = 36
    cell_width = max(sheet.width for sheet in pair_sheets.values())
    row_heights = [
        max(pair_sheets[name].height for name in layout_row)
        for layout_row in layout_rows
    ]
    columns = max(len(row) for row in layout_rows)
    width = 2 * outer_padding + columns * cell_width + (columns - 1) * column_gap
    height = 2 * outer_padding + sum(row_heights) + row_gap * (
        len(layout_rows) - 1
    )
    contact = Image.new("RGB", (width, height), "white")

    y = outer_padding
    for layout_row, row_height in zip(layout_rows, row_heights):
        for column, name in enumerate(layout_row):
            sheet = pair_sheets[name]
            cell_x = outer_padding + column * (cell_width + column_gap)
            x = cell_x + (cell_width - sheet.width) // 2
            contact.paste(sheet, (x, y + (row_height - sheet.height) // 2))
        y += row_height + row_gap
    return contact


def main():
    args = parse_args()
    layout_rows = WIDE_LAYOUT_ROWS if args.wide else LAYOUT_ROWS
    if args.layouts:
        layout_rows = tuple((name,) for name in args.layouts)
    if args.output_dir is None:
        args.output_dir = Path(
            "docs/overcooked_v3/wide_layouts"
            if args.wide
            else "paper/figures/overcooked_v3_layouts"
        )
    if args.tile_size < 8:
        raise ValueError("--tile-size must be at least 8")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_options = {"dpi": (args.dpi, args.dpi)}
    pair_sheets = {}
    for layout_row in layout_rows:
        for layout_name in layout_row:
            frames = _render_layout(layout_name, args.tile_size, args.seed)
            for label, _ in PHASES:
                phase_name = label.lower().replace(" ", "_")
                frames[label].save(
                    args.output_dir / f"{layout_name}_{phase_name}.png",
                    **save_options,
                )
            pair_sheets[layout_name] = _pair_sheet(layout_name, frames)
            pair_sheets[layout_name].save(
                args.output_dir / f"{layout_name}.png", **save_options
            )

    contact = _contact_sheet(pair_sheets, layout_rows)
    output_path = args.output_dir / (
        "layout_candidates.png" if args.layouts else
        "wide_layouts_3.png" if args.wide else "observer_layouts_6.png"
    )
    contact.save(output_path, **save_options)
    print(f"Rendered {len(pair_sheets)} layouts to {args.output_dir.resolve()}")
    print(f"Combined figure: {output_path.resolve()}")


if __name__ == "__main__":
    main()
