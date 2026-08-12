import math
from functools import partial
from typing import Optional

import imageio
import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl.viz.grid_rendering_v2 as rendering
from jaxmarl.environments.overcooked_v3.common import DynamicObject, StaticObject
from jaxmarl.environments.overcooked_v3.settings import (
    INDICATOR_ACTIVATION_TIME,
    POT_COOK_TIME,
)
from jaxmarl.environments.overcooked_v3.utils import compute_view_box
from jaxmarl.viz.window import Window

TILE_PIXELS = 32
DEFAULT_SECONDS_PER_STEP = 0.2
DEFAULT_TRANSITION_WARNING_STEPS = 20

COLORS = {
    "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
    "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
    "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
    "purple": jnp.array([160, 32, 240], dtype=jnp.uint8),
    "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
    "grey": jnp.array([100, 100, 100], dtype=jnp.uint8),
    "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
    "black": jnp.array([25, 25, 25], dtype=jnp.uint8),
    "orange": jnp.array([230, 180, 0], dtype=jnp.uint8),
    "pink": jnp.array([255, 105, 180], dtype=jnp.uint8),
    "brown": jnp.array([139, 69, 19], dtype=jnp.uint8),
    "cyan": jnp.array([0, 255, 255], dtype=jnp.uint8),
    "light_blue": jnp.array([173, 216, 230], dtype=jnp.uint8),
    "dark_green": jnp.array([0, 150, 0], dtype=jnp.uint8),
}

INGREDIENT_COLORS = jnp.array(
    [
        COLORS["yellow"],
        COLORS["dark_green"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["red"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["pink"],
        COLORS["brown"],
    ]
)


AGENT_COLORS = jnp.array(
    [
        COLORS["red"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["yellow"],
        COLORS["orange"],
    ]
)


class OvercookedV3Visualizer:
    """
    Manages a window and renders contents of EnvState instances to it.
    """

    tile_cache = {}

    def __init__(
        self,
        tile_size=TILE_PIXELS,
        subdivs=3,
        seconds_per_step=DEFAULT_SECONDS_PER_STEP,
        transition_warning_steps=DEFAULT_TRANSITION_WARNING_STEPS,
        signal_activation_time=INDICATOR_ACTIVATION_TIME,
    ):
        if seconds_per_step <= 0:
            raise ValueError("seconds_per_step must be greater than zero")
        if transition_warning_steps <= 0:
            raise ValueError("transition_warning_steps must be greater than zero")
        if signal_activation_time <= 0:
            raise ValueError("signal_activation_time must be greater than zero")
        self.window: Optional[Window] = None

        self.tile_size = tile_size
        self.subdivs = subdivs
        self.seconds_per_step = seconds_per_step
        self.transition_warning_steps = transition_warning_steps
        self.signal_activation_time = signal_activation_time

    def _lazy_init_window(self) -> Window:
        if self.window is None:
            self.window = Window("Overcooked V3")
        return self.window

    def show(self, block=False):
        self._lazy_init_window().show(block=block)

    def _caption_with_countdown_steps(self, steps_remaining, caption=""):
        if not 0 < steps_remaining <= self.transition_warning_steps:
            return caption
        seconds_remaining = steps_remaining * self.seconds_per_step
        countdown = (
            f"layout change in {steps_remaining} steps ({seconds_remaining:.1f}s)"
        )
        return f"{caption} | {countdown}" if caption else countdown

    def caption_with_countdown(self, state, caption=""):
        steps_remaining = int(np.asarray(state.steps_until_layout_change))
        return self._caption_with_countdown_steps(steps_remaining, caption)

    def render(self, state, agent_view_size=None, caption=""):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        window = self._lazy_init_window()

        img = self._render_frame(state, agent_view_size)

        window.set_caption(self.caption_with_countdown(state, caption))
        window.show_img(img)

    def animate(
        self,
        state_seq,
        filename="animation.gif",
        agent_view_size=None,
        captions=None,
    ):
        """Render a state sequence and save it as a GIF."""
        frame_seq = self._animation_frames(state_seq, agent_view_size, captions)
        frame_duration_ms = int(round(self.seconds_per_step * 1000))
        durations = [frame_duration_ms] * len(frame_seq)
        durations[-1] += 3000
        imageio.mimsave(filename, frame_seq, "GIF", duration=durations, loop=0)

    def save_video(
        self,
        state_seq,
        filename,
        agent_view_size=None,
        captions=None,
        fps=None,
        quality=5,
    ):
        """Render a state sequence and save a compact H.264 MP4."""
        if fps is None:
            fps = max(1, int(round(1.0 / self.seconds_per_step)))
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if not 0 <= quality <= 10:
            raise ValueError("quality must be between 0 and 10")

        frame_seq = self._animation_frames(state_seq, agent_view_size, captions)
        imageio.mimsave(
            filename,
            frame_seq,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=quality,
            macro_block_size=2,
        )

    def _animation_frames(self, state_seq, agent_view_size=None, captions=None):
        states = self._state_sequence_to_list(state_seq)
        frame_seq = [self._render_frame(state, agent_view_size) for state in states]
        countdown_steps = [
            int(np.asarray(state.steps_until_layout_change)) for state in states
        ]

        if captions is not None or any(
            0 < steps <= self.transition_warning_steps for steps in countdown_steps
        ):
            from PIL import Image, ImageDraw, ImageFont

            if captions is None:
                captions = [""] * len(frame_seq)
            elif len(captions) != len(frame_seq):
                raise ValueError("captions and state_seq must have the same length")
            captions = [
                self._caption_with_countdown_steps(steps, caption)
                for steps, caption in zip(countdown_steps, captions)
            ]

            font = ImageFont.load_default()
            measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            text_boxes = [
                measure.textbbox((0, 0), caption, font=font) for caption in captions
            ]
            text_width = max(box[2] - box[0] for box in text_boxes)
            text_height = max(box[3] - box[1] for box in text_boxes)
            canvas_width = max(frame_seq[0].shape[1], text_width + 12)
            canvas_height = frame_seq[0].shape[0] + text_height + 12
            canvas_width += canvas_width % 2
            canvas_height += canvas_height % 2

            captioned_frames = []
            for frame, caption in zip(frame_seq, captions):
                image = Image.fromarray(frame)
                canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
                canvas.paste(image, ((canvas_width - image.width) // 2, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (6, image.height + 6),
                    caption,
                    fill="black",
                    font=font,
                )
                captioned_frames.append(np.asarray(canvas))
            frame_seq = captioned_frames
        return frame_seq

    def render_sequence(self, state_seq, agent_view_size=None):
        states = self._state_sequence_to_list(state_seq)
        return np.stack(
            [self._render_frame(state, agent_view_size) for state in states]
        )

    @staticmethod
    def _state_sequence_to_list(state_seq):
        if isinstance(state_seq, (list, tuple)):
            return list(state_seq)
        sequence_length = int(np.asarray(state_seq.step).shape[0])
        return [
            jax.tree_util.tree_map(lambda value: value[index], state_seq)
            for index in range(sequence_length)
        ]

    def _render_frame(self, state, agent_view_size=None):
        frame = np.asarray(self._render_state(state, agent_view_size))
        frame = self._overlay_tile_countdown(frame, state)
        return self._overlay_signal_countdown(frame, state)

    def _overlay_signal_countdown(self, frame, state):
        static_objects = np.asarray(state.grid[:, :, 0])
        signal_positions = np.argwhere(
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
        )
        if not len(signal_positions):
            return frame

        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font_size = max(7, min(12, self.tile_size // 3))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        for tile_y, tile_x in signal_positions:
            steps_remaining = int(np.asarray(state.grid[tile_y, tile_x, 2]))
            if steps_remaining <= 0:
                continue
            label = f"SIG\n{steps_remaining}"
            text_box = draw.multiline_textbbox(
                (0, 0),
                label,
                font=font,
                spacing=0,
                align="center",
                stroke_width=1,
            )
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            center_x = int(tile_x * self.tile_size + self.tile_size / 2)
            center_y = int(tile_y * self.tile_size + self.tile_size / 2)
            padding = max(1, self.tile_size // 20)
            draw.rounded_rectangle(
                (
                    center_x - text_width // 2 - padding,
                    center_y - text_height // 2 - padding,
                    center_x + (text_width + 1) // 2 + padding,
                    center_y + (text_height + 1) // 2 + padding,
                ),
                radius=max(1, padding),
                fill=(0, 90, 0),
                outline=(255, 255, 255),
                width=1,
            )
            draw.multiline_text(
                (center_x - text_width / 2, center_y - text_height / 2),
                label,
                fill=(255, 255, 255),
                font=font,
                spacing=0,
                align="center",
                stroke_width=1,
                stroke_fill=(0, 60, 0),
            )
        return np.asarray(image)

    def _overlay_tile_countdown(self, frame, state):
        steps_remaining = int(np.asarray(state.steps_until_layout_change))
        if not 0 < steps_remaining <= self.transition_warning_steps:
            return frame

        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font_size = max(7, min(14, self.tile_size // 2))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        label = str(steps_remaining)
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        change_positions = np.argwhere(np.asarray(state.layout_change_mask))
        for tile_y, tile_x in change_positions:
            center_x = int(tile_x * self.tile_size + self.tile_size / 2)
            center_y = int(tile_y * self.tile_size + self.tile_size / 2)
            padding = max(1, self.tile_size // 16)
            left = center_x - text_width // 2 - padding
            top = center_y - text_height // 2 - padding
            right = center_x + (text_width + 1) // 2 + padding
            bottom = center_y + (text_height + 1) // 2 + padding
            draw.rounded_rectangle(
                (left, top, right, bottom),
                radius=max(1, padding),
                fill=(25, 25, 25),
                outline=(230, 180, 0),
                width=1,
            )
            draw.text(
                (center_x - text_width / 2, center_y - text_height / 2),
                label,
                fill=(255, 255, 255),
                font=font,
                stroke_width=1,
                stroke_fill=(25, 25, 25),
            )
        return np.asarray(image)

    @classmethod
    def _encode_agent_extras(cls, direction, idx):
        return direction | (idx << 2)

    @classmethod
    def _decode_agent_extras(cls, extras):
        direction = extras & 0x3
        idx = extras >> 2
        return direction, idx

    @partial(jax.jit, static_argnums=(0, 2))
    def _render_state(self, state, agent_view_size=None):
        """
        Render the state
        """

        grid = state.grid
        agents = state.agents
        recipe = state.recipe

        num_agents = agents.dir.shape[0]

        def _include_agents(grid, x):
            agent, idx = x
            pos = agent.pos
            inventory = agent.inventory
            direction = agent.dir

            # we have to do the encoding because we don't really have a way to also pass the agent's id
            extra_info = OvercookedV3Visualizer._encode_agent_extras(direction, idx)

            new_grid = grid.at[pos.y, pos.x].set(
                [StaticObject.AGENT, inventory, extra_info]
            )
            return new_grid, None

        grid, _ = jax.lax.scan(_include_agents, grid, (agents, jnp.arange(num_agents)))

        static_objects = grid[:, :, 0]
        ingredients = grid[:, :, 1]
        extra_info = grid[:, :, 2]

        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        button_recipe_indicator_mask = (
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
        ) & (extra_info > 0)

        new_ingredients_layer = jnp.where(
            recipe_indicator_mask | button_recipe_indicator_mask,
            recipe | DynamicObject.COOKED | DynamicObject.PLATE,
            ingredients,
        )
        grid = grid.at[:, :, 1].set(new_ingredients_layer)

        highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)
        if agent_view_size:
            for x, y in zip(agents.pos.x, agents.pos.y):
                x_low, x_high, y_low, y_high = compute_view_box(
                    x, y, agent_view_size, grid.shape[0], grid.shape[1]
                )

                row_mask = jnp.arange(grid.shape[0])
                col_mask = jnp.arange(grid.shape[1])

                row_mask = (row_mask >= y_low) & (row_mask < y_high)
                col_mask = (col_mask >= x_low) & (col_mask < x_high)

                agent_mask = row_mask[:, None] & col_mask[None, :]

                highlight_mask |= agent_mask

        warning_active = (state.steps_until_layout_change > 0) & (
            state.steps_until_layout_change <= self.transition_warning_steps
        )
        blink_on = jnp.mod(state.steps_until_layout_change, 4) < 2
        visible_change_mask = state.layout_change_mask & warning_active & blink_on

        # Render the whole grid
        img = self._render_grid(grid, highlight_mask, visible_change_mask)
        return img

    @staticmethod
    def _render_dynamic_item(
        ingredients,
        img,
        plate_fn=rendering.point_in_circle(0.5, 0.5, 0.3),
        ingredient_fn=rendering.point_in_circle(0.5, 0.5, 0.15),
        dish_positions=jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)]),
    ):
        def _no_op(img, ingredients):
            return img

        def _render_plate(img, ingredients):
            return rendering.fill_coords(img, plate_fn, COLORS["white"])

        def _render_ingredient(img, ingredients):
            idx = DynamicObject.get_ingredient_idx(ingredients)
            return rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

        def _render_dish(img, ingredients):
            img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)

            for idx, ingredient_idx in enumerate(ingredient_indices):
                color = INGREDIENT_COLORS[ingredient_idx]
                pos = dish_positions[idx]
                ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
                img_ing = rendering.fill_coords(img, ingredient_fn, color)

                img = jax.lax.select(ingredient_idx != -1, img_ing, img)

            return img

        branches = jnp.array(
            [
                ingredients == 0,
                ingredients == DynamicObject.PLATE,
                DynamicObject.is_ingredient(ingredients),
                ingredients & DynamicObject.COOKED,
            ]
        )
        branch_idx = jnp.argmax(branches)

        img = jax.lax.switch(
            branch_idx,
            [_no_op, _render_plate, _render_ingredient, _render_dish],
            img,
            ingredients,
        )

        return img

    def _render_cell(self, cell, img):
        static_object = cell[0]

        def _render_empty(cell, img):
            return img

        def _render_wall(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = OvercookedV3Visualizer._render_dynamic_item(cell[1], img)

            return img

        def _render_agent(cell, img):
            tri_fn = rendering.point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            direction, idx = OvercookedV3Visualizer._decode_agent_extras(cell[2])

            # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
            direction_reordering = jnp.array([3, 1, 0, 2])
            direction = direction_reordering[direction]

            agent_color = AGENT_COLORS[idx]

            tri_fn = rendering.rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction
            )
            img = rendering.fill_coords(img, tri_fn, agent_color)

            img = OvercookedV3Visualizer._render_dynamic_item(
                cell[1],
                img,
                plate_fn=rendering.point_in_circle(0.75, 0.75, 0.2),
                ingredient_fn=rendering.point_in_circle(0.75, 0.75, 0.15),
                dish_positions=jnp.array([(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]),
            )

            return img

        def _render_agent_self(cell, img):
            # Note: This should not ever be called
            return img

        def _render_goal(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["green"]
            )

            return img

        def _render_pot(cell, img):
            return OvercookedV3Visualizer._render_pot(cell, img)

        def _render_recipe_indicator(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
            )
            img = OvercookedV3Visualizer._render_dynamic_item(cell[1], img)

            return img

        def _render_button_recipe_indicator(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
            )
            img = OvercookedV3Visualizer._render_dynamic_item(cell[1], img)

            time_left = cell[2]
            progress_fn = rendering.point_in_rect(
                0.1,
                0.9 - (0.9 - 0.1) / self.signal_activation_time * time_left,
                0.83,
                0.88,
            )
            img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])

            button_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
            img_button = rendering.fill_coords(img, button_fn, COLORS["red"])

            img = jax.lax.select(time_left > 0, img_timer, img_button)
            return img

        def _render_plate_pile(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            plate_fns = [
                rendering.point_in_circle(*coord, 0.2)
                for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]
            ]
            for plate_fn in plate_fns:
                img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            return img

        def _render_ingredient_pile(cell, img):
            ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE

            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            ingredient_fns = [
                rendering.point_in_circle(*coord, 0.15)
                for coord in [
                    (0.5, 0.15),
                    (0.3, 0.4),
                    (0.8, 0.35),
                    (0.4, 0.8),
                    (0.75, 0.75),
                ]
            ]

            for ingredient_fn in ingredient_fns:
                img = rendering.fill_coords(
                    img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx]
                )

            return img

        render_fns_dict = {
            StaticObject.EMPTY: _render_empty,
            StaticObject.WALL: _render_wall,
            StaticObject.AGENT: _render_agent,
            StaticObject.SELF_AGENT: _render_agent_self,
            StaticObject.GOAL: _render_goal,
            StaticObject.POT: _render_pot,
            StaticObject.RECIPE_INDICATOR: _render_recipe_indicator,
            StaticObject.BUTTON_RECIPE_INDICATOR: _render_button_recipe_indicator,
            StaticObject.PLATE_PILE: _render_plate_pile,
        }

        render_fns = [_render_empty] * (max(render_fns_dict.keys()) + 2)
        for key, value in render_fns_dict.items():
            render_fns[key] = value
        render_fns[-1] = _render_ingredient_pile

        branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)

        return jax.lax.switch(
            branch_idx,
            render_fns,
            cell,
            img,
        )

    @staticmethod
    def _render_pot(cell, img):
        ingredients = cell[1]
        time_left = cell[2]

        is_cooking = time_left > 0
        is_cooked = (ingredients & DynamicObject.COOKED) != 0
        is_idle = ~is_cooking & ~is_cooked
        ingredients = DynamicObject.get_ingredient_idx_list_jit(ingredients)
        has_ingredients = ingredients[0] != -1

        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )

        ingredient_fns = [
            rendering.point_in_circle(*coord, 0.13)
            for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
        ]

        for i, ingredient_idx in enumerate(ingredients):
            img_ing = rendering.fill_coords(
                img, ingredient_fns[i], INGREDIENT_COLORS[ingredient_idx]
            )
            img = jax.lax.select(ingredient_idx != -1, img_ing, img)

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        lid_fn_open = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
        handle_fn_open = rendering.rotate_fn(
            handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi
        )
        pot_open = is_idle & has_ingredients

        img = rendering.fill_coords(img, pot_fn, COLORS["black"])

        img_closed = rendering.fill_coords(img, lid_fn, COLORS["black"])
        img_closed = rendering.fill_coords(img_closed, handle_fn, COLORS["black"])

        img_open = rendering.fill_coords(img, lid_fn_open, COLORS["black"])
        img_open = rendering.fill_coords(img_open, handle_fn_open, COLORS["black"])

        img = jax.lax.select(pot_open, img_open, img_closed)

        # Render progress bar
        progress_fn = rendering.point_in_rect(
            0.1, 0.9 - (0.9 - 0.1) / POT_COOK_TIME * time_left, 0.83, 0.88
        )
        img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])
        img = jax.lax.select(is_cooking, img_timer, img)

        return img

    def _render_tile(
        self,
        obj,
        highlight=False,
        layout_will_change=False,
    ):
        """
        Render a tile and cache the result
        """
        # key = (*obj.tolist(), highlight, tile_size)

        # if key in OvercookedV3Visualizer.tile_cache:
        #     return OvercookedV3Visualizer.tile_cache[key]

        img = jnp.zeros(
            shape=(self.tile_size * self.subdivs, self.tile_size * self.subdivs, 3),
            dtype=jnp.uint8,
        )

        # Draw the grid lines (top and left edges)
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"]
        )
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"]
        )

        img = self._render_cell(obj, img)

        img_highlight = rendering.highlight_img(img)
        img = jax.lax.select(highlight, img_highlight, img)

        warning_edges = (
            rendering.point_in_rect(0.02, 0.09, 0.02, 0.98),
            rendering.point_in_rect(0.91, 0.98, 0.02, 0.98),
            rendering.point_in_rect(0.02, 0.98, 0.02, 0.09),
            rendering.point_in_rect(0.02, 0.98, 0.91, 0.98),
        )

        def warning_border(x, y):
            return (
                warning_edges[0](x, y)
                | warning_edges[1](x, y)
                | warning_edges[2](x, y)
                | warning_edges[3](x, y)
            )

        img_warning = rendering.fill_coords(img, warning_border, COLORS["orange"])
        img = jax.lax.select(layout_will_change, img_warning, img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, self.subdivs)

        # Cache the rendered tile
        # OvercookedV3Visualizer.tile_cache[key] = img

        return img

    def _render_grid(
        self,
        grid,
        highlight_mask,
        layout_change_mask,
    ):
        img_grid = jax.vmap(jax.vmap(self._render_tile))(
            grid,
            highlight_mask,
            layout_change_mask,
        )

        # print("img_grid", img_grid.shape)

        grid_rows, grid_cols, tile_height, tile_width, channels = img_grid.shape

        big_image = img_grid.transpose(0, 2, 1, 3, 4).reshape(
            grid_rows * tile_height, grid_cols * tile_width, channels
        )

        # print("big_image", big_image.shape)

        return big_image

    def close(self):
        if self.window is not None:
            self.window.close()
