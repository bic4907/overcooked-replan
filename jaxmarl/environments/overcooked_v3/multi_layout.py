"""Overcooked V3 over a distribution of layouts.

An episode draws one layout at reset and keeps it to the end, running that
layout's own phase schedule. This is the environment the optimal behavior prior
is trained in: a self-play policy that has seen many kitchens is a far better
starting point for behavior cloning than a randomly initialised network.

Layouts differ in grid size, so every one of them is padded into a shared
canvas and surrounded by wall. Padding has to be wall rather than zero: the
static observation channels encode WALL, GOAL, POT, RECIPE_INDICATOR and
PLATE_PILE, and an empty floor tile is all-zero across them, so zero padding
would read as walkable floor.

The phase arrays of every layout are stacked into one array and addressed
through ``State.layout_base``, the index of the first phase belonging to the
episode's layout. ``phases_per_layout`` keeps "the phase after the last one"
wrapping inside its own layout instead of falling into the next.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import PRNGKeyArray

from jaxmarl.environments.overcooked_v3 import dynamic_layout_data
from jaxmarl.environments.overcooked_v3.common import (
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.dynamic_layouts import (
    DynamicLayout,
    _load_named_dynamic_layout,
)
from jaxmarl.environments.overcooked_v3.dynamic_overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3.overcooked import (
    ObservationType,
    OvercookedV3Base,
    State,
)

#: Canvas that holds every layout below. Width 13 comes from outage_wide and
#: distance_switch_wide, height 7 from split_0 and split_wide.
OBP_CANVAS_WIDTH = 13
OBP_CANVAS_HEIGHT = 7

#: The layouts the prior is trained on. Only ``_0`` and ``_2`` families: the
#: ``_1`` variants are not used by this project, and recipe_switch is excluded
#: because its two ingredients give it 38 observation channels where every
#: layout here has 31.
OBP_LAYOUT_NAMES = (
    "outage_0",
    "outage_2",
    "outage_2_plate",
    "distance_switch_0",
    "split_0",
    "outage_wide_2",
    "outage_wide_2_plate",
    "split_wide",
    "outage_wide",
    "distance_switch_wide",
)


def pad_grid(grid: str, width: int, height: int) -> str:
    """Centre a layout grid on a larger canvas, filling the rest with wall.

    Centring rather than corner-anchoring keeps every kitchen away from the
    canvas edge, so a convolution sees the same kind of wall border around each
    one that the layouts already draw for themselves.
    """
    rows = grid.strip("\n").splitlines()
    grid_height = len(rows)
    grid_width = len(rows[0])
    if any(len(row) != grid_width for row in rows):
        raise ValueError("Layout rows must all have the same width")
    if grid_width > width or grid_height > height:
        raise ValueError(
            f"Layout is {grid_width}x{grid_height}, larger than the "
            f"{width}x{height} canvas"
        )
    top = (height - grid_height) // 2
    left = (width - grid_width) // 2
    canvas = [["W"] * width for _ in range(height)]
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            canvas[top + y][left + x] = cell
    return "\n" + "\n".join("".join(row) for row in canvas) + "\n"


def padded_dynamic_layout(
    name: str, width: int = OBP_CANVAS_WIDTH, height: int = OBP_CANVAS_HEIGHT
) -> DynamicLayout:
    """Build the named layout on the shared canvas.

    Goes through the registry loader so the layout keeps the recipe table its
    name selects -- outage kitchens cook two onions, the others three.
    """
    raw = getattr(dynamic_layout_data, name, None)
    if raw is None:
        raise ValueError(f"Unknown dynamic layout: {name}")
    padded = [[pad_grid(grid, width, height), steps] for grid, steps in raw]
    return _load_named_dynamic_layout(name, padded)


def _possible_recipes(layout: DynamicLayout) -> list:
    """Recipe table of a layout, which Layout.__post_init__ always fills in."""
    possible = layout.phases[0].layout.possible_recipes
    if possible is None:
        raise ValueError("Layout has no possible_recipes")
    return possible


def _encode_recipe(recipe: Sequence[int]) -> int:
    """Bitmask the environment uses for a recipe, two bits per ingredient."""
    return sum(1 << (2 + 2 * ingredient) for ingredient in recipe)


class MultiLayoutOvercookedV3(OvercookedV3):
    """Draw a layout at reset and run its phase schedule for the episode."""

    def __init__(
        self,
        layouts: Sequence[str] = OBP_LAYOUT_NAMES,
        canvas_width: int = OBP_CANVAS_WIDTH,
        canvas_height: int = OBP_CANVAS_HEIGHT,
        **kwargs,
    ):
        names = tuple(layouts)
        if len(names) < 1:
            raise ValueError("layouts must name at least one layout")
        if len(set(names)) != len(names):
            raise ValueError(f"layouts contains duplicates: {names}")

        padded = [padded_dynamic_layout(n, canvas_width, canvas_height) for n in names]

        phase_counts = {len(layout.phases) for layout in padded}
        if len(phase_counts) != 1:
            raise ValueError(
                f"Every layout must have the same number of phases, got {phase_counts}"
            )
        durations = {tuple(phase.steps for phase in layout.phases) for layout in padded}
        if len(durations) != 1:
            raise ValueError(
                "Every layout must use the same phase schedule, so that one "
                f"cycle length applies to all of them; got {durations}"
            )

        recipes = []
        for name, layout in zip(names, padded):
            possible = _possible_recipes(layout)
            if len(possible) != 1:
                raise ValueError(
                    f"{name} has {len(possible)} possible recipes; a layout "
                    "distribution needs one fixed recipe per layout so the "
                    "episode's recipe can be installed at reset"
                )
            recipes.append(_encode_recipe(possible[0]))

        # Recipe lengths may differ (outage cooks two onions, the others
        # three). That is safe for the grid observation because cooking reads
        # the ingredient count out of State.recipe, but the featurized
        # observation uses the construction-time recipe_size and would report
        # the reference layout's length for every episode.
        recipe_sizes = {len(_possible_recipes(layout)[0]) for layout in padded}
        observation_type = kwargs.get("observation_type", ObservationType.DEFAULT)
        types = (
            observation_type
            if isinstance(observation_type, (list, tuple))
            else [observation_type]
        )
        if len(recipe_sizes) > 1 and ObservationType.FEATURIZED in types:
            raise ValueError(
                "Featurized observations hard-code one recipe length, and this "
                f"layout set mixes {sorted(recipe_sizes)}. Use the default grid "
                "observation, or restrict the layouts to one recipe length."
            )

        # Build on the layout with the longest recipe so recipe_size is never
        # short of what an episode actually needs.
        reference = max(
            range(len(padded)), key=lambda i: len(_possible_recipes(padded[i])[0])
        )
        super().__init__(layout=padded[reference], **kwargs)

        self.layout_names = names
        self.num_layouts = len(names)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.layout_recipes = jnp.asarray(recipes, dtype=jnp.int32)

        # Stack every layout's phases behind one another. phase_durations,
        # phase_ends, cycle_steps and phases_per_layout stay per-layout, so the
        # schedule within an episode is unchanged.
        self.phase_static_objects = jnp.concatenate(
            [
                jnp.asarray(
                    np.stack([p.layout.static_objects for p in layout.phases]),
                    dtype=jnp.int32,
                )
                for layout in padded
            ]
        )
        self.phase_agent_positions = jnp.concatenate(
            [
                jnp.asarray([p.agent_positions for p in layout.phases], dtype=jnp.int32)
                for layout in padded
            ]
        )
        self.phase_has_recipe = jnp.concatenate(
            [
                jnp.asarray(
                    [p.recipe is not None for p in layout.phases], dtype=jnp.bool_
                )
                for layout in padded
            ]
        )
        self.phase_recipes = jnp.concatenate(
            [
                jnp.asarray(
                    [
                        0 if p.recipe is None else _encode_recipe(p.recipe)
                        for p in layout.phases
                    ],
                    dtype=jnp.int32,
                )
                for layout in padded
            ]
        )
        self.phase_enclosed_spaces = jnp.concatenate(
            [self._enclosed_spaces_for(layout) for layout in padded]
        )

        expected = self.num_layouts * self.phases_per_layout
        for name, array in (
            ("phase_static_objects", self.phase_static_objects),
            ("phase_agent_positions", self.phase_agent_positions),
            ("phase_has_recipe", self.phase_has_recipe),
            ("phase_recipes", self.phase_recipes),
            ("phase_enclosed_spaces", self.phase_enclosed_spaces),
        ):
            if array.shape[0] != expected:
                raise RuntimeError(
                    f"{name} has {array.shape[0]} entries, expected {expected}"
                )

    @staticmethod
    def _enclosed_spaces_for(layout: DynamicLayout) -> jax.Array:
        from jaxmarl.environments.overcooked_v3.utils import compute_enclosed_spaces

        return jnp.asarray(
            np.stack(
                [
                    np.asarray(
                        compute_enclosed_spaces(
                            jnp.asarray(phase.layout.static_objects)
                            == StaticObject.EMPTY
                        )
                    )
                    for phase in layout.phases
                ]
            ),
            dtype=jnp.int32,
        )

    def layout_id(self, state: State) -> jax.Array:
        """Which layout this episode drew."""
        return state.layout_base // self.phases_per_layout

    def _install_layout(self, state: State, layout_id: jax.Array) -> State:
        """Put the drawn layout's first phase into a freshly reset state."""
        base = layout_id * self.phases_per_layout
        static_objects = self.phase_static_objects[base]
        grid = state.grid.at[:, :, 0].set(static_objects)
        grid = grid.at[:, :, 1].set(DynamicObject.EMPTY)
        grid = grid.at[:, :, 2].set(0)

        spawns = self.phase_agent_positions[base]
        positions = Position(x=spawns[:, 0], y=spawns[:, 1])

        recipe = self.layout_recipes[layout_id]
        return state.replace(
            grid=grid,
            agents=state.agents.replace(pos=positions),
            recipe=recipe,
            previous_recipe=recipe,
            next_recipe=recipe,
            legacy_recipe_deliveries_remaining=jnp.array(0, dtype=jnp.int32),
            layout_index=base,
            layout_base=base,
        )

    def reset(self, key: PRNGKeyArray):
        key, layout_key = jax.random.split(key)
        layout_id = jax.random.randint(layout_key, (), 0, self.num_layouts)
        # Skip OvercookedV3.reset: it installs phase 0 of the single layout it
        # was built with, which is one specific kitchen out of the stack.
        _, state = OvercookedV3Base.reset(self, key)
        state = self._install_layout(state, layout_id)
        state = self._set_transition_awareness(state)
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)
