"""Kitchen rendering and cell marks shared by the transition figures.

Renders a layout's phases with the native viewer and draws the marks Figure 3
uses on a kitchen: an orange frame on a cell the transition acts on, a red
cross for one that disappears. Used by results/explore_same_edit_pairs.py,
results/explore_spare_or_not.py and results/explore_random_placements.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jaxmarl import make  # noqa: E402
from jaxmarl.environments.overcooked_v3.common import Direction, Position  # noqa: E402
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer  # noqa: E402

TILE = 32
#: Taken from Figure 3 so the figures read as one family: an orange frame marks
#: a cell the transition acts on and a red cross means it disappears. Every
#: stroke carries a white outline, which keeps them legible over the dark tiles.
FRAME = "#E8890C"
CROSS = "#E8322A"
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
HALO_THIN = [pe.withStroke(linewidth=1.5, foreground="white")]


def phase_state(env, initial, index):
    """The kitchen as it stands at the start of one phase, agents reset."""
    positions = env.phase_agent_positions[index]
    grid = initial.grid.at[:, :, 0].set(env.phase_static_objects[index]).at[:, :, 1:].set(0)
    agents = initial.agents.replace(
        pos=Position(x=positions[:, 0], y=positions[:, 1]),
        dir=jnp.full_like(initial.agents.dir, Direction.UP),
        inventory=jnp.zeros_like(initial.agents.inventory),
    )
    recipe = jnp.where(env.phase_has_recipe[index], env.phase_recipes[index], initial.recipe)
    state = initial.replace(
        agents=agents, grid=grid,
        step=jnp.sum(env.phase_durations[:index], dtype=jnp.int32),
        done=jnp.array(False), recipe=recipe, previous_recipe=recipe, next_recipe=recipe,
        legacy_recipe_deliveries_remaining=jnp.array(0, dtype=jnp.int32),
        layout_index=jnp.array(index, dtype=jnp.int32),
    )
    return env._set_transition_awareness(state)


def frames(layout):
    env = make("overcooked_v3", layout=layout, max_steps=450, random_agent_positions=False)
    _, initial = env.reset(jax.random.PRNGKey(0))
    viz = OvercookedV3Visualizer(tile_size=TILE)
    return [np.asarray(viz._render_frame(phase_state(env, initial, i))) for i in (0, 1)]


def centre(cell):
    y, x = cell
    return (x + 0.5) * TILE, (y + 0.5) * TILE


def frame_cell(ax, cell, hatch=None, colour=FRAME):
    """The orange box Figure 3 puts around a cell the transition acts on."""
    cx, cy = centre(cell)
    ax.add_patch(Rectangle((cx - TILE / 2, cy - TILE / 2), TILE, TILE,
                           facecolor="white" if hatch else "none",
                           alpha=0.22 if hatch else 1.0, hatch=hatch,
                           edgecolor=colour, linewidth=1.2, zorder=5,
                           path_effects=HALO))
