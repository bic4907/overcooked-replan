"""Human demonstrations on the shared canvas, ready for behavior cloning.

The collector stores whole environment states, not just observations, which is
what lets this module put a demonstration onto the padded canvas the optimal
behavior prior is trained on. Padding the stored observation array directly
would mean knowing what each of the 31 channels should hold outside the
kitchen -- one for wall, zero for the rest, and the broadcast countdown value
for its own. Rebuilding the state and asking the padded environment for the
observation avoids that question entirely, and is checked against the stored
observations on the way through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl
from jaxmarl.environments.overcooked_v3.common import (
    Agent,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.dynamic_layouts import dynamic_layouts
from jaxmarl.environments.overcooked_v3.multi_layout import (
    OBP_CANVAS_HEIGHT,
    OBP_CANVAS_WIDTH,
    padded_dynamic_layout,
)
from jaxmarl.environments.overcooked_v3.overcooked import State

#: Rebuilt observations are allowed to differ from the stored ones by a few
#: float32 ULPs, which is what a differently fused countdown division costs.
_OBSERVATION_TOLERANCE = 1e-6

#: Every field the collector writes, so a State can be rebuilt from an archive.
_STATE_KEYS = (
    "done",
    "step",
    "grid",
    "recipe",
    "previous_recipe",
    "next_recipe",
    "legacy_recipe_deliveries_remaining",
    "new_correct_delivery",
    "layout_index",
    "steps_until_layout_change",
    "layout_change_mask",
    "agents/pos/x",
    "agents/pos/y",
    "agents/dir",
    "agents/inventory",
)


class Episode(NamedTuple):
    """One demonstration, with observations on the padded canvas."""

    episode_id: str
    layout: str
    observations: jax.Array  # [T, 2, H, W, C]
    actions: jax.Array  # [T, 2]
    human_mask: jax.Array  # [T, 2], which actions a person chose
    rewards: jax.Array  # [T, 2]
    episode_return: float


class Dataset(NamedTuple):
    """Flat behavior-cloning arrays, one row per human-chosen action."""

    observations: jax.Array  # [N, H, W, C]
    actions: jax.Array  # [N]
    episode_index: jax.Array  # [N], row back to the episode it came from
    episode_ids: tuple[str, ...]
    layouts: tuple[str, ...]


def canvas_offset(
    layout: str,
    width: int = OBP_CANVAS_WIDTH,
    height: int = OBP_CANVAS_HEIGHT,
) -> tuple[int, int]:
    """Where ``pad_grid`` puts this layout's top-left corner, as (top, left).

    Same centring rule as ``pad_grid``, read off the layout's own grid so a
    padded state lands exactly where the padded layout does.
    """
    grid_height, grid_width = np.asarray(
        dynamic_layouts[layout].phases[0].layout.static_objects
    ).shape
    if grid_width > width or grid_height > height:
        raise ValueError(
            f"{layout} is {grid_width}x{grid_height}, larger than the "
            f"{width}x{height} canvas"
        )
    return (height - grid_height) // 2, (width - grid_width) // 2


def _read_archive(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return metadata, arrays


def _state_from_arrays(arrays: dict) -> State:
    missing = [key for key in _STATE_KEYS if f"state/{key}" not in arrays]
    if missing:
        raise ValueError(f"Archive is missing state fields: {missing}")
    read = lambda key: jnp.asarray(arrays[f"state/{key}"])  # noqa: E731
    return State(
        done=read("done"),
        step=read("step"),
        agents=Agent(
            pos=Position(x=read("agents/pos/x"), y=read("agents/pos/y")),
            dir=read("agents/dir"),
            inventory=read("agents/inventory"),
        ),
        grid=read("grid"),
        recipe=read("recipe"),
        previous_recipe=read("previous_recipe"),
        next_recipe=read("next_recipe"),
        legacy_recipe_deliveries_remaining=read("legacy_recipe_deliveries_remaining"),
        new_correct_delivery=read("new_correct_delivery"),
        ingredient_permutations=None,
        layout_index=read("layout_index"),
        # The collector runs one layout, so its phase indices are already the
        # indices into that layout's own phase arrays.
        layout_base=jnp.zeros_like(read("layout_index")),
        steps_until_layout_change=read("steps_until_layout_change"),
        layout_change_mask=read("layout_change_mask"),
    )


def pad_state(state: State, top: int, left: int, height: int, width: int) -> State:
    """Place a batch of states on the canvas, filling the surround with wall."""
    batch, grid_height, grid_width, channels = state.grid.shape
    if top + grid_height > height or left + grid_width > width:
        raise ValueError(
            f"A {grid_width}x{grid_height} grid at ({left}, {top}) does not fit "
            f"the {width}x{height} canvas"
        )
    grid = jnp.zeros((batch, height, width, channels), dtype=state.grid.dtype)
    grid = grid.at[:, :, :, 0].set(StaticObject.WALL)
    grid = grid.at[:, top : top + grid_height, left : left + grid_width, :].set(
        state.grid
    )
    mask = jnp.zeros((batch, height, width), dtype=state.layout_change_mask.dtype)
    mask = mask.at[:, top : top + grid_height, left : left + grid_width].set(
        state.layout_change_mask
    )
    return state.replace(
        grid=grid,
        layout_change_mask=mask,
        agents=state.agents.replace(
            pos=Position(x=state.agents.pos.x + left, y=state.agents.pos.y + top)
        ),
    )


def load_episode(
    path: str | Path,
    width: int = OBP_CANVAS_WIDTH,
    height: int = OBP_CANVAS_HEIGHT,
    verify: bool = True,
) -> Episode:
    """Read one archive and re-render its observations on the canvas.

    With ``verify`` the rebuilt state is first replayed through the layout's own
    environment and checked against the stored observations, so a change to the
    observation encoding since collection is caught here rather than silently
    training on observations the policy will never see.
    """
    path = Path(path)
    metadata, arrays = _read_archive(path)
    env_kwargs = dict(metadata["env_kwargs"])
    layout = env_kwargs.pop("layout")
    state = _state_from_arrays(arrays)

    if verify:
        reference = jaxmarl.make("overcooked_v3", layout=layout, **env_kwargs)
        rendered = jax.jit(jax.vmap(reference.get_obs))(state)
        stacked = jnp.stack([rendered[agent] for agent in reference.agents], axis=1)
        stored = jnp.asarray(arrays["observations"])
        # Not exact equality: the countdown channel is a division whose rounding
        # depends on how XLA fuses the surrounding graph, which differs between
        # the collector and this call by about one float32 ULP.
        deviation = float(jnp.max(jnp.abs(stacked - stored)))
        if deviation > _OBSERVATION_TOLERANCE:
            raise ValueError(
                f"{path.name}: observations rebuilt from the stored state differ "
                f"from the stored ones by {deviation:.3g}, more than rounding "
                "explains, so the environment no longer encodes states the way "
                "it did during collection"
            )

    top, left = canvas_offset(layout, width, height)
    padded_env = jaxmarl.make(
        "overcooked_v3",
        layout=padded_dynamic_layout(layout, width, height),
        **env_kwargs,
    )
    padded = jax.jit(jax.vmap(padded_env.get_obs))(
        pad_state(state, top, left, height, width)
    )
    observations = jnp.stack([padded[agent] for agent in padded_env.agents], axis=1)

    actions = jnp.asarray(arrays["actions"])
    steps = actions.shape[0]
    return Episode(
        episode_id=str(metadata["episode_id"]),
        layout=layout,
        # The archive stores one more observation than there are actions; the
        # last one is what the episode ended in and has no action to pair with.
        observations=observations[:steps],
        actions=actions,
        human_mask=jnp.asarray(arrays["human_mask"]),
        rewards=jnp.asarray(arrays["rewards"]),
        episode_return=float(metadata.get("episode_return", float("nan"))),
    )


def find_episodes(root: str | Path, layouts: Sequence[str] | None = None) -> list[Path]:
    """Every archive under ``root``, optionally restricted to some layouts."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No such demonstration directory: {root}")
    paths = sorted(root.rglob("*.npz"))
    if layouts is not None:
        wanted = set(layouts)
        paths = [path for path in paths if path.parent.name in wanted]
    return paths


def build_dataset(
    episodes: Iterable[Episode], include_non_human: bool = False
) -> Dataset:
    """Flatten episodes into one row per action.

    Only actions a person chose are kept by default. Real-time collection marks
    every step as human, including the ones where no key was down, so a large
    share of the rows are deliberate waits -- that is what a person does at five
    steps a second, and the human model should predict it.
    """
    observations, actions, episode_index = [], [], []
    episode_ids, layouts = [], []
    for index, episode in enumerate(episodes):
        steps, num_agents = episode.actions.shape
        mask = (
            jnp.ones_like(episode.human_mask)
            if include_non_human
            else episode.human_mask
        )
        flat_mask = np.asarray(mask[:steps]).reshape(-1)
        flat_obs = np.asarray(episode.observations).reshape(
            steps * num_agents, *episode.observations.shape[2:]
        )
        flat_actions = np.asarray(episode.actions).reshape(-1)
        observations.append(flat_obs[flat_mask])
        actions.append(flat_actions[flat_mask])
        episode_index.append(np.full(int(flat_mask.sum()), index, dtype=np.int32))
        episode_ids.append(episode.episode_id)
        layouts.append(episode.layout)
    if not observations:
        raise ValueError("No episodes to build a dataset from")
    return Dataset(
        observations=jnp.asarray(np.concatenate(observations)),
        actions=jnp.asarray(np.concatenate(actions)),
        episode_index=jnp.asarray(np.concatenate(episode_index)),
        episode_ids=tuple(episode_ids),
        layouts=tuple(layouts),
    )


def split_by_episode(
    dataset: Dataset, validation_fraction: float, seed: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Row masks for a train/validation split that never cuts an episode.

    Two agents of one episode see the same kitchen from two sides, so keeping
    them together is what stops the validation score from being measured on
    states the training rows already contain.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    count = len(dataset.episode_ids)
    order = np.random.default_rng(seed).permutation(count)
    held_out = order[: int(round(count * validation_fraction))]
    is_validation = np.zeros(count, dtype=bool)
    is_validation[held_out] = True
    rows = np.asarray(dataset.episode_index)
    validation = jnp.asarray(is_validation[rows])
    return ~validation, validation
