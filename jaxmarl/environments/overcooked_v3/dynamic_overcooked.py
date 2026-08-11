"""Overcooked V3: Overcooked V2 dynamics with cyclic map changes."""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import PRNGKeyArray

from jaxmarl.environments.multi_agent_env import Actions
from jaxmarl.environments.overcooked_v3.common import (
    DIR_TO_VEC,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.dynamic_layouts import (
    DynamicLayout,
    dynamic_layouts,
)
from jaxmarl.environments.overcooked_v3.overcooked import (
    ObservationType,
    OvercookedV3Base,
    State,
)
from jaxmarl.environments.overcooked_v3.utils import (
    OvercookedPathPlanner,
    compute_enclosed_spaces,
)


class OvercookedV3(OvercookedV3Base):
    """Overcooked V2-compatible environment whose map changes cyclically."""

    def __init__(
        self,
        layout: Union[str, DynamicLayout] = "dynamic_cramped_room",
        include_transition_countdown: bool = True,
        include_layout_change_mask: Union[bool, None] = None,
        **kwargs,
    ):
        self.include_transition_countdown = include_transition_countdown
        self.include_layout_change_mask = (
            include_transition_countdown
            if include_layout_change_mask is None
            else include_layout_change_mask
        )
        if isinstance(layout, str):
            if layout not in dynamic_layouts:
                raise ValueError(
                    f"Invalid dynamic layout: {layout}, "
                    f"allowed layouts: {dynamic_layouts.keys()}"
                )
            dynamic_layout = dynamic_layouts[layout]
        elif isinstance(layout, DynamicLayout):
            dynamic_layout = layout
        else:
            raise ValueError(
                "layout must be a DynamicLayout or a key in dynamic_layouts"
            )

        self.dynamic_layout = dynamic_layout
        super().__init__(layout=dynamic_layout.initial_layout, **kwargs)

        self.phase_static_objects = jnp.asarray(
            np.stack([phase.layout.static_objects for phase in dynamic_layout.phases]),
            dtype=jnp.int32,
        )
        self.phase_agent_positions = jnp.asarray(
            [phase.agent_positions for phase in dynamic_layout.phases],
            dtype=jnp.int32,
        )
        self.phase_enclosed_spaces = jnp.asarray(
            np.stack(
                [
                    np.asarray(
                        compute_enclosed_spaces(
                            jnp.asarray(phase.layout.static_objects)
                            == StaticObject.EMPTY
                        )
                    )
                    for phase in dynamic_layout.phases
                ]
            ),
            dtype=jnp.int32,
        )
        self.phase_durations = jnp.asarray(
            [phase.steps for phase in dynamic_layout.phases], dtype=jnp.int32
        )
        self.phase_ends = jnp.cumsum(self.phase_durations)
        self.cycle_steps = int(dynamic_layout.cycle_steps)

    def _get_obs_shape(self):
        obs_shape = super()._get_obs_shape()

        def _append_transition_features(shape, obs_type):
            extra_features = int(self.include_transition_countdown)
            if self.include_layout_change_mask:
                extra_features += (
                    1
                    if obs_type == ObservationType.DEFAULT
                    else self.height * self.width
                )
            return (*shape[:-1], shape[-1] + extra_features)

        if isinstance(obs_shape, list):
            return [
                _append_transition_features(shape, obs_type)
                for shape, obs_type in zip(obs_shape, self.observation_type)
            ]
        return _append_transition_features(obs_shape, self.observation_type)

    def get_layout_index(self, step: jax.Array) -> jax.Array:
        cycle_step = jnp.mod(step, self.cycle_steps)
        return jnp.sum(cycle_step >= self.phase_ends).astype(jnp.int32)

    def get_steps_until_layout_change(self, step: jax.Array) -> jax.Array:
        cycle_step = jnp.mod(step, self.cycle_steps)
        layout_index = self.get_layout_index(step)
        return self.phase_ends[layout_index] - cycle_step

    def get_transition_countdown(self, step: jax.Array) -> jax.Array:
        layout_index = self.get_layout_index(step)
        steps_remaining = self.get_steps_until_layout_change(step)
        return steps_remaining.astype(jnp.float32) / self.phase_durations[
            layout_index
        ].astype(jnp.float32)

    def get_layout_change_mask(self, step: jax.Array) -> jax.Array:
        layout_index = self.get_layout_index(step)
        next_layout_index = (layout_index + 1) % self.phase_static_objects.shape[0]
        return (
            self.phase_static_objects[layout_index]
            != self.phase_static_objects[next_layout_index]
        )

    def _set_transition_awareness(self, state: State) -> State:
        return state.replace(
            steps_until_layout_change=self.get_steps_until_layout_change(state.step),
            layout_change_mask=self.get_layout_change_mask(state.step),
        )

    def _append_default_transition_features(self, obs, step):
        transition_layers = []
        if self.include_transition_countdown:
            countdown = self.get_transition_countdown(step)
            transition_layers.append(
                jnp.full((*obs.shape[:-1], 1), countdown, dtype=jnp.float32)
            )
        if self.include_layout_change_mask:
            change_mask = self.get_layout_change_mask(step).astype(jnp.float32)
            change_mask = jnp.broadcast_to(
                change_mask,
                (*obs.shape[:-3], *change_mask.shape),
            )
            transition_layers.append(change_mask[..., None])
        if not transition_layers:
            return obs
        return jnp.concatenate([obs.astype(jnp.float32), *transition_layers], axis=-1)

    def _append_featurized_transition_features(self, obs, step):
        transition_features = []
        if self.include_transition_countdown:
            countdown = self.get_transition_countdown(step)
            transition_features.append(
                jnp.full((*obs.shape[:-1], 1), countdown, dtype=jnp.float32)
            )
        if self.include_layout_change_mask:
            change_mask = self.get_layout_change_mask(step).astype(jnp.float32)
            transition_features.append(
                jnp.broadcast_to(
                    change_mask.flatten(), (*obs.shape[:-1], change_mask.size)
                )
            )
        if not transition_features:
            return obs
        return jnp.concatenate([obs.astype(jnp.float32), *transition_features], axis=-1)

    def get_obs_default(self, state: State):
        obs = super().get_obs_default(state)
        return self._append_default_transition_features(obs, state.step)

    def get_obs_featurized(self, state: State):
        obs = super().get_obs_featurized(state)
        return self._append_featurized_transition_features(obs, state.step)

    def reset(self, key: PRNGKeyArray):
        _, state = super().reset(key)
        state = self._set_transition_awareness(state)
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def _get_move_area(self, state: State) -> jax.Array:
        current_empty = state.grid[:, :, 0] == StaticObject.EMPTY
        next_layout_index = self.get_layout_index(state.step + 1)
        layout_will_change = next_layout_index != state.layout_index
        next_empty = self.phase_static_objects[next_layout_index] == StaticObject.EMPTY
        return current_empty & jnp.where(layout_will_change, next_empty, True)

    def _get_enclosed_spaces(self, state: State) -> jax.Array:
        return self.phase_enclosed_spaces[state.layout_index]

    def _get_closest_target_pos(self, state, targets, pos, direction):
        return OvercookedPathPlanner.get_closest_target_pos_static(
            state.grid[:, :, 0] == StaticObject.EMPTY,
            targets,
            pos,
            direction,
        )

    def step_env(self, key: PRNGKeyArray, state: State, actions: Actions):
        obs, state, rewards, dones, infos = super().step_env(key, state, actions)

        layout_index = self.get_layout_index(state.step)
        layout_changed = layout_index != state.layout_index
        state = lax.cond(
            layout_changed,
            self._change_layout,
            lambda current_state, _: current_state.replace(layout_index=layout_index),
            state,
            layout_index,
        )
        state = self._set_transition_awareness(state)
        obs = self.get_obs(state)
        steps_until_layout_change = state.steps_until_layout_change
        transition_countdown = self.get_transition_countdown(state.step)
        static_objects = state.grid[:, :, 0]
        layout_change_tile_count = jnp.sum(state.layout_change_mask)
        wall_tile_count = jnp.sum(static_objects == StaticObject.WALL)
        ingredient_pile_count = jnp.sum(
            static_objects >= StaticObject.INGREDIENT_PILE_BASE
        )
        signal_tile_count = jnp.sum(
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
        )
        infos = {
            **infos,
            "layout_index": jnp.full((self.num_agents,), layout_index),
            "layout_changed": jnp.full((self.num_agents,), layout_changed),
            "steps_until_layout_change": jnp.full(
                (self.num_agents,), steps_until_layout_change
            ),
            "transition_countdown": jnp.full((self.num_agents,), transition_countdown),
            "layout_change_tile_count": jnp.full(
                (self.num_agents,), layout_change_tile_count
            ),
            "wall_tile_count": jnp.full((self.num_agents,), wall_tile_count),
            "ingredient_pile_count": jnp.full(
                (self.num_agents,), ingredient_pile_count
            ),
            "signal_tile_count": jnp.full((self.num_agents,), signal_tile_count),
        }
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            infos,
        )

    def _change_layout(self, state: State, layout_index: jax.Array) -> State:
        old_static = state.grid[:, :, 0]
        new_static = self.phase_static_objects[layout_index]
        changed_cells = old_static != new_static

        grid = state.grid.at[:, :, 0].set(new_static)
        grid = grid.at[:, :, 1].set(
            jnp.where(changed_cells, DynamicObject.EMPTY, grid[:, :, 1])
        )
        grid = grid.at[:, :, 2].set(jnp.where(changed_cells, 0, grid[:, :, 2]))

        agent_positions = self._relocate_blocked_agents(
            state.agents.pos,
            state.agents.dir,
            new_static,
            self.phase_agent_positions[layout_index],
        )

        return state.replace(
            agents=state.agents.replace(pos=agent_positions),
            grid=grid,
            layout_index=layout_index,
        )

    def _relocate_blocked_agents(
        self,
        positions: Position,
        directions,
        static_objects,
        spawn_positions,
    ) -> Position:
        opposite = jnp.array([1, 0, 3, 2], dtype=jnp.int32)
        clockwise = jnp.array([2, 3, 1, 0], dtype=jnp.int32)

        def _relocate_one(agent_index, current_positions):
            x = current_positions.x[agent_index]
            y = current_positions.y[agent_index]
            occupied = (
                jnp.zeros((self.height, self.width), dtype=jnp.bool_)
                .at[current_positions.y, current_positions.x]
                .set(True)
                .at[y, x]
                .set(False)
            )
            available = (static_objects == StaticObject.EMPTY) & ~occupied

            first = opposite[directions[agent_index]]
            second = clockwise[first]
            third = clockwise[second]
            fourth = clockwise[third]
            vectors = DIR_TO_VEC[jnp.stack([first, second, third, fourth])]
            candidate_x = x + vectors[:, 0]
            candidate_y = y + vectors[:, 1]
            in_bounds = (
                (candidate_x >= 0)
                & (candidate_x < self.width)
                & (candidate_y >= 0)
                & (candidate_y < self.height)
            )
            safe_x = jnp.clip(candidate_x, 0, self.width - 1)
            safe_y = jnp.clip(candidate_y, 0, self.height - 1)
            valid_neighbor = in_bounds & available[safe_y, safe_x]
            neighbor_index = jnp.argmax(valid_neighbor)
            neighbor_x = safe_x[neighbor_index]
            neighbor_y = safe_y[neighbor_index]

            spawn_order = (
                jnp.arange(self.num_agents, dtype=jnp.int32) + agent_index
            ) % self.num_agents
            ordered_spawns = spawn_positions[spawn_order]
            valid_spawn = available[ordered_spawns[:, 1], ordered_spawns[:, 0]]
            spawn = ordered_spawns[jnp.argmax(valid_spawn)]

            target_x = jnp.where(jnp.any(valid_neighbor), neighbor_x, spawn[0])
            target_y = jnp.where(jnp.any(valid_neighbor), neighbor_y, spawn[1])
            is_blocked = static_objects[y, x] != StaticObject.EMPTY
            target_x = jnp.where(is_blocked, target_x, x)
            target_y = jnp.where(is_blocked, target_y, y)
            return Position(
                x=current_positions.x.at[agent_index].set(target_x),
                y=current_positions.y.at[agent_index].set(target_y),
            )

        return lax.fori_loop(0, self.num_agents, _relocate_one, positions)

    @property
    def name(self) -> str:
        return "Overcooked V3"


__all__ = ["OvercookedV3"]
