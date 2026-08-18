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
        include_signal_status: bool = True,
        transition_warning_steps: int = 20,
        **kwargs,
    ):
        if isinstance(transition_warning_steps, bool) or not isinstance(
            transition_warning_steps, int
        ):
            raise ValueError("transition_warning_steps must be a positive integer")
        if transition_warning_steps <= 0:
            raise ValueError("transition_warning_steps must be a positive integer")

        self.include_transition_countdown = include_transition_countdown
        self.include_signal_status = include_signal_status
        self.include_layout_change_mask = (
            include_transition_countdown
            if include_layout_change_mask is None
            else include_layout_change_mask
        )
        self.transition_warning_steps = transition_warning_steps
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
        self.has_scheduled_recipes = any(
            phase.recipe is not None for phase in dynamic_layout.phases
        )
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
        self.phase_has_recipe = jnp.asarray(
            [phase.recipe is not None for phase in dynamic_layout.phases],
            dtype=jnp.bool_,
        )
        self.phase_recipes = jnp.asarray(
            [
                0
                if phase.recipe is None
                else sum(1 << (2 + 2 * ingredient) for ingredient in phase.recipe)
                for phase in dynamic_layout.phases
            ],
            dtype=jnp.int32,
        )
        self.phase_ends = jnp.cumsum(self.phase_durations)
        self.cycle_steps = int(dynamic_layout.cycle_steps)

    def _get_obs_shape(self):
        obs_shape = super()._get_obs_shape()

        def _append_transition_features(shape, obs_type):
            extra_features = int(self.include_signal_status) + int(
                self.include_transition_countdown
            )
            if self.has_scheduled_recipes:
                extra_features += self.layout.num_ingredients
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
        steps_remaining = self.get_steps_until_layout_change(step)
        warning_active = self.get_transition_warning_active(step)
        countdown = steps_remaining.astype(jnp.float32) / float(
            self.transition_warning_steps
        )
        return jnp.where(warning_active, countdown, 0.0)

    def get_transition_warning_active(self, step: jax.Array) -> jax.Array:
        steps_remaining = self.get_steps_until_layout_change(step)
        return (steps_remaining > 0) & (
            steps_remaining <= self.transition_warning_steps
        )

    def get_layout_change_mask(self, step: jax.Array) -> jax.Array:
        layout_index = self.get_layout_index(step)
        next_layout_index = (layout_index + 1) % self.phase_static_objects.shape[0]
        static_change_mask = (
            self.phase_static_objects[layout_index]
            != self.phase_static_objects[next_layout_index]
        )
        recipe_changes = (
            self.phase_has_recipe[layout_index]
            & self.phase_has_recipe[next_layout_index]
            & (self.phase_recipes[layout_index] != self.phase_recipes[next_layout_index])
        )
        recipe_indicator_mask = (
            self.phase_static_objects[layout_index] == StaticObject.RECIPE_INDICATOR
        ) | (
            self.phase_static_objects[next_layout_index]
            == StaticObject.RECIPE_INDICATOR
        )
        return static_change_mask | (recipe_changes & recipe_indicator_mask)

    def get_observation_layout_change_mask(self, step: jax.Array) -> jax.Array:
        return self.get_layout_change_mask(step) & self.get_transition_warning_active(step)

    def _set_transition_awareness(self, state: State) -> State:
        layout_index = self.get_layout_index(state.step)
        next_layout_index = (layout_index + 1) % self.phase_recipes.shape[0]
        next_recipe = jnp.where(
            self.phase_has_recipe[next_layout_index],
            self.phase_recipes[next_layout_index],
            state.recipe,
        )
        return state.replace(
            steps_until_layout_change=self.get_steps_until_layout_change(state.step),
            layout_change_mask=self.get_layout_change_mask(state.step),
            next_recipe=next_recipe,
        )

    def _append_default_next_recipe(self, obs, state):
        if not self.has_scheduled_recipes:
            return obs
        ingredient_indices = jnp.arange(self.layout.num_ingredients)
        ingredient_counts = (state.next_recipe >> (2 + 2 * ingredient_indices)) & 0x3
        recipe_indicator_mask = (
            state.grid[:, :, 0] == StaticObject.RECIPE_INDICATOR
        )
        preview = (
            recipe_indicator_mask[..., None] * ingredient_counts[None, None, :]
        ).astype(jnp.float32)
        preview = jnp.broadcast_to(preview, (*obs.shape[:-3], *preview.shape))
        return jnp.concatenate([obs.astype(jnp.float32), preview], axis=-1)

    def _append_featurized_next_recipe(self, obs, state):
        if not self.has_scheduled_recipes:
            return obs
        ingredient_indices = jnp.arange(self.layout.num_ingredients)
        ingredient_counts = (
            (state.next_recipe >> (2 + 2 * ingredient_indices)) & 0x3
        ).astype(jnp.float32)
        preview = jnp.broadcast_to(
            ingredient_counts, (*obs.shape[:-1], ingredient_counts.size)
        )
        return jnp.concatenate([obs.astype(jnp.float32), preview], axis=-1)

    def _append_default_transition_features(self, obs, step):
        transition_layers = []
        if self.include_transition_countdown:
            countdown = self.get_transition_countdown(step)
            transition_layers.append(
                jnp.full((*obs.shape[:-1], 1), countdown, dtype=jnp.float32)
            )
        if self.include_layout_change_mask:
            change_mask = self.get_observation_layout_change_mask(step).astype(
                jnp.float32
            )
            change_mask = jnp.broadcast_to(
                change_mask,
                (*obs.shape[:-3], *change_mask.shape),
            )
            transition_layers.append(change_mask[..., None])
        if not transition_layers:
            return obs
        return jnp.concatenate([obs.astype(jnp.float32), *transition_layers], axis=-1)

    def _append_default_signal_status(self, obs, state):
        if not self.include_signal_status:
            return obs
        static_objects = state.grid[:, :, 0]
        signal_time = jnp.where(
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR,
            state.grid[:, :, 2],
            0,
        )
        signal_status = signal_time.astype(jnp.float32) / float(
            self.signal_activation_time
        )
        signal_status = jnp.broadcast_to(
            signal_status,
            (*obs.shape[:-3], *signal_status.shape),
        )
        return jnp.concatenate(
            [obs.astype(jnp.float32), signal_status[..., None]], axis=-1
        )

    def _append_featurized_transition_features(self, obs, step):
        transition_features = []
        if self.include_transition_countdown:
            countdown = self.get_transition_countdown(step)
            transition_features.append(
                jnp.full((*obs.shape[:-1], 1), countdown, dtype=jnp.float32)
            )
        if self.include_layout_change_mask:
            change_mask = self.get_observation_layout_change_mask(step).astype(
                jnp.float32
            )
            transition_features.append(
                jnp.broadcast_to(
                    change_mask.flatten(), (*obs.shape[:-1], change_mask.size)
                )
            )
        if not transition_features:
            return obs
        return jnp.concatenate([obs.astype(jnp.float32), *transition_features], axis=-1)

    def _append_featurized_signal_status(self, obs, state):
        if not self.include_signal_status:
            return obs
        static_objects = state.grid[:, :, 0]
        signal_time = jnp.max(
            jnp.where(
                static_objects == StaticObject.BUTTON_RECIPE_INDICATOR,
                state.grid[:, :, 2],
                0,
            )
        )
        signal_status = signal_time.astype(jnp.float32) / float(
            self.signal_activation_time
        )
        signal_feature = jnp.full(
            (*obs.shape[:-1], 1), signal_status, dtype=jnp.float32
        )
        return jnp.concatenate([obs.astype(jnp.float32), signal_feature], axis=-1)

    def get_obs_default(self, state: State):
        obs = super().get_obs_default(state)
        obs = self._append_default_signal_status(obs, state)
        obs = self._append_default_transition_features(obs, state.step)
        return self._append_default_next_recipe(obs, state)

    def get_obs_featurized(self, state: State):
        obs = super().get_obs_featurized(state)
        obs = self._append_featurized_signal_status(obs, state)
        obs = self._append_featurized_transition_features(obs, state.step)
        return self._append_featurized_next_recipe(obs, state)

    def reset(self, key: PRNGKeyArray):
        _, state = super().reset(key)
        initial_recipe = jnp.where(
            self.phase_has_recipe[0], self.phase_recipes[0], state.recipe
        )
        state = state.replace(
            recipe=initial_recipe,
            previous_recipe=initial_recipe,
            legacy_recipe_deliveries_remaining=0,
        )
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
        recipe_before_step = state.recipe
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
        recipe_changed = state.recipe != recipe_before_step
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
        signal_steps_remaining = jnp.max(
            jnp.where(
                static_objects == StaticObject.BUTTON_RECIPE_INDICATOR,
                state.grid[:, :, 2],
                0,
            )
        )
        signal_active = signal_steps_remaining > 0
        signal_activated = signal_steps_remaining == self.signal_activation_time
        center_column = static_objects.shape[1] // 2
        column_indices = jnp.arange(static_objects.shape[1])[None, :]
        left_mask = column_indices < center_column
        right_mask = column_indices > center_column
        workload_mask = (
            (static_objects == StaticObject.POT)
            | (static_objects == StaticObject.PLATE_PILE)
            | (static_objects == StaticObject.GOAL)
        )
        ingredient_pile_mask = StaticObject.is_ingredient_pile(static_objects)
        left_workload_tile_count = jnp.sum(workload_mask & left_mask)
        right_workload_tile_count = jnp.sum(workload_mask & right_mask)
        left_ingredient_pile_count = jnp.sum(ingredient_pile_mask & left_mask)
        right_ingredient_pile_count = jnp.sum(ingredient_pile_mask & right_mask)
        infos = {
            **infos,
            "layout_index": jnp.full((self.num_agents,), layout_index),
            "layout_changed": jnp.full((self.num_agents,), layout_changed),
            "recipe_changed": jnp.full((self.num_agents,), recipe_changed),
            "recipe_onion_count": jnp.full(
                (self.num_agents,), (state.recipe >> 2) & 0x3
            ),
            "recipe_tomato_count": jnp.full(
                (self.num_agents,), (state.recipe >> 4) & 0x3
            ),
            "legacy_recipe_deliveries_remaining": jnp.full(
                (self.num_agents,), state.legacy_recipe_deliveries_remaining
            ),
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
            "signal_steps_remaining": jnp.full(
                (self.num_agents,), signal_steps_remaining
            ),
            "signal_active": jnp.full((self.num_agents,), signal_active),
            "signal_activated": jnp.full((self.num_agents,), signal_activated),
            "left_workload_tile_count": jnp.full(
                (self.num_agents,), left_workload_tile_count
            ),
            "right_workload_tile_count": jnp.full(
                (self.num_agents,), right_workload_tile_count
            ),
            "left_ingredient_pile_count": jnp.full(
                (self.num_agents,), left_ingredient_pile_count
            ),
            "right_ingredient_pile_count": jnp.full(
                (self.num_agents,), right_ingredient_pile_count
            ),
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

        next_recipe = jnp.where(
            self.phase_has_recipe[layout_index],
            self.phase_recipes[layout_index],
            state.recipe,
        )
        recipe_changed = next_recipe != state.recipe
        old_cooked_recipe = state.recipe | DynamicObject.COOKED
        old_plated_recipe = old_cooked_recipe | DynamicObject.PLATE
        dynamic_objects = grid[:, :, 1]
        extra_info = grid[:, :, 2]
        old_started_pots = (
            (new_static == StaticObject.POT)
            & (dynamic_objects == state.recipe)
            & (extra_info > 0)
        )
        old_cooked_pots = (
            (new_static == StaticObject.POT)
            & (dynamic_objects == old_cooked_recipe)
        )
        old_plated_on_grid = dynamic_objects == old_plated_recipe
        old_plated_in_inventory = state.agents.inventory == old_plated_recipe
        legacy_deliveries = (
            jnp.sum(old_started_pots)
            + jnp.sum(old_cooked_pots)
            + jnp.sum(old_plated_on_grid)
            + jnp.sum(old_plated_in_inventory)
        ).astype(jnp.int32)

        agent_positions = self._relocate_blocked_agents(
            state.agents.pos,
            state.agents.dir,
            new_static,
            self.phase_agent_positions[layout_index],
        )

        return state.replace(
            agents=state.agents.replace(pos=agent_positions),
            grid=grid,
            recipe=next_recipe,
            previous_recipe=jnp.where(
                recipe_changed, state.recipe, state.previous_recipe
            ),
            legacy_recipe_deliveries_remaining=jnp.where(
                recipe_changed,
                legacy_deliveries,
                state.legacy_recipe_deliveries_remaining,
            ),
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
