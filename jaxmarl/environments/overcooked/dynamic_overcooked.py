"""Original Overcooked environment with cyclic, step-based map changes."""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import PRNGKeyArray

from jaxmarl.environments.multi_agent_env import Actions
from jaxmarl.environments.overcooked.common import (
    COLOR_TO_INDEX,
    DIR_TO_VEC,
    OBJECT_INDEX_TO_VEC,
    OBJECT_TO_INDEX,
    make_overcooked_map,
)
from jaxmarl.environments.overcooked.dynamic_layouts import (
    DynamicLayout,
    DynamicLayoutPhase,
    dynamic_layouts,
)
from jaxmarl.environments.overcooked.overcooked import (
    POT_EMPTY_STATUS,
    Overcooked,
    State,
)


class DynamicOvercooked(Overcooked):
    """Single-recipe V1 Overcooked whose complete map changes cyclically."""

    def __init__(
        self,
        layout: Union[str, DynamicLayout] = "dynamic_cramped_room",
        **kwargs,
    ):
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

        phase_data = [self._build_phase_data(phase) for phase in dynamic_layout.phases]
        self.phase_static_objects = jnp.asarray(
            np.stack([data[0] for data in phase_data]), dtype=jnp.int32
        )
        self.phase_wall_maps = jnp.asarray(
            np.stack([data[1] for data in phase_data]), dtype=jnp.bool_
        )
        self.phase_maze_templates = jnp.asarray(
            np.stack([data[2] for data in phase_data]), dtype=jnp.uint8
        )
        self.phase_goal_positions = jnp.asarray(
            np.stack([data[3] for data in phase_data]), dtype=jnp.uint32
        )
        self.phase_pot_positions = jnp.asarray(
            np.stack([data[4] for data in phase_data]), dtype=jnp.uint32
        )
        self.phase_agent_positions = jnp.asarray(
            [phase.agent_positions for phase in dynamic_layout.phases],
            dtype=jnp.uint32,
        )
        self.phase_durations = jnp.asarray(
            [phase.steps for phase in dynamic_layout.phases], dtype=jnp.int32
        )
        self.phase_end_steps = jnp.cumsum(self.phase_durations)
        self.cycle_steps = dynamic_layout.cycle_steps

    def _positions(self, indices):
        indices = jnp.asarray(indices, dtype=jnp.uint32)
        return jnp.stack([indices % self.width, indices // self.width], axis=-1)

    def _build_phase_data(self, phase: DynamicLayoutPhase):
        layout = phase.layout
        all_positions = np.arange(self.height * self.width)
        wall_map = np.zeros_like(all_positions, dtype=bool)
        wall_map[np.asarray(layout["wall_idx"])] = True
        wall_map = wall_map.reshape(self.height, self.width)

        agent_positions = jnp.asarray(phase.agent_positions, dtype=jnp.uint32)
        goal_positions = self._positions(layout["goal_idx"])
        plate_positions = self._positions(layout["plate_pile_idx"])
        onion_positions = self._positions(layout["onion_pile_idx"])
        pot_positions = self._positions(layout["pot_idx"])
        pot_status = jnp.full(
            (pot_positions.shape[0],), POT_EMPTY_STATUS, dtype=jnp.uint8
        )
        template = make_overcooked_map(
            wall_map,
            goal_positions,
            agent_positions,
            jnp.zeros((self.num_agents,), dtype=jnp.int32),
            plate_positions,
            onion_positions,
            pot_positions,
            pot_status,
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size,
        )

        padding = self.agent_view_size - 1
        empty = OBJECT_INDEX_TO_VEC[OBJECT_TO_INDEX["empty"]]
        template = template.at[
            padding + agent_positions[:, 1],
            padding + agent_positions[:, 0],
        ].set(empty)

        return (
            phase.static_objects,
            wall_map,
            np.asarray(template),
            np.asarray(goal_positions),
            np.asarray(pot_positions),
        )

    def get_layout_index(self, step: jax.Array) -> jax.Array:
        cycle_step = jnp.mod(step, self.cycle_steps)
        return jnp.sum(cycle_step >= self.phase_end_steps).astype(jnp.int32)

    def _get_move_area(self, state: State) -> jax.Array:
        old_empty = super()._get_move_area(state)
        current_layout_index = self.get_layout_index(state.step)
        next_layout_index = self.get_layout_index(state.step + 1)
        layout_will_change = current_layout_index != next_layout_index
        next_empty = (
            self.phase_static_objects[next_layout_index]
            == OBJECT_TO_INDEX["empty"]
        )
        return old_empty & jnp.where(layout_will_change, next_empty, True)

    def step_env(self, key: PRNGKeyArray, state: State, actions: Actions):
        obs, state, rewards, dones, infos = super().step_env(key, state, actions)

        layout_index = self.get_layout_index(state.step)
        previous_layout_index = self.get_layout_index(state.step - 1)
        layout_changed = layout_index != previous_layout_index
        state = lax.cond(
            layout_changed,
            self._change_layout,
            lambda current_state, _: current_state,
            state,
            layout_index,
        )
        obs = self.get_obs(state)
        infos = {
            **infos,
            "layout_index": jnp.full((self.num_agents,), layout_index),
            "layout_changed": jnp.full((self.num_agents,), layout_changed),
        }
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            infos,
        )

    def _change_layout(self, state: State, layout_index: jax.Array) -> State:
        previous_layout_index = self.get_layout_index(state.step - 1)
        old_static = self.phase_static_objects[previous_layout_index]
        new_static = self.phase_static_objects[layout_index]
        changed_cells = old_static != new_static

        padding = self.agent_view_size - 1
        interior = state.maze_map[
            padding : padding + self.height,
            padding : padding + self.width,
        ]
        template = self.phase_maze_templates[layout_index][
            padding : padding + self.height,
            padding : padding + self.width,
        ]
        interior = jnp.where(changed_cells[..., None], template, interior)
        maze_map = state.maze_map.at[
            padding : padding + self.height,
            padding : padding + self.width,
        ].set(interior)

        agent_positions = self._relocate_blocked_agents(
            state.agent_pos,
            state.agent_dir_idx,
            new_static,
            self.phase_agent_positions[layout_index],
        )

        empty = OBJECT_INDEX_TO_VEC[OBJECT_TO_INDEX["empty"]]
        agent_mask = maze_map[..., 0] == OBJECT_TO_INDEX["agent"]
        maze_map = jnp.where(agent_mask[..., None], empty, maze_map)

        def _agent_tile(direction, agent_index):
            return jnp.array(
                [
                    OBJECT_TO_INDEX["agent"],
                    COLOR_TO_INDEX["red"] + agent_index * 2,
                    direction,
                ],
                dtype=jnp.uint8,
            )

        agent_tiles = jax.vmap(_agent_tile)(
            state.agent_dir_idx, jnp.arange(self.num_agents)
        )
        maze_map = maze_map.at[
            padding + agent_positions[:, 1],
            padding + agent_positions[:, 0],
        ].set(agent_tiles)

        return state.replace(
            agent_pos=agent_positions.astype(jnp.uint32),
            goal_pos=self.phase_goal_positions[layout_index],
            pot_pos=self.phase_pot_positions[layout_index],
            wall_map=self.phase_wall_maps[layout_index],
            maze_map=maze_map,
        )

    def _relocate_blocked_agents(
        self,
        positions,
        directions,
        static_objects,
        spawn_positions,
    ):
        opposite = jnp.array([1, 0, 3, 2], dtype=jnp.int32)
        clockwise = jnp.array([2, 3, 1, 0], dtype=jnp.int32)

        def _relocate_one(agent_index, current_positions):
            x, y = current_positions[agent_index]
            occupied = (
                jnp.zeros((self.height, self.width), dtype=jnp.bool_)
                .at[current_positions[:, 1], current_positions[:, 0]]
                .set(True)
                .at[y, x]
                .set(False)
            )
            available = (
                static_objects == OBJECT_TO_INDEX["empty"]
            ) & ~occupied

            first = opposite[directions[agent_index]]
            second = clockwise[first]
            third = clockwise[second]
            fourth = clockwise[third]
            vectors = DIR_TO_VEC[jnp.stack([first, second, third, fourth])]
            candidates = current_positions[agent_index] + vectors
            candidate_x = candidates[:, 0]
            candidate_y = candidates[:, 1]
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
            neighbor = jnp.array([safe_x[neighbor_index], safe_y[neighbor_index]])

            spawn_order = (
                jnp.arange(self.num_agents, dtype=jnp.int32) + agent_index
            ) % self.num_agents
            ordered_spawns = spawn_positions[spawn_order]
            valid_spawn = available[ordered_spawns[:, 1], ordered_spawns[:, 0]]
            spawn = ordered_spawns[jnp.argmax(valid_spawn)]

            target = jnp.where(jnp.any(valid_neighbor), neighbor, spawn)
            is_blocked = static_objects[y, x] != OBJECT_TO_INDEX["empty"]
            target = jnp.where(is_blocked, target, current_positions[agent_index])
            return current_positions.at[agent_index].set(
                target.astype(current_positions.dtype)
            )

        return lax.fori_loop(0, self.num_agents, _relocate_one, positions)

    @property
    def name(self) -> str:
        return "Dynamic Overcooked"


__all__ = ["DynamicOvercooked"]
