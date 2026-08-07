"""Validated cyclic layouts for the original single-recipe Overcooked env."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from flax.core.frozen_dict import FrozenDict

from jaxmarl.environments.overcooked import dynamic_layout_data
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
from jaxmarl.environments.overcooked.layouts import layout_grid_to_dict

_STATIC_OBJECTS = {
    " ": OBJECT_TO_INDEX["empty"],
    "A": OBJECT_TO_INDEX["empty"],
    "W": OBJECT_TO_INDEX["wall"],
    "X": OBJECT_TO_INDEX["goal"],
    "B": OBJECT_TO_INDEX["plate_pile"],
    "O": OBJECT_TO_INDEX["onion_pile"],
    "0": OBJECT_TO_INDEX["onion_pile"],
    "P": OBJECT_TO_INDEX["pot"],
}


def _parse_grid(grid: str) -> tuple[FrozenDict, np.ndarray, Tuple[Tuple[int, int], ...]]:
    rows = grid.splitlines()
    while rows and not rows[0]:
        rows = rows[1:]
    while rows and not rows[-1]:
        rows = rows[:-1]
    if not rows:
        raise ValueError("Dynamic layout map must not be empty")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("All rows in a dynamic layout map must have the same width")

    unknown = sorted({cell for row in rows for cell in row if cell not in _STATIC_OBJECTS})
    if unknown:
        raise ValueError(
            "V1 dynamic layouts support only W, A, X, B, O/0, P and spaces; "
            f"unsupported symbols: {unknown}"
        )

    normalized = "\n".join(row.replace("0", "O") for row in rows)
    layout = layout_grid_to_dict(normalized)
    static_objects = np.asarray(
        [[_STATIC_OBJECTS[cell] for cell in row] for row in rows],
        dtype=np.int32,
    )
    agent_positions = tuple(
        (int(index % layout["width"]), int(index // layout["width"]))
        for index in np.asarray(layout["agent_idx"])
    )
    return layout, static_objects, agent_positions


@dataclass(frozen=True)
class DynamicLayoutPhase:
    layout: FrozenDict
    static_objects: np.ndarray
    agent_positions: Tuple[Tuple[int, int], ...]
    steps: int
    name: str = ""

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("steps must be an integer")
        if self.steps <= 0:
            raise ValueError("steps must be greater than zero")

    @classmethod
    def from_grid(cls, grid: str, steps: int, name: str = ""):
        layout, static_objects, agent_positions = _parse_grid(grid)
        return cls(layout, static_objects, agent_positions, steps, name)


@dataclass(frozen=True)
class DynamicLayout:
    phases: Tuple[DynamicLayoutPhase, ...]

    def __post_init__(self):
        phases = tuple(self.phases)
        object.__setattr__(self, "phases", phases)
        if not phases:
            raise ValueError("A dynamic layout must contain at least one phase")

        first = phases[0]
        first_shape = first.static_objects.shape
        num_agents = len(first.agent_positions)
        num_goals = len(first.layout["goal_idx"])
        num_pots = len(first.layout["pot_idx"])
        if num_agents != 2:
            raise ValueError(
                "Phase 0 must contain exactly two agents; "
                f"found {num_agents}"
            )

        for phase_index, phase in enumerate(phases):
            if phase.static_objects.shape != first_shape:
                raise ValueError(
                    f"Phase {phase_index} has size {phase.static_objects.shape}; "
                    f"expected {first_shape}"
                )
            if len(phase.agent_positions) != num_agents:
                raise ValueError(
                    f"Phase {phase_index} has {len(phase.agent_positions)} agents; "
                    f"expected {num_agents}"
                )
            if len(phase.layout["goal_idx"]) != num_goals:
                raise ValueError(
                    f"Phase {phase_index} has {len(phase.layout['goal_idx'])} goals; "
                    f"expected {num_goals}"
                )
            if len(phase.layout["pot_idx"]) != num_pots:
                raise ValueError(
                    f"Phase {phase_index} has {len(phase.layout['pot_idx'])} pots; "
                    f"expected {num_pots}"
                )
            if len(set(phase.agent_positions)) != num_agents:
                raise ValueError(
                    f"Phase {phase_index} agent start positions must be unique"
                )
            for x, y in phase.agent_positions:
                if not (0 <= y < first_shape[0] and 0 <= x < first_shape[1]):
                    raise ValueError(
                        f"Phase {phase_index} agent start position {(x, y)} "
                        "is outside the map"
                    )
                if phase.static_objects[y, x] != OBJECT_TO_INDEX["empty"]:
                    raise ValueError(
                        f"Phase {phase_index} agent start position {(x, y)} "
                        "is not an empty cell"
                    )
            if np.count_nonzero(
                phase.static_objects == OBJECT_TO_INDEX["empty"]
            ) < num_agents:
                raise ValueError(
                    f"Phase {phase_index} must have at least one empty cell per agent"
                )

    @classmethod
    def from_data(
        cls,
        data: Sequence[Sequence[object]],
        names: Sequence[str] | None = None,
    ) -> "DynamicLayout":
        if names is None:
            names = ("",) * len(data)
        elif len(names) != len(data):
            raise ValueError("names and dynamic layout entries must have the same length")

        phases = []
        for index, (entry, name) in enumerate(zip(data, names)):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"Dynamic layout entry {index} must be [map_string, steps]"
                )
            grid, steps = entry
            if not isinstance(grid, str):
                raise TypeError(f"Dynamic layout entry {index} map must be a string")
            phases.append(DynamicLayoutPhase.from_grid(grid, steps, name))
        return cls(tuple(phases))

    @property
    def initial_layout(self) -> FrozenDict:
        return self.phases[0].layout

    @property
    def cycle_steps(self) -> int:
        return sum(phase.steps for phase in self.phases)


dynamic_layouts = {
    "dynamic_cramped_room": DynamicLayout.from_data(
        [
            ["""
WWPWW
OA AO
W   W
WBWXW
""", 100],
            ["""
WWPWW
OA AO
W W W
WBWXW
""", 100],
            ["""
WWPWW
OA AO
W B W
WBWXW
""", 100],
        ],
        names=("open", "wall", "plate_pile"),
    )
}

def _load_named_dynamic_layout(name, data):
    try:
        return DynamicLayout.from_data(data)
    except (TypeError, ValueError) as error:
        raise type(error)(f"Invalid dynamic layout {name!r}: {error}") from error


dynamic_layouts.update(
    {
        name: _load_named_dynamic_layout(name, data)
        for name, data in vars(dynamic_layout_data).items()
        if not name.startswith("_") and isinstance(data, (list, tuple))
    }
)


__all__ = ["DynamicLayout", "DynamicLayoutPhase", "dynamic_layouts"]
