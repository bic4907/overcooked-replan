"""Validated cyclic layouts for the Overcooked V3 environment."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from jaxmarl.environments.overcooked_v3 import dynamic_layout_data
from jaxmarl.environments.overcooked_v3.common import StaticObject
from jaxmarl.environments.overcooked_v3.layouts import Layout

_ALLOWED_SYMBOLS = set(" WAXBP RLO0123456789")
_DEFAULT_RECIPES = [[0, 0, 0]]


def _parse_grid(grid: str) -> tuple[Layout, Tuple[Tuple[int, int], ...]]:
    rows = grid.splitlines()
    while rows and not rows[0]:
        rows = rows[1:]
    while rows and not rows[-1]:
        rows = rows[:-1]
    if not rows:
        raise ValueError("Dynamic layout map must not be empty")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("All rows in a dynamic layout map must have the same width")

    unknown = sorted(
        {cell for row in rows for cell in row if cell not in _ALLOWED_SYMBOLS}
    )
    if unknown:
        raise ValueError(f"Unsupported Overcooked V3 layout symbols: {unknown}")

    normalized = "\n".join(rows)
    layout = Layout.from_string(normalized, possible_recipes=_DEFAULT_RECIPES)
    agent_positions = tuple((int(x), int(y)) for x, y in layout.agent_positions)
    return layout, agent_positions


@dataclass(frozen=True)
class DynamicLayoutPhase:
    layout: Layout
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
        layout, agent_positions = _parse_grid(grid)
        return cls(layout, agent_positions, steps, name)


@dataclass(frozen=True)
class DynamicLayout:
    phases: Tuple[DynamicLayoutPhase, ...]

    def __post_init__(self):
        phases = tuple(self.phases)
        object.__setattr__(self, "phases", phases)
        if not phases:
            raise ValueError("A dynamic layout must contain at least one phase")

        first = phases[0]
        first_shape = first.layout.static_objects.shape
        num_agents = len(first.agent_positions)
        if num_agents != 2:
            raise ValueError(
                f"Phase 0 must contain exactly two agents; found {num_agents}"
            )

        for phase_index, phase in enumerate(phases):
            layout = phase.layout
            if layout.static_objects.shape != first_shape:
                raise ValueError(
                    "All phases must have the same size; "
                    f"phase {phase_index} has {layout.static_objects.shape}, "
                    f"expected {first_shape}"
                )
            if len(phase.agent_positions) != num_agents:
                raise ValueError(
                    f"Phase {phase_index} has {len(phase.agent_positions)} agents; "
                    f"expected {num_agents}"
                )
            if layout.num_ingredients != first.layout.num_ingredients:
                raise ValueError(
                    "All phases must expose the same number of ingredients; "
                    f"phase {phase_index} has {layout.num_ingredients}, "
                    f"expected {first.layout.num_ingredients}"
                )
            if layout.possible_recipes != first.layout.possible_recipes:
                raise ValueError("All phases must use the same possible recipes")
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
                if layout.static_objects[y, x] != StaticObject.EMPTY:
                    raise ValueError(
                        f"Phase {phase_index} agent start position {(x, y)} "
                        "is not an empty cell"
                    )
            if (
                np.count_nonzero(layout.static_objects == StaticObject.EMPTY)
                < num_agents
            ):
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
            raise ValueError(
                "names and dynamic layout entries must have the same length"
            )

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
    def initial_layout(self) -> Layout:
        return self.phases[0].layout

    @property
    def cycle_steps(self) -> int:
        return sum(phase.steps for phase in self.phases)


dynamic_layouts = {
    "dynamic_cramped_room": DynamicLayout.from_data(
        [
            [
                """
WWPWW
0A A0
W   W
WBWXW
""",
                100,
            ],
            [
                """
WWPWW
0A A0
W W W
WBWXW
""",
                100,
            ],
        ],
        names=("open", "wall"),
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

ROLE_SCENARIO_LAYOUTS = {
    family: (f"{family}_0",)
    for family in ("splitnosig", "splitsig", "outagenosig", "outagesig")
}
ROLE_SCENARIO_LAYOUT_NAMES = tuple(
    name for names in ROLE_SCENARIO_LAYOUTS.values() for name in names
)

__all__ = [
    "DynamicLayout",
    "DynamicLayoutPhase",
    "ROLE_SCENARIO_LAYOUTS",
    "ROLE_SCENARIO_LAYOUT_NAMES",
    "dynamic_layouts",
]
