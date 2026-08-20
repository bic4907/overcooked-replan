"""Validated cyclic layouts for the Overcooked V3 environment."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from jaxmarl.environments.overcooked_v3 import dynamic_layout_data
from jaxmarl.environments.overcooked_v3.common import StaticObject
from jaxmarl.environments.overcooked_v3.layouts import Layout

_ALLOWED_SYMBOLS = set(" WAXBP RLSO0123456789")
_DEFAULT_RECIPES = [[0, 0, 0]]
_OUTAGE_RECIPES = [[0, 0]]
_RECIPE_SWITCH_RECIPES = [[0, 0, 1], [0, 1, 1]]


def _parse_grid(
    grid: str,
    possible_recipes: Sequence[Sequence[int]] = _DEFAULT_RECIPES,
) -> tuple[Layout, Tuple[Tuple[int, int], ...]]:
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
    layout = Layout.from_string(
        normalized,
        possible_recipes=[list(recipe) for recipe in possible_recipes],
    )
    agent_positions = tuple((int(x), int(y)) for x, y in layout.agent_positions)
    return layout, agent_positions


@dataclass(frozen=True)
class DynamicLayoutPhase:
    layout: Layout
    agent_positions: Tuple[Tuple[int, int], ...]
    steps: int
    name: str = ""
    recipe: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("steps must be an integer")
        if self.steps <= 0:
            raise ValueError("steps must be greater than zero")
        if self.recipe is not None:
            recipe = tuple(self.recipe)
            object.__setattr__(self, "recipe", recipe)
            if list(recipe) not in self.layout.possible_recipes:
                raise ValueError(
                    f"Phase recipe {list(recipe)} is not in possible_recipes"
                )

    @classmethod
    def from_grid(
        cls,
        grid: str,
        steps: int,
        name: str = "",
        possible_recipes: Sequence[Sequence[int]] = _DEFAULT_RECIPES,
        recipe: Optional[Sequence[int]] = None,
    ):
        layout, agent_positions = _parse_grid(grid, possible_recipes)
        return cls(
            layout,
            agent_positions,
            steps,
            name,
            None if recipe is None else tuple(recipe),
        )


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
        possible_recipes: Sequence[Sequence[int]] = _DEFAULT_RECIPES,
    ) -> "DynamicLayout":
        if names is None:
            names = ("",) * len(data)
        elif len(names) != len(data):
            raise ValueError(
                "names and dynamic layout entries must have the same length"
            )

        phases = []
        for index, (entry, name) in enumerate(zip(data, names)):
            if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
                raise ValueError(
                    "Dynamic layout entry "
                    f"{index} must be [map_string, steps] or "
                    "[map_string, steps, recipe]"
                )
            grid, steps = entry[:2]
            recipe = entry[2] if len(entry) == 3 else None
            if not isinstance(grid, str):
                raise TypeError(f"Dynamic layout entry {index} map must be a string")
            phases.append(
                DynamicLayoutPhase.from_grid(
                    grid,
                    steps,
                    name,
                    possible_recipes=possible_recipes,
                    recipe=recipe,
                )
            )
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
        if name.startswith("outage"):
            possible_recipes = _OUTAGE_RECIPES
        elif name.startswith("recipe_switch"):
            possible_recipes = _RECIPE_SWITCH_RECIPES
        else:
            possible_recipes = _DEFAULT_RECIPES
        return DynamicLayout.from_data(data, possible_recipes=possible_recipes)
    except (TypeError, ValueError) as error:
        raise type(error)(f"Invalid dynamic layout {name!r}: {error}") from error


dynamic_layouts.update(
    {
        name: _load_named_dynamic_layout(name, data)
        for name, data in vars(dynamic_layout_data).items()
        if not name.startswith("_") and isinstance(data, (list, tuple))
    }
)


POLICY_SWITCH_BASE_LAYOUTS = tuple(dynamic_layouts)
_STATIC_POLICY_PHASE_STEPS = 1_000_000_000


def _phase_policy_signature(phase: DynamicLayoutPhase) -> tuple:
    """Describe every phase property that changes the policy's task."""
    static_objects = np.asarray(phase.layout.static_objects)
    return (
        static_objects.shape,
        static_objects.dtype.str,
        static_objects.tobytes(),
        tuple(phase.agent_positions),
        phase.recipe,
    )


def phase_policy_layout_name(base_layout: str, policy_index: int) -> str:
    """Return the no-transition training layout for one unique phase policy."""
    if base_layout not in POLICY_SWITCH_BASE_LAYOUTS:
        raise ValueError(f"Unsupported policy-switch layout: {base_layout}")
    if isinstance(policy_index, bool) or not isinstance(
        policy_index, (int, np.integer)
    ):
        raise TypeError("policy_index must be an integer")
    policy_index = int(policy_index)
    if policy_index < 0:
        raise ValueError("policy_index must be non-negative")
    return f"{base_layout}_policy_{policy_index}"


def phase_policy_sequence(base_layout: str) -> Tuple[int, ...]:
    """Map every dynamic phase to a deduplicated static policy index."""
    if base_layout not in POLICY_SWITCH_BASE_LAYOUTS:
        raise ValueError(f"Unsupported policy-switch layout: {base_layout}")
    signatures = []
    sequence = []
    for phase in dynamic_layouts[base_layout].phases:
        signature = _phase_policy_signature(phase)
        try:
            policy_index = signatures.index(signature)
        except ValueError:
            policy_index = len(signatures)
            signatures.append(signature)
        sequence.append(policy_index)
    return tuple(sequence)


def _register_static_phase_policy_layouts() -> None:
    """Expose one transition-free training layout per unique dynamic phase."""
    for base_layout in POLICY_SWITCH_BASE_LAYOUTS:
        sequence = phase_policy_sequence(base_layout)
        phases = dynamic_layouts[base_layout].phases
        for policy_index in range(max(sequence) + 1):
            source_phase_index = sequence.index(policy_index)
            source_phase = phases[source_phase_index]
            static_phase = DynamicLayoutPhase(
                layout=source_phase.layout,
                agent_positions=source_phase.agent_positions,
                steps=_STATIC_POLICY_PHASE_STEPS,
                name=f"policy_{policy_index}",
                recipe=source_phase.recipe,
            )
            dynamic_layouts[phase_policy_layout_name(base_layout, policy_index)] = (
                DynamicLayout((static_phase,))
            )


_register_static_phase_policy_layouts()

ROLE_SCENARIO_LAYOUTS = {
    **{
        family: tuple(f"{family}_{variant}" for variant in range(3))
        for family in ("splitnosig", "splitsig", "outagenosig", "outagesig")
    },
    "recipe_switch": tuple(f"recipe_switch_{variant}" for variant in range(3)),
    "distance_switch": tuple(
        f"distance_switch_{variant}" for variant in range(10)
    ),
}
ROLE_SCENARIO_LAYOUT_NAMES = tuple(
    name for names in ROLE_SCENARIO_LAYOUTS.values() for name in names
)

__all__ = [
    "DynamicLayout",
    "DynamicLayoutPhase",
    "POLICY_SWITCH_BASE_LAYOUTS",
    "ROLE_SCENARIO_LAYOUTS",
    "ROLE_SCENARIO_LAYOUT_NAMES",
    "dynamic_layouts",
    "phase_policy_layout_name",
    "phase_policy_sequence",
]
