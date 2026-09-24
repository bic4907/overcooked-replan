"""Describe static floor-graph costs for the two deadline-handoff maps.

This is a geometric analysis, not an environment rollout or policy benchmark.
Costs exclude facing, interaction, occupied floors, and opportunity cost of
leaving production early. Starts and staging cells survive the boundary.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import runpy


def shortest_route(grid, start, target):
    floor = {
        (x, y) for y, row in enumerate(grid.strip("\n").splitlines())
        for x, symbol in enumerate(row) if symbol in {" ", "A"}
    }
    if start not in floor or target not in floor:
        raise ValueError(f"Route endpoint is not a floor: {start} -> {target}")
    pending = deque([(start, [start])])
    visited = {start}
    while pending:
        (x, y), route = pending.popleft()
        if (x, y) == target:
            return route
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in floor and neighbor not in visited:
                visited.add(neighbor)
                pending.append((neighbor, [*route, neighbor]))
    raise ValueError(f"No floor route: {start} -> {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = runpy.run_path(str(root / "jaxmarl/environments/overcooked_v3/dynamic_layout_data.py"))
    rows = []
    for name in ("distance_2", "distance_3"):
        grid_rows = data[name][0][0].strip("\n").splitlines()
        center, middle = len(grid_rows[0]) // 2, len(grid_rows) // 2
        for transition, before, after, start, staging, source, handoff in (
            ("A->B: incoming supplier B", 0, 1,
             (center + 2, middle - 1), (center + 2, middle + 1),
             (center + 1, middle + 2), (center + 1, middle + 1)),
            ("B->A: incoming supplier A", 1, 0,
             (center - 2, middle + 1), (center - 2, middle - 1),
             (center - 1, middle - 2), (center - 1, middle - 1)),
        ):
            pre_route = shortest_route(data[name][before][0], start, staging)
            late_route = shortest_route(data[name][after][0], start, staging)
            ready_route = shortest_route(data[name][after][0], staging, handoff)
            unprepared_route = shortest_route(data[name][after][0], start, handoff)
            prepared_source = shortest_route(data[name][after][0], staging, source)
            late_source = shortest_route(data[name][after][0], start, source)
            source_to_handoff = shortest_route(data[name][after][0], source, handoff)
            pre_cost, late_cost = len(pre_route) - 1, len(late_route) - 1
            ready_cost, unprepared_cost = len(ready_route) - 1, len(unprepared_route) - 1
            rows.append({
                "layout": name, "transition": transition,
                "start": start, "staging": staging, "next_handoff_floor": handoff,
                "cross_before": pre_cost, "cross_after": late_cost,
                "cross_multiplier": late_cost / pre_cost,
                "prepared_post_change_moves": ready_cost,
                "unprepared_post_change_moves": unprepared_cost,
                "prepared_total_moves": pre_cost + ready_cost,
                "net_moves_saved": unprepared_cost - pre_cost - ready_cost,
                "new_source_floor": source,
                "first_supply_prepared_post_moves": len(prepared_source) + len(source_to_handoff) - 2,
                "first_supply_unprepared_post_moves": len(late_source) + len(source_to_handoff) - 2,
                "first_supply_prepared_total_moves": pre_cost + len(prepared_source) + len(source_to_handoff) - 2,
                "first_supply_net_moves_saved": len(late_source) - len(prepared_source) - pre_cost,
                "pre_route": pre_route, "late_route": late_route,
            })
    result = {
        "revision": "handoff-site-compact-11x7-v4",
        "kind": "static floor graph, no policy rollouts",
        "excluded": ["facing", "interaction", "other-agent occupancy", "production opportunity cost"],
        "routes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"{row['layout']} {row['transition']}: cross {row['cross_before']} -> {row['cross_after']}; "
              f"post-change handoff {row['prepared_post_change_moves']} prepared vs "
              f"{row['unprepared_post_change_moves']} unprepared; net saving {row['net_moves_saved']}")
        print(f"  first onion to handoff after switch: {row['first_supply_prepared_post_moves']} prepared vs "
              f"{row['first_supply_unprepared_post_moves']} unprepared moves; "
              f"total saving {row['first_supply_net_moves_saved']}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
