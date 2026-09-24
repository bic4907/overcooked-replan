"""Render the compact dual-handoff candidate and measure its floor routes."""

from collections import deque
import html
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/layouts/handoff_convention_0924"
DATA = runpy.run_path(str(ROOT / "jaxmarl/environments/overcooked_v3/dynamic_layout_data.py"))
GRIDS = {
    name: [phase[0].strip("\n").splitlines() for phase in DATA[name][:2]]
    for name in ("distance_7", "distance_8")
}


def distance(grid, start, target):
    floors = {
        (x, y)
        for y, row in enumerate(grid)
        for x, cell in enumerate(row)
        if cell in {" ", "A"}
    }
    if start not in floors or target not in floors:
        return None
    pending = deque([(start, 0)])
    seen = {start}
    while pending:
        (x, y), steps = pending.popleft()
        if (x, y) == target:
            return steps
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in floors and neighbor not in seen:
                seen.add(neighbor)
                pending.append((neighbor, steps + 1))
    return None


def routes(layout_name):
    records = []
    for phase, grid in enumerate(GRIDS[layout_name]):
        for side, start, source_x, handoff_x in (
            ("left", (1, 3), 4, 4),
            ("right", (9, 3), 6, 6),
        ):
            records.append(
                {
                    "phase": "A" if phase == 0 else "B",
                    "side": side,
                    "start_to_upper_source": distance(grid, start, (source_x, 1)),
                    "start_to_lower_source": distance(grid, start, (source_x, 5)),
                    "start_to_upper_handoff": distance(grid, start, (handoff_x, 2)),
                    "start_to_lower_handoff": distance(grid, start, (handoff_x, 4)),
                    "switch_between_handoffs": distance(
                        grid, (handoff_x, 2), (handoff_x, 4)
                    ),
                }
            )
    return records


def svg(layout_name):
    cell, left_margin, top = 42, 42, 78
    gap = 80
    colors = {
        "N": ("#242d39", ""),
        " ": ("#f5f7fb", ""),
        "A": ("#d7f7d8", "A"),
        "W": ("#a6ebef", "H"),
        "0": ("#ffd59b", "O"),
        "P": ("#f4a4a7", "P"),
        "X": ("#dfc6fb", "X"),
        "B": ("#d4ee9a", "B"),
        "R": ("#eee8fc", "R"),
    }
    width = left_margin * 2 + 11 * cell * 2 + gap
    height = top + 7 * cell + 114
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1d2939}.title{font-weight:700;font-size:23px}.phase{font-weight:700;font-size:19px}.tile{font-size:17px;font-weight:700;text-anchor:middle;dominant-baseline:middle}.note{font-size:15px}</style>',
        f'<text class="title" x="{width/2}" y="34" text-anchor="middle">{layout_name}: dual-handoff convention candidate</text>',
    ]
    for phase, grid in enumerate(GRIDS[layout_name]):
        x0 = left_margin + phase * (11 * cell + gap)
        parts.append(
            f'<text class="phase" x="{x0}" y="65">Phase {"A" if phase == 0 else "B"}: {"left" if phase == 0 else "right"} supplies</text>'
        )
        for y, row in enumerate(grid):
            for x, symbol in enumerate(row):
                fill, label = colors[symbol]
                if symbol == "W" and x in (0, 10):
                    label = "S"
                px, py = x0 + x * cell, top + y * cell
                parts.append(
                    f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" fill="{fill}" stroke="#c6cfda"/>'
                )
                if label:
                    parts.append(
                        f'<text class="tile" x="{px + cell/2}" y="{py + cell/2}">{html.escape(label)}</text>'
                    )
    y = top + 7 * cell + 28
    parts.extend(
        [
            f'<text class="note" x="{left_margin}" y="{y}">H: handoff  S: storage  O: onion  P: private pot  A: spawn  X: delivery  B: plate</text>',
            f'<text class="note" x="{left_margin}" y="{y+26}">Upper/lower routes are equal from spawn; switching lanes after the gate closes costs 12 floor moves.</text>',
            f'<text class="note" x="{left_margin}" y="{y+52}">A/B changes who can reach onions; partner pairs must agree on the upper or lower lane.</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in GRIDS:
        (OUT / f"{name}.svg").write_text(svg(name), encoding="utf-8")
        route_name = "route_costs.json" if name == "distance_7" else f"route_costs_{name}.json"
        (OUT / route_name).write_text(
            json.dumps({"layout": name, "floor_routes": routes(name)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
