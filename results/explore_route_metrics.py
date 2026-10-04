"""Grid and route helpers for the kitchen figures.

Reads a kitchen written as rows of characters, finds which cells a cook can
walk and which stations it can reach, and walks shortest routes between them.
Used by results/explore_same_edit_pairs.py and results/explore_spare_or_not.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np

#: A cook stands on the floor and reaches a station, so only these are walkable.
FLOOR = {" ", "A"}
#: Anything a cook uses. "W" walls and "N" blockers are neither walkable nor used.
STATIONS = {"P": "pot", "B": "plate", "X": "serve", "R": "recipe"}


def parse(grid):
    rows = [list(r) for r in grid]
    w = max(len(r) for r in rows)
    for r in rows:
        r.extend(" " * (w - len(r)))
    return rows


def walkable(rows):
    return np.array([[c in FLOOR for c in row] for row in rows])


def stations(rows):
    out = {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            # "O" is how the original kitchens write an onion pile; the loader
            # rewrites it to "0", and missing that here loses every V1 pile.
            kind = STATIONS.get(ch) or ("ingredient" if ch.isdigit() or ch == "O" else None)
            if kind:
                out.setdefault(kind, []).append((y, x))
    return out


def access(rows, cell, free):
    y, x = cell
    out = []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < free.shape[0] and 0 <= nx < free.shape[1] and free[ny, nx]:
            out.append((ny, nx))
    return out


def bfs(free, sources):
    dist = np.full(free.shape, np.inf)
    q = deque()
    for s in sources:
        if free[s]:
            dist[s] = 0
            q.append(s)
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]
                    and free[ny, nx] and dist[ny, nx] == np.inf):
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return dist


def shortest_path(free, sources, targets):
    """One shortest route from any source to any target, as a list of cells."""
    prev = {}
    seen = set()
    q = deque()
    for s in sources:
        if free[s]:
            seen.add(s)
            q.append(s)
    goal = None
    targets = {t for t in targets if free[t]}
    while q:
        cur = q.popleft()
        if cur in targets:
            goal = cur
            break
        y, x = cur
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (y + dy, x + dx)
            if (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]
                    and free[nxt] and nxt not in seen):
                seen.add(nxt)
                prev[nxt] = cur
                q.append(nxt)
    if goal is None:
        return []
    path = [goal]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    return path[::-1]


def station_route(rows, free, kind_a, kind_b):
    """The route a cook walks from one station kind to another."""
    st = stations(rows)
    src = [c for s in st.get(kind_a, []) for c in access(rows, s, free)]
    dst = [c for s in st.get(kind_b, []) for c in access(rows, s, free)]
    if not src or not dst:
        return []
    return shortest_path(free, src, dst)
