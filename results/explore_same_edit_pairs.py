"""The same edit where the kitchen has a spare and where it has none.

Every panel is a kitchen that was trained and evaluated: the left column is a
V1 diagnostic cell, the right column one of the benchmark's own transitions.
Each panel is the kitchen after the boundary, with the edit marked on it and
what the edit did to the cooking:

    wall, dispenser   a tick on each station a cook can still use, a cross
                      on each one it cannot, and on the spot a lost station
                      stood -- so a complete set of ticks is a soup that can
                      still be made
    swap              the trip each cook makes for the job it had before;
                      a trip that grows is drawn in red

    python results/explore_same_edit_pairs.py
"""

from __future__ import annotations

import ast
import re
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/paper/explore"

from results.explore_maps_over_bars import CROSS, FRAME, TILE, centre, frames  # noqa: E402
from results.explore_route_metrics import access, bfs, parse, shortest_path, stations, walkable  # noqa: E402

OK = "#2a9d8f"
BAD = CROSS
#: Every tile is drawn the same size in every panel, and every stroke, badge
#: and ring is sized from that one number, so the marks keep their proportion
#: to the kitchen whatever the figure is scaled to.
TILE_IN = 0.155            # one tile, in inches, in every panel
PT = TILE_IN * 72          # one tile, in points
#: Pixels per tile the viewer renders at. The overlays are laid out in units of
#: the viewer's usual 32-pixel tile, so a sharper render is fitted back into
#: that frame with ``extent`` rather than changing every coordinate. At 128 a
#: tile embeds at about 830 dpi, enough that a PDF viewer never has to blur it.
RENDER_PX = 128


def sharp_frame(layout, index):
    """One phase of a registered kitchen, rendered at RENDER_PX per tile."""
    import jax
    from jaxmarl import make
    from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
    from results.explore_maps_over_bars import phase_state
    env = make("overcooked_v3", layout=layout, max_steps=450, random_agent_positions=False)
    _, initial = env.reset(jax.random.PRNGKey(0))
    viz = OvercookedV3Visualizer(tile_size=RENDER_PX)
    return np.asarray(viz._render_frame(phase_state(env, initial, index)))


def lw(frac):
    """A line width given as a fraction of one tile."""
    return frac * PT


def halo(frac):
    return [pe.withStroke(linewidth=lw(frac), foreground="white")]
#: The four stations a soup needs, in the order a cook visits them.
NEEDED = ("ingredient", "pot", "plate", "serve")

#: Panels where each cook is walked to a station that is no longer there.
DEAD_ENDS = {"outage_0"}
#: label, then (layout, reason) for the column with a spare and the one without.
ROWS = (
    ("Add a wall",
     ("rs_v1_coord_ring_block", ""),
     ("rs_v2_test_time_wide_block", ""),
     ("split_0", "")),
)


def phase_grids(name):
    """The two phases of an evaluated cell, whichever table defines it."""
    if name.startswith("rs_v"):
        src = (ROOT / "jaxmarl/environments/overcooked_v3/shift_diagnostic_data.py").read_text()

        def table(table_name):
            m = re.search(rf"^{table_name} = (\{{.*?^\}})", src, re.S | re.M)
            assert m is not None, table_name
            return ast.literal_eval(m.group(1))

        family = "V2_" if name.startswith("rs_v2") else ""
        base, edit = name.split("_", 2)[2].rsplit("_", 1)
        return (table(f"{family}ORIGINAL_KITCHENS")[base].strip().split("\n"),
                table(f"{family}DRAWN_KITCHENS")[(base, edit)].strip().split("\n"))
    import jaxmarl.environments.overcooked_v3.dynamic_layout_data as data
    obj = getattr(data, name)
    raw = obj["phases"] if isinstance(obj, dict) and "phases" in obj else obj
    seen = []
    for p in raw:
        g = p[0] if isinstance(p, (list, tuple)) else p
        t = str(g).strip().split("\n")
        if t not in seen:
            seen.append(t)
    return seen[0], seen[1]


def blank_agents(image, rows):
    """Paint the cooks out; the swap row draws its own start points instead."""
    fy, fx = next((y, x) for y, r in enumerate(rows) for x, c in enumerate(r) if c == " ")
    floor = image[fy * TILE:(fy + 1) * TILE, fx * TILE:(fx + 1) * TILE].copy()
    out = image.copy()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "A":
                out[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE] = floor
    return out


def regions(free):
    seen = np.zeros_like(free, bool)
    out = []
    for y in range(free.shape[0]):
        for x in range(free.shape[1]):
            if free[y, x] and not seen[y, x]:
                q, group = deque([(y, x)]), []
                seen[y, x] = True
                while q:
                    cy, cx = q.popleft()
                    group.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]
                                and free[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append(group)
    return sorted(out, key=len, reverse=True)


def badge(ax, cell, good):
    """A tick or a cross in the corner of a station tile."""
    y, x = cell
    cx, cy = (x + 0.80) * TILE, (y + 0.22) * TILE
    ax.add_patch(Circle((cx, cy), TILE * 0.21, facecolor=OK if good else BAD,
                        edgecolor="white", linewidth=lw(0.05), zorder=8))
    # The tick and cross are drawn as strokes, not glyphs: Times has neither,
    # and a fallback font would be the only non-Times text in the figure.
    r = TILE * 0.11
    if good:
        xs = [cx - r, cx - r * 0.25, cx + r]
        ys = [cy, cy + r * 0.75, cy - r * 0.7]
        ax.plot(xs, ys, color="white", lw=lw(0.045), solid_capstyle="round",
                solid_joinstyle="round", zorder=9)
    else:
        for dx in (-1, 1):
            ax.plot([cx - dx * r * 0.8, cx + dx * r * 0.8], [cy - r * 0.8, cy + r * 0.8],
                    color="white", lw=lw(0.045), solid_capstyle="round", zorder=9)


def kind_of(ch):
    """Station kind of a grid character; each ingredient type is its own kind."""
    if ch.isdigit() or ch == "O":
        return f"ingredient {0 if ch == 'O' else int(ch)}"
    return {"P": "pot", "B": "plate", "X": "serve"}.get(ch)


def needed(name, rows):
    """Station kinds a soup needs here: pot, plate, serving, and the ingredient
    types the kitchen's recipes use. V2 kitchens stock ingredients no recipe
    asks for, and a cook walled off from those has lost nothing."""
    types = {kind_of(c) for r in rows for c in r if kind_of(c) and kind_of(c).startswith("ingredient")}
    if name.startswith("rs_v2"):
        from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
        base = name.split("_", 2)[2].rsplit("_", 1)[0]
        recipes = overcooked_v2_layouts[base].possible_recipes
        types = {f"ingredient {i}" for recipe in recipes for i in recipe}
    return {"pot", "plate", "serve"} | types


def reach(rows, cook, kinds):
    """Which of these kinds a cook standing at ``cook`` can use."""
    r = parse(rows)
    free = walkable(r)
    home = {c for piece in regions(free) if cook in piece for c in piece}
    got = set()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            k = kind_of(ch)
            if k in kinds and set(access(r, (y, x), free)) & home:
                got.add(k)
    return got


def working_cook(name, before):
    """The red cook, the one every panel is read from, and its seat."""
    return 0, next((y, x) for y, row in enumerate(before) for x, c in enumerate(row) if c == "A")


def usability(name, before, after):
    """Mark what the red cook can still use, one verdict per needed kind.

    A kind counts as usable if the cook reaches any one of its stations; those
    stations get a tick. A kind the cook reaches nowhere gets a cross on each of
    its stations, and on the spot one stood if it vanished. Other stations are
    left unmarked: a kitchen that keeps one reachable pot has lost nothing by
    walling off another.
    """
    _, cook = working_cook(name, before)
    # Only what the cook had before can be lost; a station it never reached is
    # the kitchen's design, not the edit's doing, and is left unmarked.
    kinds = reach(before, cook, needed(name, before))
    ra = parse(after)
    free = walkable(ra)
    home = {c for piece in regions(free) if cook in piece for c in piece}
    where = {}
    for y, row in enumerate(after):
        for x, ch in enumerate(row):
            k = kind_of(ch)
            if k in kinds:
                where.setdefault(k, []).append((y, x))
    marks = []
    for k, cells in where.items():
        reached = [c for c in cells if set(access(ra, c, free)) & home]
        marks += [(c, True) for c in reached] if reached else [(c, False) for c in cells]
    for y, row in enumerate(before):
        for x, ch in enumerate(row):
            k = kind_of(ch)
            if k in kinds and k not in where:
                marks.append(((y, x), False))
    return marks


def jobs(before, after):
    """Each cook's trip to the end of the work it was nearer before the edit."""
    out = []
    for rows in (before, after):
        r = parse(rows)
        free = walkable(r)
        st = stations(r)
        out.append((r, free, st))
    (rb, fb, sb), (ra, fa, sa) = out
    trips = []
    for start in [(y, x) for y, row in enumerate(before) for x, c in enumerate(row) if c == "A"]:
        near = {}
        for kind in ("ingredient", "serve"):
            cells = [c for s in sb.get(kind, []) for c in access(rb, s, fb)]
            d = bfs(fb, [start])
            near[kind] = min((d[c] for c in cells if np.isfinite(d[c])), default=np.inf)
        job = min(near, key=lambda k: near[k])
        was = shortest_path(fb, [start], [c for s in sb[job] for c in access(rb, s, fb)])
        now = shortest_path(fa, [start], [c for s in sa[job] for c in access(ra, s, fa)])
        trips.append((start, was, now))
    return trips


def route(ax, start, cells, colour, dashed=False, end=None):
    """A cook's trip from where it stands; the cook itself is drawn by the viewer."""
    path = [start] + list(cells)
    if end is not None:
        path.append(end)
    if len(path) < 2:
        path = [start, start]
    ax.plot([(x + 0.5) * TILE for _, x in path], [(y + 0.5) * TILE for y, _ in path],
            color=colour, lw=2.0, zorder=6, solid_capstyle="round",
            linestyle=(0, (2.4, 1.6)) if dashed else "-",
            path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
    if end is None:
        y, x = path[-1]
        ax.add_patch(Circle(((x + 0.5) * TILE, (y + 0.5) * TILE), TILE * 0.13,
                            facecolor=colour, edgecolor="white", lw=0.8, zorder=7))


def mark_edit(ax, before, after):
    """On the after frame: what the edit changed, and where.

    A new wall is hatched; a station that is gone is framed and crossed where it
    stood; a station that moved is framed at its new place.
    """
    lost = []
    for y, (ra, rb) in enumerate(zip(before, after)):
        for x, (was, now) in enumerate(zip(ra, rb)):
            if was == now:
                continue
            hatched = was in " A"
            # Transparency goes on the fill alone: a patch-wide alpha would fade
            # the orange frame and the hatching along with it.
            ax.add_patch(Rectangle((x * TILE - 0.5, y * TILE - 0.5), TILE, TILE,
                                   facecolor=(1, 1, 1, 0.22) if hatched else "none",
                                   hatch="////" if hatched else None,
                                   edgecolor=FRAME, linewidth=lw(0.07), zorder=5,
                                   path_effects=halo(0.12)))
            if was not in " AN" and now in "W N":
                lost.append((y, x))
                cx, cy = centre((y, x))
                r = TILE * 0.28
                for dx in (-1, 1):
                    ax.plot([cx - dx * r, cx + dx * r], [cy - r, cy + r], color=CROSS,
                            lw=lw(0.08), solid_capstyle="round", zorder=7,
                            path_effects=halo(0.14))
    return set(lost)



#: Figure 7's convention: the red cook solid, its partner dashed.
SEAT_COLOUR = ("#E8322A", "#2233CC")


def trip(ax, cells, seat, dashed=None):
    """One cook's walk, in the style of the paired-trajectory figure."""
    if len(cells) < 2:
        return
    dashed = seat == 1 if dashed is None else dashed
    colour = SEAT_COLOUR[seat]
    xs = [(x + 0.5) * TILE for _, x in cells]
    ys = [(y + 0.5) * TILE for y, _ in cells]
    ax.plot(xs, ys, color=colour, lw=lw(0.085), zorder=6, solid_capstyle="round",
            dash_capstyle="round", linestyle=(0, (2.2, 1.4)) if dashed else "-",
            path_effects=halo(0.16))
    ax.add_patch(Circle((xs[-1], ys[-1]), TILE * 0.10, facecolor=colour,
                        edgecolor="white", linewidth=lw(0.03), zorder=7))


def paths(ax, name, before, after):
    """The walking a cook has to do after the edit.

    Where the kitchen keeps a way round, the red cook's trip from the pot to
    the serving window, which the wall either lengthens or never meets. Where
    the floor splits, the hand-off: the partner carries a plate to the counter
    between the halves, and the red cook fetches it from beside its pot.
    """
    r = parse(after)
    free = walkable(r)
    st = stations(r)
    cooks = [(y, x) for y, row in enumerate(after) for x, c in enumerate(row) if c == "A"]
    home = {c: i for i, piece in enumerate(regions(free)) for c in piece}
    near = lambda kind, start: [c for s in st.get(kind, []) for c in access(r, s, free)
                                if c in home and home[c] == home[start]]
    # Hand-off only where the edit took something from the red cook; a kitchen
    # built with two rooms is not split by an edit that leaves them as they were.
    kinds = needed(name, before)
    lost_kinds = reach(before, cooks[0], kinds) - reach(after, cooks[0], kinds)
    if not lost_kinds:
        red = cooks[0]
        leg = shortest_path(free, near("pot", red), near("serve", red))
        trip(ax, leg, 0)
        return
    # The counter the two halves share that makes the hand-off shortest.
    red, blue = cooks[0], cooks[1]
    best = None
    best_counter = [None]
    for y, row in enumerate(after):
        for x, ch in enumerate(row):
            if ch not in "WN":
                continue
            left = [c for c in access(r, (y, x), free) if c in home and home[c] == home[red]]
            right = [c for c in access(r, (y, x), free) if c in home and home[c] == home[blue]]
            if not left or not right:
                continue
            a = shortest_path(free, near("pot", red), left)
            b = shortest_path(free, near("plate", blue), right)
            c = shortest_path(free, right, near("serve", blue))
            if a and b and c:
                cost = len(a) + len(b) + len(c)
                if best is None or cost < best[0]:
                    best = (cost, a, b, c)
                    best_counter[0] = (y, x)
    if best:
        # Run both trips onto the counter itself, so the two lines meet where
        # the plate changes hands.
        _, a, b, c = best
        counter = best_counter[0]
        trip(ax, a + [counter], 0)
        trip(ax, b + [counter], 1)
        trip(ax, [counter] + c, 1)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def steps_label(ax, cells, text, colour):
    """The step count beside a trip.

    Above the far end for a long trip; for a trip of a tile or two that spot
    sits on the start marker, so the count goes to the side of the end instead.
    """
    y, x = cells[-1] if cells else (0, 0)
    if len(cells) > 3:
        ax.text((x + 0.5) * TILE, (y - 0.15) * TILE, text, ha="center", va="bottom",
                fontsize=5.8, color=colour, fontweight="bold", zorder=9,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    else:
        ax.text((x + 1.02) * TILE, (y + 0.5) * TILE, text, ha="left", va="center",
                fontsize=5.8, color=colour, fontweight="bold", zorder=9,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def panel_into(ax, name, swap=False, walks=False):
    """The kitchen after the edit, with what the edit did to the cooking on it."""
    before, after = phase_grids(name)
    ax.imshow(sharp_frame(name, 1), interpolation="none",
              extent=(-0.5, len(after[0]) * TILE - 0.5, len(after) * TILE - 0.5, -0.5))
    lost = mark_edit(ax, before, after)
    # The ticks and crosses are read from the red cook's side, so that cook is
    # framed. The viewer draws the first cook, in grid order, in red.
    seat, (y, x) = working_cook(name, before)
    # A ring in the cook's own colour with a white outline, the way Figure 3
    # singles out a cook. The viewer draws seat 0 red and seat 1 blue.
    ax.add_patch(Circle(((x + 0.5) * TILE, (y + 0.5) * TILE), TILE * 0.46, fill=False,
                        edgecolor=(CROSS, "#2233CC")[seat], linewidth=lw(0.08), zorder=6,
                        path_effects=halo(0.14)))
    if swap:
        for start, was, now in jobs(before, after):
            # A path lists its cells, so steps are one fewer. A trip counts as
            # grown only if it adds real walking: a cook standing beside a pile
            # that moves one tile away has not had its job changed.
            steps_was, steps_now = len(was) - 1, len(now) - 1
            grew = steps_now >= steps_was + 2 and steps_now >= 1.5 * max(steps_was, 1)
            colour = BAD if grew else OK
            route(ax, start, now, colour)
            steps_label(ax, now, f"{steps_was}\u2192{steps_now}", colour)
    else:
        for cell, good in usability(name, before, after):
            if cell not in lost:
                badge(ax, cell, good)
        if walks:
            paths(ax, name, before, after)
        if name in DEAD_ENDS:
            # Walk each cook to where the station stood: the trip ends on the
            # cross, because there is nothing left there to pick up.
            ra = parse(after)
            free = walkable(ra)
            for start in [(y, x) for y, row in enumerate(after) for x, c in enumerate(row) if c == "A"]:
                for cell in lost:
                    path = shortest_path(free, [start], access(ra, cell, free))
                    if path:
                        route(ax, start, path[1:], BAD, dashed=True, end=cell)
    bare(ax)
    ax.set_xlim(-0.5, len(after[0]) * TILE - 0.5)
    ax.set_ylim(len(after) * TILE - 0.5, -0.5)
    return ax


def main(walks=False):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    panels = [name for _, *cells in ROWS for name, _ in cells]
    shapes = [(len(phase_grids(n)[1]), len(phase_grids(n)[1][0])) for n in panels]
    gap, margin, below = 0.14, 0.03, 0.30          # inches
    width = sum(w for _, w in shapes) * TILE_IN + gap * (len(shapes) - 1) + 2 * margin
    height = max(h for h, _ in shapes) * TILE_IN + below + margin
    plt.rcParams["hatch.linewidth"] = lw(0.04)
    fig = plt.figure(figsize=(width, height))
    #: Named the way the bar figure names its groups, so the two read together.
    captions = ("Carroll et al.,\nrandom edit", "Gessler et al.,\nrandom edit",
                "Allocation-aware edit")
    x = margin
    for name, (h, w), caption in zip(panels, shapes, captions):
        # Each panel is exactly its grid at the common tile size, sitting on
        # a shared baseline, so a tile is the same size wherever it is.
        ax = fig.add_axes([x / width, below / height, w * TILE_IN / width, h * TILE_IN / height])
        panel_into(ax, name, walks=walks)
        fig.text((x + w * TILE_IN / 2) / width, (below - 0.04) / height, caption,
                 ha="center", va="top", fontsize=7.0, linespacing=1.1)
        x += w * TILE_IN + gap
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "same_edit_pairs_paths" if walks else "same_edit_pairs"
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)
    print("figure ->", OUT / f"{stem}.png")


if __name__ == "__main__":
    main(walks="--paths" in sys.argv)
