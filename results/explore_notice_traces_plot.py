"""Drawing for the paired Notice / Masked trajectories of the Section 6.2 probe.

Reads what results/explore_notice_traces.py wrote: for a probe state, the
kitchen as the countdown starts, the informed cook's route solid red and its
alert-blind H0 partner's dashed blue, tiles the boundary will change framed
and anything set down in the window marked. No figure of its own; the paper
figure is results/plot_notice_combined.py.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment/notice"))

import probe_preparation as pp  # noqa: E402
from common import build_env  # noqa: E402
from jaxmarl.environments.overcooked_v3.common import DynamicObject, Position, StaticObject  # noqa: E402
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer  # noqa: E402

TRACES = ROOT / "outputs/notice/traces"
PX = 128                       # rendered pixels per tile
TILE_IN = 0.16                 # one tile, in inches, in every panel
PT = TILE_IN * 72
#: FRAME marks the tiles the boundary changes: the Google palette's Orange.
RED, BLUE, FRAME = "#E8322A", "#2F5FD0", "#E37400"
#: The frame round an item set down in the window, in the informed cook's red.
PLACED = RED
#: The viewer's own cook colours, so a ghost is the same shape and hue as the
#: cook it trails.
AGENT_RED, AGENT_BLUE = "#ff0000", "#0000ff"
#: Write the step labels (takes, puts, pot states, the scored act) into the
#: panels. Off for the paper figure, which leaves the reading to the caption.
LABELS = False

POT_INK = "#333333"
#: How far, in tiles, each route sits off the cell centres (along both axes).
ROUTE_OFFSET = 0.10   # the informed cook at 0.4 of a tile, its partner at 0.6


def lw(frac):
    return frac * PT


def halo(frac):
    return [pe.withStroke(linewidth=lw(frac), foreground="white")]


def probe_frame(record):
    """The kitchen at the probe state, rendered with the viewer."""
    env = build_env(record["layout"], "none")
    stacked, _ = pp.probe_states(env, 0, record["probe_step"], pp.INFORMED_START.get(record["layout"]))
    state = jax.tree_util.tree_map(lambda a: a[record["state"]], stacked)
    xy = np.asarray(record["ON"]["xy"][0])
    state = state.replace(agents=state.agents.replace(
        pos=Position(x=jnp.asarray(xy[:, 0]), y=jnp.asarray(xy[:, 1]))))
    return np.asarray(OvercookedV3Visualizer(tile_size=PX)._render_frame(state))


def boundary_frame(record, arm, partner=False, step=None, cooks_at=None):
    """The kitchen as the boundary arrives, drawn on its phase-A layout.

    The stored render is the state just after the boundary step, which
    already carries phase B's tiles. This redraws that state -- where the
    informed cook stands, what it holds, what lies on every counter and in
    every pot -- on the layout the cooks were working in, so the changed
    tiles show as they were and the frames mark what is about to go.
    With ``partner`` the partner is rendered too, as it stands; otherwise
    it is left out for the figure to draw itself.
    """
    env = build_env(record["layout"], "none")
    _, state = env.reset(jax.random.PRNGKey(0))
    pre = record["pre"] if step is None else step
    rec = record[arm]
    grid = np.array(state.grid)
    grid[:, :, 0] = np.asarray(record["static_a"])
    grid[:, :, 1] = np.asarray(rec["items"][pre])
    grid[:, :, 2] = np.asarray(rec["timer"][pre])
    # The cooks may be drawn at another step than the kitchen around them.
    pre = pre if cooks_at is None else cooks_at
    n = 2 if partner else 1
    at = lambda v: jnp.asarray(np.asarray(v)[pre, :n])  # noqa: E731
    xy = np.asarray(rec["xy"])
    state = state.replace(grid=jnp.asarray(grid), agents=state.agents.replace(
        pos=Position(x=jnp.asarray(xy[pre, :n, 0]), y=jnp.asarray(xy[pre, :n, 1])),
        dir=at(rec["dir"]), inventory=at(rec["inventory"])))
    agents = jax.tree_util.tree_map(lambda a: a[:n] if getattr(a, "ndim", 0) and a.shape[0] == 2 else a,
                                    state.agents)
    return np.asarray(OvercookedV3Visualizer(tile_size=PX)._render_frame(state.replace(agents=agents)))


def regions(static):
    free = np.asarray(static) == StaticObject.EMPTY
    label = -np.ones(free.shape, int)
    n = 0
    for y in range(free.shape[0]):
        for x in range(free.shape[1]):
            if free[y, x] and label[y, x] < 0:
                q = deque([(y, x)])
                label[y, x] = n
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]
                                and free[ny, nx] and label[ny, nx] < 0):
                            label[ny, nx] = n
                            q.append((ny, nx))
                n += 1
    return label


def event_step(record, arm, kind):
    """When the informed cook does the scored thing within the window, if it does."""
    xy = np.asarray(record[arm]["xy"])[: record["pre"] + 1, 0]
    if kind == "cross":
        label = regions(record["static_b"])
        start = label[xy[0, 1], xy[0, 0]]
        for t, (x, y) in enumerate(xy):
            if label[y, x] >= 0 and start >= 0 and label[y, x] != start:
                return t
        return None
    if kind == "wait":
        # Table 1's Inversion endpoint: standing, at the boundary, as close to
        # the pile about to open as the kitchen allows. The step is when the
        # cook arrived there and stayed.
        env = build_env(record["layout"], "none")
        _, reach, _, best, _ = pp.activating_source(env, tuple(int(v) for v in xy[0]))
        close = np.array([reach[y, x] <= best for x, y in xy])
        if not close[-1]:
            return None
        return int(len(close) - np.argmin(close[::-1])) if not close.all() else 0
    inv = np.asarray(record[arm]["inventory"])[: record["pre"] + 1, 0]
    for t in range(1, len(inv)):
        if inv[t] & DynamicObject.PLATE and not inv[t - 1] & DynamicObject.PLATE:
            return t
    return None


#: Rotation of the viewer's right-pointing triangle for each facing, in degrees
#: on image axes (y down): UP, DOWN, RIGHT, LEFT.
FACING = {0: 270, 1: 90, 2: 0, 3: 180}
AGENT = ((0.87, 0.50), (0.12, 0.19), (0.12, 0.81))


#: The viewer's ingredient colours, in ingredient order.
INGREDIENT = ("#ffff00", "#009600", "#a020f0", "#00ffff", "#ff0000", "#e6b400")
VEC = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}


def held(ax, x, y, inv, alpha, z):
    """What a cook carries, drawn where the viewer draws it on the cook's tile."""
    from matplotlib.patches import Circle as C
    inv = int(inv)
    if inv == 0:
        return
    at = lambda u, v: ((x + u) * PX, (y + v) * PX)
    if inv & DynamicObject.PLATE:
        ax.add_patch(C(at(0.75, 0.75), 0.20 * PX, facecolor="white", edgecolor="none",
                       alpha=alpha, zorder=z))
        rest, dots = inv >> 2, []
        for idx in range(6):
            dots += [idx] * (rest & 0x3)
            rest >>= 2
        for (u, v), idx in zip(((0.65, 0.65), (0.85, 0.65), (0.75, 0.85)), dots):
            ax.add_patch(C(at(u, v), 0.10 * PX, facecolor=INGREDIENT[idx], edgecolor="none",
                           alpha=alpha, zorder=z + 0.1))
        return
    rest, idx = inv >> 2, 0
    while rest and not rest & 0x3:
        rest >>= 2
        idx += 1
    ax.add_patch(C(at(0.75, 0.75), 0.15 * PX, facecolor=INGREDIENT[idx], edgecolor="none",
                   alpha=alpha, zorder=z))


def what(inv):
    inv = int(inv)
    if inv == 0:
        return None
    if inv == DynamicObject.PLATE:
        return "plate"
    if inv & DynamicObject.PLATE:
        return "dish"
    return "ingr."


def actions(record, arm, seat):
    """(step, faced cell, verb) for every pick-up and hand-down in the window.

    Read off the inventory: empty to holding is a take, holding to empty a
    put, plate to dish plating a soup. The cell is the one the cook faced when
    it interacted, and what stands there names the put: a counter, a pot or
    the serving window.
    """
    xy = np.asarray(record[arm]["xy"])[: record["pre"] + 1, seat]
    facing = np.asarray(record[arm]["dir"])[: record["pre"] + 1, seat]
    inv = np.asarray(record[arm]["inventory"])[: record["pre"] + 1, seat]
    static = np.asarray(record["static_a"])
    out = []
    for t in range(1, len(inv)):
        before, after = what(inv[t - 1]), what(inv[t])
        if before == after:
            continue
        dx, dy = VEC[int(facing[t - 1])]
        cell = (int(xy[t - 1][0]) + dx, int(xy[t - 1][1]) + dy)
        target = int(static[cell[1], cell[0]])
        if before is None:
            verb = f"take {after}"
        elif after is None:
            verb = {StaticObject.POT: "into pot", StaticObject.GOAL: "serve"}.get(target, f"set {before} down")
        elif before == "plate" and after == "dish":
            verb = "plate soup"
        else:
            verb = f"{before} to {after}"
        out.append((t, cell, verb))
    return out


def pot_events(record, arm):
    """(step, pot cell, what) for how each pot stands and changes in the window.

    A pot already busy when the window opens is named at +0; after that, the
    steps at which a full pot starts to cook and a cooking one finishes.
    Filling and emptying are the cooks' own takes and puts, labelled there.
    """
    static = np.asarray(record["static_a"])
    items = np.asarray(record[arm]["items"])
    timer = np.asarray(record[arm]["timer"])
    out = []
    state = lambda t, y, x: ("ready" if items[t, y, x] & DynamicObject.COOKED  # noqa: E731
                             else "cooking" if timer[t, y, x] > 0 else None)
    for y, x in zip(*np.where(static == StaticObject.POT)):
        if state(0, y, x):
            out.append((0, (int(x), int(y)), state(0, y, x)))
        for t in range(1, len(items)):
            now = state(t, y, x)
            if now and now != state(t - 1, y, x):
                out.append((t, (int(x), int(y)), now))
    return out


def ghosts(ax, xy, facing, colour, inventory=None, scale=1.0):
    """Earlier steps as the viewer's own cook shape, fading with age.

    The solid cook at the end of the trail is the rendered kitchen at the
    boundary; the ghosts are every step before it, the oldest faintest, so a
    cook that lingers darkens its cell and one that walks leaves a trail.
    """
    from matplotlib.patches import Polygon
    steps = len(xy) - 1
    for t in range(steps):
        a = np.deg2rad(FACING[int(facing[t])])
        c, s_ = np.cos(a), np.sin(a)
        (x, y) = xy[t]
        pts = [((x + 0.5 + (u - 0.5) * c - (v - 0.5) * s_) * PX,
                (y + 0.5 + (u - 0.5) * s_ + (v - 0.5) * c) * PX) for u, v in AGENT]
        alpha = scale * (0.10 + 0.45 * t / max(steps - 1, 1))
        ax.add_patch(Polygon(pts, closed=True, facecolor=colour, edgecolor="none",
                             alpha=alpha, zorder=4))
        if inventory is not None:
            held(ax, x, y, inventory[t], scale * min(1.0, alpha / scale + 0.25), 4.5)


def walk(ax, xy, colour, dashed=False, placed=None, z=6, offset=0.0):
    """The route as one line: a hollow dot where it starts and a small
    arrowhead on each straight leg; the cook drawn at the boundary ends it.

    ``placed`` collects where arrowheads and dots already sit in this panel,
    so a leg walked back over, or one the other cook shares, puts its
    arrowhead elsewhere along it -- or leaves it out when there is no room.
    """
    from matplotlib.patches import Polygon
    placed = [] if placed is None else placed
    pts = (xy + 0.5 + offset) * PX
    # The route ends at the centre of the last cell, under the cook drawn
    # there, rather than off to one side where it would show past it.
    pts[-1] = (xy[-1] + 0.5) * PX
    ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=lw(0.05), solid_capstyle="round",
            solid_joinstyle="round", dash_capstyle="round",
            linestyle=(0, (2.4, 1.6)) if dashed else "-", zorder=z)
    placed += [pts[0], pts[-1]]
    corners = [pts[0]]
    for q in pts[1:]:
        if (q != corners[-1]).any():
            corners.append(q)
    legs, start = [], 0
    for k in range(1, len(corners)):
        turn = k + 1 < len(corners) and not np.array_equal(
            np.sign(corners[k + 1] - corners[k]), np.sign(corners[k] - corners[k - 1]))
        if turn or k + 1 == len(corners):
            legs.append((corners[start], corners[k]))
            start = k
    length, width = 0.34 * PX, 0.28 * PX
    for a, b in legs:
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        for f in (0.5, 0.3, 0.7, 0.2, 0.8):
            tip = a + (b - a) * f + d * length / 2
            if all(np.linalg.norm(tip - q) > 0.45 * PX for q in placed):
                break
        else:
            continue
        placed.append(tip)
        base = tip - d * length
        ax.add_patch(Polygon([tip, base + n * width / 2, base - n * width / 2], closed=True,
                             facecolor=colour, edgecolor="none", zorder=z + 0.1))
    ax.plot(*pts[0], marker="o", ms=lw(0.20), mfc="white", mec=colour, mew=lw(0.05), zorder=z + 0.2)


def cook(ax, xy, facing, inv, colour, alpha=1.0, z=4):
    """A cook as the viewer draws it -- its triangle and what it carries --
    so it can be laid over the routes instead of under them."""
    from matplotlib.patches import Polygon
    a = np.deg2rad(FACING[int(facing)])
    c, s_ = np.cos(a), np.sin(a)
    x, y = xy
    pts = [((x + 0.5 + (u - 0.5) * c - (v - 0.5) * s_) * PX,
            (y + 0.5 + (u - 0.5) * s_ + (v - 0.5) * c) * PX) for u, v in AGENT]
    ax.add_patch(Polygon(pts, closed=True, facecolor=colour, edgecolor="none",
                         alpha=alpha, zorder=z))
    held(ax, x, y, inv, alpha, z + 0.2)


def partner_at_boundary(ax, xy, facing, inv, alpha=0.30):
    """The partner where the boundary finds it, drawn faded over the kitchen."""
    cook(ax, xy, facing, inv, AGENT_BLUE, alpha, 4)


def placed_items(ax, record, arm):
    """Ring what the window left on a counter that was not there at its start.

    A plate set aside before a blackout is the act the figure is about, and on
    the rendered kitchen it is just another plate; the ring says it is new.
    """
    from matplotlib.patches import Circle as C
    static = np.asarray(record["static_a"])
    items = np.asarray(record[arm]["items"])
    fresh = (static == StaticObject.WALL) & (items[-1] != 0) & (items[-1] != items[0])
    for y, x in zip(*np.where(fresh)):
        # The kitchen may be drawn as the countdown starts, before the thing
        # was put there, so draw it as well.
        v = int(items[-1, y, x])
        ax.add_patch(C(((x + 0.5) * PX, (y + 0.5) * PX), 0.30 * PX, edgecolor="none", zorder=6.5,
                       facecolor="white" if v & DynamicObject.PLATE else INGREDIENT[0]))
        if v & DynamicObject.PLATE:
            # A dish carries its soup as dots, as the viewer draws them.
            rest, dots = v >> 2, []
            for idx in range(len(INGREDIENT)):
                dots += [idx] * (rest & 0x3)
                rest >>= 2
            for (u, w), idx in zip(((0.40, 0.42), (0.60, 0.42), (0.50, 0.60)), dots):
                ax.add_patch(C(((x + u) * PX, (y + w) * PX), 0.10 * PX, edgecolor="none",
                               facecolor=INGREDIENT[idx], zorder=6.6))
        # Framed like the tiles the boundary changes, in the informed cook's
        # colour.
        ax.add_patch(Rectangle((x * PX, y * PX), PX, PX, fill=False, edgecolor=PLACED,
                               linewidth=lw(0.07), zorder=7))


def passing_items(ax, record, arm):
    """Things that sat on a counter at some step but are gone by the boundary.

    The rendered kitchen shows what the boundary finds; a plate that was
    there at the start and taken, or set down and picked up again, would
    otherwise leave no mark. Each is drawn faded, dashed, where it lay.
    """
    from matplotlib.patches import Circle as C
    static = np.asarray(record["static_a"])
    items = np.asarray(record[arm]["items"])
    for y, x in zip(*np.where((static != StaticObject.POT) & (items != items[-1]).any(0))):
        for v in sorted({int(v) for v in items[:, y, x]} - {0, int(items[-1, y, x])}):
            centre = ((x + 0.5) * PX, (y + 0.5) * PX)
            if v & DynamicObject.PLATE:
                ax.add_patch(C(centre, 0.30 * PX, facecolor=(1, 1, 1, 0.45), edgecolor="#555",
                               linestyle=(0, (2, 1.5)), linewidth=lw(0.02), zorder=3))
            else:
                ax.add_patch(C(centre, 0.22 * PX, facecolor=(1, 1, 0, 0.45), edgecolor="#555",
                               linestyle=(0, (2, 1.5)), linewidth=lw(0.02), zorder=3))


def panel(ax, record, image, arm, kind, trail="line"):
    h, w = len(record["static_a"]), len(record["static_a"][0])
    ax.imshow(image, interpolation="none", extent=(0, w * PX, h * PX, 0))
    a, b = np.asarray(record["static_a"]), np.asarray(record["static_b"])
    for y, x in zip(*np.where(a != b)):
        ax.add_patch(Rectangle((x * PX, y * PX), PX, PX, fill=False, edgecolor=FRAME,
                               linewidth=lw(0.07), zorder=5))
    xy = np.asarray(record[arm]["xy"])[: record["pre"] + 1]
    if trail == "ghost":
        facing = np.asarray(record[arm]["dir"])[: record["pre"] + 1]
        inv = np.asarray(record[arm]["inventory"])[: record["pre"] + 1]
        passing_items(ax, record, arm)
        placed_items(ax, record, arm)
        # The partner is context, not the subject: no trail, only where the
        # boundary finds it, faded.
        partner_at_boundary(ax, xy[-1, 1], facing[-1, 1], inv[-1, 1])
        ghosts(ax, xy[:, 0], facing[:, 0], AGENT_RED, inv[:, 0])
    else:
        facing = np.asarray(record[arm]["dir"])[: record["pre"] + 1]
        inv = np.asarray(record[arm]["inventory"])[: record["pre"] + 1]
        passing_items(ax, record, arm)
        placed_items(ax, record, arm)
        # Both routes through cell centres, the partner's dashed.
        # The informed cook's route first, so its arrowheads get the places;
        # the partner's sits underneath.
        placed = []
        # Each cook's route sits off the cell centres, the two to opposite
        # sides, so a cell both walk through shows both lines.
        # From the bottom up: the partner's route, the partner, the informed
        # cook's route, the informed cook -- each cook over the routes that
        # reach it, and the informed route over everything of the partner's.
        # Routes that never share a cell keep to the cell centres.
        shared = {tuple(c) for c in xy[:, 0].tolist()} & {tuple(c) for c in xy[:, 1].tolist()}
        apart = ROUTE_OFFSET if shared else 0.0
        walk(ax, xy[:, 0], RED, placed=placed, z=6.0, offset=-apart)
        walk(ax, xy[:, 1], BLUE, dashed=True, placed=placed, z=5.0, offset=apart)
        # Each cook in its route's colour, so the line and the cook read as one.
        cook(ax, xy[-1, 1], facing[-1, 1], inv[-1, 1], BLUE, z=5.5)
        cook(ax, xy[-1, 0], facing[-1, 0], inv[-1, 0], RED, z=6.5)
    if LABELS:
        label_events(ax, record, arm, xy, kind)
    ax.set_xlim(0, w * PX)
    ax.set_ylim(h * PX, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def label_events(ax, record, arm, xy, kind):
    # Every take and put by the informed cook, on the cell it was done at,
    # so a plate set down and picked up again inside the window still shows;
    # and how the pots stand. The partner's own acts are left unlabelled.
    stacked = {}
    events = [(t, c, v, POT_INK) for t, c, v in pot_events(record, arm)]
    events += [(t, c, v, RED) for t, c, v in actions(record, arm, 0)]
    for t, (cx, cy), verb, colour in sorted(events, key=lambda e: e[0]):
        k = stacked.get((cx, cy), 0)
        stacked[(cx, cy)] = k + 1
        width = len(record["static_a"][0])
        ha, u = ("left", 0.02) if cx == 0 else ("right", 0.98) if cx == width - 1 else ("center", 0.5)
        ax.text((cx + u) * PX, (cy + 0.08 + 0.30 * k) * PX, f"+{t} {verb}",
                ha=ha, va="top", fontsize=0.24 * PT, color=colour,
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor=colour, linewidth=lw(0.015)))
    t = event_step(record, arm, kind)
    if t is not None and kind in ("cross", "wait"):
        x, y = xy[t, 0]
        ax.text((x + 0.5) * PX, (y + 0.05) * PX, f"+{t} {kind}", ha="center", va="bottom",
                fontsize=0.34 * PT, color=RED, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=RED,
                          linewidth=lw(0.02)))

