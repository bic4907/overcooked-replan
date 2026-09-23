"""The paper's scripted cook, as a stand-in for a person.

The paper does not evaluate collaborative agents against a cloned human. It
evaluates them against a scripted one (Appendix B.2): at every timestep the
model "re-plan[s] an action sequence to complete the next high level task
(selected greedily), and take[s] the first action in that plan". Three dials set
how far from optimal it plays:

    prob_wait   how often it does nothing at all
    lltemp      Boltzmann noise on the move, proportional to how much worse
                that move is
    hltemp      Boltzmann noise on the goal, so it sometimes walks to a further
                dispenser

The paper's H is H0: prob_wait fit to the proportion of the time real people
wait, both temperatures zero. With every dial at zero this is the optimal
greedy player, which is the reference the human model is read against.

One mechanism comes from the implementation the paper builds on rather than
from the paper's own text. Two independent greedy cooks in a one-wide corridor
push against each other forever, each planning as though the other were not
there, and no amount of waiting breaks it because neither is moving. The
reference GreedyHumanModel carries an `auto_unstuck` rule for exactly this: when
a step changes nothing about where anyone is standing or facing, take a random
action that does not collide. That rule is here, and it is why this planner
needs a key and a carry.

We have the opposite problem the paper had. Our demonstrations are real people,
which the paper's were not, but there are only forty episodes, so holding some
out to build an evaluation partner leaves a partner too weak to evaluate
against. A planner needs no data at all.

What follows the paper's three dials is a set of rules for the two things the
paper's kitchen never asked: who does which job, and what to do about a
partner that will not move. Each was measured on the six 75-step kitchens,
six seeds apiece, and the ones that cost more than they earned are still here
as options, defaulted off, with what they measured:

    stage_while_cooking   carry the next batch to the counters by the pot
                          while it cooks -- split_0 -50, and still -43 after
                          the collision rules were rewritten
    helper_stages         the plate cook carries ingredients for the loader
                          -- split_0 -57, and -150 after the rewrite
    loader_cap            stop fetching once the counters hold a batch -- -20
    relay                 hand a thing over a counter when the other cook
                          would finish it sooner -- distance_0 -40
    plate_timing          fetch the plate only once the timer is short enough
                          to walk it -- split_0 -20
    idle_turn_cost        price a turn that brings the goal no nearer, to
                          stop a cook facing a wall through a standoff --
                          outage_0 -280, since it also stops a cook turning
                          to the counter it is working at
    aside_wait            stand aside for a moment after giving way -- -50
    defer_to_yielder      leave a cook that always gives way to give way
                          again -- no gain
    yield_turn="state"    decide who gives way by what each cook carries
                          rather than by seat -- -28; the give-up rule is
                          what makes an unknown partner safe in any case
    pile_by_round_trip    draw from the pile that is nearest the pot rather
                          than the one nearest the cook -- outage_0 -27
    long_haul_specialist
        =False            let both cooks load on a kitchen where the pile is
                          far from the pot -- outage_1 -47

One more was tried and taken out rather than left as an option: refusing to
fetch a plate while any pot still wanted filling. It reads right, and it cost
thirteen points a run -- on a kitchen with two pots the plate fetched early is
waiting at the pot when the first soup comes out, and the walk it saves is
worth more than the loading it delays.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked_v3.common import (
    ACTION_TO_DIRECTION,
    MAX_INGREDIENTS,
    Direction,
    DynamicObject,
    OvercookedActionsEnum,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.settings import POT_COOK_TIME
from jaxmarl.environments.overcooked_v3.utils import (
    OvercookedPathPlanner,
    compute_enclosed_spaces,
    mark_adjacent_cells,
)

#: Move actions, in the order their Direction is numbered.
MOVES = jnp.array(
    [
        OvercookedActionsEnum.up,
        OvercookedActionsEnum.down,
        OvercookedActionsEnum.right,
        OvercookedActionsEnum.left,
    ]
)
DIRECTIONS = jnp.array([Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT])


class PlannerCarry(NamedTuple):
    """What the unstuck rule needs to remember: where everyone last stood."""

    positions: jax.Array  # (num_agents, 2)
    directions: jax.Array  # (num_agents,)
    region: jax.Array  # (num_agents,) which half of the kitchen each seat serves
    moved: jax.Array  # (num_agents,) whether the last action was a move
    shared: jax.Array  # (num_agents,) ingredients handed across this outage
    given: jax.Array  # (num_agents, H, W) counters this seat set something on for the other
    streak: jax.Array  # (num_agents,) consecutive steps spent blocked
    loader: jax.Array  # (num_agents,) which seat is loading the pots
    neglect: jax.Array  # (H, W) steps a finished soup has sat in a pot or a dish on a counter
    yield_rate: jax.Array  # (num_agents,) how often the other cook has given way in a standoff
    yielded: jax.Array  # (num_agents,) whether this seat stepped aside last step
    waiting: jax.Array  # (num_agents,) steps left to stand aside before rejoining
    target: jax.Array  # (num_agents,) the cell each seat was walking to, flattened
    idle_for: jax.Array  # (num_agents,) steps since this seat's hands last changed
    inventory: jax.Array  # (num_agents,) what each seat held last step
    pot_contents: jax.Array  # (H, W) what the pots held last step
    stall: jax.Array  # () steps since any pot took or gave up anything
    best_cost: jax.Array  # (num_agents,) closest this seat has been to its goal
    stuck_for: jax.Array  # (num_agents,) steps since it last got closer to it


def ingredient_counts(encoding):
    """How many of each ingredient a recipe, a pot, or a hand holds.

    Ingredients are packed two bits apiece above the plate and cooked flags,
    and quantities are stored by adding, so a count reads straight out.
    """
    shifts = 2 + 2 * jnp.arange(MAX_INGREDIENTS)
    return (encoding >> shifts) & 0x3


def ingredient_on_counter_all(held):
    """Which cells hold a bare ingredient (no plate, not cooked)."""
    return jax.vmap(DynamicObject.is_ingredient)(held.reshape(-1)).reshape(held.shape)


def _manhattan_to(mask):
    """For every cell, the Manhattan distance to the nearest cell of ``mask``."""
    height, width = mask.shape
    ys, xs = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    flat_y, flat_x = ys.reshape(-1), xs.reshape(-1)
    distance = jnp.abs(flat_y[:, None] - flat_y[None, :]) + jnp.abs(
        flat_x[:, None] - flat_x[None, :]
    )
    return jnp.min(
        jnp.where(mask.reshape(-1)[None, :], distance, jnp.inf), axis=1
    ).reshape(mask.shape)


def _min_manhattan(mask_a, mask_b):
    """Shortest Manhattan distance between a cell of each mask (inf if none)."""
    height, width = mask_a.shape
    ys, xs = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    ya, xa = ys.reshape(-1), xs.reshape(-1)
    distance = jnp.abs(ya[:, None] - ya[None, :]) + jnp.abs(xa[:, None] - xa[None, :])
    valid = mask_a.reshape(-1)[:, None] & mask_b.reshape(-1)[None, :]
    return jnp.min(jnp.where(valid, distance, jnp.inf))


def _boltzmann(costs, temperature, key):
    """Pick an index, noisily in proportion to how much worse it is."""
    finite = jnp.isfinite(costs)
    safe = jnp.where(finite, costs, 0.0)
    logits = jnp.where(finite, -safe / temperature, -jnp.inf)
    return jax.random.categorical(key, logits)


class GreedyPlanner:
    """One greedy cook per seat, replanned from scratch every step."""

    def __init__(
        self,
        env,
        prob_wait: float = 0.0,
        lltemp: float = 0.0,
        hltemp: float = 0.0,
        home_bias: float = 4.0,
        free_movement: bool | None = None,
        pot_specialist: int = 2,
        pot_role_connected_only: bool = True,
        pot_role_helper_escape: bool = True,
        return_margin: float = 4.0,
        local_needs: bool = True,
        share_mode: str = "continuous",
        role_margin: int = 2,
        role_hysteresis: int = 6,
        cross_on_countdown: bool = True,
        advantage_roles: bool = True,
        advantage_margin: float = 0.0,
        plate_stock: int | None = None,
        stage_while_cooking: bool = False,
        long_haul_specialist: bool = True,
        helper_stages: bool = False,
        plate_cap_joined: bool = True,
        loader_cap: bool = False,
        hold_ahead: bool = True,
        neglect_steps: int = 30,
        yield_style: str = "adaptive",
        yield_turn: str = "seat",
        aside_style: str = "random",
        idle_turn_cost: float = 0.0,
        heading_cost: float = 20.0,
        hold_max: int = 3,
        defer_to_yielder: bool = False,
        aside_wait: int = 0,
        stubborn_steps: int = 4,
        partner_idle_steps: int = 40,
        stall_steps: int = 40,
        no_progress_steps: int = 24,
        advantage_everywhere: bool = False,
        relay: bool = False,
        relay_margin: float = 4.0,
        stage_margin: float = 2.0,
        plate_timing: bool = False,
        plate_lead: float = 4.0,
        pile_by_round_trip: bool = False,
        seat: int | None = None,
    ):
        if not 0.0 <= prob_wait <= 1.0:
            raise ValueError("prob_wait is a probability")
        if lltemp < 0.0 or hltemp < 0.0:
            raise ValueError("temperatures cannot be negative")
        self.env = env
        self.prob_wait = float(prob_wait)
        self.lltemp = float(lltemp)
        self.hltemp = float(hltemp)
        self.home_bias = float(home_bias)
        # Cooks go wherever the work is while the halves are joined, and only
        # split the kitchen between them once a door shuts.
        self.free_movement = True if free_movement is None else bool(free_movement)
        self.pot_specialist = int(pot_specialist)
        self.pot_role_connected_only = bool(pot_role_connected_only)
        self.pot_role_helper_escape = bool(pot_role_helper_escape)
        self.return_margin = float(return_margin)
        self.local_needs = bool(local_needs)
        self.role_margin = int(role_margin)
        # Once a cook is loading it keeps loading. Re-decided by distance
        # every step, the job changes hands whenever the loader walks off to
        # a far pot and the other cook happens to pass the pile -- both then
        # carry ingredients, both then wait with plates, and the pipeline the
        # division was meant to give never forms. The other cook takes over
        # only when it is nearer the pile by a clear margin.
        self.role_hysteresis = int(role_hysteresis)
        # Being on the wrong side when a door shuts costs the whole phase: the
        # far room's pots and piles go untouched while both cooks work one half.
        # So when the phase ahead would leave them sharing a room, whichever of
        # them can get across soonest does, and it leaves in time rather than on
        # a fixed warning -- twenty steps is not enough on a wide floor and is
        # wasted on a narrow one.
        self.cross_on_countdown = bool(cross_on_countdown)
        # Who fetches the plate is not about who is nearer the plate: it is
        # about who loses less by walking the soup to the serving window. Where
        # the ingredients sit at one end of the floor and the window at the
        # other, the cook beside the pile pays for that trip twice -- once
        # walking it and once in the pot that stands empty meanwhile. Each cook
        # compares its own two errands, window against pile, and the one with
        # relatively less to lose delivers while the other keeps loading.
        self.advantage_roles = bool(advantage_roles)
        # How much cheaper that has to be before the other cook stops taking
        # plates at all. Without a margin a single step of difference hands
        # every soup to one of them, and two soups finishing together leave the
        # second standing.
        self.advantage_margin = float(advantage_margin)
        # A pot takes twenty steps to cook and a cook with nothing to load
        # spends them standing at it holding a plate. That time is better
        # spent walking the next batch of ingredients to the counters beside
        # the pot, so that when the soup is out the pot is refilled in a few
        # steps instead of a few round trips. The plate is fetched only once
        # the timer is down to what the plate errand takes, plus a margin.
        # A blackout takes the plate pile for a whole phase, and the twenty
        # steps of warning are not enough to lay in what that phase will eat:
        # measured, one plate crosses the boundary and the pots then sit
        # cooked for forty steps. So from the start of a phase whose successor
        # has no plates, the plate cook keeps this many plates out on the
        # counters whenever no soup is waiting for one. Left unset, the stock
        # is two plates a pot: about what a pot turns out in one 75-step phase.
        self.plate_stock = None if plate_stock is None else int(plate_stock)
        self.stage_while_cooking = bool(stage_while_cooking)
        self.long_haul_specialist = bool(long_haul_specialist)
        self.helper_stages = bool(helper_stages)
        self.plate_cap_joined = bool(plate_cap_joined)
        self.loader_cap = bool(loader_cap)
        # A loader that fetched ahead and finds the counters by the pot
        # already holding a batch keeps the ingredient in hand and waits at
        # the cooking pot, rather than adding it to the pile on the counters.
        self.hold_ahead = bool(hold_ahead)
        # Whatever the division of labour says, a soup that has sat finished
        # this long -- in a pot, or plated on a counter for the other cook --
        # is nobody's job any more, and this cook takes it if it can reach
        # the window. The wait is long enough that it never fires while the
        # other cook is merely on its way, and short enough that a partner
        # who never delivers costs a phase, not the episode.
        self.neglect_steps = int(neglect_steps)
        # Who gives way in a standoff is not a convention the two share -- a
        # stranger has none -- but a reading of the other cook. Each seat
        # keeps an estimate of how often the other has given way before, and
        # holds its ground for longer the more the other tends to yield: up
        # to 1 + hold_max steps against a cook that always yields, one step
        # against one that never does. Past that it steps aside with a
        # probability that rises as the other's yielding falls, so two of
        # these break their symmetry by chance and a cook that never yields
        # is given way to at once. "never" and "always" are the two extremes,
        # kept for measuring what the adaptive rule does against them.
        if yield_style not in ("adaptive", "never", "always"):
            raise ValueError("yield_style is 'adaptive', 'never' or 'always'")
        self.yield_style = yield_style
        self.yield_turn = yield_turn  # "state" (carrying less / more room yields) or "seat" (measurement only)
        self.hold_max = int(hold_max)
        self.defer_to_yielder = bool(defer_to_yielder)
        # Having stepped aside, stand there this many steps: a cook that
        # steps back into the cell on the very next move is in the way again
        # before the other has taken a step, and the yield was for nothing.
        self.aside_wait = int(aside_wait)
        # A cook that has been blocked this long is not going to be let past:
        # the other one has shown it will not move. Rather than push at it for
        # the rest of the phase, the cell it stands in is treated as a wall,
        # so the plan routes round it or, where there is no way round, settles
        # on some other job this cook can still do.
        self.stubborn_steps = int(stubborn_steps)
        # A partner whose hands have not changed for this long is not doing
        # its half: standing in a corridor, giving way for ever, or simply
        # stopped. The division of labour is between two cooks who are both
        # working, so with one of them idle this cook drops it and does the
        # whole job -- loading, plating and serving -- rather than waiting
        # for a pot that nobody is filling.
        self.partner_idle_steps = int(partner_idle_steps)
        # And a kitchen where no pot has taken or given up anything for this
        # long has stopped, whatever either cook's hands have been doing. A
        # partner that shuffles about picking things up and putting them down
        # again reads as busy by its hands alone.
        self.stall_steps = int(stall_steps)
        # Getting no closer for this long counts as blocked too. Two cooks
        # facing each other in a corridor can each turn one step and step
        # back the next, so neither is frozen twice running and the count of
        # blocked steps never builds. It has to be longer than the wait a
        # cook can legitimately spend going round something, or the planner
        # takes the long way round a ring corridor for nothing.
        self.no_progress_steps = int(no_progress_steps)
        # Whether the comparative rule -- whoever loses less by walking the
        # soup out delivers -- is also asked on a kitchen that never divides.
        self.advantage_everywhere = bool(advantage_everywhere)
        # Handing a thing over a counter is not only for a shut door. Where
        # the serving window is at one end of a long floor and the piles at
        # the other, the cook that is holding a soup may be the wrong one to
        # walk it: setting it on a counter the other passes anyway saves both
        # of them the length of the kitchen. The margin covers the two
        # interacts the handover costs and keeps a soup from being passed
        # back and forth over a step of difference.
        self.relay = bool(relay)
        self.relay_margin = float(relay_margin)
        self.aside_style = aside_style  # "goal" (away from the other's goal) or "random"
        self.idle_turn_cost = float(idle_turn_cost)
        # The cell the other cook is expected to step into next is charged
        # less than the cell it stands in: the expectation is this planner's
        # guess, and a stranger, or a cook standing aside, may not move.
        self.heading_cost = float(heading_cost)
        self.stage_margin = float(stage_margin)
        self.plate_timing = bool(plate_timing)
        self.plate_lead = float(plate_lead)
        self.pile_by_round_trip = bool(pile_by_round_trip)
        if share_mode not in ("two", "continuous"):
            raise ValueError("share_mode is 'two' or 'continuous'")
        self.share_mode = share_mode
        # Planning for one seat only, with whoever else is playing treated as
        # part of the kitchen rather than as a partner that can be told
        # anything.
        self.seat = seat

        # Which half of the kitchen a cook is responsible for, phase by phase.
        # A switching layout shuts a door and reopens it, and the right play is
        # not the same on both sides of that: with the door open the two of
        # them work the whole kitchen the way the reference model does, and
        # with it shut each works its own half and passes the rest over the
        # counter. Deciding this once for the whole episode costs either the
        # pipeline or the open door -- measured, four fifths of one seat's
        # time. A kitchen that never closes has a single region throughout.
        self.phase_static = jnp.asarray(env.phase_static_objects)
        self.phase_empty = self.phase_static == StaticObject.EMPTY
        self.persistent = jnp.all(self.phase_empty, axis=0)
        self.regions = compute_enclosed_spaces(self.persistent)
        # Per phase as well: whether the two cooks are on the same side right
        # now cannot be read off the objects they share, because the counters
        # between them belong to both halves and are reachable from either --
        # so a shut door still looks joined.
        self.phase_regions = jax.vmap(compute_enclosed_spaces)(self.phase_empty)
        # A kitchen divided in every phase never lets the two cooks share a
        # room: one works the end with the ingredients, the other the end with
        # the window, and they pass things over the counters between. That is
        # where dividing the jobs by what each loses pays. Where a door opens
        # both of them can reach both ends, and splitting the jobs there leaves
        # one of them standing -- measured, split_0 220 -> 180 under the same
        # rule.
        self.always_divided = all(
            len({int(room) for room in np.asarray(phase).reshape(-1) if room >= 0}) > 1
            for phase in self.phase_regions
        )
        # The distinct halves, fixed for the episode, so a seat can be told to
        # take one that is not the other seat's.
        self.region_ids = jnp.asarray(
            sorted(int(i) for i in set(np.asarray(self.regions).reshape(-1).tolist()) if i >= 0)
        )

    def _region_objects(self, region_id):
        """The object cells a seat serving this region can stand and face."""
        inside = self.regions == region_id
        return mark_adjacent_cells(inside) & ~self.persistent

    def initial_carry(self, state) -> PlannerCarry:
        return PlannerCarry(
            positions=jnp.stack([state.agents.pos.x, state.agents.pos.y], axis=-1),
            directions=state.agents.dir,
            region=self.regions[state.agents.pos.y, state.agents.pos.x],
            moved=jnp.zeros((self.env.num_agents,), dtype=bool),
            shared=jnp.zeros((self.env.num_agents,), dtype=jnp.int32),
            given=jnp.zeros((self.env.num_agents, *state.grid.shape[:2]), dtype=bool),
            streak=jnp.zeros((self.env.num_agents,), dtype=jnp.int32),
            # Nobody yet: the first decision falls to the distance rule.
            loader=jnp.zeros((self.env.num_agents,), dtype=bool),
            neglect=jnp.zeros(state.grid.shape[:2], dtype=jnp.int32),
            yield_rate=jnp.full((self.env.num_agents,), 0.5),
            yielded=jnp.zeros((self.env.num_agents,), dtype=bool),
            waiting=jnp.zeros((self.env.num_agents,), dtype=jnp.int32),
            target=jnp.full((self.env.num_agents,), -1, dtype=jnp.int32),
            idle_for=jnp.zeros((self.env.num_agents,), dtype=jnp.int32),
            inventory=state.agents.inventory,
            pot_contents=jnp.where(
                state.grid[:, :, 0] == StaticObject.POT, state.grid[:, :, 1], 0
            ),
            stall=jnp.zeros((), dtype=jnp.int32),
            best_cost=jnp.full((self.env.num_agents,), jnp.inf),
            stuck_for=jnp.zeros((self.env.num_agents,), dtype=jnp.int32),
        )

    # -- what the kitchen looks like right now ---------------------------

    def _kitchen(self, state):
        static = state.grid[:, :, 0]
        held = state.grid[:, :, 1]
        timer = state.grid[:, :, 2]

        is_pot = static == StaticObject.POT
        cooked = is_pot & ((held & DynamicObject.COOKED) != 0)
        cooking = is_pot & (timer > 0)
        idle = is_pot & ~cooked & ~cooking

        wanted = ingredient_counts(state.recipe)
        inside = jax.vmap(ingredient_counts)(held.reshape(-1))
        inside = inside.reshape(*held.shape, MAX_INGREDIENTS)
        # A pot wants an ingredient when it is idle and holds fewer than the
        # recipe asks for. An idle pot with a wrong mixture wants nothing and
        # is simply ignored.
        short = idle[..., None] & (inside < wanted)
        return {
            "static": static,
            "held": held,
            "timer": timer,
            "recipe": state.recipe,
            "cooked": cooked,
            "cooking": cooking,
            "idle": idle,
            "short": short,
        }

    # -- which cells this seat wants to face -----------------------------

    def _goal_cells(
        self,
        kitchen,
        inventory,
        partner_inventory,
        mine,
        theirs,
        here,
        partner_here,
        first,
        countdown,
        together,
        stock_ingredients,
        stock_plates,
        share_ingredients,
        shared,
        given,
        vanishing_wall,
        displaced,
        crossing,
        cross_cells,
        was_loader,
        partner_was_loader,
        plates_vanish_ahead,
        neglected,
        partner_working,
    ):
        """The next high level task, as the set of cells that complete it.

        Both seats' distance fields are needed, because of the one task
        neither the paper's environment nor its planner has: the handover. A
        switching layout can shut the door between the two halves of a kitchen
        while somebody holds a soup the serving hatch is now walled off from,
        and the only way through is to set it on a counter both cooks can reach
        and let the other carry it. Without this the planner stands still.

        Whether to hand over is decided the same way every other choice here is
        decided -- by cost. A cook passes the item on when the other is closer
        to finishing with it, which makes the pipeline on a split kitchen fall
        out of the greedy rule rather than being written into it, and keeps the
        two of them from passing the same soup back and forth: the comparison
        is strict, so it cannot be true for both at once.
        """
        static = kitchen["static"]
        held = kitchen["held"]
        plate_pile = static == StaticObject.PLATE_PILE
        goal = static == StaticObject.GOAL
        # Whether this seat has a pot at all is a fact about the kitchen, not
        # about this second, so gating on it cannot pull a cook off the pots it
        # does own -- which gating on "no pot needs filling right now" does.
        # A seat holding more than one pot is the cook: filling them is a full
        # job on its own, and a second pair of hands in the same pots only gets
        # in the way. The other seat then leaves the ingredients alone and runs
        # With both cooks free to reach the same pots, both walk to them and
        # their routes run over each other. Where one cook can keep two pots
        # going alone, one of them should: the nearer takes the loading and the
        # other leaves those pots alone and runs plates and deliveries. Only
        # the loading step is assigned -- whatever the other ends up holding it
        # still deals with. It applies only while the halves are joined, since
        # a shut door already divides the work.
        # Being on the wrong side when the door shuts is worse than anything a
        # cook could be doing instead, and how early it has to start back
        # depends on how far away it is -- a fixed warning is too late for a
        # cook at the far end and too early for one already home. So the walk
        # back is priced: once the steps left are no more than it costs to get
        # to something in its own half, plus a little, the halves are treated
        # as separate and it works its own.
        reachable = jnp.isfinite(here)
        # Once the countdown is running nobody crosses, whatever the distance
        # says: a cook that wanders over with time to spare has no slack left
        # for anything going wrong on the way back, and the door does not wait.
        # Once the countdown is running nobody crosses, whatever the distance
        # says: a cook that wanders over with time to spare has no slack for
        # anything going wrong on the way back, and the door does not wait. On
        # a kitchen that never splits this changes nothing -- with one region,
        # what a cook may reach and what it may cross to are the same cells.
        in_time = countdown > self.env.transition_warning_steps
        # Two different questions. Whether the halves are one room decides
        # how the work is divided; whether a cook may still cross decides how
        # far it may go for it. The countdown changes the second and not the
        # first -- a kitchen that never splits has no far side to stay out of,
        # and its roles have no reason to change when a pile is about to go.
        joined = together & self.free_movement
        connected = joined & in_time
        # What this cook can actually get to: everything within reach while the
        # halves are joined, its own half once they are not.
        usable = jnp.where(joined, reachable, mine)


        holding_something_early = inventory != 0
        holding_dish = (inventory & DynamicObject.COOKED) != 0
        holding_plate = inventory == DynamicObject.PLATE
        holding_ingredient = DynamicObject.is_ingredient(inventory)

        # Carrying an ingredient: any idle pot still short of it.
        carried = ingredient_counts(inventory)
        pot_for_carried = jnp.any(kitchen["short"] & (carried > 0), axis=-1)

        # Empty handed: a plate if a soup is ready, otherwise the first
        # ingredient some pot is still short of. Both questions are asked about
        # this seat's own half of the kitchen. A cook with no pot to fill has
        # nothing better to do than fetch the plate early, and a cook with one
        # does.
        any_short = jnp.any(kitchen["short"], axis=(0, 1))
        first_short = jnp.argmax(any_short)
        pile_for_first = static == StaticObject.ingredient_pile(first_short)
        # Asked of the pots this cook can actually get to. Asked globally, the
        # cook on the half with the plates and no pots decides it ought to be
        # fetching ingredients, finds the pile walled off, and stands there.
        short_anywhere = jnp.any(kitchen["short"], axis=-1)
        needs_filling = jnp.any(
            short_anywhere & usable if self.local_needs else short_anywhere
        )
        soup_ready = jnp.any(kitchen["cooked"])
        # The reference greedy model watches the other cook for exactly one
        # thing: if they already hold a plate, fetching a second leaves nobody
        # filling pots. A plate already waiting on a counter counts the same
        # way, and not checking it is worse than wasteful here -- plates pile
        # up on the very counters a handover needs.
        #
        partner_has_plate = partner_inventory == DynamicObject.PLATE
        # A seat with no pot of its own has nothing to do until a soup is
        # ready, and measured, that is four steps in five -- while a finished
        # pot sits waiting for a plate about a quarter of the episode. So it
        # stages plates on the counters ahead of time. The cap is the number of
        # pots: one plate per pot is all that can ever be in flight, and plates
        # beyond that cover the very counters a handover needs.
        is_pot = static == StaticObject.POT
        is_plate_counter = (static == StaticObject.WALL) & (held == DynamicObject.PLATE)
        plate_waiting = jnp.any(is_plate_counter)
        staged_plates = jnp.sum(is_plate_counter)
        pots = jnp.sum(static == StaticObject.POT)

        # Who loads the pots is decided at the ingredients, not at the pots: the
        # loading loop starts and ends at the pile, so the cook standing nearer
        # to it is the one whose round trip is shorter.
        enough_pots = (jnp.sum(is_pot) >= self.pot_specialist) & partner_working
        # One pot is also a full-time job when the pile is a long way from
        # it: a batch of round trips that outlasts the cooking time leaves the
        # pot idle whenever the loader is on the road, and a second loader in
        # the same corridor mostly gets in the first one's way. Then the other
        # cook keeps to plates and serving.
        long_haul = jnp.bool_(False)
        if self.long_haul_specialist:
            round_trips = 2 * _min_manhattan(pile_for_first, is_pot) * jnp.sum(ingredient_counts(kitchen["recipe"]))
            long_haul = round_trips >= POT_COOK_TIME
            enough_pots = enough_pots | long_haul
        if self.pot_role_connected_only:
            enough_pots = enough_pots & joined
        my_pile_cost = jnp.min(jnp.where(pile_for_first, here, jnp.inf))
        their_pile_cost = jnp.min(jnp.where(pile_for_first, partner_here, jnp.inf))
        # Ties go to the first seat, so exactly one of them loads -- and the
        # other has to be clearly nearer, not just nearer, before the job
        # changes hands. Decided by exact distance, the job flips with every
        # step the cooks take and a cook sets off for the pile, turns round,
        # and sets off again.
        by_distance = jnp.where(
            first,
            my_pile_cost <= their_pile_cost + self.role_margin,
            my_pile_cost + self.role_margin < their_pile_cost,
        )
        # With a loader already chosen, it stays chosen unless the other cook
        # is nearer the pile by the hysteresis; with none chosen yet (the
        # first step), distance decides.
        undecided = ~was_loader & ~partner_was_loader
        keep = jnp.where(
            was_loader,
            my_pile_cost <= their_pile_cost + self.role_hysteresis,
            my_pile_cost + self.role_hysteresis < their_pile_cost,
        )
        nearer = jnp.where(undecided, by_distance, keep)

        # Three plates in circulation is all a split kitchen can use; more only
        # covers the counters they have to cross.
        no_plate_in_flight = ~partner_has_plate & ~plate_waiting
        # Three across is all a shut door needs; the cook on the plate side has
        # nothing else to do and will otherwise carry them until they cover
        # every counter between the halves. The cap has to sit outside the
        # disjunction, because a cook with no pot within reach satisfies the
        # first clause forever.
        # Three empty plates on the counters is the most that is ever any use;
        # past that the cook is hoarding. Keeping the cap on while the halves
        # are joined costs a little on split_0 -- the surplus happens to carry
        # the shut phase -- but stacks of idle plates are not what a cook does.
        # Plates are only ever set down while the door is shut -- with it open
        # the cook carries one to the pot itself -- and two across is all the
        # far side can use at once. Past that it is hoarding.
        # With the door open the plate is carried to the pot, so a plate
        # only ends up on a counter when no pot could take it -- and a cook
        # with nothing but plates to do will fetch one after another. Two a
        # pot in circulation, counting the one the other cook holds, is all
        # a kitchen can use; past that it is hoarding, and the counters a
        # handover needs fill up.
        plate_cap = 2 * pots
        in_circulation = staged_plates + partner_has_plate.astype(jnp.int32)
        # A soup that is done and has nobody bringing it a plate is worth a
        # plate however many are already out: the cap is there to stop a cook
        # hoarding, not to stop it plating. It counts what the other cook
        # carries, so the two do not both fetch for one soup.
        # Counted, not merely checked: two soups done and one plate on its
        # way still leaves one soup standing, so the second cook fetches too.
        soup_ready_here = jnp.any(kitchen["cooked"] & usable)
        plate_urgent = soup_ready_here & ~partner_has_plate & (inventory == 0)
        room_for_plates = plate_urgent | jnp.where(
            stock_plates, True,
            jnp.where(joined, (in_circulation < plate_cap) | ~self.plate_cap_joined, staged_plates < 2),
        )
        if self.advantage_roles and (self.always_divided or self.advantage_everywhere):
            serving = static == StaticObject.GOAL
            my_serve = jnp.min(jnp.where(serving, here, jnp.inf))
            their_serve = jnp.min(jnp.where(serving, partner_here, jnp.inf))
            my_pile = jnp.min(jnp.where(pile_for_first, here, jnp.inf))
            their_pile = jnp.min(jnp.where(pile_for_first, partner_here, jnp.inf))
            my_edge = my_serve - my_pile
            their_edge = their_serve - their_pile
            # Ties, and anything inside the margin, leave both of them free to
            # take a plate; only a clear difference divides the jobs.
            mine_is_cheaper = jnp.where(
                first,
                my_edge <= their_edge + self.advantage_margin,
                my_edge < their_edge - self.advantage_margin,
            )
            # The work does not have to exist yet. A pot that is cooking will
            # want filling the moment its soup is lifted out, and that is
            # exactly when this cook should already be carrying the first
            # ingredient rather than walking a plate to the far end.
            # The work does not have to exist yet. A pot that is cooking will
            # want filling the moment its soup is lifted out, and that is
            # exactly when the cook nearer the pile should already be walking
            # there. Tying this to the trip length instead -- defer only if the
            # pot comes free within a round trip -- reads the case backwards:
            # the cook that is near the pile has the shortest trip and is the
            # one that should be making it (measured, distance_0 500 -> 400).
            work_coming = (
                jnp.any(jnp.any(kitchen["short"], axis=-1))
                | jnp.any(kitchen["cooking"])
                | jnp.any(kitchen["cooked"])
            )
            # One cook can only carry one soup. When more are ready than the
            # deliverer can lift, the second stands cooling however the errands
            # compare, so this cook takes a plate too. Counted against what the
            # other cook is already carrying: a soup it is walking out is one it
            # has in hand, not one still in a pot.
            ready = jnp.sum(kitchen["cooked"])
            partner_carrying = (partner_inventory & DynamicObject.COOKED) != 0
            more_than_one_can_lift = ready > jnp.where(partner_carrying, 0, 1)
            defer_plate = (
                jnp.isfinite(my_serve)
                & jnp.isfinite(their_serve)
                & work_coming
                & ~mine_is_cheaper
                & ~more_than_one_can_lift
                & ~jnp.any(neglected & kitchen["cooked"] & usable)
            )
        else:
            defer_plate = jnp.bool_(False)

        # A cook doing both jobs loads pots before it fetches plates: a soup
        # that is done keeps, an empty pot earns nothing until it is filled.
        # The plate is fetched once nothing is left to load.
        fetch_plate = ~needs_filling & room_for_plates & ~defer_plate

        # Waiting at a pot with a plate is only worth it once the soup is
        # nearly out. While the timer still has more on it than the walk to
        # the plates and back, the second cook is better off loading the
        # other pot, and fetches the plate on the way past.
        plate_errand_now = jnp.min(jnp.where(plate_pile & usable, here, jnp.inf))
        soonest_cooking = jnp.min(jnp.where(kitchen["cooking"] & usable, kitchen["timer"], jnp.inf))
        soup_soon = soup_ready | (soonest_cooking <= plate_errand_now + self.plate_lead)
        helper_plate = jnp.where(
            self.plate_timing, soup_soon, soup_ready | jnp.any(kitchen["cooking"])
        ) & no_plate_in_flight
        # Ahead of a blackout the plate cook has a job of its own whether or
        # not a soup is cooking: laying in the stock. Plates already out on
        # the counters are that stock, not a plate in flight, so they must not
        # read as "somebody has the plate job covered".
        plate_stock = 2 * pots if self.plate_stock is None else self.plate_stock
        stock_early = (
            plates_vanish_ahead
            & (enough_pots & ~nearer | ~enough_pots)
            & ~soup_ready
            & (staged_plates + holding_plate.astype(jnp.int32) < plate_stock)
            & jnp.any(plate_pile & usable)
        )
        # While a pot cooks, the next batch goes onto the counters beside it.
        # The counters that count are the ones a cook loading the pot can
        # reach in a step or two: those next to the cells next to the pot.
        pot_access = mark_adjacent_cells(is_pot) & (static == StaticObject.EMPTY)
        near_pot = mark_adjacent_cells(pot_access) & (static == StaticObject.EMPTY)
        stage_counters = (static == StaticObject.WALL) & mark_adjacent_cells(near_pot) & ~vanishing_wall
        staged_here = jnp.sum(stage_counters & ingredient_on_counter_all(held))
        batch = jnp.sum(ingredient_counts(kitchen["recipe"]))
        cooking_usable = kitchen["cooking"] & usable
        soonest = jnp.min(jnp.where(cooking_usable, kitchen["timer"], jnp.inf))
        # Timed from the moment the cook reaches the pile: what is left on the
        # clock then has to cover the walk from the pile past the pot to the
        # plates and back to the pot with the plate. Being late to a finished
        # soup by up to a pile round trip costs nothing overall, because that
        # is the round trip saved when the pot is refilled from the counter.
        # Both the timer and the walk to the pile fall by one a step, so the
        # test reads the same all the way there and the cook does not turn
        # round halfway.
        my_pile = jnp.min(jnp.where(pile_for_first & usable, here, jnp.inf))
        plate_errand_steps = jnp.min(jnp.where(plate_pile & usable, here, jnp.inf)) + self.stage_margin
        pile_to_plate = _min_manhattan(pile_for_first & usable, plate_pile & usable)
        plate_to_pot = _min_manhattan(plate_pile & usable, cooking_usable)
        pile_to_pot = _min_manhattan(pile_for_first & usable, cooking_usable)
        errand = pile_to_plate + plate_to_pot + self.stage_margin - 2 * pile_to_pot
        # Only the loader stages, and only where the jobs are divided: a
        # cook that also fetches plates has no idle stretch to fill.
        # Only where the haul is long. On a kitchen where the pile is a few
        # steps from the pot, the round trip fits between soups anyway and
        # the detour to a counter costs more than it saves -- measured,
        # fifty points a run on split_0.
        stage_now = (
            self.stage_while_cooking
            & long_haul
            & enough_pots
            & nearer
            & ~needs_filling
            & jnp.any(cooking_usable)
            & ~soup_ready
            & (soonest - my_pile > errand)
            & (staged_here < batch)
            & jnp.any(pile_for_first & usable)
        )
        # The plate cook, with its plates out and no soup to plate, carries
        # the next batch to the counters beside the pot. On a kitchen where
        # the pile is a long way from the pot that is most of the work, and
        # the loader then refills from a step away instead of a round trip.
        ingredients_out = jnp.sum(ingredient_on_counter_all(held) & usable)
        helper_stage = (
            self.helper_stages
            & enough_pots
            & ~nearer
            & ~stock_early
            & ~soup_ready
            & ~(jnp.any(cooking_usable) & no_plate_in_flight & (soonest <= plate_errand_steps))
            & (staged_here < batch)
            & jnp.any(pile_for_first & usable)
            & jnp.any(stage_counters & (held == 0) & usable)
        )
        # And the loader does not fetch ahead of what the counters can take:
        # ingredients with nowhere to go pile up on the handover counters.
        loader_ahead = self.loader_cap & enough_pots & nearer & ~needs_filling & (ingredients_out >= batch)
        fetch_plate = fetch_plate & ~stage_now & ~(enough_pots & nearer & long_haul)
        # A plate is worth fetching only when a soup is on its way to want
        # it, or when the pile is about to go. Fetched with every pot still
        # being filled, it has nowhere to go: the cook carries it to a
        # counter, puts it down, finds itself empty handed with plates
        # available, and fetches the same plate again.
        # A soup nobody has come for is this cook's to plate, whatever its job.
        # Not while the other cook is walking a plate over: that soup is
        # about to be taken, however long it has waited.
        soup_neglected = jnp.any(neglected & kitchen["cooked"] & usable) & ~partner_has_plate
        fetch_plate = fetch_plate | (soup_neglected & ~holding_something_early)
        # Standing back is only worth it while there is a plate to go and get.
        # With nothing cooking there is no plate job, and a cook waiting for one
        # anyway is half the kitchen doing nothing.
        loads_pots = ~enough_pots | nearer
        if self.pot_role_helper_escape:
            # Not on a long haul: a second loader in that corridor is in the
            # first one's way more than it is at the pile.
            loads_pots = loads_pots | (~helper_plate & ~stock_early & ~long_haul)

        # The cook that is not loading does not pick ingredients up either --
        # holding one it cannot place is how it gets stuck. It fetches the plate
        # the pots are going to want and waits at the pot holding it.
        # Whoever is not loading the pots is the one who fetches plates from
        # the pile. It does not decide who may take one off a counter: a plate
        # put across for a cook beside a finished pot is for that cook, whatever
        # its usual job, and refusing it leaves both of them waiting.
        # With the plate piles gone, whatever is on the counters is all there
        # is, and the role that normally reserves plates for one cook would
        # leave the other unable to touch them.
        plates_are_mine = ~enough_pots | ~loads_pots | ~jnp.any(plate_pile) | jnp.any(neglected & usable)
        # Two ingredients across, ahead of this cook's own pots, and then its
        # own work. Keeping the other half supplied for the whole outage reads
        # as the generous thing and is not: one ingredient per round trip
        # never adds up to a soup over there, and the pots on this side sit
        # empty the whole time -- measured, forty points a phase against a
        # hundred.
        if self.share_mode == "two":
            share_now = share_ingredients & (shared < 2)
        else:
            # Keep up to two on the boundary for as long as the other half's
            # pots can take them.
            on_boundary = (static == StaticObject.WALL) & mine & theirs & (
                jax.vmap(DynamicObject.is_ingredient)(held.reshape(-1)).reshape(held.shape)
            )
            partner_room = jnp.any(kitchen["short"] & theirs[..., None])
            share_now = share_ingredients & (jnp.sum(on_boundary) < 2) & partner_room
        pot_for_carried = pot_for_carried & (loads_pots | stock_ingredients) & ~share_now
        # When the piles are about to go, laying in a stock comes before
        # anything else and before any division of roles: once they are gone
        # nothing can be cooked at all, and whatever is on the counters by then
        # is the whole of what the next phase has to work with.
        # Stocking plates is the plate cook's job where the jobs are divided:
        # the loader that goes for plates too leaves the pots short at the
        # boundary and, on a narrow floor, stands in the plate cook's way.
        stock_plates_mine = stock_plates & plates_are_mine
        # And ahead of the warning: with the plates about to go, the plate
        # cook lays in the stock whenever no soup is waiting for a plate,
        # counting what it carries as already out.
        stock_plates_mine = stock_plates_mine | stock_early
        empty_target = jnp.where(
            stock_ingredients | share_now,
            pile_for_first,
            jnp.where(
                stock_plates_mine,
                plate_pile,
                jnp.where(
                    stage_now | helper_stage,
                    pile_for_first,
                    jnp.where(
                        loads_pots & ~loader_ahead,
                        jnp.where(fetch_plate, plate_pile, pile_for_first),
                        # The plate cook is capped too, or the cap is no cap:
                        # this branch is the one that fetches plate after plate.
                        # A loader waiting on its counters wants nothing.
                        jnp.where(
                            loader_ahead,
                            jnp.zeros_like(plate_pile),
                            jnp.where(room_for_plates, plate_pile, pile_for_first),
                        ),
                    ),
                ),
            ),
        )

        # Carrying a plate: the pot that is done, else the one that is cooking,
        # so the wait happens at the pot. Waiting at an idle pot as well reads
        # like an improvement and is not -- the carrier parks in front of it and
        # the cook cannot get an ingredient in.
        plate_target = jnp.where(soup_ready, kitchen["cooked"], kitchen["cooking"])
        # A hand carrying the wrong thing in the run-up to an outage is a hand
        # not carrying the right one, so it puts it down on a counter -- and a
        # plate being stocked goes down rather than waiting at a pot.
        # While stocking, a plate in hand goes onto a counter rather than to a
        # pot: the point is to have them out when the piles are gone, and a
        # plate carried to a pot is one that never got put across. Leaving it to
        # chance -- the plate only ends up on a counter when no pot happens to
        # want it -- stocks four plates on one kitchen and none on another.
        plate_target = jnp.where(
            stock_ingredients | (stock_plates_mine & ~soup_ready),
            jnp.zeros_like(plate_target),
            plate_target,
        )

        # An ingredient with no pot to take it goes onto a counter beside a
        # pot, where the next load will find it.
        empty_stage = stage_counters & (held == 0) & usable
        stager = (self.stage_while_cooking & enough_pots & nearer) | (self.helper_stages & enough_pots & ~nearer)
        ingredient_target = jnp.where(
            jnp.any(pot_for_carried) | share_now | ~jnp.any(empty_stage) | ~stager,
            pot_for_carried,
            empty_stage,
        )
        # An ingredient fetched while the pot is still cooking is carried to
        # the pot and held there, not set down: the pot takes it the moment
        # the soup comes out, and a counter would only add two more interacts
        # and a detour to the same trip.
        hold_it = (
            self.hold_ahead & enough_pots & nearer & ~share_now & ~stock_ingredients
            & ~jnp.any(pot_for_carried) & jnp.any(cooking_usable)
            # Not on a long haul: there the trip back to the pile is what the
            # cooking time is for, and standing at the pot with one
            # ingredient wastes it.
            & (~long_haul | (ingredients_out >= batch))
        )
        ingredient_target = jnp.where(hold_it, cooking_usable, ingredient_target)
        wanted = jnp.where(
            holding_dish,
            goal,
            jnp.where(
                holding_plate,
                plate_target,
                jnp.where(holding_ingredient, ingredient_target, empty_target),
            ),
        )

        # An empty counter on the boundary between the two halves. If every
        # one of those is taken -- an outage stock and a soup can fill them
        # between them -- any empty counter this cook can reach will do: it
        # cannot hand the thing over, but it can at least put it down and get
        # on with something. Standing still holding it is how both cooks spend
        # a hundred and thirty steps doing nothing.
        # The counters both halves can reach are the only way anything crosses
        # a shut door, and there are about three of them. Plates and soups have
        # to cross; surplus ingredients do not, and putting them there is how
        # the crossing silts up -- measured, both cooks then stand still for a
        # hundred and thirty steps with a plate on one side and cooked pots on
        # the other. So ingredients go inward and the boundary is kept for the
        # things that have to pass.
        is_counter = static == StaticObject.WALL
        # A wall that is about to become floor is no place to set anything
        # down: whatever is on it when the door reopens is gone with the wall.
        empty_counter = is_counter & (held == 0) & ~vanishing_wall
        boundary = empty_counter & mine & theirs
        inner = empty_counter & usable & ~boundary
        must_cross = (
            (inventory == DynamicObject.PLATE)
            | ((inventory & DynamicObject.COOKED) != 0)
            | (share_now & holding_ingredient)
        )
        handover = jnp.where(
            must_cross,
            jnp.where(jnp.any(boundary), boundary, inner),
            jnp.where(jnp.any(inner), inner, boundary),
        )

        # Picking something up off a counter is only worth doing if this seat
        # can finish what the item is for. That condition is also what stops a
        # cook putting a soup down and immediately taking it back: it puts it
        # down precisely because the hatch is out of reach.
        # Whether taking something off a counter is worth doing is asked of
        # what this cook can actually get to now, not of the half it nominally
        # serves. Asked of the half, a cook picks up a plate it has no pot for,
        # finds nothing to do with it, puts it back on the same counter, and
        # takes it again -- an interact repeated against a wall for the rest of
        # the episode.
        # What this cook set on the boundary for the other one is given, not
        # stored, and it is not taken back. Stock it put on an inner counter
        # for itself is the other case, and that it takes. Without the
        # distinction a cook hands an ingredient across, sees an ingredient it
        # could use, takes it, and hands it across again for the whole outage.
        # What this cook set down for the other is not taken back -- unless
        # a soup here is waiting for exactly that plate, or the thing has
        # been sitting long enough that nobody is coming for it.
        counter_item = is_counter & (held != 0) & usable & (
            ~given | neglected | (plate_urgent & (held == DynamicObject.PLATE))
        )
        can_deliver = jnp.any(goal & usable)
        can_reach_soup = jnp.any((kitchen["cooked"] | kitchen["cooking"]) & usable)
        can_fill_pot = jnp.any(jnp.any(kitchen["short"], axis=-1) & usable)
        # And only for work this cook is actually going to do. Picking a plate
        # up when plating the soup is somebody else's job leaves it holding
        # something with nowhere to take it, and the same goes for ingredients
        # when the pots are not its to load.
        ingredient_on_counter = jax.vmap(DynamicObject.is_ingredient)(
            held.reshape(-1)
        ).reshape(held.shape)
        useful = counter_item & (
            (((held & DynamicObject.COOKED) != 0) & can_deliver)
            | (
                (held == DynamicObject.PLATE)
                & can_reach_soup
                & (plates_are_mine | ~joined)
                # A plate off the counter is for a soup that is done, or for
                # a cook with nothing left to load. Taken while a pot stands
                # empty it buys a wait at the cooking pot and leaves the
                # other pot cold for the whole of it.
                # ... unless there is no plate pile left anywhere -- an
                # outage has taken it -- and the counters are all the plates
                # the kitchen still has.
                & (soup_ready_here | ~needs_filling | ~jnp.any(plate_pile))
            )
            | (ingredient_on_counter & can_fill_pot & loads_pots)
        )

        holding_something = inventory != 0

        # Free movement is the normal case: while the halves are joined the
        # kitchen is one room and both cooks go wherever the work is. Only a
        # shut door makes each of them work the half it can finish in and put
        # the rest on a counter for the other, which is the pipeline.
        scoped = wanted & jnp.where(connected, reachable, mine)
        # The same errand, as the other cook could do it: the cells are the
        # same, what differs is whose half they fall in.
        wanted_by_partner = wanted & jnp.where(connected, jnp.isfinite(partner_here), theirs)

        # Assigned the other half and still standing in this one, a cook walks
        # over by aiming at what is over there -- that is the one crossing
        # the countdown must not stop, since two cooks left on one side of a
        # shut door is the worst place a split can leave them. Not at a
        # counter: interacting with one while holding something puts it down.
        must_move = displaced & ~jnp.any(scoped & reachable)
        wanted = jnp.where(must_move, mine & ~is_counter, scoped)

        pass_it_on = holding_something & jnp.any(handover) & ~jnp.any(scoped & reachable)

        # And when the other cook would finish this errand sooner: the cost
        # of walking it there myself against the cost of setting it down
        # where the other will pass and its own walk from there.
        if self.relay:
            my_finish = jnp.min(jnp.where(scoped, here, jnp.inf))
            their_finish = jnp.min(jnp.where(wanted_by_partner, partner_here, jnp.inf))
            drop_cost = jnp.min(jnp.where(handover, here, jnp.inf))
            worth_relaying = (
                holding_something
                # Ingredients only, and only where the haul is long enough
                # that half of it is worth a pair of interacts. A soup or a
                # plate passed this way only adds a stop to a short trip.
                & holding_ingredient
                & long_haul
                & jnp.any(handover)
                & jnp.isfinite(my_finish)
                & jnp.isfinite(their_finish)
                & (drop_cost + their_finish + self.relay_margin < my_finish)
                & ~share_now
            )
            pass_it_on = pass_it_on | worth_relaying

        wanted = jnp.where(
            pass_it_on,
            handover,
            jnp.where(~holding_something & jnp.any(useful), useful, wanted),
        )

        # The door is about to shut on both of them in one room and this is
        # the cook that reaches the other one first. Nothing it could be doing
        # here is worth a phase with that room empty.
        wanted = jnp.where(crossing & jnp.any(cross_cells), cross_cells, wanted)

        # Crossing is allowed but not free. A cook that wanders into the other
        # half for a job barely closer is out of position when the door shuts,
        # and the pipeline that shut door needs has to be built again from
        # nothing. Charging a few steps for leaving home keeps both of them
        # where they will be needed without forbidding the trip.
        # Where interacting right now actually does something. A cook can
        # stand at a cooking pot holding a plate -- that is the right place to
        # wait -- but pressing interact there every step does nothing and
        # reads as a cook fiddling with a pot it has no business with.
        # Nothing to do is a place to stand, not a reason to stand still: the
        # cook walks to where its next job will start -- the pot that is
        # cooking, else the piles it will draw from -- so the walk is already
        # done when the job appears. It only positions there; picking
        # anything up is still decided by the rules above.
        nothing_to_do = ~jnp.any(wanted)
        # Where its own next job starts: the pot for whoever plates, the
        # pile for whoever loads. Waiting at the other cook's station is how
        # an idle cook ends up in the way of the one still working.
        waiting_room = jnp.where(
            loads_pots,
            pile_for_first & usable,
            (kitchen["cooking"] | kitchen["cooked"]) & usable,
        )
        wanted = jnp.where(nothing_to_do & jnp.any(waiting_room), waiting_room, wanted)
        idle_pot = kitchen["cooking"] & ((inventory == DynamicObject.PLATE) | holding_ingredient)
        actionable = wanted & ~idle_pot & ~nothing_to_do
        # Which pile to draw from is the round trip, not the walk out: with
        # two piles and a pot beside one of them, the near pile is the one
        # that saves a walk on the way back as well.
        if self.pile_by_round_trip:
            to_pot = _manhattan_to(jnp.any(kitchen["short"], axis=-1))
            pile_penalty = jnp.where(
                pile_for_first & jnp.isfinite(to_pot), to_pot, 0.0
            )
        else:
            pile_penalty = 0.0
        penalty = jnp.where(mine, 0.0, self.home_bias) + pile_penalty
        return wanted, penalty, actionable, nearer

    # -- one step towards it ---------------------------------------------

    def _cost_grid(self, move_area, pos, direction):
        """For every cell that cannot be walked on, steps to stand and face it."""
        return OvercookedPathPlanner._compute_min_moves(pos, direction, move_area)

    def _act(self, move_area, goals, penalty, actionable, here, pos, direction, key, occupied, heading=None, give_up=False):
        goal_key, move_key = jax.random.split(key)
        # Where the other cook will not move, plan as though it were a wall:
        # its cell is impassable and so is everything only reachable through
        # it, which leaves this cook the goals it can still get to on its own.
        move_area = jnp.where(give_up, move_area & ~occupied, move_area)
        here = jnp.where(give_up, self._cost_grid(move_area, pos, direction), here)
        costs = jnp.where(goals, here + penalty, jnp.inf)

        if self.hltemp > 0.0:
            flat = _boltzmann(costs.reshape(-1), self.hltemp, goal_key)
        else:
            flat = jnp.argmin(costs.reshape(-1))
        target = jnp.unravel_index(flat, costs.shape)
        reachable = jnp.isfinite(costs.reshape(-1)[flat])
        at_goal = here[target] == 0

        def step_cost(move_direction):
            moved = pos.move_in_bounds(move_direction, self.env.width, self.env.height)
            blocked = ~move_area[moved.y, moved.x]
            landed = Position(
                x=jnp.where(blocked, pos.x, moved.x),
                y=jnp.where(blocked, pos.y, moved.y),
            )
            cost = self._cost_grid(move_area, landed, move_direction)[target]
            # Cells the other cook is standing in, or is about to step into,
            # are avoided before the two of them meet rather than after. The
            # cost is heavy but finite, so the one route through a corridor
            # stays usable when there is no way round.
            # A move into a wall only turns the cook on the spot. With the
            # way through occupied, that turn is the cheapest move on offer,
            # and a cook can spend a standoff turning towards a wall -- never
            # pushing, so never registering as blocked, never giving way.
            # A turn that brings the goal no nearer is priced above pushing
            # through; a turn that does (facing the counter it stands at) is
            # left alone.
            idle_turn = blocked & (cost >= here[target])
            expected = jnp.zeros_like(occupied) if heading is None else heading
            return (
                cost
                + jnp.where(occupied[landed.y, landed.x], 20.0, 0.0)
                + jnp.where(expected[landed.y, landed.x] & ~occupied[landed.y, landed.x], self.heading_cost, 0.0)
                + jnp.where(idle_turn, self.idle_turn_cost, 0.0)
            )

        step_costs = jax.vmap(step_cost)(DIRECTIONS)
        if self.lltemp > 0.0:
            move = MOVES[_boltzmann(step_costs, self.lltemp, move_key)]
        else:
            move = MOVES[jnp.argmin(step_costs)]

        does_something = actionable.reshape(-1)[flat]
        action = jnp.where(
            ~reachable,
            OvercookedActionsEnum.stay,
            jnp.where(
                at_goal,
                jnp.where(does_something, OvercookedActionsEnum.interact, OvercookedActionsEnum.stay),
                move,
            ),
        ).astype(jnp.int32)
        return action, jnp.where(reachable, flat, -1)

    # -- the same choice, as probabilities ---------------------------------

    def seat_features(self, carry: PlannerCarry, state):
        """What each seat's action distribution is made from, dial-free.

        Fitting the three dials to people means scoring every recorded action
        under many settings; everything expensive -- the distance fields, the
        halves, the goal sets -- does not depend on the dials, so it is computed
        once here and ``probs_from_features`` is the cheap part that is
        repeated. Goal claiming and collision avoidance between the two seats
        use the first seat's most likely goal and move rather than a sample, so
        the features are a deterministic function of the state.
        """
        inputs = self._seat_inputs(carry, state)
        move_area, costs = inputs["move_area"], inputs["costs"]
        height, width = state.grid.shape[:2]
        claimed = jnp.zeros((height, width), dtype=bool)
        heading = jnp.zeros((height, width), dtype=bool)
        features = []
        for index in range(self.env.num_agents):
            goals, penalty, actionable = inputs["seats"][index]
            free = goals & ~claimed
            goals = jnp.where(jnp.any(free), free, goals)
            pos = Position(x=state.agents.pos.x[index], y=state.agents.pos.y[index])
            others = (
                jnp.zeros((height, width), dtype=bool)
                .at[state.agents.pos.y, state.agents.pos.x]
                .set(True)
                .at[pos.y, pos.x]
                .set(False)
            ) | heading
            here = costs[index]
            goal_costs = jnp.where(goals, here + penalty, jnp.inf).reshape(-1)

            def step_grid(move_direction):
                moved = pos.move_in_bounds(move_direction, self.env.width, self.env.height)
                blocked = ~move_area[moved.y, moved.x]
                landed = Position(
                    x=jnp.where(blocked, pos.x, moved.x),
                    y=jnp.where(blocked, pos.y, moved.y),
                )
                grid = self._cost_grid(move_area, landed, move_direction)
                idle_turn = blocked & (grid >= here)
                penalty = jnp.where(others[landed.y, landed.x], 20.0, 0.0) + jnp.where(idle_turn, self.idle_turn_cost, 0.0)
                return (grid + penalty).reshape(-1), landed

            step_costs, landed = jax.vmap(step_grid)(DIRECTIONS)  # (4, H*W)
            features.append(
                dict(
                    goal_costs=goal_costs,
                    step_costs=step_costs,
                    at_goal=(here == 0).reshape(-1),
                    does_something=actionable.reshape(-1),
                )
            )
            # The next seat plans around this one's most likely choice.
            best = jnp.argmin(goal_costs)
            reachable = jnp.isfinite(goal_costs[best])
            claimed = claimed | jnp.zeros((height, width), dtype=bool).reshape(-1).at[best].set(reachable).reshape(height, width)
            move = jnp.argmin(step_costs[:, best])
            heading = heading.at[landed.y[move], landed.x[move]].set(reachable & ~(here == 0).reshape(-1)[best])
        return features

    @staticmethod
    def probs_from_features(features, hltemp, lltemp, prob_wait):
        """The (6,) action distribution of one seat under the three dials.

        The goal is drawn by Boltzmann over its cost (argmin at hltemp 0), the
        move by Boltzmann over how much each step lengthens the way to that
        goal (argmin at lltemp 0); at the goal the seat interacts when that does
        anything and otherwise stays. ``prob_wait`` mixes a stay in on top.
        """
        goal_costs = features["goal_costs"]
        finite = jnp.isfinite(goal_costs)
        safe = jnp.where(finite, goal_costs, 0.0)
        if hltemp > 0.0:
            logits = jnp.where(finite, -safe / hltemp, -1e9)
            p_goal = jax.nn.softmax(logits)
        else:
            p_goal = jax.nn.one_hot(jnp.argmin(jnp.where(finite, safe, 1e9)), goal_costs.shape[0])
        p_goal = jnp.where(finite, p_goal, 0.0)
        p_goal = jnp.where(jnp.any(finite), p_goal, 0.0)

        step_costs = features["step_costs"]  # (4, G)
        # A goal no step reaches has weight zero below; its column is only
        # kept finite so that zero does not become nan.
        step_finite = jnp.isfinite(step_costs)
        step_safe = jnp.where(step_finite, step_costs, 0.0)
        if lltemp > 0.0:
            logits = jnp.where(step_finite, -step_safe / lltemp, -1e9)
            p_move = jax.nn.softmax(logits, axis=0)
        else:
            p_move = jax.nn.one_hot(jnp.argmin(jnp.where(step_finite, step_safe, 1e9), axis=0), 4).T
        # (4, G) -> (6, G): move actions in Direction order.
        p_move_actions = jnp.zeros((6, step_costs.shape[1])).at[MOVES].set(p_move)

        at_goal = features["at_goal"]
        does = features["does_something"]
        p_at = (
            jnp.zeros((6, step_costs.shape[1]))
            .at[OvercookedActionsEnum.interact]
            .set(jnp.where(does, 1.0, 0.0))
            .at[OvercookedActionsEnum.stay]
            .set(jnp.where(does, 0.0, 1.0))
        )
        p_given_goal = jnp.where(at_goal[None, :], p_at, p_move_actions)
        probs = p_given_goal @ p_goal  # (6,)
        # No reachable goal at all: the seat stays.
        stay = jax.nn.one_hot(OvercookedActionsEnum.stay, 6)
        probs = jnp.where(jnp.any(finite), probs, stay)
        return prob_wait * stay + (1.0 - prob_wait) * probs

    def action_probs(self, carry: PlannerCarry, state):
        """(num_agents, 6) action distribution under this planner's dials.

        The unstuck rule is left out: it reads the previous step and rolls
        dice, so it has no place in a per-state likelihood.
        """
        features = self.seat_features(carry, state)
        return jnp.stack(
            [
                self.probs_from_features(f, self.hltemp, self.lltemp, self.prob_wait)
                for f in features
            ]
        )

    # -- the unstuck rule -------------------------------------------------

    def _step_aside(self, move_area, state, index):
        """Move away from the other cook, for a seat with nothing to do.

        A cook that is out of work and stands where it stopped is in the way:
        the pots are full, or a soup has to be carried past it, and the one
        still working has to go round. Backing off costs the idle seat nothing.
        """
        pos = Position(x=state.agents.pos.x[index], y=state.agents.pos.y[index])
        occupied = (
            jnp.zeros_like(move_area)
            .at[state.agents.pos.y, state.agents.pos.x]
            .set(True)
            .at[pos.y, pos.x]
            .set(False)
        )
        others = jnp.stack([state.agents.pos.x, state.agents.pos.y], axis=-1)

        def gap(move_direction):
            moved = pos.move_in_bounds(
                move_direction, self.env.width, self.env.height
            )
            free = move_area[moved.y, moved.x] & ~occupied[moved.y, moved.x]
            distance = jnp.min(
                jnp.abs(others[:, 0] - moved.x)
                + jnp.abs(others[:, 1] - moved.y)
                + jnp.where(jnp.arange(self.env.num_agents) == index, 1e6, 0.0)
            )
            return jnp.where(free, distance, -jnp.inf)

        room = jax.vmap(gap)(DIRECTIONS)
        here = jnp.min(
            jnp.abs(others[:, 0] - pos.x)
            + jnp.abs(others[:, 1] - pos.y)
            + jnp.where(jnp.arange(self.env.num_agents) == index, 1e6, 0.0)
        )
        best = jnp.argmax(room)
        return jnp.where(
            room[best] > here, MOVES[best], OvercookedActionsEnum.stay
        ).astype(jnp.int32)

    def _unstuck(self, move_area, state, key, planned, streak, yield_rate, targets):
        """Break a standoff: hold, then give way, as the other cook warrants.

        A cook blocked by the other holds its plan for a number of steps set
        by how often that cook has given way before, then steps aside with a
        probability that is higher the less the other yields. Stepping aside
        means the free neighbouring cell furthest from the other cook, so the
        way is actually cleared rather than shuffled.
        """
        occupied = (
            jnp.zeros_like(move_area)
            .at[state.agents.pos.y, state.agents.pos.x]
            .set(True)
        )
        others = jnp.stack([state.agents.pos.x, state.agents.pos.y], axis=-1)

        height, width = move_area.shape
        partner_of = self.env.num_agents - 1 - jnp.arange(self.env.num_agents)

        def aside(index, own_key):
            pos = Position(x=state.agents.pos.x[index], y=state.agents.pos.y[index])
            # Where the other cook is going, when this planner knows (both
            # seats planned here); otherwise the cell it is facing.
            other = partner_of[index]
            target = targets[other]
            facing = Position(
                x=state.agents.pos.x[other], y=state.agents.pos.y[other]
            ).move_in_bounds(state.agents.dir[other], width, height)
            goal_y, goal_x = jnp.where(
                target >= 0, jnp.stack(jnp.unravel_index(jnp.maximum(target, 0), (height, width))),
                jnp.stack([facing.y, facing.x]),
            )

            def room(move_direction):
                moved = pos.move_in_bounds(move_direction, width, height)
                free = move_area[moved.y, moved.x] & ~occupied[moved.y, moved.x]
                free = free & ((moved.x != pos.x) | (moved.y != pos.y))
                # Out of the other cook's way means away from where it is
                # going first, and away from where it stands second.
                from_goal = jnp.abs(moved.x - goal_x) + jnp.abs(moved.y - goal_y)
                from_them = jnp.min(
                    jnp.abs(others[:, 0] - moved.x) + jnp.abs(others[:, 1] - moved.y)
                    + jnp.where(jnp.arange(self.env.num_agents) == index, 1e6, 0.0)
                )
                score = 10.0 * from_goal + from_them if self.aside_style == "goal" else 0.0
                return jnp.where(free, score, -jnp.inf)

            gaps = jax.vmap(room)(DIRECTIONS)
            best = jnp.max(gaps)
            logits = jnp.where(gaps >= best - 1e-6, 0.0, -jnp.inf)
            return jnp.where(
                jnp.isfinite(best),
                MOVES[jax.random.categorical(own_key, logits)],
                OvercookedActionsEnum.stay,
            ).astype(jnp.int32)

        keys = jax.random.split(key, self.env.num_agents)
        stepping_aside = jnp.stack([aside(i, keys[i]) for i in range(self.env.num_agents)])

        # Whose turn it is to give way is read off the state both can see,
        # not off a seat number: the cook carrying less yields to the one
        # carrying more (a soup outranks a plate outranks an ingredient), and
        # between equals the one with more room to step into. An exact tie is
        # settled by one shared coin, so two of these planners never both
        # step aside at once; a stranger reading it differently is caught by
        # the estimate below.
        inventory = state.agents.inventory
        rank = jnp.where(
            (inventory & DynamicObject.COOKED) != 0, 3,
            jnp.where(inventory == DynamicObject.PLATE, 2, jnp.where(inventory != 0, 1, 0)),
        )

        def free_room(index):
            pos = Position(x=state.agents.pos.x[index], y=state.agents.pos.y[index])

            def free(move_direction):
                moved = pos.move_in_bounds(move_direction, self.env.width, self.env.height)
                return move_area[moved.y, moved.x] & ~occupied[moved.y, moved.x] & (
                    (moved.x != pos.x) | (moved.y != pos.y)
                )

            return jnp.sum(jax.vmap(free)(DIRECTIONS))

        room_of = jnp.stack([free_room(i) for i in range(self.env.num_agents)])
        partner = self.env.num_agents - 1 - jnp.arange(self.env.num_agents)
        shared_coin = jax.random.bernoulli(jax.random.fold_in(key, 11))
        my_turn = (
            (rank < rank[partner])
            | ((rank == rank[partner]) & (room_of > room_of[partner]))
            | ((rank == rank[partner]) & (room_of == room_of[partner])
               & (jnp.arange(self.env.num_agents) == jnp.where(shared_coin, 1, 0)))
        )
        if self.yield_turn == "seat":
            my_turn = jnp.arange(self.env.num_agents) == 1
        elif self.yield_turn == "coin":
            my_turn = jnp.arange(self.env.num_agents) == jnp.where(shared_coin, 1, 0)
        coin = jax.random.uniform(jax.random.fold_in(key, 7), (self.env.num_agents,))
        if self.yield_style == "never":
            gives_way = jnp.zeros((self.env.num_agents,), dtype=bool)
        elif self.yield_style == "always":
            gives_way = jnp.ones((self.env.num_agents,), dtype=bool)
        else:
            # Not my turn: hold for as long as the other cook's record says it
            # is likely to give way, then step aside on a coin -- against a
            # cook that never yields the record soon says so and the hold is
            # a single step.
            hold = 1 + jnp.round(self.hold_max * yield_rate)
            # Whose turn it is by convention still reads the other cook: a
            # partner that has always given way is left to give way again,
            # else the two of them step aside together and mirror each other.
            partner_reliable = (yield_rate > 0.75) & self.defer_to_yielder
            gives_way = (my_turn & ~partner_reliable) | ((streak > hold) & (coin < 0.5))
        return jnp.where(gives_way, stepping_aside, planned), gives_way

    # -- the policy -------------------------------------------------------

    def _seat_inputs(self, carry: PlannerCarry, state):
        """Everything a seat's decision is made from, before any dial is read.

        The distance fields, the halves, the outage flags and each seat's goal
        set depend on the kitchen alone; the temperatures and the wait
        probability only decide how those are turned into an action. Splitting
        it here lets the same inputs be sampled from (``actions``) or scored
        (``action_probs``), which is what fitting the dials to people needs.
        """
        kitchen = self._kitchen(state)
        move_area = self.env._get_move_area(state)

        # One distance field per seat, reused for choosing the task and the
        # step, and read by the other seat to see what a handover can reach.
        costs = jax.vmap(
            lambda x, y, d: self._cost_grid(move_area, Position(x=x, y=y), d)
        )(state.agents.pos.x, state.agents.pos.y, state.agents.dir)
        # Which half a cook serves follows where it is standing, not where it
        # started, so the two can swap sides. The doorway belongs to neither,
        # so standing in it keeps the half from the step before.
        #
        # The two of them never take the same half. If they are both in one,
        # the second seat is sent to whichever other half it can reach most
        # cheaply -- a shut door with both cooks on the same side of it leaves
        # the other side with nobody, and nothing there gets done at all.
        standing = self.regions[state.agents.pos.y, state.agents.pos.x]
        region = jnp.where(standing < 0, carry.region, standing)
        if self.env.num_agents == 2 and self.region_ids.size > 1:
            clash = region[1] == region[0]
            # Whichever of the two is nearer to another half is the one that
            # goes: the seat number has nothing to do with who should walk.
            def nearest_other(seat):
                away = jnp.where(self.region_ids == region[seat], jnp.inf, 0.0) + jax.vmap(
                    lambda rid: jnp.min(
                        jnp.where(self._region_objects(rid), costs[seat], jnp.inf)
                    )
                )(self.region_ids)
                return self.region_ids[jnp.argmin(away)], jnp.min(away)

            (other0, cost0), (other1, cost1) = nearest_other(0), nearest_other(1)
            # And once chosen, the mover stays chosen until the two are apart.
            # Re-picked by distance every step, the job passes to the other
            # cook as soon as the first one starts walking, and neither ever
            # arrives.
            was_moving = carry.region != standing
            mover = jnp.where(
                jnp.any(was_moving),
                jnp.argmax(was_moving),
                jnp.where(cost1 < cost0, 1, 0),
            )
            other = jnp.where(mover == 1, other1, other0)
            # Only if that other half can actually be got to. Sent to a half
            # behind a shut door, the mover has nothing it can reach and spends
            # the phase standing still.
            can_get_there = jnp.isfinite(jnp.minimum(cost0, cost1))
            region = jnp.where(
                clash & can_get_there,
                region.at[mover].set(other),
                region,
            )
        owned = jax.vmap(self._region_objects)(region)

        # Seats are served in order, each striking out what the ones before it
        # claimed. Two cooks running the same rule on the same kitchen walk to
        # the same onion otherwise, and one of those trips is wasted.
        # An outage takes the ingredient piles away for a whole phase, and
        # nothing can be cooked while they are gone -- measured, the phase after
        # one scores a quarter of what the others do, on the two pots that
        # happened to be full. So from the moment the countdown shows it
        # coming, both cooks do nothing but carry ingredients: into the pots
        # while they have room, onto the counters after that, as many as the
        # countdown allows. There is no cap; the counters run out first.
        following = jnp.mod(state.layout_index + 1, self.phase_static.shape[0])
        now, soon = state.grid[:, :, 0], self.phase_static[following]
        warned = state.steps_until_layout_change <= self.env.transition_warning_steps

        def vanishing(here_mask, there_mask):
            return jnp.any(here_mask) & ~jnp.any(there_mask) & warned

        stock_ingredients = vanishing(
            now >= StaticObject.INGREDIENT_PILE_BASE,
            soon >= StaticObject.INGREDIENT_PILE_BASE,
        )
        stock_plates = vanishing(
            now == StaticObject.PLATE_PILE, soon == StaticObject.PLATE_PILE
        )
        # The whole of the phase ahead has no plate pile, warning or not: the
        # stock has to be laid in over the phase, not over the last twenty
        # steps of it.
        plates_vanish_ahead = jnp.any(now == StaticObject.PLATE_PILE) & ~jnp.any(
            soon == StaticObject.PLATE_PILE
        )
        # An outage that takes every pile is one thing; one that takes the pile
        # from one half and leaves the other is a different job. There the cook
        # that keeps a pile fills its own pots first and carries the rest to
        # the counters for the one that does not.
        pile_now = now >= StaticObject.INGREDIENT_PILE_BASE
        vanishing_wall = warned & (now == StaticObject.WALL) & (soon != StaticObject.WALL)

        here_regions = self.phase_regions[state.layout_index]
        sides = here_regions[state.agents.pos.y, state.agents.pos.x]
        same_side = sides[0] == sides[1]

        # The rooms of the phase this one is about to become. Only a change
        # that leaves the two of them in one room matters; the countdown also
        # runs before the door reopens, and crossing then is what puts both of
        # them on one side for a whole phase.
        next_index = (state.layout_index + 1) % self.phase_regions.shape[0]
        next_rooms = self.phase_regions[next_index]
        seat_rooms = next_rooms[state.agents.pos.y, state.agents.pos.x]
        labelled = jnp.where(next_rooms >= 0, next_rooms, jnp.max(next_rooms))
        several_rooms = jnp.max(next_rooms) > jnp.min(labelled)
        # A cook standing in the doorway belongs to no room of the phase
        # ahead: the cell it is on is about to be a wall, and it will be put
        # into one of the two rooms without being asked which. Read as "not
        # sharing", that cook keeps working where it is and the door shuts
        # with both of them on the same side -- measured, a whole phase with
        # nobody at the serving end. So an unplaced cook counts as sharing,
        # and the crossing is decided from whichever of the two does have a
        # room.
        unplaced = seat_rooms < 0
        reference_room = jnp.where(seat_rooms[0] >= 0, seat_rooms[0], seat_rooms[1])
        sharing = (
            (seat_rooms[0] == seat_rooms[1]) | jnp.any(unplaced)
        ) & (reference_room >= 0)
        splitting_ahead = (
            several_rooms & sharing & (state.steps_until_layout_change > 0)
        )
        if self.cross_on_countdown and self.env.num_agents == 2:
            elsewhere = (next_rooms >= 0) & (next_rooms != reference_room)
            # What only the far side can reach. A counter on the boundary is
            # next to both rooms, and a cook sent to one of those stands on its
            # own side facing it and calls the crossing done. And a goal has to
            # be a cell that cannot be walked on, since that is what the
            # distance field measures -- aiming at the floor over there leaves
            # every distance infinite and the rule never fires.
            cross_cells = (
                mark_adjacent_cells(elsewhere)
                & ~mark_adjacent_cells(next_rooms == reference_room)
                & (state.grid[:, :, 0] != StaticObject.EMPTY)
            )
            reach = jnp.stack(
                [jnp.min(jnp.where(cross_cells, costs[seat], jnp.inf)) for seat in range(2)]
            )
            first_there = jnp.argmin(reach)
            leaving_now = state.steps_until_layout_change <= (
                reach[first_there] + self.return_margin
            )
            crossing_now = (
                splitting_ahead
                & leaving_now
                & jnp.isfinite(reach[first_there])
                & (jnp.arange(self.env.num_agents) == first_there)
            )
        else:
            crossing_now = jnp.zeros((self.env.num_agents,), dtype=bool)
            cross_cells = jnp.zeros(state.grid.shape[:2], dtype=bool)

        seats = []
        roles = []
        for index in range(self.env.num_agents):
            partner = self.env.num_agents - 1 - index
            goals, penalty, actionable, role = self._goal_cells(
                kitchen,
                state.agents.inventory[index],
                state.agents.inventory[partner],
                owned[index],
                owned[partner],
                costs[index],
                costs[partner],
                index == 0,
                state.steps_until_layout_change,
                same_side,
                stock_ingredients,
                stock_plates,
                # The other half has pots and no ingredients while this half
                # still has a pile: whatever this cook fetches beyond what its
                # own pots take goes across. No need to start ahead of time --
                # the carrying runs for as long as the outage does.
                ~jnp.any(pile_now & owned[partner])
                & jnp.any(pile_now & owned[index])
                & jnp.any((now == StaticObject.POT) & owned[partner]),
                carry.shared[index],
                carry.given[index],
                vanishing_wall,
                standing[index] != region[index],
                crossing_now[index],
                cross_cells,
                carry.loader[index],
                carry.loader[partner],
                plates_vanish_ahead,
                carry.neglect >= self.neglect_steps,
                (carry.idle_for[partner] < self.partner_idle_steps)
                & (carry.stall < self.stall_steps),
            )
            seats.append((goals, penalty, actionable))
            roles.append(role)
        return dict(
            move_area=move_area, costs=costs, region=region, owned=owned,
            pile_now=pile_now, seats=seats, loader=jnp.stack(roles),
        )

    def actions(self, carry: PlannerCarry, state, key):
        """One action per agent, and the carry the unstuck rule needs."""
        plan_key, stuck_key, wait_key = jax.random.split(key, 3)
        keys = jax.random.split(plan_key, self.env.num_agents)
        inputs = self._seat_inputs(carry, state)
        move_area, costs = inputs["move_area"], inputs["costs"]
        region, owned, pile_now = inputs["region"], inputs["owned"], inputs["pile_now"]

        # Seats are served in order, each striking out what the ones before it
        # claimed. Two cooks running the same rule on the same kitchen walk to
        # the same onion otherwise, and one of those trips is wasted.
        claimed = jnp.zeros(state.grid.shape[0] * state.grid.shape[1], dtype=bool)
        heading = jnp.zeros(state.grid.shape[:2], dtype=bool)
        picked = []
        targets = []
        costs_now = []
        for index in range(self.env.num_agents):
            if self.seat is not None and index != self.seat:
                picked.append(jnp.int32(OvercookedActionsEnum.stay))
                targets.append(jnp.int32(-1))
                costs_now.append(jnp.inf)
                continue
            goals, penalty, actionable = inputs["seats"][index]
            free = goals & ~claimed.reshape(goals.shape)
            # Rather duplicate than stand still.
            goals = jnp.where(jnp.any(free), free, goals)
            others = (
                jnp.zeros(state.grid.shape[:2], dtype=bool)
                .at[state.agents.pos.y, state.agents.pos.x]
                .set(True)
                .at[state.agents.pos.y[index], state.agents.pos.x[index]]
                .set(False)
            )
            action, target = self._act(
                move_area,
                goals,
                penalty,
                actionable,
                costs[index],
                Position(x=state.agents.pos.x[index], y=state.agents.pos.y[index]),
                state.agents.dir[index],
                keys[index],
                others,
                heading,
                (carry.streak[index] >= self.stubborn_steps)
                | (carry.stuck_for[index] >= self.no_progress_steps),
            )
            claimed = jnp.where(target >= 0, claimed.at[target].set(True), claimed)
            targets.append(target)
            costs_now.append(
                jnp.where(target >= 0, costs[index].reshape(-1)[jnp.maximum(target, 0)], jnp.inf)
            )
            action = jnp.where(
                target < 0, self._step_aside(move_area, state, index), action
            )
            direction = ACTION_TO_DIRECTION[action]
            landing = Position(
                x=state.agents.pos.x[index], y=state.agents.pos.y[index]
            ).move_in_bounds(direction, self.env.width, self.env.height)
            heading = jnp.where(
                direction >= 0, heading.at[landing.y, landing.x].set(True), heading
            )
            picked.append(action)
        chosen = jnp.stack(picked)

        # A seat that tried to move and is standing exactly where it was, still
        # facing the same way, is blocked -- by the other cook, since the walls
        # are already in the distance field. Asking this per seat rather than
        # about the pair catches the common case where one gets through and the
        # other does not.
        positions = jnp.stack([state.agents.pos.x, state.agents.pos.y], axis=-1)
        # A move that left the cook where it stood is a blocked move, whether
        # or not it turned on the spot: with the other cook on the only way
        # through, turning towards a wall is the cheapest move on offer, and
        # counting a turn as progress would let a standoff go on for ever.
        frozen = (
            carry.moved
            & jnp.all(positions == carry.positions, axis=-1)
            & (state.agents.dir == carry.directions)
        )
        streak = jnp.where(frozen, carry.streak + 1, 0)
        # Blocked is not the only way to get nowhere. Two cooks facing each
        # other in a corridor can each turn one step and step back the next,
        # so neither is ever frozen twice running while neither gets any
        # closer. What the give-up rule wants to know is whether this cook is
        # making progress towards what it is walking to, so that is what is
        # counted: the steps since it was last closer to its goal than it has
        # ever been on this errand.
        target_now = jnp.stack(targets)
        cost_now = jnp.stack(costs_now)
        # A cook already standing at its goal is not stuck, however long it
        # waits: waiting at a cooking pot with a plate is the plan, not a
        # failure to get there.
        arrived = cost_now <= 0
        fresh = (target_now != carry.target) | ~jnp.isfinite(carry.best_cost) | arrived
        closer = cost_now < carry.best_cost
        best_cost = jnp.where(fresh | closer, cost_now, carry.best_cost)
        stuck_for = jnp.where(fresh | closer, 0, carry.stuck_for + 1)
        unstuck, gave_way = self._unstuck(
            move_area, state, stuck_key, chosen, streak, carry.yield_rate, jnp.stack(targets)
        )
        chosen = jnp.where(frozen, unstuck, chosen)
        yielded = frozen & gave_way
        # Stand aside for a moment after yielding, unless something can be
        # done from here (an interact is not a step back into the way).
        standing_aside = (carry.waiting > 0) & ~yielded
        chosen = jnp.where(
            standing_aside & (chosen < OvercookedActionsEnum.stay),
            OvercookedActionsEnum.stay,
            chosen,
        )
        waiting = jnp.where(yielded, self.aside_wait, jnp.maximum(carry.waiting - 1, 0))
        # What the other cook did about the last standoff. Blocked a step
        # ago and free now without having stepped aside oneself: the other
        # gave way. Blocked a step ago and blocked still: it did not.
        partner = self.env.num_agents - 1 - jnp.arange(self.env.num_agents)
        was_blocked = carry.streak > 0
        # The other cook gave way if it moved and ended further from where
        # this cook stood -- not merely into the cell this cook vacated.
        partner_moved = jnp.any(positions[partner] != carry.positions[partner], axis=-1)
        they_yielded = was_blocked & ~frozen & ~carry.yielded & partner_moved
        they_held = was_blocked & frozen
        observed = they_yielded | they_held
        yield_rate = jnp.where(
            observed,
            0.7 * carry.yield_rate + 0.3 * they_yielded.astype(jnp.float32),
            carry.yield_rate,
        )

        if self.prob_wait > 0.0:
            waits = jax.random.uniform(wait_key, chosen.shape) < self.prob_wait
            chosen = jnp.where(waits, OvercookedActionsEnum.stay, chosen)

        # A handover is counted when a cook holding an ingredient interacts
        # facing a boundary counter; the count resets whenever there is no
        # other half to share with, so it starts fresh for each outage.
        holding_ingredient = jax.vmap(DynamicObject.is_ingredient)(state.agents.inventory)
        facing = jax.vmap(
            lambda x, y, d: Position(x=x, y=y).move_in_bounds(d, self.env.width, self.env.height)
        )(state.agents.pos.x, state.agents.pos.y, state.agents.dir)
        # On a kitchen with one room every counter is reachable from both
        # sides, so this marks everything a cook sets down. That is not the
        # handover it was written for, but it earns its keep there all the
        # same: it stops a cook picking straight back up what it has just put
        # down -- measured, thirty points a run on the Blackout kitchens.
        boundary_all = (state.grid[:, :, 0] == StaticObject.WALL) & owned[0] & owned[-1]
        handed = (
            (chosen == OvercookedActionsEnum.interact)
            & holding_ingredient
            & boundary_all[facing.y, facing.x]
        )
        sharing_on = jnp.stack([
            ~jnp.any(pile_now & owned[self.env.num_agents - 1 - i])
            & jnp.any(pile_now & owned[i])
            for i in range(self.env.num_agents)
        ])
        shared = jnp.where(sharing_on, carry.shared + handed.astype(jnp.int32), 0)
        holding_any = state.agents.inventory != 0
        placed = (
            (chosen == OvercookedActionsEnum.interact)
            & holding_any
            & boundary_all[facing.y, facing.x]
        )
        marks = jnp.zeros_like(carry.given)
        marks = marks.at[jnp.arange(self.env.num_agents), facing.y, facing.x].set(placed)
        # Given until the other one has taken it: an empty counter carries no
        # mark, so what comes back to it later is somebody else's.
        # Old marks lapse once the counter is empty; a new mark is made on the
        # counter as it was before this step, which is still empty, so it must
        # not be put through the same test.
        given = (carry.given & (state.grid[:, :, 1] != 0)[None]) | marks
        # How long a finished soup has been waiting, in a pot or on a counter.
        static_now, held_now = state.grid[:, :, 0], state.grid[:, :, 1]
        finished = ((held_now & DynamicObject.COOKED) != 0) & (
            (static_now == StaticObject.POT) | (static_now == StaticObject.WALL)
        )
        neglect = jnp.where(finished, carry.neglect + 1, 0)
        hands_changed = state.agents.inventory != carry.inventory
        idle_for = jnp.where(hands_changed, 0, carry.idle_for + 1)
        pot_contents = jnp.where(
            state.grid[:, :, 0] == StaticObject.POT, state.grid[:, :, 1], 0
        )
        stall = jnp.where(jnp.any(pot_contents != carry.pot_contents), 0, carry.stall + 1)
        return (
            PlannerCarry(
                positions=positions,
                directions=state.agents.dir,
                region=region,
                moved=chosen < OvercookedActionsEnum.stay,
                shared=shared,
                given=given,
                streak=streak,
                loader=inputs["loader"],
                neglect=neglect,
                yield_rate=yield_rate,
                yielded=yielded,
                waiting=waiting,
                target=jnp.stack(targets),
                idle_for=idle_for,
                inventory=state.agents.inventory,
                pot_contents=pot_contents,
                stall=stall,
                best_cost=best_cost,
                stuck_for=stuck_for,
            ),
            chosen,
        )
