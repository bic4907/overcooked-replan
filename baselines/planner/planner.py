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


def ingredient_counts(encoding):
    """How many of each ingredient a recipe, a pot, or a hand holds.

    Ingredients are packed two bits apiece above the plate and cooked flags,
    and quantities are stored by adding, so a count reads straight out.
    """
    shifts = 2 + 2 * jnp.arange(MAX_INGREDIENTS)
    return (encoding >> shifts) & 0x3


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
        staged = jnp.sum(is_plate_counter)
        pots = jnp.sum(static == StaticObject.POT)

        # Who loads the pots is decided at the ingredients, not at the pots: the
        # loading loop starts and ends at the pile, so the cook standing nearer
        # to it is the one whose round trip is shorter.
        enough_pots = jnp.sum(is_pot) >= self.pot_specialist
        if self.pot_role_connected_only:
            enough_pots = enough_pots & joined
        my_pile_cost = jnp.min(jnp.where(pile_for_first, here, jnp.inf))
        their_pile_cost = jnp.min(jnp.where(pile_for_first, partner_here, jnp.inf))
        # Ties go to the first seat, so exactly one of them loads -- and the
        # other has to be clearly nearer, not just nearer, before the job
        # changes hands. Decided by exact distance, the job flips with every
        # step the cooks take and a cook sets off for the pile, turns round,
        # and sets off again.
        nearer = jnp.where(
            first,
            my_pile_cost <= their_pile_cost + self.role_margin,
            my_pile_cost + self.role_margin < their_pile_cost,
        )

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
        room_for_plates = jnp.where(
            stock_plates, True, jnp.where(joined, True, staged_plates < 2)
        )
        # A cook doing both jobs loads pots before it fetches plates: a soup
        # that is done keeps, an empty pot earns nothing until it is filled.
        # The plate is fetched once nothing is left to load.
        fetch_plate = ~needs_filling & room_for_plates
        # Standing back is only worth it while there is a plate to go and get.
        # With nothing cooking there is no plate job, and a cook waiting for one
        # anyway is half the kitchen doing nothing.
        helper_plate = (soup_ready | jnp.any(kitchen["cooking"])) & no_plate_in_flight
        loads_pots = ~enough_pots | nearer
        if self.pot_role_helper_escape:
            loads_pots = loads_pots | ~helper_plate

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
        plates_are_mine = ~enough_pots | ~loads_pots | ~jnp.any(plate_pile)
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
        empty_target = jnp.where(
            stock_ingredients | share_now,
            pile_for_first,
            jnp.where(
                stock_plates,
                plate_pile,
                jnp.where(
                    loads_pots,
                    jnp.where(fetch_plate, plate_pile, pile_for_first),
                    # The plate cook is capped too, or the cap is no cap: this
                    # branch is the one that fetches plate after plate.
                    jnp.where(room_for_plates, plate_pile, pile_for_first),
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
            stock_ingredients | (stock_plates & ~soup_ready),
            jnp.zeros_like(plate_target),
            plate_target,
        )

        wanted = jnp.where(
            holding_dish,
            goal,
            jnp.where(
                holding_plate,
                plate_target,
                jnp.where(holding_ingredient, pot_for_carried, empty_target),
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
        counter_item = is_counter & (held != 0) & usable & ~given
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
            | ((held == DynamicObject.PLATE) & can_reach_soup & (plates_are_mine | ~joined))
            | (ingredient_on_counter & can_fill_pot & loads_pots)
        )

        holding_something = inventory != 0

        # Free movement is the normal case: while the halves are joined the
        # kitchen is one room and both cooks go wherever the work is. Only a
        # shut door makes each of them work the half it can finish in and put
        # the rest on a counter for the other, which is the pipeline.
        scoped = wanted & jnp.where(connected, reachable, mine)

        # Assigned the other half and still standing in this one, a cook walks
        # over by aiming at what is over there -- that is the one crossing
        # the countdown must not stop, since two cooks left on one side of a
        # shut door is the worst place a split can leave them. Not at a
        # counter: interacting with one while holding something puts it down.
        must_move = displaced & ~jnp.any(scoped & reachable)
        wanted = jnp.where(must_move, mine & ~is_counter, scoped)

        pass_it_on = holding_something & jnp.any(handover) & ~jnp.any(scoped & reachable)

        wanted = jnp.where(
            pass_it_on,
            handover,
            jnp.where(~holding_something & jnp.any(useful), useful, wanted),
        )

        # Crossing is allowed but not free. A cook that wanders into the other
        # half for a job barely closer is out of position when the door shuts,
        # and the pipeline that shut door needs has to be built again from
        # nothing. Charging a few steps for leaving home keeps both of them
        # where they will be needed without forbidding the trip.
        # Where interacting right now actually does something. A cook can
        # stand at a cooking pot holding a plate -- that is the right place to
        # wait -- but pressing interact there every step does nothing and
        # reads as a cook fiddling with a pot it has no business with.
        idle_pot = kitchen["cooking"] & (inventory == DynamicObject.PLATE)
        actionable = wanted & ~idle_pot
        return wanted, jnp.where(mine, 0.0, self.home_bias), actionable

    # -- one step towards it ---------------------------------------------

    def _cost_grid(self, move_area, pos, direction):
        """For every cell that cannot be walked on, steps to stand and face it."""
        return OvercookedPathPlanner._compute_min_moves(pos, direction, move_area)

    def _act(self, move_area, goals, penalty, actionable, here, pos, direction, key, occupied):
        goal_key, move_key = jax.random.split(key)
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
            return cost + jnp.where(occupied[landed.y, landed.x], 20.0, 0.0)

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
                return (grid + jnp.where(others[landed.y, landed.x], 20.0, 0.0)).reshape(-1), landed

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

    def _unstuck(self, move_area, state, key, planned, streak):
        """Break a standoff by having one cook give way rather than both.

        Two cooks pushing into the same cell both stop, and both re-planning
        the same way next step keeps them there. If both then pick at random
        they are as likely to collide again. So the first seat keeps its plan
        and the others step aside, which clears the way in one step instead of
        waiting for the coin to land right.
        """
        occupied = (
            jnp.zeros_like(move_area)
            .at[state.agents.pos.y, state.agents.pos.x]
            .set(True)
        )

        def per_agent(x, y, own_key):
            pos = Position(x=x, y=y)

            def usable(move_direction):
                moved = pos.move_in_bounds(
                    move_direction, self.env.width, self.env.height
                )
                free = move_area[moved.y, moved.x] & ~occupied[moved.y, moved.x]
                return free & ((moved.x != x) | (moved.y != y))

            allowed = jax.vmap(usable)(DIRECTIONS)
            logits = jnp.where(allowed, 0.0, -jnp.inf)
            # Nowhere to go is possible; standing still is then the only option.
            return jnp.where(
                jnp.any(allowed),
                MOVES[jax.random.categorical(own_key, logits)],
                OvercookedActionsEnum.stay,
            ).astype(jnp.int32)

        keys = jax.random.split(key, self.env.num_agents)
        stepping_aside = jax.vmap(per_agent)(
            state.agents.pos.x, state.agents.pos.y, keys
        )
        # With two planners the first seat holds its plan and the rest give
        # way, which clears a standoff in one step. Alone, this seat cannot
        # count on the other giving way, so it is the one that moves.
        # Priority is a first-step courtesy, not a right. Blocked once, the
        # first seat holds its plan and the other gives way; blocked again on
        # the very next step, nothing has improved and the first seat gives way
        # instead -- and the other then holds its plan. Exactly one of them
        # yields at a time: both stepping aside at once clears the cell and
        # puts them straight back into each other on the next step.
        if self.seat is not None:
            keeps_plan = jnp.zeros((self.env.num_agents,), dtype=bool)
        else:
            first_yields = streak[0] >= 2
            keeps_plan = jnp.where(
                jnp.arange(self.env.num_agents) == 0, ~first_yields, first_yields
            )
        return jnp.where(keeps_plan, planned, stepping_aside)

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

        # An outage that takes every pile is one thing; one that takes the pile
        # from one half and leaves the other is a different job. There the cook
        # that keeps a pile fills its own pots first and carries the rest to
        # the counters for the one that does not.
        pile_now = now >= StaticObject.INGREDIENT_PILE_BASE
        pile_soon = soon >= StaticObject.INGREDIENT_PILE_BASE
        vanishing_wall = warned & (now == StaticObject.WALL) & (soon != StaticObject.WALL)

        here_regions = self.phase_regions[state.layout_index]
        sides = here_regions[state.agents.pos.y, state.agents.pos.x]
        same_side = sides[0] == sides[1]

        seats = []
        for index in range(self.env.num_agents):
            partner = self.env.num_agents - 1 - index
            goals, penalty, actionable = self._goal_cells(
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
            )
            seats.append((goals, penalty, actionable))
        return dict(
            move_area=move_area, costs=costs, region=region, owned=owned,
            pile_now=pile_now, seats=seats,
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
        for index in range(self.env.num_agents):
            if self.seat is not None and index != self.seat:
                picked.append(jnp.int32(OvercookedActionsEnum.stay))
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
            ) | heading
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
            )
            claimed = jnp.where(target >= 0, claimed.at[target].set(True), claimed)
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
        frozen = (
            carry.moved
            & jnp.all(positions == carry.positions, axis=-1)
            & (state.agents.dir == carry.directions)
        )
        streak = jnp.where(frozen, carry.streak + 1, 0)
        chosen = jnp.where(
            frozen, self._unstuck(move_area, state, stuck_key, chosen, streak), chosen
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
        return (
            PlannerCarry(
                positions=positions,
                directions=state.agents.dir,
                region=region,
                moved=chosen < OvercookedActionsEnum.stay,
                shared=shared,
                given=given,
                streak=streak,
            ),
            chosen,
        )
