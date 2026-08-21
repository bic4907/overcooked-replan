"""Stable event vocabulary for Overcooked V3 coordination research.

The order mirrors ``SHAPED_INFOS`` in the CooT/HSP supplementary release.  It
is intentionally independent of the environment's scalar shaped reward: HSP
uses counts of these events as features of a hidden utility function.

Do not reorder or remove entries.  Stored trajectories and population
metadata use the integer positions as a serialization contract.
"""

from enum import IntEnum

import jax.numpy as jnp


class OvercookedV3Event(IntEnum):
    PUT_ONION_ON_COUNTER = 0
    PUT_TOMATO_ON_COUNTER = 1
    PUT_DISH_ON_COUNTER = 2
    PUT_SOUP_ON_COUNTER = 3
    PICKUP_ONION_FROM_COUNTER = 4
    PICKUP_ONION_FROM_DISPENSER = 5
    PICKUP_TOMATO_FROM_COUNTER = 6
    PICKUP_TOMATO_FROM_DISPENSER = 7
    PICKUP_DISH_FROM_COUNTER = 8
    PICKUP_DISH_FROM_DISPENSER = 9
    PICKUP_SOUP_FROM_COUNTER = 10
    USEFUL_DISH_PICKUP = 11
    SOUP_PICKUP = 12
    PLACEMENT_IN_POT = 13
    VIABLE_PLACEMENT = 14
    OPTIMAL_PLACEMENT = 15
    CATASTROPHIC_PLACEMENT = 16
    USELESS_PLACEMENT = 17
    POTTING_ONION = 18
    POTTING_TOMATO = 19
    COOK = 20
    DELIVERY = 21
    DELIVER_SIZE_TWO_ORDER = 22
    DELIVER_SIZE_THREE_ORDER = 23
    DELIVER_USELESS_ORDER = 24
    STAY = 25
    MOVEMENT = 26
    IDLE_MOVEMENT = 27
    IDLE_INTERACT = 28


# Supplement-compatible names are kept verbatim for checkpoint/dataset
# metadata.  The clearer enum names above are used inside V3 implementation.
EVENT_NAMES = (
    "put_onion_on_X",
    "put_tomato_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_tomato_from_X",
    "pickup_tomato_from_T",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",
    "SOUP_PICKUP",
    "PLACEMENT_IN_POT",
    "viable_placement",
    "optimal_placement",
    "catastrophic_placement",
    "useless_placement",
    "potting_onion",
    "potting_tomato",
    "cook",
    "delivery",
    "deliver_size_two_order",
    "deliver_size_three_order",
    "deliver_useless_order",
    "STAY",
    "MOVEMENT",
    "IDLE_MOVEMENT",
    "IDLE_INTERACT",
)

NUM_EVENTS = len(EVENT_NAMES)
EVENT_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}


def empty_event_vector():
    """Return one zero-initialized event vector with the stable float dtype."""

    return jnp.zeros((NUM_EVENTS,), dtype=jnp.float32)


__all__ = [
    "EVENT_INDEX",
    "EVENT_NAMES",
    "NUM_EVENTS",
    "OvercookedV3Event",
    "empty_event_vector",
]
