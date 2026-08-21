"""Small W&B safety helpers shared by Overcooked V3 experiment entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def require_sweep_target(
    run: Any,
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail before training when a sweep was registered in the wrong project.

    A W&B agent owns the actual entity/project and ignores the values passed to
    ``wandb.init``. Comparing the created run to the Hydra sweep parameters
    prevents a CooT job from silently landing in an FCP/SP project while its
    recorded config claims otherwise.
    """

    if environ is None:
        environ = os.environ
    if not environ.get("WANDB_SWEEP_ID") or run is None:
        return
    expected = {
        "entity": str(config.get("ENTITY") or ""),
        "project": str(config.get("PROJECT") or ""),
    }
    actual = {
        "entity": str(getattr(run, "entity", "") or ""),
        "project": str(getattr(run, "project", "") or ""),
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, actual {actual[name]!r}"
        for name in ("entity", "project")
        if expected[name] and actual[name] and expected[name] != actual[name]
    ]
    if not mismatches:
        return
    message = (
        "W&B sweep target mismatch; register the sweep with the entity/project "
        "declared in its YAML (" + "; ".join(mismatches) + ")"
    )
    finish = getattr(run, "finish", None)
    if callable(finish):
        finish(exit_code=1)
    raise RuntimeError(message)
