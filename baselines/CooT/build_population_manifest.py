"""Build the paper-matched CooT population and response-job manifests.

This utility deliberately implements the supplement's HSP selection rule as
``greedy_normalized_l1``.  It is *not* a DPP selector: DPP is an evaluation
metric in the CooT release and must not be substituted for population
construction.

The catalog format is intentionally small.  A catalog is a JSON object with a
``layout`` and a ``candidates`` list.  Each candidate needs an ``id`` and, for
HSP, a non-negative ``selection_features`` vector.  Partner policies may be
written either as nested policy specs::

    {
      "id": "hsp_000",
      "selection_features": [1, 0, 4],
      "partner": {
        "mid": {"checkpoint": "checkpoints/hsp_000_mid.safetensors"},
        "final": {"checkpoint": "checkpoints/hsp_000_final.safetensors"}
      },
      "best_response": {
        "mid": {"checkpoint": "responses/hsp_000_mid.safetensors"},
        "final": {"checkpoint": "responses/hsp_000_final.safetensors"}
      }
    }

or with ``partner_mid_checkpoint`` / ``partner_final_checkpoint`` fields.  A
separate response-result JSON can use the atomic job schema emitted by the
``response-jobs`` subcommand.  Mark a completed job with ``status=completed``
and retain its ``best_response_checkpoint`` field.

Relative checkpoint paths in every input are interpreted relative to that
input JSON.  Relative checkpoint paths in generated manifests are always
rebased to the generated manifest's parent directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PAPER_HSP_COUNT = 21
PAPER_MEP_COUNT = 15
PAPER_SELECTOR_SEED = 0
RELEASE_MIN_HSP_REFERENCE_RETURN = 0.1
HSP_MID_ROLLOUTS = 30
HSP_FINAL_ROLLOUTS = 220
MEP_FINAL_ROLLOUTS = 200
POLICY_FIELDS = (
    "architecture",
    "activation",
    "fc_dim_size",
    "gru_hidden_dim",
    "stochastic",
)
COMPLETED_STATUSES = {"complete", "completed", "finished", "succeeded", "success"}


@dataclass
class Candidate:
    """Normalized candidate assembled from catalog and response-result entries."""

    identifier: str
    population_type: str
    partner: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_response: dict[str, dict[str, Any]] = field(default_factory=dict)
    selection_features: list[float] | None = None
    feature_priority: int = -1
    reference_return: float | None = None
    reference_return_priority: int = -1


def _read_json(path: Path) -> tuple[Any, Path]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON input not found: {resolved}")
    try:
        return json.loads(resolved.read_text(encoding="utf-8")), resolved
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc


def _entries(payload: Any, *, path: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, Mapping):
        raw_entries = None
        for key in (
            "candidates",
            "policies",
            "entries",
            "jobs",
            "results",
            "responses",
        ):
            value = payload.get(key)
            if value is not None:
                raw_entries = value
                break
        if raw_entries is None and (
            "original_job" in payload
            or "partner_id" in payload
            or "candidate_id" in payload
            or "id" in payload
        ):
            # Per-run BR trainers write one result object rather than a list.
            # Treat it as one atomic response entry; _merge_response_results
            # flattens original_job/resolved_job below.
            raw_entries = [payload]
        if raw_entries is None:
            raise ValueError(
                f"{path} needs one of candidates/policies/entries/jobs/results/"
                "responses"
            )
    else:
        raise ValueError(f"{path} must contain a JSON object or list")
    if not isinstance(raw_entries, list):
        raise ValueError(f"The entry collection in {path} must be a JSON list")
    if not raw_entries:
        raise ValueError(f"The entry collection in {path} is empty")
    if not all(isinstance(entry, Mapping) for entry in raw_entries):
        raise ValueError(f"Every entry in {path} must be a JSON object")
    return list(raw_entries)


def _entry_identifier(entry: Mapping[str, Any], *, path: Path) -> str:
    value = entry.get("id", entry.get("candidate_id", entry.get("partner_id")))
    if value is None or not str(value).strip():
        raise ValueError(f"Every entry in {path} needs id, candidate_id, or partner_id")
    return str(value).strip()


def _population_type(
    entry: Mapping[str, Any], *, expected: str | None, path: Path
) -> str:
    value = entry.get("population_type", entry.get("source", expected))
    if value is None:
        raise ValueError(
            f"Entry {_entry_identifier(entry, path=path)!r} needs population_type"
        )
    normalized = str(value).lower()
    if normalized not in {"hsp", "mep"}:
        raise ValueError(f"Unsupported population_type {value!r} in {path}")
    if expected is not None and normalized != expected:
        raise ValueError(
            f"Entry {_entry_identifier(entry, path=path)!r} in {path} says "
            f"population_type={normalized!r}; expected {expected!r}"
        )
    return normalized


def _skill(entry: Mapping[str, Any], *, default: str = "final") -> str:
    value = str(
        entry.get("skill", entry.get("skill_level", entry.get("variant", default)))
    )
    normalized = value.lower()
    aliases = {"intermediate": "mid", "middle": "mid", "last": "final"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"mid", "final"}:
        raise ValueError(f"Unsupported policy skill {value!r}; expected mid or final")
    return normalized


def _policy_defaults(entry: Mapping[str, Any], role: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    generic = entry.get("policy")
    if isinstance(generic, Mapping):
        defaults.update({key: generic[key] for key in POLICY_FIELDS if key in generic})
    named = entry.get(f"{role}_policy")
    if isinstance(named, Mapping):
        defaults.update({key: named[key] for key in POLICY_FIELDS if key in named})
    defaults.update({key: entry[key] for key in POLICY_FIELDS if key in entry})
    return defaults


def _normalize_policy_spec(
    value: Any,
    *,
    defaults: Mapping[str, Any],
    base_dir: Path,
    context: str,
) -> dict[str, Any]:
    if isinstance(value, (str, os.PathLike)):
        raw: dict[str, Any] = {"checkpoint": str(value)}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError(f"{context} must be a checkpoint string or policy object")
    checkpoint_value = raw.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError(f"{context} needs a checkpoint")
    checkpoint = Path(str(checkpoint_value)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (base_dir / checkpoint).resolve()
    else:
        checkpoint = checkpoint.resolve()

    merged = dict(defaults)
    merged.update({key: raw[key] for key in POLICY_FIELDS if key in raw})
    architecture = str(merged.get("architecture", "rnn")).lower()
    if architecture not in {"rnn", "cnn"}:
        raise ValueError(f"{context} has unsupported architecture {architecture!r}")
    fc_dim_size = int(merged.get("fc_dim_size", 128))
    gru_hidden_dim = int(merged.get("gru_hidden_dim", 128))
    if fc_dim_size < 1 or gru_hidden_dim < 1:
        raise ValueError(f"{context} policy dimensions must be positive")
    return {
        "checkpoint": str(checkpoint),
        "architecture": architecture,
        "activation": str(merged.get("activation", "relu")),
        "fc_dim_size": fc_dim_size,
        "gru_hidden_dim": gru_hidden_dim,
        "stochastic": bool(merged.get("stochastic", False)),
    }


def _nested_variants(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    variants: dict[str, Any] = {}
    for raw_skill, normalized_skill in (
        ("mid", "mid"),
        ("intermediate", "mid"),
        ("final", "final"),
    ):
        if raw_skill in value:
            variants[normalized_skill] = value[raw_skill]
    return variants


def _extract_policy_variants(
    entry: Mapping[str, Any],
    *,
    role: str,
    base_dir: Path,
    context: str,
) -> dict[str, dict[str, Any]]:
    if role == "partner":
        container_keys = ("partner", "partners", "partner_checkpoints")
        direct_checkpoint_keys = ("partner_checkpoint",)
    elif role == "best_response":
        container_keys = (
            "best_response",
            "best_responses",
            "response",
            "responses",
            "best_response_checkpoints",
        )
        direct_checkpoint_keys = ("best_response_checkpoint", "response_checkpoint")
    else:  # pragma: no cover - private API guard
        raise AssertionError(role)

    defaults = _policy_defaults(entry, role)
    raw_variants: dict[str, Any] = {}
    for key in container_keys:
        if key not in entry:
            continue
        value = entry[key]
        nested = _nested_variants(value)
        if nested:
            raw_variants.update(nested)
        else:
            raw_variants[_skill(entry)] = value
        break

    for skill in ("mid", "final"):
        for key in (
            f"{role}_{skill}",
            f"{skill}_{role}",
            f"{role}_{skill}_checkpoint",
            f"{skill}_{role}_checkpoint",
        ):
            if key in entry:
                raw_variants[skill] = entry[key]
                break

    if not raw_variants:
        for key in direct_checkpoint_keys:
            if key in entry:
                raw_variants[_skill(entry)] = entry[key]
                break

    return {
        skill: _normalize_policy_spec(
            value,
            defaults=defaults,
            base_dir=base_dir,
            context=f"{context} {role}/{skill}",
        )
        for skill, value in raw_variants.items()
    }


def _feature_vector(entry: Mapping[str, Any], *, context: str) -> list[float] | None:
    value: Any = None
    for key in (
        "selection_features",
        "event_features",
        "behavior_features",
        "features",
    ):
        if key in entry:
            value = entry[key]
            break
    if value is None and isinstance(entry.get("metrics"), Mapping):
        metrics = entry["metrics"]
        for key in ("selection_features", "event_features", "behavior_features"):
            if key in metrics:
                value = metrics[key]
                break
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} selection features must be a JSON number list")
    try:
        features = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} selection features must contain only numbers"
        ) from exc
    if not features or any(not math.isfinite(item) or item < 0.0 for item in features):
        raise ValueError(
            f"{context} selection features must be a non-empty, finite, "
            "non-negative vector"
        )
    return features


def _merge_features(
    candidate: Candidate,
    features: list[float] | None,
    *,
    priority: int,
    context: str,
) -> None:
    if features is None or priority < candidate.feature_priority:
        return
    if (
        priority == candidate.feature_priority
        and candidate.selection_features != features
    ):
        raise ValueError(
            f"Conflicting equal-priority selection feature vectors for {context}"
        )
    candidate.selection_features = features
    candidate.feature_priority = priority


def _reference_return(entry: Mapping[str, Any], *, context: str) -> float | None:
    """Read the stochastic BR-vs-partner sparse return emitted by the scorer."""

    value = entry.get("reference_return")
    if value is None and isinstance(entry.get("metrics"), Mapping):
        value = entry["metrics"].get("reference_return")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} reference_return must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} reference_return must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} reference_return must be a finite number")
    return result


def _merge_reference_return(
    candidate: Candidate,
    value: float | None,
    *,
    priority: int,
    context: str,
) -> None:
    if value is None or priority < candidate.reference_return_priority:
        return
    if (
        priority == candidate.reference_return_priority
        and candidate.reference_return != value
    ):
        raise ValueError(
            f"Conflicting equal-priority reference_return values for {context}"
        )
    candidate.reference_return = value
    candidate.reference_return_priority = priority


def _load_catalog(
    path: Path, *, expected_population_type: str
) -> tuple[dict[str, Candidate], Any, Path]:
    payload, resolved = _read_json(path)
    candidates: dict[str, Candidate] = {}
    for entry in _entries(payload, path=resolved):
        identifier = _entry_identifier(entry, path=resolved)
        population_type = _population_type(
            entry, expected=expected_population_type, path=resolved
        )
        candidate = candidates.setdefault(
            identifier,
            Candidate(identifier=identifier, population_type=population_type),
        )
        context = f"{resolved}:{identifier}"
        candidate.partner.update(
            _extract_policy_variants(
                entry,
                role="partner",
                base_dir=resolved.parent,
                context=context,
            )
        )
        candidate.best_response.update(
            _extract_policy_variants(
                entry,
                role="best_response",
                base_dir=resolved.parent,
                context=context,
            )
        )
        feature_priority = 3 if _skill(entry) == "final" else 2
        if (
            "skill" not in entry
            and "skill_level" not in entry
            and "variant" not in entry
        ):
            feature_priority = 4
        _merge_features(
            candidate,
            _feature_vector(entry, context=context),
            priority=feature_priority,
            context=context,
        )
        _merge_reference_return(
            candidate,
            _reference_return(entry, context=context),
            priority=feature_priority,
            context=context,
        )
    return candidates, payload, resolved


def _result_entry_is_complete(entry: Mapping[str, Any]) -> bool:
    if "status" not in entry:
        return True
    return str(entry["status"]).lower() in COMPLETED_STATUSES


def _flatten_response_result(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one BR sidecar, keeping top-level result fields authoritative."""

    flattened: dict[str, Any] = {}
    original_job = entry.get("original_job")
    if isinstance(original_job, Mapping):
        flattened.update(original_job)
    resolved_job = entry.get("resolved_job")
    if isinstance(resolved_job, Mapping):
        flattened.update(resolved_job)
    flattened.update(entry)
    return flattened


def _expand_response_result_paths(
    response_paths: Iterable[str | Path],
) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in response_paths:
        raw_value = os.path.expanduser(str(raw_path))
        candidate = Path(raw_value)
        if glob.has_magic(raw_value):
            matches = [Path(value) for value in glob.glob(raw_value, recursive=True)]
            if not matches:
                raise FileNotFoundError(
                    f"Response-result glob matched no files: {raw_path}"
                )
        elif candidate.is_dir():
            matches = sorted(candidate.rglob("response_job*.json"))
            if not matches:
                raise FileNotFoundError(
                    "Response-result directory contains no recursive "
                    f"response_job*.json files: {candidate}"
                )
        else:
            matches = [candidate]
        for match in matches:
            resolved = match.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Response result not found: {resolved}")
            if resolved not in seen:
                seen.add(resolved)
                expanded.append(resolved)
    return expanded


def _merge_response_results(
    response_paths: Iterable[str | Path],
    *,
    hsp: dict[str, Candidate],
    mep: dict[str, Candidate],
) -> tuple[list[Path], list[Any]]:
    consumed: list[Path] = []
    payloads: list[Any] = []
    for response_path in _expand_response_result_paths(response_paths):
        payload, resolved = _read_json(response_path)
        consumed.append(resolved)
        payloads.append(payload)
        for raw_entry in _entries(payload, path=resolved):
            if not _result_entry_is_complete(raw_entry):
                continue
            entry = _flatten_response_result(raw_entry)
            identifier_value = entry.get(
                "partner_id", entry.get("candidate_id", entry.get("id"))
            )
            if identifier_value is None or not str(identifier_value).strip():
                identifier = _entry_identifier(entry, path=resolved)
            else:
                identifier = str(identifier_value).strip()
            declared_type = entry.get("population_type", entry.get("source"))
            if declared_type is None:
                matches = [mapping for mapping in (hsp, mep) if identifier in mapping]
                if len(matches) != 1:
                    raise ValueError(
                        f"Response result {identifier!r} in {resolved} needs "
                        "population_type because its id is missing or ambiguous"
                    )
                candidate = matches[0][identifier]
            else:
                population_type = _population_type(entry, expected=None, path=resolved)
                mapping = hsp if population_type == "hsp" else mep
                if identifier not in mapping:
                    raise ValueError(
                        f"Response result {population_type}/{identifier} in {resolved} "
                        "does not match a catalog candidate"
                    )
                candidate = mapping[identifier]
            context = f"{resolved}:{candidate.population_type}/{identifier}"
            responses = _extract_policy_variants(
                entry,
                role="best_response",
                base_dir=resolved.parent,
                context=context,
            )
            if not responses:
                raise ValueError(
                    f"Completed response result {context} has no "
                    "best_response_checkpoint"
                )
            candidate.best_response.update(responses)
            priority = 3 if _skill(entry) == "final" else 2
            _merge_features(
                candidate,
                _feature_vector(entry, context=context),
                priority=priority,
                context=context,
            )
            _merge_reference_return(
                candidate,
                _reference_return(entry, context=context),
                priority=priority,
                context=context,
            )
    return consumed, payloads


def _payload_layouts(payload: Any) -> set[str]:
    layouts: set[str] = set()
    if isinstance(payload, Mapping) and payload.get("layout"):
        layouts.add(str(payload["layout"]))
    if isinstance(payload, Mapping):
        for nested_key in ("original_job", "resolved_job"):
            nested = payload.get(nested_key)
            if isinstance(nested, Mapping) and nested.get("layout"):
                layouts.add(str(nested["layout"]))
        for key in (
            "candidates",
            "policies",
            "entries",
            "jobs",
            "results",
            "responses",
        ):
            values = payload.get(key)
            if isinstance(values, list):
                for entry in values:
                    if isinstance(entry, Mapping) and entry.get("layout"):
                        layouts.add(str(entry["layout"]))
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, Mapping) and entry.get("layout"):
                layouts.add(str(entry["layout"]))
    return layouts


def _resolve_layout(explicit: str | None, payloads: Iterable[Any]) -> str:
    layouts = set().union(*(_payload_layouts(payload) for payload in payloads))
    if explicit is not None:
        if layouts and layouts != {explicit}:
            raise ValueError(
                f"--layout={explicit!r} conflicts with catalog layouts "
                f"{sorted(layouts)!r}"
            )
        return explicit
    if len(layouts) != 1:
        raise ValueError(
            "Specify --layout unless all input catalogs declare one identical layout; "
            f"found {sorted(layouts)!r}"
        )
    return next(iter(layouts))


def _natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.lower())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in parts if part
    )


def greedy_normalized_l1_select(
    candidates: Sequence[Candidate], *, count: int, seed: int
) -> list[Candidate]:
    """Reproduce the release's seeded greedy normalized-L1 HSP selector.

    The legacy code normalizes each feature column by ``max + 1e-3``, samples
    the first item with ``np.random.seed(seed); np.random.randint(n)``, and then
    repeatedly maximizes summed L1 distance to the selected set.  RandomState
    is used intentionally because ``default_rng`` gives a different seed-0
    first index.
    """

    if count < 1:
        raise ValueError("HSP selection count must be positive")
    ordered = sorted(
        candidates, key=lambda candidate: _natural_key(candidate.identifier)
    )
    if len(ordered) < count:
        raise ValueError(
            f"Need at least {count} eligible HSP candidates; found {len(ordered)}"
        )
    missing = [
        candidate.identifier
        for candidate in ordered
        if candidate.selection_features is None
    ]
    if missing:
        raise ValueError(
            "Every eligible HSP candidate needs selection_features before selection; "
            f"missing for {missing}"
        )
    dimensions = {len(candidate.selection_features or []) for candidate in ordered}
    if len(dimensions) != 1:
        raise ValueError(
            f"HSP selection feature dimensions must match; found {sorted(dimensions)}"
        )
    matrix = np.asarray(
        [candidate.selection_features for candidate in ordered], dtype=np.float64
    )
    normalized = matrix / (np.max(matrix, axis=0, keepdims=True) + 1e-3)
    rng = np.random.RandomState(seed)
    selected_indices = [int(rng.randint(0, len(ordered)))]
    while len(selected_indices) < count:
        scores = np.full((len(ordered),), -np.inf, dtype=np.float64)
        for index in range(len(ordered)):
            if index in selected_indices:
                continue
            scores[index] = sum(
                float(np.abs(normalized[index] - normalized[chosen]).sum())
                for chosen in selected_indices
            )
        selected_indices.append(int(np.argmax(scores)))
    # The release sorts the selected run identifiers before materializing the
    # population.  Natural catalog-id ordering is the generic equivalent.
    return sorted(
        (ordered[index] for index in selected_indices),
        key=lambda candidate: _natural_key(candidate.identifier),
    )


def _load_exclusions(values: Sequence[str], file_path: Path | None) -> set[str]:
    exclusions = {str(value).strip() for value in values if str(value).strip()}
    if file_path is None:
        return exclusions
    payload, resolved = _read_json(file_path)
    if isinstance(payload, list):
        file_values = payload
    elif isinstance(payload, Mapping):
        file_values = payload.get("exclude_ids", payload.get("ids"))
    else:
        file_values = None
    if not isinstance(file_values, list):
        raise ValueError(f"{resolved} must be a list or contain exclude_ids/ids")
    exclusions.update(str(value).strip() for value in file_values if str(value).strip())
    return exclusions


def _filter_scored_hsp_by_return(
    candidates: Sequence[Candidate],
    *,
    minimum_return: float,
    context: str,
) -> tuple[list[Candidate], list[str]]:
    """Apply the release's pre-selection sparse-return viability filter."""

    missing = [
        candidate.identifier
        for candidate in candidates
        if candidate.reference_return is None
    ]
    if missing:
        raise ValueError(
            f"{context} requires scorer-produced reference_return for every HSP "
            f"candidate before selection; missing for {missing}"
        )
    filtered = [
        candidate.identifier
        for candidate in candidates
        if float(candidate.reference_return) <= minimum_return
    ]
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate.reference_return) > minimum_return
    ]
    return eligible, sorted(filtered, key=_natural_key)


def _assert_skills(
    candidate: Candidate, skills: Sequence[str], *, responses: bool
) -> None:
    missing_partner = [skill for skill in skills if skill not in candidate.partner]
    if missing_partner:
        raise ValueError(
            f"{candidate.population_type}/{candidate.identifier} is missing partner "
            f"checkpoint(s) for {missing_partner}"
        )
    if responses:
        missing_response = [
            skill for skill in skills if skill not in candidate.best_response
        ]
        if missing_response:
            raise ValueError(
                f"{candidate.population_type}/{candidate.identifier} is missing "
                f"completed best-response checkpoint(s) for {missing_response}; "
                "generate response-jobs and pass their completed results with "
                "--response-results"
            )


def _verify_policy_checkpoints(
    candidates: Iterable[Candidate], *, include_responses: bool
) -> None:
    for candidate in candidates:
        policies = list(candidate.partner.values())
        if include_responses:
            policies.extend(candidate.best_response.values())
        for policy in policies:
            checkpoint = Path(str(policy["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Checkpoint for {candidate.population_type}/"
                    f"{candidate.identifier} does not exist: {checkpoint}"
                )


def _relative_checkpoint(checkpoint: str, *, output_dir: Path) -> str:
    return Path(os.path.relpath(checkpoint, output_dir)).as_posix()


def _serialize_policy(policy: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    serialized = dict(policy)
    serialized["checkpoint"] = _relative_checkpoint(
        str(policy["checkpoint"]), output_dir=output_dir
    )
    return serialized


def _pair_identifier(candidate: Candidate) -> str:
    prefix = f"{candidate.population_type}_"
    if candidate.identifier.lower().startswith(prefix):
        return candidate.identifier
    return f"{prefix}{candidate.identifier}"


def _deviations(
    *,
    hsp_count: int,
    mep_count: int,
    seed: int,
    hsp_only: bool,
    minimum_hsp_reference_return: float,
) -> list[dict[str, Any]]:
    deviations: list[dict[str, Any]] = []
    if hsp_count != PAPER_HSP_COUNT:
        deviations.append(
            {
                "code": "non_paper_hsp_count",
                "paper_value": PAPER_HSP_COUNT,
                "used_value": hsp_count,
            }
        )
    if seed != PAPER_SELECTOR_SEED:
        deviations.append(
            {
                "code": "non_paper_selector_seed",
                "paper_value": PAPER_SELECTOR_SEED,
                "used_value": seed,
            }
        )
    if minimum_hsp_reference_return != RELEASE_MIN_HSP_REFERENCE_RETURN:
        deviations.append(
            {
                "code": "non_release_hsp_return_filter",
                "release_value": RELEASE_MIN_HSP_REFERENCE_RETURN,
                "used_value": minimum_hsp_reference_return,
                "comparison": "reference_return > threshold",
            }
        )
    if hsp_only:
        deviations.append(
            {
                "code": "hsp_only_proxy",
                "description": (
                    "MEP is absent. This is an explicit HSP-only proxy, not the "
                    "paper's 21-HSP + 15-MEP CooT training population."
                ),
                "missing_population_type": "mep",
                "paper_mep_count": PAPER_MEP_COUNT,
            }
        )
    elif mep_count != PAPER_MEP_COUNT:
        deviations.append(
            {
                "code": "non_paper_mep_count",
                "paper_value": PAPER_MEP_COUNT,
                "used_value": mep_count,
            }
        )
    return deviations


def _construction_metadata(
    *,
    selected_hsp: Sequence[Candidate],
    selected_mep: Sequence[Candidate],
    eligible_hsp_count: int,
    exclusions: set[str],
    return_filtered_ids: Sequence[str],
    minimum_hsp_reference_return: float,
    seed: int,
    requested_hsp_count: int,
    requested_mep_count: int,
    hsp_only: bool,
) -> dict[str, Any]:
    return {
        "paper_target": {
            "hsp": PAPER_HSP_COUNT,
            "mep": PAPER_MEP_COUNT,
            "total": PAPER_HSP_COUNT + PAPER_MEP_COUNT,
        },
        "population_mode": "hsp_only_proxy" if hsp_only else "hsp_plus_mep",
        "hsp_selector": {
            "name": "greedy_normalized_l1",
            "is_dpp": False,
            "seed": seed,
            "normalization": "feature / (column_max + 1e-3)",
            "first_candidate": "numpy RandomState(seed).randint(candidate_count)",
            "candidate_count_after_exclusions": (
                eligible_hsp_count + len(return_filtered_ids)
            ),
            "candidate_count_after_return_filter": eligible_hsp_count,
            "minimum_reference_return_exclusive": minimum_hsp_reference_return,
            "return_filtered_ids": list(return_filtered_ids),
            "selected_count": len(selected_hsp),
            "selected_ids": [candidate.identifier for candidate in selected_hsp],
        },
        "mep": {
            "selection": "catalog_population_no_additional_selector",
            "selected_count": len(selected_mep),
            "selected_ids": [candidate.identifier for candidate in selected_mep],
        },
        "excluded_hsp_ids": sorted(exclusions, key=_natural_key),
        "rollout_recipe": {
            "hsp": {
                "total": HSP_MID_ROLLOUTS + HSP_FINAL_ROLLOUTS,
                "mid": HSP_MID_ROLLOUTS,
                "final": HSP_FINAL_ROLLOUTS,
                "mid_weight": HSP_MID_ROLLOUTS
                / (HSP_MID_ROLLOUTS + HSP_FINAL_ROLLOUTS),
                "final_weight": HSP_FINAL_ROLLOUTS
                / (HSP_MID_ROLLOUTS + HSP_FINAL_ROLLOUTS),
                "mid_overrides_partner_and_best_response": True,
            },
            "mep": {"total": MEP_FINAL_ROLLOUTS, "skill": "final"},
        },
        "deviations": _deviations(
            hsp_count=requested_hsp_count,
            mep_count=requested_mep_count,
            seed=seed,
            hsp_only=hsp_only,
            minimum_hsp_reference_return=minimum_hsp_reference_return,
        ),
    }


def _atomic_write_json(
    payload: Mapping[str, Any], path: Path, *, overwrite: bool
) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {resolved}; pass --overwrite")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def _load_population_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Candidate],
    dict[str, Candidate],
    list[Any],
    list[Path],
    bool,
]:
    hsp, hsp_payload, hsp_path = _load_catalog(
        args.hsp_catalog, expected_population_type="hsp"
    )
    payloads = [hsp_payload]
    source_paths = [hsp_path]
    mep: dict[str, Candidate] = {}
    hsp_only = args.mep_catalog is None
    if hsp_only:
        hsp_preselection = bool(
            args.command == "response-jobs"
            and getattr(args, "all_hsp_candidates", False)
        )
        if not args.allow_hsp_only and not hsp_preselection:
            raise ValueError(
                "The paper population requires a 15-policy MEP catalog. Provide "
                "--mep-catalog or explicitly acknowledge the proxy with "
                "--allow-hsp-only."
            )
    else:
        mep, mep_payload, mep_path = _load_catalog(
            args.mep_catalog, expected_population_type="mep"
        )
        payloads.append(mep_payload)
        source_paths.append(mep_path)
    return hsp, mep, payloads, source_paths, hsp_only


def _build_pairs(args: argparse.Namespace) -> Path:
    hsp, mep, payloads, source_paths, hsp_only = _load_population_inputs(args)
    response_paths, response_payloads = _merge_response_results(
        args.response_results,
        hsp=hsp,
        mep=mep,
    )
    payloads.extend(response_payloads)
    layout = _resolve_layout(args.layout, payloads)
    exclusions = _load_exclusions(args.exclude_id, args.exclude_ids_file)
    unknown_exclusions = exclusions.difference(hsp)
    if unknown_exclusions:
        raise ValueError(
            "Excluded HSP ids are not in the catalog: "
            f"{sorted(unknown_exclusions, key=_natural_key)}"
        )
    hsp_after_explicit_exclusions = [
        candidate
        for identifier, candidate in hsp.items()
        if identifier not in exclusions
    ]
    eligible_hsp, return_filtered_ids = _filter_scored_hsp_by_return(
        hsp_after_explicit_exclusions,
        minimum_return=args.minimum_hsp_reference_return,
        context="build-pairs HSP selection",
    )
    selected_hsp = greedy_normalized_l1_select(
        eligible_hsp, count=args.hsp_count, seed=args.selector_seed
    )

    if not hsp_only and len(mep) != args.mep_count:
        raise ValueError(
            f"MEP catalog must contain exactly --mep-count={args.mep_count} entries; "
            f"found {len(mep)}"
        )
    selected_mep = sorted(
        mep.values(), key=lambda candidate: _natural_key(candidate.identifier)
    )

    for candidate in selected_hsp:
        _assert_skills(candidate, ("mid", "final"), responses=True)
    for candidate in selected_mep:
        _assert_skills(candidate, ("final",), responses=True)
    if args.verify_checkpoints:
        _verify_policy_checkpoints(
            [*selected_hsp, *selected_mep], include_responses=True
        )

    output = args.output.expanduser().resolve()
    output_dir = output.parent
    pairs: list[dict[str, Any]] = []
    for candidate in selected_hsp:
        pair = {
            "id": _pair_identifier(candidate),
            "split": "train",
            "population_type": "hsp",
            "num_rollouts": HSP_MID_ROLLOUTS + HSP_FINAL_ROLLOUTS,
            "partner": _serialize_policy(
                candidate.partner["final"], output_dir=output_dir
            ),
            "best_response": _serialize_policy(
                candidate.best_response["final"], output_dir=output_dir
            ),
            "rollout_variants": [
                {
                    "name": "mid",
                    "weight": HSP_MID_ROLLOUTS
                    / (HSP_MID_ROLLOUTS + HSP_FINAL_ROLLOUTS),
                    "num_rollouts": HSP_MID_ROLLOUTS,
                    # PORTING NOTE: CooT's intermediate HSP rollout is a
                    # matched pair.  Both checkpoints must be overridden;
                    # mixing a mid partner with the final BR is not faithful.
                    "partner": _serialize_policy(
                        candidate.partner["mid"], output_dir=output_dir
                    ),
                    "best_response": _serialize_policy(
                        candidate.best_response["mid"], output_dir=output_dir
                    ),
                },
                {
                    "name": "final",
                    "weight": HSP_FINAL_ROLLOUTS
                    / (HSP_MID_ROLLOUTS + HSP_FINAL_ROLLOUTS),
                    "num_rollouts": HSP_FINAL_ROLLOUTS,
                },
            ],
        }
        if candidate.reference_return is not None:
            pair["reference_return"] = candidate.reference_return
        pairs.append(pair)
    for candidate in selected_mep:
        pair = {
            "id": _pair_identifier(candidate),
            "split": "train",
            "population_type": "mep",
            "num_rollouts": MEP_FINAL_ROLLOUTS,
            "partner": _serialize_policy(
                candidate.partner["final"], output_dir=output_dir
            ),
            "best_response": _serialize_policy(
                candidate.best_response["final"], output_dir=output_dir
            ),
        }
        if candidate.reference_return is not None:
            pair["reference_return"] = candidate.reference_return
        pairs.append(pair)
    pair_ids = [pair["id"] for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"Generated pair ids are not unique: {pair_ids}")

    manifest = {
        "format_version": 1,
        "kind": "coot_pair_manifest",
        "layout": layout,
        "checkpoint_path_base": "manifest_parent",
        "population_construction": _construction_metadata(
            selected_hsp=selected_hsp,
            selected_mep=selected_mep,
            eligible_hsp_count=len(eligible_hsp),
            exclusions=exclusions,
            return_filtered_ids=return_filtered_ids,
            minimum_hsp_reference_return=args.minimum_hsp_reference_return,
            seed=args.selector_seed,
            requested_hsp_count=args.hsp_count,
            requested_mep_count=args.mep_count,
            hsp_only=hsp_only,
        ),
        "provenance": {
            "catalogs": [str(path) for path in source_paths],
            "response_results": [str(path) for path in response_paths],
        },
        "pairs": pairs,
    }
    return _atomic_write_json(manifest, output, overwrite=args.overwrite)


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not normalized:
        raise ValueError(f"Cannot derive a filesystem-safe name from id {value!r}")
    return normalized


def _response_jobs(args: argparse.Namespace) -> Path:
    hsp, mep, payloads, source_paths, hsp_only = _load_population_inputs(args)
    layout = _resolve_layout(args.layout, payloads)
    exclusions = _load_exclusions(args.exclude_id, args.exclude_ids_file)
    unknown_exclusions = exclusions.difference(hsp)
    if unknown_exclusions:
        raise ValueError(
            "Excluded HSP ids are not in the HSP catalog: "
            f"{sorted(unknown_exclusions, key=_natural_key)}"
        )
    if not hsp_only and len(mep) != args.mep_count:
        raise ValueError(
            f"MEP catalog must contain exactly --mep-count={args.mep_count} entries; "
            f"found {len(mep)}"
        )

    hsp_after_explicit_exclusions = [
        candidate
        for identifier, candidate in hsp.items()
        if identifier not in exclusions
    ]
    if args.all_hsp_candidates:
        # This preselection stage intentionally precedes the BR rollout that
        # produces reference_return, so every explicit candidate is scheduled.
        eligible_hsp = hsp_after_explicit_exclusions
        return_filtered_ids: list[str] = []
        selected_hsp = sorted(
            eligible_hsp,
            key=lambda candidate: _natural_key(candidate.identifier),
        )
    else:
        eligible_hsp, return_filtered_ids = _filter_scored_hsp_by_return(
            hsp_after_explicit_exclusions,
            minimum_return=args.minimum_hsp_reference_return,
            context="response-jobs HSP selection",
        )
        selected_hsp = greedy_normalized_l1_select(
            eligible_hsp,
            count=args.hsp_count,
            seed=args.selector_seed,
        )
    ordered_mep = sorted(
        mep.values(), key=lambda candidate: _natural_key(candidate.identifier)
    )
    requested_hsp_skills = set(args.hsp_skill or ("mid", "final"))
    hsp_skills = tuple(
        skill for skill in ("mid", "final") if skill in requested_hsp_skills
    )
    for candidate in selected_hsp:
        _assert_skills(candidate, hsp_skills, responses=False)
    for candidate in ordered_mep:
        _assert_skills(candidate, ("final",), responses=False)
    if args.verify_checkpoints:
        _verify_policy_checkpoints(
            [*selected_hsp, *ordered_mep], include_responses=False
        )

    output = args.output.expanduser().resolve()
    output_dir = output.parent
    response_root = args.response_output_root.expanduser()
    if not response_root.is_absolute():
        response_root = (Path.cwd() / response_root).resolve()
    jobs: list[dict[str, Any]] = []
    seen_job_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for candidate in [*selected_hsp, *ordered_mep]:
        skills = hsp_skills if candidate.population_type == "hsp" else ("final",)
        for skill in skills:
            safe_id = _safe_identifier(candidate.identifier)
            job_id = f"{candidate.population_type}__{safe_id}__{skill}"
            expected_response = (
                response_root
                / layout
                / candidate.population_type
                / f"{safe_id}_{skill}.safetensors"
            )
            expected_relative = _relative_checkpoint(
                str(expected_response), output_dir=output_dir
            )
            if job_id in seen_job_ids:
                raise ValueError(f"Generated duplicate response job id: {job_id}")
            if expected_relative in seen_outputs:
                raise ValueError(
                    "Response checkpoint path collision after id sanitization: "
                    f"{expected_relative}"
                )
            seen_job_ids.add(job_id)
            seen_outputs.add(expected_relative)
            # train_bias_agents_br.sh uses --seed ${i} for its 1-based job
            # loop. Keep that initialization contract inside the atomic job
            # instead of applying one sweep-global seed to every response.
            response_seed = len(jobs) + 1
            partner_policy = _serialize_policy(
                candidate.partner[skill], output_dir=output_dir
            )
            partner_checkpoint = str(partner_policy["checkpoint"])
            jobs.append(
                {
                    "job_id": job_id,
                    "layout": layout,
                    "partner_id": candidate.identifier,
                    "population_type": candidate.population_type,
                    "skill": skill,
                    "response_seed": response_seed,
                    # Keep the scalar checkpoint field for sweep tools while
                    # also exposing the full mapping consumed by the response
                    # trainer (architecture metadata must not be discarded).
                    "partner_checkpoint": partner_checkpoint,
                    "partner": partner_policy,
                    "best_response_checkpoint": expected_relative,
                    "best_response_policy": {
                        # The FCP BR trainer uses one network tree for the
                        # trainable response and frozen partner.  Copy the
                        # partner shape exactly; a global 128-wide default
                        # breaks HSP checkpoints trained with width 64.
                        "architecture": partner_policy["architecture"],
                        "activation": partner_policy["activation"],
                        "fc_dim_size": partner_policy["fc_dim_size"],
                        "gru_hidden_dim": partner_policy["gru_hidden_dim"],
                        # Supplementary collection and behavior scoring sample
                        # actions from both the partner and its response.
                        "stochastic": True,
                    },
                    "status": "pending",
                    "atomic": True,
                }
            )

    hsp_candidate_preselection = bool(args.all_hsp_candidates and hsp_only)
    deviations = _deviations(
        hsp_count=args.hsp_count,
        mep_count=args.mep_count,
        seed=args.selector_seed,
        # A final-only BR sweep over every HSP candidate precedes selection;
        # it is not itself the final paper population and does not require MEP.
        hsp_only=hsp_only and not hsp_candidate_preselection,
        minimum_hsp_reference_return=args.minimum_hsp_reference_return,
    )
    if args.all_hsp_candidates:
        deviations.append(
            {
                "code": "all_hsp_response_candidates",
                "description": (
                    "Preselection generated responses for every eligible HSP "
                    "candidate so response event counts can drive greedy selection."
                ),
                "used_count": len(selected_hsp),
                "paper_selected_count": PAPER_HSP_COUNT,
            }
        )
    if hsp_candidate_preselection:
        population_mode = "hsp_candidate_preselection"
    elif hsp_only:
        population_mode = "hsp_only_proxy"
    else:
        population_mode = "hsp_plus_mep"
    manifest = {
        "format_version": 1,
        "kind": "coot_response_jobs",
        "layout": layout,
        "schema": {
            "job_atomicity": (
                "Each job trains one best response against exactly one partner "
                "checkpoint at one skill level."
            ),
            "checkpoint_paths": (
                "All relative partner_checkpoint and best_response_checkpoint paths "
                "are resolved against this response-job manifest's parent directory."
            ),
            "completion": (
                "Set status to completed after writing best_response_checkpoint; the "
                "completed manifest can be passed to build-pairs --response-results."
            ),
        },
        "population_construction": {
            "paper_target": {
                "hsp": PAPER_HSP_COUNT,
                "mep": PAPER_MEP_COUNT,
                "total": PAPER_HSP_COUNT + PAPER_MEP_COUNT,
            },
            "population_mode": population_mode,
            "hsp_selector": {
                "name": "greedy_normalized_l1",
                "is_dpp": False,
                "seed": args.selector_seed,
                "normalization": "feature / (column_max + 1e-3)",
                "candidate_count_after_exclusions": len(hsp_after_explicit_exclusions),
                "candidate_count_after_return_filter": len(eligible_hsp),
                "minimum_reference_return_exclusive": (
                    None
                    if args.all_hsp_candidates
                    else args.minimum_hsp_reference_return
                ),
                "return_filter_applied": not args.all_hsp_candidates,
                "return_filtered_ids": return_filtered_ids,
                "selection_applied": not args.all_hsp_candidates,
                "selected_count": len(selected_hsp),
                "selected_ids": [candidate.identifier for candidate in selected_hsp],
            },
            "mep": {
                "selected_count": len(ordered_mep),
                "selected_ids": [candidate.identifier for candidate in ordered_mep],
            },
            "excluded_hsp_ids": sorted(exclusions, key=_natural_key),
            "hsp_response_skills": list(hsp_skills),
            "job_count": len(jobs),
            "deviations": deviations,
        },
        "provenance": {"catalogs": [str(path) for path in source_paths]},
        "jobs": jobs,
    }
    return _atomic_write_json(manifest, output, overwrite=args.overwrite)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _add_population_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hsp-catalog", type=Path, required=True)
    parser.add_argument("--mep-catalog", type=Path)
    parser.add_argument("--layout", help="Override/validate the catalog layout.")
    parser.add_argument("--hsp-count", type=_positive_int, default=PAPER_HSP_COUNT)
    parser.add_argument("--mep-count", type=_positive_int, default=PAPER_MEP_COUNT)
    parser.add_argument("--selector-seed", type=int, default=PAPER_SELECTOR_SEED)
    parser.add_argument(
        "--minimum-hsp-reference-return",
        type=_nonnegative_float,
        default=RELEASE_MIN_HSP_REFERENCE_RETURN,
        help=(
            "Exclude HSP candidates whose scored sparse reference return is at "
            "or below this threshold before greedy selection (release: 0.1)."
        ),
    )
    parser.add_argument(
        "--exclude-id",
        action="append",
        default=[],
        help="Exclude one catalog id (repeatable). Primarily for held-out HSP ids.",
    )
    parser.add_argument("--exclude-ids-file", type=Path)
    parser.add_argument(
        "--allow-hsp-only",
        action="store_true",
        help=(
            "Explicitly permit an HSP-only proxy when --mep-catalog is absent. "
            "The generated manifest records this paper deviation. This flag is "
            "not needed for --all-hsp-candidates preselection jobs."
        ),
    )
    parser.add_argument(
        "--verify-checkpoints",
        action="store_true",
        help="Require every consumed input checkpoint to exist locally.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create CooT best-response jobs or a paper-matched HSP+MEP pair manifest."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    jobs = subparsers.add_parser(
        "response-jobs",
        help="Emit one atomic response-training job per candidate checkpoint/skill.",
    )
    _add_population_arguments(jobs)
    jobs.add_argument(
        "--response-output-root",
        type=Path,
        default=Path("saves/coot_responses"),
    )
    jobs.add_argument(
        "--all-hsp-candidates",
        action="store_true",
        help=(
            "Preselection/diagnostic mode: emit the requested --hsp-skill jobs "
            "for every eligible HSP candidate instead of only the "
            "greedy-selected --hsp-count candidates."
        ),
    )
    jobs.add_argument(
        "--hsp-skill",
        action="append",
        choices=("mid", "final"),
        help=(
            "HSP skill checkpoint to schedule (repeatable; default: mid and "
            "final). Use final for the preselection BR stage and mid after "
            "greedy selection. MEP remains final-only."
        ),
    )
    pairs = subparsers.add_parser(
        "build-pairs",
        help="Select 21 HSP candidates, merge 15 MEP candidates, and emit pairs.",
    )
    _add_population_arguments(pairs)
    pairs.add_argument(
        "--response-results",
        type=str,
        action="append",
        default=[],
        help=(
            "Completed atomic response-job/result JSON, directory, or glob "
            "(repeatable). Directories recursively load response_job*.json. "
            "Responses may instead be embedded in a catalog."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "response-jobs":
        output = _response_jobs(args)
    elif args.command == "build-pairs":
        output = _build_pairs(args)
    else:  # pragma: no cover - argparse enforces subcommands
        raise AssertionError(args.command)
    print(output)


if __name__ == "__main__":
    main()
