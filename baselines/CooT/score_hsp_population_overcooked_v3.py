"""Merge HSP sidecars or score candidates with exact V3 event rollouts.

The CooT supplement selects HSP policies from normalized event-count features.
This port obtains those features directly from the response agent's
``info["event_vector"]``. Each candidate's final partner/response pair is
sampled stochastically in both physical-seat permutations. Event vectors are
summed within episodes, averaged within each seat, and concatenated in the
release's partner-seat-0 then partner-seat-1 order. The shared sparse team
return, counted once per step, becomes ``reference_return``.

If a catalog has no ``best_response.final`` policy, the final HSP partner is
used for both agents only as an explicit V3 shared-IPPO fallback.  The release
uses its separated runner and separately extracted actors, so this fallback is
not paper-equivalent and every use is recorded in output metadata.

``--merge-only`` stops after sidecar normalization.  It writes a raw aggregate
catalog (including nullable ``selection_features``) for the all-candidate final
BR stage without creating an environment or loading any checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

import jaxmarl
from jaxmarl.environments.overcooked_v3.events import EVENT_NAMES, NUM_EVENTS

try:
    from .build_population_manifest import _extract_policy_variants
    from .runtime import CheckpointPolicy, PolicySpec
except ImportError:  # Direct execution: python baselines/CooT/<script>.py
    from build_population_manifest import _extract_policy_variants
    from runtime import CheckpointPolicy, PolicySpec


COLLECTION_KEYS = ("candidates", "policies", "entries", "results")
IDENTIFIER_KEYS = ("id", "candidate_id", "profile_id", "partner_id", "run_id")
CHECKPOINT_MAP_KEYS = {
    "checkpoints",
    "checkpoint_paths",
    "actor_checkpoints",
    "partner_checkpoints",
    "best_response_checkpoints",
    "response_checkpoints",
}
POLICY_CONTAINER_KEYS = {
    "partner",
    "partners",
    "best_response",
    "best_responses",
    "response",
    "responses",
}
PROFILE_KEYS = (
    "profile",
    "utility_profile",
    "utility_weights",
    "resolved_utility",
    "reward_weights",
    "candidate_seed",
)
COMPLETED_STATUSES = {"complete", "completed", "finished", "succeeded", "success"}


@dataclass
class CandidateRecord:
    identifier: str
    entry: dict[str, Any]
    partner: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_response: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_paths: list[Path] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _identifier(entry: Mapping[str, Any], *, source: Path) -> str:
    for key in IDENTIFIER_KEYS:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(
        f"Candidate entry in {source} needs one of {', '.join(IDENTIFIER_KEYS)}"
    )


def _catalog_entries(payload: Any, *, source: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = None
        for key in COLLECTION_KEYS:
            if key in payload:
                values = payload[key]
                break
        if values is None and any(key in payload for key in IDENTIFIER_KEYS):
            values = [payload]
        if values is None:
            raise ValueError(
                f"{source} needs a candidate collection ({COLLECTION_KEYS})"
            )
    else:
        raise ValueError(f"{source} must contain a JSON object or list")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Candidate collection in {source} must be a non-empty list")
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError(f"Every candidate in {source} must be a JSON object")
    return list(values)


def _candidate_result_entries(payload: Any, *, source: Path) -> list[Mapping[str, Any]]:
    """Extract candidate entries while allowing unrelated JSON in a directory."""

    try:
        entries = _catalog_entries(payload, source=source)
    except ValueError:
        entries = []
    filtered = [entry for entry in entries if _looks_like_policy_result(entry)]
    if filtered:
        return filtered
    if not isinstance(payload, Mapping):
        return []
    for key in ("candidate", "result", "resolved_candidate"):
        nested = payload.get(key)
        if not isinstance(nested, Mapping):
            continue
        merged = dict(payload)
        merged.pop(key, None)
        merged.update(nested)
        if any(
            identifier_key in merged for identifier_key in IDENTIFIER_KEYS
        ) and _looks_like_policy_result(merged):
            return [merged]
    return []


def _looks_like_policy_result(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if _is_checkpoint_key(key) and isinstance(raw_value, (str, os.PathLike)):
            return True
        if key in CHECKPOINT_MAP_KEYS and isinstance(raw_value, Mapping):
            if any(
                isinstance(item, (str, os.PathLike)) or _looks_like_policy_result(item)
                for item in raw_value.values()
            ):
                return True
        if key in POLICY_CONTAINER_KEYS and isinstance(raw_value, (str, os.PathLike)):
            return True
        if key in POLICY_CONTAINER_KEYS and isinstance(raw_value, Mapping):
            if any(
                (
                    str(variant) in {"mid", "intermediate", "final"}
                    and isinstance(policy, (str, os.PathLike))
                )
                or _looks_like_policy_result(policy)
                for variant, policy in raw_value.items()
            ):
                return True
    return False


def _is_checkpoint_key(key: str) -> bool:
    return (
        key == "checkpoint"
        or key.endswith("_checkpoint")
        or key.endswith("_checkpoint_path")
    )


def _resolve_checkpoint(value: Any, *, base_dir: Path, context: str) -> str:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{context} must be a non-empty checkpoint path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _absolutize_checkpoint_paths(
    value: Any,
    *,
    base_dir: Path,
    context: str,
    checkpoint_map: bool = False,
    policy_variants: bool = False,
) -> Any:
    """Resolve all supported catalog checkpoint aliases against their JSON."""

    if checkpoint_map and isinstance(value, (str, os.PathLike)):
        return _resolve_checkpoint(value, base_dir=base_dir, context=context)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            item_context = f"{context}.{key}"
            if _is_checkpoint_key(key):
                normalized[key] = _resolve_checkpoint(
                    raw_value, base_dir=base_dir, context=item_context
                )
            elif checkpoint_map and isinstance(raw_value, (str, os.PathLike)):
                normalized[key] = _resolve_checkpoint(
                    raw_value, base_dir=base_dir, context=item_context
                )
            elif (
                policy_variants
                and key in {"mid", "intermediate", "final"}
                and isinstance(raw_value, (str, os.PathLike))
            ):
                normalized[key] = _resolve_checkpoint(
                    raw_value, base_dir=base_dir, context=item_context
                )
            else:
                normalized[key] = _absolutize_checkpoint_paths(
                    raw_value,
                    base_dir=base_dir,
                    context=item_context,
                    checkpoint_map=(
                        key in CHECKPOINT_MAP_KEYS
                        or (checkpoint_map and isinstance(raw_value, list))
                    ),
                    policy_variants=key in POLICY_CONTAINER_KEYS,
                )
        return normalized
    if isinstance(value, list):
        return [
            _absolutize_checkpoint_paths(
                item,
                base_dir=base_dir,
                context=f"{context}[{index}]",
                checkpoint_map=checkpoint_map,
                policy_variants=policy_variants,
            )
            for index, item in enumerate(value)
        ]
    return value


def _serialize_checkpoint_paths(
    value: Any,
    *,
    output_dir: Path,
    checkpoint_map: bool = False,
    policy_variants: bool = False,
) -> Any:
    if checkpoint_map and isinstance(value, (str, os.PathLike)):
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ValueError(f"Internal checkpoint path was not normalized: {value}")
        return Path(os.path.relpath(path, output_dir)).as_posix()
    if isinstance(value, Mapping):
        serialized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            is_path = _is_checkpoint_key(key)
            is_map_path = checkpoint_map and isinstance(raw_value, (str, os.PathLike))
            is_variant_path = (
                policy_variants
                and key in {"mid", "intermediate", "final"}
                and isinstance(raw_value, (str, os.PathLike))
            )
            if is_path or is_map_path or is_variant_path:
                path = Path(str(raw_value)).expanduser()
                if not path.is_absolute():
                    raise ValueError(
                        f"Internal checkpoint path was not normalized: {raw_value}"
                    )
                serialized[key] = Path(os.path.relpath(path, output_dir)).as_posix()
            else:
                serialized[key] = _serialize_checkpoint_paths(
                    raw_value,
                    output_dir=output_dir,
                    checkpoint_map=(
                        key in CHECKPOINT_MAP_KEYS
                        or (checkpoint_map and isinstance(raw_value, list))
                    ),
                    policy_variants=key in POLICY_CONTAINER_KEYS,
                )
        return serialized
    if isinstance(value, list):
        return [
            _serialize_checkpoint_paths(
                item,
                output_dir=output_dir,
                checkpoint_map=checkpoint_map,
                policy_variants=policy_variants,
            )
            for item in value
        ]
    return value


def _canonicalize_checkpoint_aliases(entry: dict[str, Any]) -> None:
    """Translate common per-run sidecar checkpoint names for the builder."""

    skill = str(
        entry.get("skill", entry.get("skill_level", entry.get("variant", "final")))
    ).lower()
    skill = "mid" if skill in {"mid", "middle", "intermediate"} else "final"

    if "partner" not in entry:
        for container_key in ("checkpoints", "checkpoint_paths", "actor_checkpoints"):
            if isinstance(entry.get(container_key), Mapping):
                entry["partner"] = copy.deepcopy(entry[container_key])
                break

    for raw_skill, normalized_skill in (
        ("mid", "mid"),
        ("intermediate", "mid"),
        ("final", "final"),
    ):
        for suffix in ("checkpoint", "checkpoint_path"):
            alias = f"{raw_skill}_{suffix}"
            if alias in entry:
                entry[f"partner_{normalized_skill}_checkpoint"] = entry[alias]
                break
    if "partner" not in entry and not any(
        key in entry for key in ("partner_mid_checkpoint", "partner_final_checkpoint")
    ):
        for alias in ("policy_checkpoint", "checkpoint"):
            if alias in entry:
                entry[f"partner_{skill}_checkpoint"] = entry[alias]
                break

    for role in ("partner", "best_response", "response"):
        value = entry.get(role)
        if not isinstance(value, Mapping):
            continue
        for variant in ("mid", "intermediate", "final"):
            policy = value.get(variant)
            if isinstance(policy, Mapping) and "checkpoint" not in policy:
                if "path" in policy:
                    policy = dict(policy)
                    policy["checkpoint"] = policy["path"]
                    value = dict(value)
                    value[variant] = policy
                    entry[role] = value


def _policy_variants(
    entry: Mapping[str, Any], *, role: str, source: Path, identifier: str
) -> dict[str, dict[str, Any]]:
    prepared = copy.deepcopy(dict(entry))
    container_keys = (
        ("partner", "partners", "partner_checkpoints")
        if role == "partner"
        else (
            "best_response",
            "best_responses",
            "response",
            "responses",
            "best_response_checkpoints",
        )
    )
    for key in container_keys:
        value = prepared.get(key)
        if value is None:
            prepared.pop(key, None)
        elif isinstance(value, Mapping):
            prepared[key] = {
                variant: policy
                for variant, policy in value.items()
                if policy is not None
            }
    return _extract_policy_variants(
        prepared,
        role=role,
        base_dir=source.parent,
        context=f"{source}:{identifier}",
    )


def _merge_policy_variants(
    target: dict[str, dict[str, Any]],
    incoming: Mapping[str, dict[str, Any]],
    *,
    context: str,
) -> None:
    for skill, policy in incoming.items():
        if skill in target and target[skill] != policy:
            raise ValueError(
                f"Conflicting {context}/{skill} checkpoints or policy metadata"
            )
        target[skill] = dict(policy)


def _new_record(entry: Mapping[str, Any], *, source: Path) -> CandidateRecord:
    normalized_entry = _absolutize_checkpoint_paths(
        copy.deepcopy(dict(entry)),
        base_dir=source.parent,
        context=str(source),
    )
    _canonicalize_checkpoint_aliases(normalized_entry)
    identifier = _identifier(normalized_entry, source=source)
    population_type = str(normalized_entry.get("population_type", "hsp")).lower()
    if population_type != "hsp":
        raise ValueError(
            f"Scorer only accepts HSP candidates; {source}:{identifier} is "
            f"population_type={population_type!r}"
        )
    normalized_entry.setdefault("id", identifier)
    normalized_entry.setdefault("population_type", "hsp")
    return CandidateRecord(
        identifier=identifier,
        entry=normalized_entry,
        partner=_policy_variants(
            normalized_entry,
            role="partner",
            source=source,
            identifier=identifier,
        ),
        best_response=_policy_variants(
            normalized_entry,
            role="best_response",
            source=source,
            identifier=identifier,
        ),
        source_paths=[source],
    )


def _merge_sidecar_entry(
    record: CandidateRecord, entry: Mapping[str, Any], *, source: Path
) -> None:
    normalized = _absolutize_checkpoint_paths(
        copy.deepcopy(dict(entry)),
        base_dir=source.parent,
        context=str(source),
    )
    _canonicalize_checkpoint_aliases(normalized)
    identifier = _identifier(normalized, source=source)
    if identifier != record.identifier:
        raise ValueError(
            f"Cannot merge sidecar id {identifier!r} into {record.identifier!r}"
        )
    population_type = str(normalized.get("population_type", "hsp")).lower()
    if population_type != "hsp":
        raise ValueError(f"Sidecar {source} is not an HSP candidate result")
    _merge_policy_variants(
        record.partner,
        _policy_variants(
            normalized,
            role="partner",
            source=source,
            identifier=identifier,
        ),
        context=f"{identifier}/partner",
    )
    _merge_policy_variants(
        record.best_response,
        _policy_variants(
            normalized,
            role="best_response",
            source=source,
            identifier=identifier,
        ),
        context=f"{identifier}/best_response",
    )
    for key in PROFILE_KEYS:
        if key not in normalized:
            continue
        if key in record.entry and record.entry[key] != normalized[key]:
            raise ValueError(f"Conflicting {key} for HSP candidate {identifier}")
        record.entry[key] = copy.deepcopy(normalized[key])
    record.source_paths.append(source)


def _root_layouts(payload: Any) -> set[str]:
    layouts: set[str] = set()
    if not isinstance(payload, Mapping):
        return layouts
    if payload.get("layout"):
        layouts.add(str(payload["layout"]))
    for key in ("original_job", "resolved_job"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested.get("layout"):
            layouts.add(str(nested["layout"]))
    return layouts


def _entry_layout(entry: Mapping[str, Any]) -> str | None:
    value = entry.get("layout")
    return str(value) if value is not None else None


def _has_glob_magic(value: str) -> bool:
    return glob.has_magic(value)


def _expand_candidate_result_sources(values: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_value in values:
        expanded_value = os.path.expanduser(raw_value)
        raw_path = Path(expanded_value)
        if _has_glob_magic(expanded_value):
            matches = [
                Path(value) for value in glob.glob(expanded_value, recursive=True)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"--candidate-result glob matched no files: {raw_value}"
                )
        elif raw_path.is_dir():
            matches = sorted(raw_path.rglob("*candidate*.json"))
            if not matches:
                raise FileNotFoundError(
                    "--candidate-result directory has no recursive "
                    f"*candidate*.json files: {raw_path}"
                )
        else:
            matches = [raw_path]
        for match in matches:
            path = match.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Candidate result JSON not found: {path}")
            if path not in seen:
                seen.add(path)
                resolved.append(path)
    return resolved


def _expand_response_result_sources(values: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_value in values:
        expanded_value = os.path.expanduser(raw_value)
        raw_path = Path(expanded_value)
        if _has_glob_magic(expanded_value):
            matches = [
                Path(value) for value in glob.glob(expanded_value, recursive=True)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"--response-result glob matched no files: {raw_value}"
                )
        elif raw_path.is_dir():
            matches = sorted(raw_path.rglob("response_job*.json"))
            if not matches:
                raise FileNotFoundError(
                    "--response-result directory has no recursive "
                    f"response_job*.json files: {raw_path}"
                )
        else:
            matches = [raw_path]
        for match in matches:
            path = match.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Response result JSON not found: {path}")
            if path not in seen:
                seen.add(path)
                resolved.append(path)
    return resolved


def _response_entries(payload: Any, *, source: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("jobs"), list):
        entries = payload["jobs"]
    elif isinstance(payload, Mapping):
        entries = [payload]
    else:
        raise ValueError(f"Response result {source} must be a JSON object or list")
    if not entries or not all(isinstance(entry, Mapping) for entry in entries):
        raise ValueError(f"Response entries in {source} must be JSON objects")
    return list(entries)


def _flatten_response_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    original_job = entry.get("original_job")
    if isinstance(original_job, Mapping):
        flattened.update(original_job)
    resolved_job = entry.get("resolved_job")
    if isinstance(resolved_job, Mapping):
        flattened.update(resolved_job)
    flattened.update(entry)
    return flattened


def _merge_response_result(
    records: Mapping[str, CandidateRecord],
    raw_entry: Mapping[str, Any],
    *,
    source: Path,
) -> str | None:
    status = raw_entry.get("status")
    if status is not None and str(status).lower() not in COMPLETED_STATUSES:
        return None
    flattened = _flatten_response_entry(raw_entry)
    identifier_value = flattened.get(
        "partner_id", flattened.get("candidate_id", flattened.get("id"))
    )
    if identifier_value is None or not str(identifier_value).strip():
        raise ValueError(f"Response result {source} has no partner_id")
    identifier = str(identifier_value).strip()
    if identifier not in records:
        raise ValueError(
            f"Response result {source} references unknown HSP partner {identifier!r}"
        )
    population_type = str(flattened.get("population_type", "hsp")).lower()
    if population_type != "hsp":
        raise ValueError(
            f"Response result {source} is {population_type!r}, not an HSP BR"
        )

    # The trainer's top-level checkpoint/mapping is authoritative.  Do not
    # reuse original_job.best_response_checkpoint: it is only the planned
    # output path from the pre-training job manifest.
    top_response = raw_entry.get("best_response")
    top_checkpoint = raw_entry.get("best_response_checkpoint")
    if isinstance(top_response, Mapping):
        response_value = dict(top_response)
        if top_checkpoint is not None:
            response_value["checkpoint"] = top_checkpoint
    elif top_checkpoint is not None:
        response_value = {"checkpoint": top_checkpoint}
        for key in (
            "architecture",
            "activation",
            "fc_dim_size",
            "gru_hidden_dim",
            "stochastic",
        ):
            if key in raw_entry:
                response_value[key] = raw_entry[key]
    else:
        raise ValueError(
            f"Completed response result {source} needs a top-level "
            "best_response_checkpoint"
        )
    raw_skill = str(
        flattened.get(
            "skill", flattened.get("skill_level", flattened.get("variant", "final"))
        )
    ).lower()
    aliases = {"middle": "mid", "intermediate": "mid", "last": "final"}
    skill = aliases.get(raw_skill, raw_skill)
    if skill not in {"mid", "final"}:
        raise ValueError(f"Response result {source} has invalid skill {raw_skill!r}")
    response_entry = {
        "id": identifier,
        "population_type": "hsp",
        "skill": skill,
        "best_response": response_value,
    }
    normalized = _absolutize_checkpoint_paths(
        response_entry,
        base_dir=source.parent,
        context=str(source),
    )
    incoming = _policy_variants(
        normalized,
        role="best_response",
        source=source,
        identifier=identifier,
    )
    _merge_policy_variants(
        records[identifier].best_response,
        incoming,
        context=f"{identifier}/best_response",
    )
    records[identifier].source_paths.append(source)
    return identifier


def _load_population(
    catalog_path: Path | None,
    candidate_result_values: Sequence[str],
    response_result_values: Sequence[str],
    *,
    layout: str,
) -> tuple[
    dict[str, Any],
    list[CandidateRecord],
    list[Path],
    list[Path],
]:
    payload: Any = {}
    catalog: Path | None = None
    declared_layouts: set[str] = set()
    records: dict[str, CandidateRecord] = {}
    order: list[str] = []
    if catalog_path is not None:
        catalog = catalog_path.expanduser().resolve()
        if not catalog.is_file():
            raise FileNotFoundError(f"HSP catalog not found: {catalog}")
        payload = _read_json(catalog)
        declared_layouts.update(_root_layouts(payload))
        for raw_entry in _catalog_entries(payload, source=catalog):
            entry_layout = _entry_layout(raw_entry)
            if entry_layout:
                declared_layouts.add(entry_layout)
            record = _new_record(raw_entry, source=catalog)
            if record.identifier in records:
                raise ValueError(
                    f"Duplicate candidate id in {catalog}: {record.identifier}"
                )
            records[record.identifier] = record
            order.append(record.identifier)

    result_paths = _expand_candidate_result_sources(candidate_result_values)
    consumed_results: list[Path] = []
    for result_path in result_paths:
        result_payload = _read_json(result_path)
        result_entries = _candidate_result_entries(result_payload, source=result_path)
        if not result_entries:
            # Directories/globs can contain configs and W&B metadata.  Only
            # candidate-shaped JSON sidecars participate in the merge.
            continue
        consumed_results.append(result_path)
        declared_layouts.update(_root_layouts(result_payload))
        for raw_entry in result_entries:
            entry_layout = _entry_layout(raw_entry)
            if entry_layout:
                declared_layouts.add(entry_layout)
            normalized_probe = _absolutize_checkpoint_paths(
                copy.deepcopy(dict(raw_entry)),
                base_dir=result_path.parent,
                context=str(result_path),
            )
            identifier = _identifier(normalized_probe, source=result_path)
            if identifier not in records:
                record = _new_record(raw_entry, source=result_path)
                records[identifier] = record
                order.append(identifier)
            else:
                _merge_sidecar_entry(records[identifier], raw_entry, source=result_path)

    consumed_responses: list[Path] = []
    for response_path in _expand_response_result_sources(response_result_values):
        response_payload = _read_json(response_path)
        declared_layouts.update(_root_layouts(response_payload))
        merged_any = False
        for raw_entry in _response_entries(response_payload, source=response_path):
            declared_layouts.update(_root_layouts(raw_entry))
            merged_identifier = _merge_response_result(
                records,
                raw_entry,
                source=response_path,
            )
            merged_any = merged_any or merged_identifier is not None
        if merged_any:
            consumed_responses.append(response_path)
    if declared_layouts and declared_layouts != {layout}:
        raise ValueError(
            f"--layout={layout!r} conflicts with catalog/sidecar layouts "
            f"{sorted(declared_layouts)!r}"
        )
    return (
        copy.deepcopy(dict(payload)) if isinstance(payload, Mapping) else {},
        [records[identifier] for identifier in order],
        consumed_results,
        consumed_responses,
    )


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _policy_identity(spec: PolicySpec) -> dict[str, Any]:
    return {
        "checkpoint": _file_identity(spec.checkpoint),
        "architecture": spec.architecture,
        "activation": spec.activation,
        "fc_dim_size": spec.fc_dim_size,
        "gru_hidden_dim": spec.gru_hidden_dim,
    }


def _candidate_seed(seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _cache_path(cache_dir: Path, identifier: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("._") or "candidate"
    suffix = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{slug}-{suffix}.json"


def _atomic_write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_cached_result(
    path: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or payload.get("identity") != identity:
        return None
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    features = result.get("selection_features")
    if not isinstance(features, list) or len(features) != 2 * NUM_EVENTS:
        return None
    if result.get("reference_return") is None:
        return None
    return dict(result)


def _rollout_candidate(
    env: Any,
    env_step: Any,
    *,
    partner_spec: PolicySpec,
    response_spec: PolicySpec,
    episodes: int,
    max_steps: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    action_dim = int(env.action_space("agent_0").n)
    partner = CheckpointPolicy(replace(partner_spec, stochastic=True), action_dim)
    response = CheckpointPolicy(replace(response_spec, stochastic=True), action_dim)
    selection_features: list[float] = []
    episode_event_counts: list[list[float]] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    seat_results: list[dict[str, Any]] = []

    # Release fidelity: gen_hsp_S2_ymls.py keeps a separate response-event
    # block for each physical seat occupied by w0.  For two agents its stable
    # feature order is [partner@0/response@1 events, partner@1/response@0
    # events], not one seat average.  Each permutation receives the requested
    # number of stochastic episodes.
    for partner_seat in (0, 1):
        response_seat = 1 - partner_seat
        partner_agent = f"agent_{partner_seat}"
        response_agent = f"agent_{response_seat}"
        key = jax.random.fold_in(jax.random.PRNGKey(evaluation_seed), partner_seat)
        seat_event_counts: list[list[float]] = []
        seat_returns: list[float] = []
        seat_lengths: list[int] = []

        for _episode in range(episodes):
            key, reset_key = jax.random.split(key)
            observations, state = env.reset(reset_key)
            partner.reset()
            response.reset()
            event_count = np.zeros((NUM_EVENTS,), dtype=np.float64)
            episode_return = 0.0

            for step in range(max_steps):
                key, partner_key, response_key, step_key = jax.random.split(key, 4)
                partner_action, _ = partner.act(
                    observations[partner_agent], partner_key
                )
                response_action, _ = response.act(
                    observations[response_agent], response_key
                )
                actions = {
                    partner_agent: partner_action,
                    response_agent: response_action,
                }
                observations, state, rewards, dones, info = env_step(
                    step_key,
                    state,
                    actions,
                )
                try:
                    step_events = info["event_vector"][response_agent]
                except (KeyError, TypeError) as exc:
                    raise KeyError(
                        "Overcooked V3 step info must expose a response-seat "
                        f"event vector at info['event_vector']['{response_agent}']"
                    ) from exc
                step_events_array = np.asarray(
                    jax.device_get(step_events), dtype=np.float64
                )
                if step_events_array.shape != (NUM_EVENTS,):
                    raise ValueError(
                        f"Unexpected {response_agent} event vector shape: "
                        f"{step_events_array.shape}; expected {(NUM_EVENTS,)}"
                    )
                event_count += step_events_array
                # V3 exposes one shared sparse team reward under both keys;
                # count it once rather than doubling the team return.
                episode_return += float(rewards[response_agent])
                if bool(dones["__all__"]):
                    break

            seat_event_counts.append(event_count.tolist())
            seat_returns.append(episode_return)
            seat_lengths.append(step + 1)

        seat_feature = np.mean(
            np.asarray(seat_event_counts, dtype=np.float64), axis=0
        ).tolist()
        selection_features.extend(seat_feature)
        episode_event_counts.extend(seat_event_counts)
        episode_returns.extend(seat_returns)
        episode_lengths.extend(seat_lengths)
        seat_results.append(
            {
                "partner_seat": partner_seat,
                "response_seat": response_seat,
                "selection_features": seat_feature,
                "reference_return": float(np.mean(seat_returns)),
                "return_std": float(np.std(seat_returns)),
                "mean_episode_length": float(np.mean(seat_lengths)),
                "episode_event_counts": seat_event_counts,
                "episode_returns": seat_returns,
                "episode_lengths": seat_lengths,
            }
        )

    return {
        "selection_features": selection_features,
        "reference_return": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episodes_per_seat": episodes,
        "seat_results": seat_results,
        "episode_event_counts": episode_event_counts,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }


def _policy_spec(policy: Mapping[str, Any]) -> PolicySpec:
    return PolicySpec.from_mapping(policy, base_dir=Path("/"))


def _output_root(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(payload))
    for key in COLLECTION_KEYS:
        output.pop(key, None)
    return output


def _base_output_entry(record: CandidateRecord, *, output_dir: Path) -> dict[str, Any]:
    output_entry = copy.deepcopy(record.entry)
    output_entry["partner"] = copy.deepcopy(record.partner)
    if record.best_response:
        output_entry["best_response"] = copy.deepcopy(record.best_response)
    output_entry.setdefault("selection_features", None)
    return _serialize_checkpoint_paths(output_entry, output_dir=output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge HSP candidate sidecars, then optionally roll out final policies "
            "in Overcooked V3 and write event-count selection features."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        help=(
            "Optional aggregate HSP catalog. Omit when --candidate-result "
            "sidecars contain the full candidate population."
        ),
    )
    parser.add_argument(
        "--candidate-result",
        action="append",
        default=[],
        help=(
            "Candidate sidecar JSON, directory, or glob (repeatable). Directories "
            "are searched recursively for *candidate*.json files."
        ),
    )
    parser.add_argument(
        "--response-result",
        action="append",
        default=[],
        help=(
            "Completed response_job JSON, directory, or glob (repeatable). "
            "Directories are searched recursively for response_job*.json."
        ),
    )
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Only normalize/merge sidecars and rebase checkpoints. Do not create "
            "the V3 environment, load checkpoints, score features, or use cache."
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.merge_only:
        if args.episodes < 1:
            raise ValueError("--episodes must be positive")
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
    if args.catalog is None and not args.candidate_result:
        raise ValueError("Provide --catalog and/or at least one --candidate-result")

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output catalog already exists: {output_path}; pass --overwrite"
        )
    root_payload, records, consumed_results, consumed_responses = _load_population(
        args.catalog,
        args.candidate_result,
        args.response_result,
        layout=args.layout,
    )
    if not records:
        raise ValueError("HSP catalog contains no candidates")

    if args.merge_only:
        merged_entries = [
            _base_output_entry(record, output_dir=output_path.parent)
            for record in records
        ]
        output_payload = _output_root(root_payload)
        output_payload.update(
            {
                "format_version": 1,
                "kind": "coot_merged_hsp_catalog",
                "layout": args.layout,
                "checkpoint_path_base": "manifest_parent",
                "merge": {
                    "merge_only": True,
                    "candidate_count": len(merged_entries),
                    "selection_features_may_be_null": True,
                    "environment_created": False,
                    "checkpoints_loaded": False,
                },
                "provenance": {
                    "catalog": (
                        str(args.catalog.expanduser().resolve())
                        if args.catalog is not None
                        else None
                    ),
                    "candidate_results": [str(path) for path in consumed_results],
                    "response_results": [str(path) for path in consumed_responses],
                },
                "candidates": merged_entries,
            }
        )
        _atomic_write_json(output_path, output_payload, overwrite=args.overwrite)
        print(output_path)
        return

    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else output_path.with_suffix(f"{output_path.suffix}.cache")
    )

    env = jaxmarl.make(
        "overcooked_v3",
        layout=args.layout,
        max_steps=args.max_steps,
        random_agent_positions=False,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        include_event_vector=True,
        transition_warning_steps=20,
    )
    env_step = jax.jit(env.step_env)
    scored_entries: list[dict[str, Any]] = []
    fallback_ids: list[str] = []

    for index, record in enumerate(records):
        if "final" not in record.partner:
            raise ValueError(
                f"HSP candidate {record.identifier!r} has no partner.final checkpoint"
            )
        partner_spec = _policy_spec(record.partner["final"])
        if "final" in record.best_response:
            response_spec = _policy_spec(record.best_response["final"])
            response_source = "best_response.final"
        else:
            # PORTING NOTE: the release uses the separated runner and separate
            # actor extraction.  Reusing this port's shared-IPPO HSP checkpoint
            # for agent_1 is only a V3 emergency fallback, not paper-equivalent.
            response_spec = partner_spec
            response_source = "partner.final_v3_shared_ippo_approximation"
            fallback_ids.append(record.identifier)

        evaluation_seed = _candidate_seed(args.seed, record.identifier)
        partner_identity = _policy_identity(partner_spec)
        response_identity = (
            partner_identity
            if response_spec == partner_spec
            else _policy_identity(response_spec)
        )
        identity = {
            "schema_version": 2,
            "candidate_id": record.identifier,
            "layout": args.layout,
            "episodes_per_seat": args.episodes,
            "seat_permutations": [
                {"partner_seat": 0, "response_seat": 1},
                {"partner_seat": 1, "response_seat": 0},
            ],
            "selection_feature_order": [
                f"partner_seat_{partner_seat}/response_seat_{1 - partner_seat}/{name}"
                for partner_seat in (0, 1)
                for name in EVENT_NAMES
            ],
            "max_steps": args.max_steps,
            "seed": args.seed,
            "candidate_evaluation_seed": evaluation_seed,
            "stochastic": True,
            "event_source": "info.event_vector[response_agent]",
            "partner": partner_identity,
            "response": response_identity,
            "response_source": response_source,
        }
        candidate_cache = _cache_path(cache_dir, record.identifier)
        result = _load_cached_result(candidate_cache, identity) if args.resume else None
        cache_status = "hit" if result is not None else "miss"
        print(
            f"[{index + 1}/{len(records)}] {record.identifier} "
            f"({response_source}, cache={cache_status})",
            flush=True,
        )
        if result is None:
            result = _rollout_candidate(
                env,
                env_step,
                partner_spec=partner_spec,
                response_spec=response_spec,
                episodes=args.episodes,
                max_steps=args.max_steps,
                evaluation_seed=evaluation_seed,
            )
            _atomic_write_json(
                candidate_cache,
                {"identity": identity, "result": result},
            )

        output_entry = _base_output_entry(record, output_dir=output_path.parent)
        output_entry["selection_features"] = result["selection_features"]
        output_entry["selection_feature_status"] = "scored"
        output_entry["selection_feature_note"] = (
            "Seat-conditioned mean response event counts from both stochastic "
            "final-pair permutations; partner@0/response@1 precedes "
            "partner@1/response@0."
        )
        output_entry["reference_return"] = result["reference_return"]
        output_entry["score_metadata"] = {
            "response_policy_source": response_source,
            "episodes_per_seat": args.episodes,
            "total_episodes": 2 * args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "candidate_evaluation_seed": evaluation_seed,
            "stochastic": True,
            "seat_permutations": [
                {"partner_seat": 0, "response_seat": 1},
                {"partner_seat": 1, "response_seat": 0},
            ],
            "event_source": "info['event_vector'][response_agent]",
            "event_aggregation": "per-seat episode-count mean, then concatenate",
            "selection_feature_names": identity["selection_feature_order"],
            "sparse_team_return_source": "reward[response_agent]",
            "return_std": result["return_std"],
            "mean_episode_length": result["mean_episode_length"],
            "source_results": [str(path) for path in record.source_paths[1:]],
        }
        scored_entries.append(output_entry)

    output_payload = _output_root(root_payload)
    output_payload.update(
        {
            "format_version": 1,
            "kind": "coot_scored_hsp_catalog",
            "layout": args.layout,
            "checkpoint_path_base": "manifest_parent",
            "scoring": {
                "episodes_per_seat": args.episodes,
                "total_episodes_per_candidate": 2 * args.episodes,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "stochastic": True,
                "seat_permutations": [
                    {"partner_seat": 0, "response_seat": 1},
                    {"partner_seat": 1, "response_seat": 0},
                ],
                "selection_features": {
                    "source": "info['event_vector'][response_agent]",
                    "aggregation": (
                        "mean of per-episode event-count sums within each seat; "
                        "concatenate partner-seat 0 then partner-seat 1"
                    ),
                    "event_names": list(EVENT_NAMES),
                    "feature_names": [
                        f"partner_seat_{partner_seat}/"
                        f"response_seat_{1 - partner_seat}/{name}"
                        for partner_seat in (0, 1)
                        for name in EVENT_NAMES
                    ],
                },
                "reference_return": {
                    "source": "shared sparse team reward via reward[response_agent]",
                    "aggregation": "mean over both seat permutations",
                },
                "candidate_count": len(scored_entries),
                "deviations": (
                    [
                        {
                            "code": "v3_shared_ippo_response_approximation",
                            "description": (
                                "Candidates without best_response.final were "
                                "evaluated with partner.final controlling both roles, "
                                "an explicit V3 shared-IPPO fallback that is not "
                                "equivalent to the release's separated runner and "
                                "separately extracted actors."
                            ),
                            "candidate_ids": fallback_ids,
                            "count": len(fallback_ids),
                        }
                    ]
                    if fallback_ids
                    else []
                ),
            },
            "provenance": {
                "catalog": (
                    str(args.catalog.expanduser().resolve())
                    if args.catalog is not None
                    else None
                ),
                "candidate_results": [str(path) for path in consumed_results],
                "response_results": [str(path) for path in consumed_responses],
                "cache_dir": str(cache_dir),
            },
            "candidates": scored_entries,
        }
    )
    _atomic_write_json(output_path, output_payload, overwrite=args.overwrite)
    print(output_path)


if __name__ == "__main__":
    main()
