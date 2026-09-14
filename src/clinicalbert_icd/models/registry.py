"""Validation for the exact offline five-system comparison registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

from clinicalbert_icd.models.protocol import ModelSpec


_SYSTEM_DEFINITIONS = {
    "rule": ("rule", "rule"),
    "domain_lora": ("domain", "lora"),
    "domain_full": ("domain", "full"),
    "general_lora": ("general", "lora"),
    "general_full": ("general", "full"),
}
_NEURAL_SYSTEMS = frozenset(_SYSTEM_DEFINITIONS) - {"rule"}
_COMMIT_REVISION = re.compile(r"commit:[0-9a-f]{40,64}\Z")
_SHA256_REVISION = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class RegistryAudit:
    """Content-free result of validating declarative model specifications."""

    allowed: bool
    reasons: tuple[str, ...]
    system_count: int
    logical_systems: tuple[str, ...]


def validate_five_system_registry(specs: Iterable[ModelSpec]) -> RegistryAudit:
    """Validate exact membership, immutable neural revisions, and shared contracts."""

    reasons: list[str] = []
    try:
        items = tuple(specs)
    except TypeError:
        items = ()
        reasons.append("registry_invalid")

    if any(not isinstance(item, ModelSpec) for item in items):
        reasons.append("registry_invalid")
    model_specs = tuple(item for item in items if isinstance(item, ModelSpec))
    valid_items = tuple(item for item in model_specs if _spec_shape_valid(item))
    if len(valid_items) != len(model_specs):
        reasons.append("registry_invalid")

    logical_systems = tuple(item.logical_system for item in valid_items)
    if len(logical_systems) != len(set(logical_systems)):
        reasons.append("duplicate_logical_system")
    if set(logical_systems) != set(_SYSTEM_DEFINITIONS):
        reasons.append("exact_system_set_required")

    for item in valid_items:
        expected = _SYSTEM_DEFINITIONS.get(item.logical_system)
        if expected is None or (item.family, item.adaptation) != expected:
            reasons.append("logical_system_definition_invalid")
        if item.logical_system == "rule" and (
            item.tokenizer_reference is not None
            or item.tokenizer_revision is not None
        ):
            reasons.append("rule_tokenizer_forbidden")
        if not _required_string(item.label_contract_digest):
            reasons.append("label_contract_invalid")
        if not _required_string(item.output_contract):
            reasons.append("output_contract_invalid")
        if item.logical_system in _NEURAL_SYSTEMS and not _neural_revisions_are_pinned(
            item
        ):
            reasons.append("neural_revision_not_immutable")

    label_contracts = {item.label_contract_digest for item in valid_items}
    output_contracts = {item.output_contract for item in valid_items}
    if len(label_contracts) > 1:
        reasons.append("label_contract_mismatch")
    if len(output_contracts) > 1:
        reasons.append("output_contract_mismatch")

    by_system = {item.logical_system: item for item in valid_items}
    for family in ("domain", "general"):
        lora = by_system.get(f"{family}_lora")
        full = by_system.get(f"{family}_full")
        if lora is not None and full is not None and _revision_identity(lora) != _revision_identity(full):
            reasons.append("family_revision_mismatch")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return RegistryAudit(
        allowed=not unique_reasons,
        reasons=unique_reasons,
        system_count=len(items),
        logical_systems=tuple(sorted(logical_systems)),
    )


def _required_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _spec_shape_valid(spec: ModelSpec) -> bool:
    return all(
        _required_string(value)
        for value in (
            spec.logical_system,
            spec.family,
            spec.adaptation,
            spec.model_reference,
            spec.model_revision,
            spec.label_contract_digest,
            spec.output_contract,
        )
    )


def _immutable_revision(value: object) -> bool:
    if not _required_string(value):
        return False
    normalized = value.strip().casefold()
    return (
        _COMMIT_REVISION.fullmatch(normalized) is not None
        or _SHA256_REVISION.fullmatch(normalized) is not None
    )


def _neural_revisions_are_pinned(spec: ModelSpec) -> bool:
    return (
        _required_string(spec.model_reference)
        and _immutable_revision(spec.model_revision)
        and _required_string(spec.tokenizer_reference)
        and _immutable_revision(spec.tokenizer_revision)
    )


def _revision_identity(spec: ModelSpec) -> tuple[object, ...]:
    return (
        spec.model_reference,
        spec.model_revision,
        spec.tokenizer_reference,
        spec.tokenizer_revision,
    )
