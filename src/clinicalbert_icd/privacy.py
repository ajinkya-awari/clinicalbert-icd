"""Strict aggregate-only privacy audit for candidate public artifacts.

Passing this audit is not release authorization. The caller must separately
obtain the approvals declared by the artifact policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re


ALLOWED_PUBLIC_ARTIFACT_TYPES = frozenset(
    {"aggregate_metrics", "documentation", "non_sensitive_provenance"}
)
REQUIRED_FALSE_FLAGS = frozenset(
    {
        "contains_raw_data",
        "contains_identifiers",
        "contains_row_level_data",
        "contains_checkpoints",
    }
)
ALLOWED_AGGREGATE_METRIC_FIELDS = frozenset(
    {
        "macro_f1",
        "micro_f1",
        "parameter_count",
        "precision_at_10",
        "precision_at_5",
        "record_count",
        "runtime_seconds",
        "uncertainty",
    }
)

_PUBLIC_DESTINATIONS = frozenset(
    {"public_model_card", "public_report", "public_repository"}
)
_COMMON_ROOT_KEYS = REQUIRED_FALSE_FLAGS | {"artifact_type"}
_TYPE_ROOT_KEYS = {
    "aggregate_metrics": frozenset({"metrics", "limitations"}),
    "documentation": frozenset({"title", "summary", "limitations"}),
    "non_sensitive_provenance": frozenset({"provenance"}),
}
_PROVENANCE_KEYS = frozenset(
    {
        "artifact_digest",
        "command_digest",
        "config_digest",
        "created_on",
        "manifest_digest",
        "model_revision",
        "runtime",
        "seed",
        "tokenizer_revision",
    }
)
_FLAG_REASONS = {
    "contains_raw_data": "raw_data_flag_forbidden",
    "contains_identifiers": "identifiers_flag_forbidden",
    "contains_row_level_data": "row_level_data_flag_forbidden",
    "contains_checkpoints": "checkpoints_flag_forbidden",
}
_NOTE_KEYS = frozenset(
    {
        "clinical_note",
        "clinical_text",
        "discharge_summary",
        "note",
        "note_content",
        "note_text",
        "notes",
        "raw_note",
        "raw_notes",
        "text",
    }
)
_IDENTIFIER_KEYS = frozenset(
    {
        "admission_id",
        "admission_ids",
        "hadm_id",
        "hadm_ids",
        "mrn",
        "patient_id",
        "patient_ids",
        "record_id",
        "record_ids",
        "subject_id",
        "subject_ids",
    }
)
_SECRET_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "client_secret",
        "credentials",
        "hf_token",
        "password",
        "private_key",
        "secret",
        "wandb_api_key",
    }
)
_ROW_MEMBERSHIP_KEYS = frozenset(
    {
        "content",
        "membership",
        "partition_membership",
        "row_ids",
        "row_membership",
        "rows",
        "split_membership",
    }
)
_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "checkpoints",
        "model_state",
        "optimizer_state",
        "state_dict",
        "weight_path",
        "weights",
    }
)
_CHECKPOINT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)
_SECRET_PATTERNS = (
    re.compile(r"SYNTHETIC_SECRET_CANARY", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9_-]{8,})\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)
_NOTE_PATTERNS = (re.compile(r"SYNTHETIC_NOTE_TEXT_CANARY", re.IGNORECASE),)


@dataclass(frozen=True, slots=True)
class PrivacyAudit:
    """A content-free summary of privacy findings."""

    allowed: bool
    reasons: tuple[str, ...]
    finding_count: int


def validate_public_artifact(value: object, destination: str) -> PrivacyAudit:
    """Validate one in-memory JSON-compatible candidate, defaulting to deny."""

    findings: list[str] = []
    normalized_destination = (
        destination.strip().casefold() if isinstance(destination, str) else ""
    )
    if normalized_destination not in _PUBLIC_DESTINATIONS:
        findings.append("destination_not_public")

    _inspect_json_value(value, findings, active_ids=set())
    if not isinstance(value, Mapping):
        findings.append("artifact_root_invalid")
    else:
        _validate_root_contract(value, findings)

    reasons = tuple(dict.fromkeys(findings))
    return PrivacyAudit(
        allowed=not findings,
        reasons=reasons,
        finding_count=len(findings),
    )


def _validate_root_contract(value: Mapping[object, object], findings: list[str]) -> None:
    artifact_type = value.get("artifact_type")
    artifact_type_allowed = isinstance(
        artifact_type, str
    ) and artifact_type in ALLOWED_PUBLIC_ARTIFACT_TYPES
    if not artifact_type_allowed:
        findings.append("artifact_type_not_allowed")

    for flag in sorted(REQUIRED_FALSE_FLAGS):
        if flag not in value or type(value[flag]) is not bool:
            findings.append("artifact_flags_invalid")
        elif value[flag]:
            findings.append(_FLAG_REASONS[flag])

    type_keys = (
        _TYPE_ROOT_KEYS.get(artifact_type, frozenset())
        if artifact_type_allowed
        else frozenset()
    )
    allowed_keys = _COMMON_ROOT_KEYS | type_keys
    if any(not isinstance(key, str) or key not in allowed_keys for key in value):
        findings.append("artifact_structure_invalid")

    if artifact_type == "aggregate_metrics":
        metrics = value.get("metrics")
        if not isinstance(metrics, Mapping):
            findings.append("artifact_structure_invalid")
        else:
            _validate_numeric_mapping(metrics, findings)
        if "limitations" in value and not isinstance(value["limitations"], str):
            findings.append("artifact_structure_invalid")
    elif artifact_type == "documentation":
        document_keys = ("title", "summary", "limitations")
        if not any(key in value for key in document_keys):
            findings.append("artifact_structure_invalid")
        if any(
            key in value and not isinstance(value[key], str)
            for key in document_keys
        ):
            findings.append("artifact_structure_invalid")
    elif artifact_type == "non_sensitive_provenance":
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            findings.append("artifact_structure_invalid")
        else:
            if any(
                not isinstance(key, str) or key not in _PROVENANCE_KEYS
                for key in provenance
            ):
                findings.append("artifact_structure_invalid")
            if any(not _is_json_scalar(item) for item in provenance.values()):
                findings.append("artifact_structure_invalid")


def _validate_numeric_mapping(
    value: Mapping[object, object],
    findings: list[str],
) -> None:
    for key, item in value.items():
        if not isinstance(key, str) or key not in ALLOWED_AGGREGATE_METRIC_FIELDS:
            findings.append("artifact_structure_invalid")
        if not _is_aggregate_number(item):
            findings.append("artifact_structure_invalid")


def _inspect_json_value(
    value: object,
    findings: list[str],
    *,
    active_ids: set[int],
) -> None:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_ids:
            findings.append("cyclic_artifact_structure")
            return
        active_ids.add(identity)
        try:
            for key, child in value.items():
                normalized_key = _normalize_key(key)
                _record_key_findings(normalized_key, findings)
                _inspect_json_value(child, findings, active_ids=active_ids)
        finally:
            active_ids.remove(identity)
        return

    if isinstance(value, list):
        identity = id(value)
        if identity in active_ids:
            findings.append("cyclic_artifact_structure")
            return
        active_ids.add(identity)
        try:
            for child in value:
                _inspect_json_value(child, findings, active_ids=active_ids)
        finally:
            active_ids.remove(identity)
        return

    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _NOTE_PATTERNS):
            findings.append("note_text_forbidden")
        if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            findings.append("secret_forbidden")
        if value.casefold().endswith(_CHECKPOINT_SUFFIXES):
            findings.append("checkpoint_forbidden")
        return

    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    findings.append("unsupported_artifact_value")


def _record_key_findings(normalized_key: str, findings: list[str]) -> None:
    if normalized_key in _NOTE_KEYS:
        findings.append("note_text_forbidden")
    if normalized_key in _IDENTIFIER_KEYS:
        findings.append("identifier_forbidden")
    if normalized_key in _SECRET_KEYS:
        findings.append("secret_forbidden")
    if normalized_key in _ROW_MEMBERSHIP_KEYS:
        findings.append("row_membership_forbidden")
    if normalized_key in _CHECKPOINT_KEYS:
        findings.append("checkpoint_forbidden")


def _normalize_key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _is_aggregate_number(value: object) -> bool:
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _is_json_scalar(value: object) -> bool:
    return (
        isinstance(value, str)
        or value is None
        or type(value) is bool
        or _is_aggregate_number(value)
    )
