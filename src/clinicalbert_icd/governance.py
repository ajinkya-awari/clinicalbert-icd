"""Fail-closed rights and processing-environment contracts.

This module evaluates already-supplied, non-sensitive manifest values. It does
not open data, read approval evidence, or authorize an external action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date


_REQUIRED_FIELDS = (
    "schema_version",
    "fixture_only",
    "dataset_reference",
    "source_revision",
    "data_use_terms_reference",
    "rights_status",
    "rights_expires_on",
    "decision_reference",
    "approved_locations",
    "kaggle_allowed",
    "raw_transfer_allowed_locations",
    "retention_policy_reference",
    "artifact_policy_reference",
)

_RIGHTS_STATUSES = frozenset({"approved", "pending", "denied", "revoked", "expired"})


@dataclass(frozen=True, slots=True)
class GateDecision:
    """An aggregate-only authorization result with stable reason codes."""

    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RightsEnvironmentManifest:
    """Validated non-sensitive rights and environment decision metadata."""

    schema_version: int
    fixture_only: bool
    dataset_reference: str
    source_revision: str
    data_use_terms_reference: str
    rights_status: str
    rights_expires_on: date
    decision_reference: str
    approved_locations: frozenset[str]
    kaggle_allowed: bool
    raw_transfer_allowed_locations: frozenset[str]
    retention_policy_reference: str
    artifact_policy_reference: str

    def __post_init__(self) -> None:
        """Keep direct construction as strict as mapping construction."""

        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported_schema_version")
        if type(self.fixture_only) is not bool:
            raise ValueError("fixture_only_must_be_boolean")
        if (
            type(self.rights_status) is not str
            or self.rights_status not in _RIGHTS_STATUSES
        ):
            raise ValueError("invalid_rights_status")
        if type(self.rights_expires_on) is not date:
            raise ValueError("rights_expiry_date_invalid")
        if type(self.kaggle_allowed) is not bool:
            raise ValueError("kaggle_allowed_must_be_boolean")

        for value in (
            self.dataset_reference,
            self.source_revision,
            self.data_use_terms_reference,
            self.decision_reference,
            self.retention_policy_reference,
            self.artifact_policy_reference,
        ):
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError("required_string_invalid")

        _validate_direct_locations(self.approved_locations, allow_empty=False)
        _validate_direct_locations(
            self.raw_transfer_allowed_locations,
            allow_empty=True,
        )
        if not self.raw_transfer_allowed_locations.issubset(
            self.approved_locations
        ):
            raise ValueError("raw_transfer_location_not_approved")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "RightsEnvironmentManifest":
        """Validate manifest metadata without resolving referenced evidence."""

        if not isinstance(value, Mapping):
            raise ValueError("manifest_must_be_mapping")
        if any(field not in value for field in _REQUIRED_FIELDS):
            raise ValueError("missing_required_fields")
        if set(value) != set(_REQUIRED_FIELDS):
            raise ValueError("unexpected_manifest_fields")

        schema_version = value["schema_version"]
        fixture_only = value["fixture_only"]
        rights_status = _required_string(value["rights_status"])
        rights_expires_on = _required_date(value["rights_expires_on"])
        approved_locations = _location_set(value["approved_locations"], allow_empty=False)
        raw_transfer_locations = _location_set(
            value["raw_transfer_allowed_locations"], allow_empty=True
        )
        kaggle_allowed = value["kaggle_allowed"]

        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("unsupported_schema_version")
        if type(fixture_only) is not bool:
            raise ValueError("fixture_only_must_be_boolean")
        if rights_status not in _RIGHTS_STATUSES:
            raise ValueError("invalid_rights_status")
        if type(kaggle_allowed) is not bool:
            raise ValueError("kaggle_allowed_must_be_boolean")
        if not raw_transfer_locations.issubset(approved_locations):
            raise ValueError("raw_transfer_location_not_approved")

        return cls(
            schema_version=schema_version,
            fixture_only=fixture_only,
            dataset_reference=_required_string(value["dataset_reference"]),
            source_revision=_required_string(value["source_revision"]),
            data_use_terms_reference=_required_string(
                value["data_use_terms_reference"]
            ),
            rights_status=rights_status,
            rights_expires_on=rights_expires_on,
            decision_reference=_required_string(value["decision_reference"]),
            approved_locations=approved_locations,
            kaggle_allowed=kaggle_allowed,
            raw_transfer_allowed_locations=raw_transfer_locations,
            retention_policy_reference=_required_string(
                value["retention_policy_reference"]
            ),
            artifact_policy_reference=_required_string(
                value["artifact_policy_reference"]
            ),
        )


def evaluate_data_access(
    manifest: RightsEnvironmentManifest,
    requested_location: str,
    raw_transfer_requested: bool = False,
    *,
    as_of_date: date,
) -> GateDecision:
    """Evaluate access; rights expire at the start of their expiry date."""

    if type(manifest) is not RightsEnvironmentManifest:
        return GateDecision(allowed=False, reasons=("manifest_invalid",))

    reasons: list[str] = []
    location = _requested_location(requested_location)

    if manifest.fixture_only:
        reasons.append("fixture_data_access_forbidden")
    if manifest.rights_status != "approved":
        reasons.append("rights_not_approved")
    if type(as_of_date) is not date:
        reasons.append("as_of_date_invalid")
    elif manifest.rights_expires_on <= as_of_date:
        reasons.append("rights_expired")
    if location is None:
        reasons.append("requested_location_invalid")
    elif location not in manifest.approved_locations:
        reasons.append("location_not_approved")
    if location is not None and _is_kaggle(location) and not manifest.kaggle_allowed:
        reasons.append("kaggle_not_allowed")
    if type(raw_transfer_requested) is not bool:
        reasons.append("raw_transfer_request_invalid")
    elif (
        raw_transfer_requested
        and location is not None
        and location not in manifest.raw_transfer_allowed_locations
    ):
        reasons.append("raw_transfer_not_allowed")

    return GateDecision(allowed=not reasons, reasons=tuple(reasons))


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("required_string_invalid")
    return value.strip()


def _required_date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("rights_expiry_invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("rights_expiry_invalid") from exc


def _location_set(value: object, *, allow_empty: bool) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("locations_must_be_sequence")
    normalized = tuple(_required_string(item).casefold() for item in value)
    locations = frozenset(normalized)
    if len(locations) != len(normalized):
        raise ValueError("duplicate_locations")
    if not allow_empty and not locations:
        raise ValueError("approved_locations_empty")
    return locations


def _validate_direct_locations(value: object, *, allow_empty: bool) -> None:
    if type(value) is not frozenset:
        raise ValueError("locations_must_be_frozenset")
    if not allow_empty and not value:
        raise ValueError("approved_locations_empty")
    if any(
        type(item) is not str
        or not item.strip()
        or item != item.strip().casefold()
        for item in value
    ):
        raise ValueError("location_value_invalid")


def _requested_location(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().casefold()


def _is_kaggle(location: str) -> bool:
    return location == "kaggle" or location.startswith("kaggle:")
