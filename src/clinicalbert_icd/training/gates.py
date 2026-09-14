"""Offline training authorization guard; this module contains no trainer."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from clinicalbert_icd.governance import (
    GateDecision,
    RightsEnvironmentManifest,
    evaluate_data_access,
)


_RUNTIME_FIELDS = (
    "location",
    "network_enabled",
    "external_logging",
    "raw_transfer_requested",
    "synthetic_only",
    "gpu_job",
    "restricted_runtime_authorized",
)
_LOCAL_SYNTHETIC_LOCATION = "synthetic-secure-local"


def authorize_training(
    manifest: object,
    runtime: object,
    *,
    as_of_date: date,
) -> GateDecision:
    """Authorize only an approved, offline, non-logging, no-transfer runtime."""

    if not isinstance(manifest, RightsEnvironmentManifest):
        return GateDecision(False, ("manifest_invalid",))
    if not isinstance(runtime, Mapping) or set(runtime) != set(_RUNTIME_FIELDS):
        return GateDecision(False, ("runtime_invalid",))

    location = runtime["location"]
    network_enabled = runtime["network_enabled"]
    external_logging = runtime["external_logging"]
    raw_transfer_requested = runtime["raw_transfer_requested"]
    synthetic_only = runtime["synthetic_only"]
    gpu_job = runtime["gpu_job"]
    restricted_runtime_authorized = runtime["restricted_runtime_authorized"]
    if (
        not isinstance(location, str)
        or type(network_enabled) is not bool
        or type(external_logging) is not bool
        or type(raw_transfer_requested) is not bool
        or type(synthetic_only) is not bool
        or type(gpu_job) is not bool
        or type(restricted_runtime_authorized) is not bool
    ):
        return GateDecision(False, ("runtime_invalid",))

    access = evaluate_data_access(manifest, location, as_of_date=as_of_date)
    reasons = [
        reason
        for reason in access.reasons
        if reason != "fixture_data_access_forbidden"
    ]
    if not manifest.fixture_only:
        reasons.append("live_manifest_training_forbidden")
    if location.strip().casefold() != _LOCAL_SYNTHETIC_LOCATION:
        reasons.append("fixture_location_forbidden")
    if network_enabled:
        reasons.append("network_must_be_disabled")
    if external_logging:
        reasons.append("external_logging_forbidden")
    if raw_transfer_requested:
        reasons.append("training_raw_transfer_forbidden")
    if not synthetic_only:
        reasons.append("synthetic_only_required")
    if gpu_job:
        reasons.append("gpu_job_forbidden")
    if restricted_runtime_authorized:
        reasons.append("restricted_runtime_forbidden")

    return GateDecision(allowed=not reasons, reasons=tuple(reasons))
