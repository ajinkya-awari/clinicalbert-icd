"""Synthetic tests for fail-closed rights and environment decisions."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime

from clinicalbert_icd.governance import (
    RightsEnvironmentManifest,
    evaluate_data_access,
)


def synthetic_manifest_mapping(**overrides: object) -> dict[str, object]:
    """Return an unmistakably synthetic, in-memory authorization fixture."""

    value: dict[str, object] = {
        "schema_version": 1,
        "fixture_only": True,
        "dataset_reference": "synthetic-dataset-only",
        "source_revision": "synthetic-revision-v1",
        "data_use_terms_reference": "synthetic-terms-reference",
        "rights_status": "approved",
        "rights_expires_on": "2099-12-31",
        "decision_reference": "synthetic-decision-reference",
        "approved_locations": ["synthetic-secure-local"],
        "kaggle_allowed": False,
        "raw_transfer_allowed_locations": [],
        "retention_policy_reference": "synthetic-retention-policy",
        "artifact_policy_reference": "manifests/artifact_policy.json",
    }
    value.update(overrides)
    return value


class RightsEnvironmentManifestTests(unittest.TestCase):
    def test_missing_rights_field_fails_before_access(self) -> None:
        value = synthetic_manifest_mapping()
        del value["data_use_terms_reference"]

        with self.assertRaisesRegex(ValueError, "missing_required_fields"):
            RightsEnvironmentManifest.from_mapping(value)

    def test_unapproved_rights_are_denied(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(fixture_only=False, rights_status="pending")
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("rights_not_approved",))

    def test_rights_are_denied_after_expiry(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(
                fixture_only=False,
                rights_expires_on="2030-01-01",
            )
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            as_of_date=date(2030, 1, 2),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("rights_expired",))

    def test_unapproved_location_is_denied(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(fixture_only=False)
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-other-location",
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("location_not_approved",))

    def test_kaggle_requires_individual_allowance(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(
                fixture_only=False,
                approved_locations=["kaggle"],
            )
        )

        decision = evaluate_data_access(
            manifest,
            "kaggle",
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("kaggle_not_allowed",))

    def test_raw_transfer_requires_location_allowance(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(fixture_only=False)
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            raw_transfer_requested=True,
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("raw_transfer_not_allowed",))

    def test_individually_allowed_hypothetical_kaggle_transfer_can_pass_gate(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(
                fixture_only=False,
                approved_locations=["synthetic-secure-local", "kaggle"],
                kaggle_allowed=True,
                raw_transfer_allowed_locations=["kaggle"],
            )
        )

        decision = evaluate_data_access(
            manifest,
            "kaggle",
            raw_transfer_requested=True,
            as_of_date=date(2030, 1, 1),
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reasons, ())

    def test_fixture_manifest_never_authorizes_data_access(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping()
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("fixture_data_access_forbidden",))

    def test_unexpected_manifest_field_is_rejected(self) -> None:
        value = synthetic_manifest_mapping(unexpected_field="synthetic-value")

        with self.assertRaisesRegex(ValueError, "unexpected_manifest_fields"):
            RightsEnvironmentManifest.from_mapping(value)

    def test_duplicate_locations_are_rejected_in_mapping(self) -> None:
        value = synthetic_manifest_mapping(
            approved_locations=[
                "synthetic-secure-local",
                "SYNTHETIC-SECURE-LOCAL",
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate_locations"):
            RightsEnvironmentManifest.from_mapping(value)

    def test_rights_are_valid_before_expiry(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(
                fixture_only=False,
                rights_expires_on="2030-01-02",
            )
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            as_of_date=date(2030, 1, 1),
        )

        self.assertTrue(decision.allowed)

    def test_rights_are_denied_on_expiry_date(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(
                fixture_only=False,
                rights_expires_on="2030-01-01",
            )
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("rights_expired",))

    def test_datetime_as_of_value_is_denied_with_stable_reason(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(fixture_only=False)
        )

        decision = evaluate_data_access(
            manifest,
            "synthetic-secure-local",
            as_of_date=datetime(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("as_of_date_invalid",))

    def test_direct_manifest_rejects_datetime_expiry(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping()
        )

        with self.assertRaisesRegex(ValueError, "rights_expiry_date_invalid"):
            replace(manifest, rights_expires_on=datetime(2099, 12, 31))

    def test_direct_manifest_rejects_mutable_location_set(self) -> None:
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping()
        )

        with self.assertRaisesRegex(ValueError, "locations_must_be_frozenset"):
            replace(manifest, approved_locations={"synthetic-secure-local"})

    def test_invalid_manifest_input_returns_stable_denial(self) -> None:
        decision = evaluate_data_access(
            None,  # type: ignore[arg-type]
            "synthetic-secure-local",
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("manifest_invalid",))

    def test_invalid_location_with_raw_transfer_request_does_not_add_duplicate_reason(self) -> None:
        # An empty location normalises to None → requested_location_invalid.
        # raw_transfer_requested=True must not also append raw_transfer_not_allowed
        # (that would be a spurious second reason for the same root cause).
        manifest = RightsEnvironmentManifest.from_mapping(
            synthetic_manifest_mapping(fixture_only=False)
        )

        decision = evaluate_data_access(
            manifest,
            "",  # empty string → _requested_location returns None
            raw_transfer_requested=True,
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertIn("requested_location_invalid", decision.reasons)
        self.assertNotIn("raw_transfer_not_allowed", decision.reasons)


if __name__ == "__main__":
    unittest.main()
