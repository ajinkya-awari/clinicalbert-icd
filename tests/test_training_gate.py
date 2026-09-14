"""Synthetic tests for the offline training authorization guard."""

from __future__ import annotations

import unittest
from datetime import date

from clinicalbert_icd.governance import RightsEnvironmentManifest
from clinicalbert_icd.training.gates import authorize_training


def synthetic_manifest(**overrides: object) -> RightsEnvironmentManifest:
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
    return RightsEnvironmentManifest.from_mapping(value)


def offline_runtime(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "location": "synthetic-secure-local",
        "network_enabled": False,
        "external_logging": False,
        "raw_transfer_requested": False,
        "synthetic_only": True,
        "gpu_job": False,
        "restricted_runtime_authorized": False,
    }
    value.update(overrides)
    return value


class TrainingGateTests(unittest.TestCase):
    def test_approved_offline_synthetic_runtime_passes(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(),
            as_of_date=date(2030, 1, 1),
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reasons, ())

    def test_unapproved_location_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(location="synthetic-other-location"),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertIn("location_not_approved", decision.reasons)

    def test_network_enabled_runtime_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(network_enabled=True),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("network_must_be_disabled",))

    def test_external_logging_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(external_logging=True),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("external_logging_forbidden",))

    def test_raw_transfer_state_is_rejected_even_if_access_policy_allows_it(self) -> None:
        manifest = synthetic_manifest(
            raw_transfer_allowed_locations=["synthetic-secure-local"]
        )

        decision = authorize_training(
            manifest,
            offline_runtime(raw_transfer_requested=True),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("training_raw_transfer_forbidden",))

    def test_missing_runtime_field_fails_closed(self) -> None:
        runtime = offline_runtime()
        del runtime["network_enabled"]

        decision = authorize_training(
            synthetic_manifest(),
            runtime,
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("runtime_invalid",))

    def test_invalid_manifest_returns_stable_denial(self) -> None:
        decision = authorize_training(
            object(),
            offline_runtime(),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("manifest_invalid",))

    def test_gpu_job_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(gpu_job=True),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("gpu_job_forbidden",))

    def test_restricted_runtime_flag_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(restricted_runtime_authorized=True),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("restricted_runtime_forbidden",))

    def test_non_synthetic_runtime_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(synthetic_only=False),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("synthetic_only_required",))

    def test_unknown_runtime_field_is_rejected(self) -> None:
        decision = authorize_training(
            synthetic_manifest(),
            offline_runtime(unknown_field="synthetic-value"),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("runtime_invalid",))

    def test_nonfixture_manifest_cannot_authorize_local_contract_training(self) -> None:
        decision = authorize_training(
            synthetic_manifest(fixture_only=False),
            offline_runtime(),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("live_manifest_training_forbidden",))

    def test_fixture_manifest_cannot_authorize_kaggle_contract_training(self) -> None:
        manifest = synthetic_manifest(
            approved_locations=["kaggle"],
            kaggle_allowed=True,
        )

        decision = authorize_training(
            manifest,
            offline_runtime(location="kaggle"),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("fixture_location_forbidden",))

    def test_training_denies_rights_on_expiry_date(self) -> None:
        decision = authorize_training(
            synthetic_manifest(rights_expires_on="2030-01-01"),
            offline_runtime(),
            as_of_date=date(2030, 1, 1),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reasons, ("rights_expired",))


if __name__ == "__main__":
    unittest.main()
