"""Synthetic tests for aggregate-only public artifact auditing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from clinicalbert_icd import privacy
from clinicalbert_icd.privacy import validate_public_artifact


def aggregate_artifact(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "artifact_type": "aggregate_metrics",
        "contains_raw_data": False,
        "contains_identifiers": False,
        "contains_row_level_data": False,
        "contains_checkpoints": False,
        "metrics": {"micro_f1": 0.5, "record_count": 4},
        "limitations": "Synthetic aggregate fixture only.",
    }
    value.update(overrides)
    return value


class PublicArtifactPrivacyTests(unittest.TestCase):
    def test_aggregate_report_passes_privacy_audit(self) -> None:
        artifact = aggregate_artifact()

        audit = validate_public_artifact(artifact, destination="public_report")

        self.assertTrue(audit.allowed)
        self.assertEqual(audit.reasons, ())
        self.assertEqual(audit.finding_count, 0)

    def test_non_public_destination_is_rejected(self) -> None:
        audit = validate_public_artifact(
            aggregate_artifact(),
            destination="synthetic-internal-folder",
        )

        self.assertFalse(audit.allowed)
        self.assertEqual(audit.reasons, ("destination_not_public",))

    def test_note_text_is_rejected_without_echoing_content(self) -> None:
        canary = "SYNTHETIC_NOTE_TEXT_CANARY_DO_NOT_USE"

        audit = validate_public_artifact(
            aggregate_artifact(note_text=canary),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("note_text_forbidden", audit.reasons)
        self.assertNotIn(canary, repr(audit))

    def test_identifier_key_is_rejected_without_echoing_identifier(self) -> None:
        identifier = "SYNTHETIC-SUBJECT-0001"

        audit = validate_public_artifact(
            aggregate_artifact(metrics={"subject_id": identifier}),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("identifier_forbidden", audit.reasons)
        self.assertNotIn(identifier, repr(audit))

    def test_secret_key_and_secret_canary_are_rejected(self) -> None:
        secret = "SYNTHETIC_SECRET_CANARY_DO_NOT_USE"

        audit = validate_public_artifact(
            aggregate_artifact(api_key=secret, summary=secret),
            destination="public_model_card",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("secret_forbidden", audit.reasons)
        self.assertGreaterEqual(audit.finding_count, 2)
        self.assertNotIn(secret, repr(audit))

    def test_row_membership_is_rejected(self) -> None:
        audit = validate_public_artifact(
            aggregate_artifact(
                split_membership={"train": ["SYNTHETIC-SUBJECT-0001"]}
            ),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("row_membership_forbidden", audit.reasons)

    def test_checkpoint_key_or_path_is_rejected(self) -> None:
        audit = validate_public_artifact(
            aggregate_artifact(
                checkpoint_path="synthetic-model.ckpt",
                artifact_path="synthetic-weights.safetensors",
            ),
            destination="public_repository",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("checkpoint_forbidden", audit.reasons)
        self.assertGreaterEqual(audit.finding_count, 2)

    def test_scalar_or_sequence_root_is_rejected(self) -> None:
        for value in (None, 7, "synthetic-scalar", ["synthetic-list"]):
            with self.subTest(value_type=type(value).__name__):
                audit = validate_public_artifact(value, destination="public_report")
                self.assertFalse(audit.allowed)
                self.assertIn("artifact_root_invalid", audit.reasons)

    def test_unknown_artifact_type_is_rejected(self) -> None:
        audit = validate_public_artifact(
            aggregate_artifact(artifact_type="synthetic-unknown-type"),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("artifact_type_not_allowed", audit.reasons)

    def test_unhashable_artifact_type_is_denied_without_error(self) -> None:
        audit = validate_public_artifact(
            aggregate_artifact(artifact_type=[]),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("artifact_type_not_allowed", audit.reasons)

    def test_each_required_false_flag_fails_closed(self) -> None:
        cases = {
            "contains_raw_data": "raw_data_flag_forbidden",
            "contains_identifiers": "identifiers_flag_forbidden",
            "contains_row_level_data": "row_level_data_flag_forbidden",
            "contains_checkpoints": "checkpoints_flag_forbidden",
        }
        for field, reason in cases.items():
            with self.subTest(field=field):
                audit = validate_public_artifact(
                    aggregate_artifact(**{field: True}),
                    destination="public_report",
                )
                self.assertFalse(audit.allowed)
                self.assertIn(reason, audit.reasons)

    def test_missing_required_false_flag_is_rejected(self) -> None:
        artifact = aggregate_artifact()
        del artifact["contains_raw_data"]

        audit = validate_public_artifact(artifact, destination="public_report")

        self.assertFalse(audit.allowed)
        self.assertIn("artifact_flags_invalid", audit.reasons)

    def test_rows_or_content_cannot_hide_raw_like_values(self) -> None:
        for field in ("rows", "content"):
            with self.subTest(field=field):
                audit = validate_public_artifact(
                    aggregate_artifact(
                        **{field: ["SYNTHETIC_NOTE_TEXT_CANARY_DO_NOT_USE"]}
                    ),
                    destination="public_report",
                )
                self.assertFalse(audit.allowed)
                self.assertIn("artifact_structure_invalid", audit.reasons)

    def test_unsupported_nested_values_are_rejected(self) -> None:
        unsupported = (b"synthetic-bytes", {"synthetic-set"}, object())
        for value in unsupported:
            with self.subTest(value_type=type(value).__name__):
                audit = validate_public_artifact(
                    aggregate_artifact(metrics={"unsupported": value}),
                    destination="public_report",
                )
                self.assertFalse(audit.allowed)
                self.assertIn("unsupported_artifact_value", audit.reasons)

    def test_unknown_nested_metric_structure_is_rejected(self) -> None:
        audit = validate_public_artifact(
            aggregate_artifact(metrics={"unknown_nested": {"synthetic": 1}}),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("artifact_structure_invalid", audit.reasons)

    def test_boolean_and_null_are_not_aggregate_metric_numbers(self) -> None:
        for value in (True, False, None):
            with self.subTest(value=value):
                audit = validate_public_artifact(
                    aggregate_artifact(metrics={"micro_f1": value}),
                    destination="public_report",
                )

                self.assertFalse(audit.allowed)
                self.assertIn("artifact_structure_invalid", audit.reasons)

    def test_cyclic_container_is_rejected_without_recursion_error(self) -> None:
        cyclic_metrics: dict[str, object] = {}
        cyclic_metrics["cycle"] = cyclic_metrics

        audit = validate_public_artifact(
            aggregate_artifact(metrics=cyclic_metrics),
            destination="public_report",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("cyclic_artifact_structure", audit.reasons)

    def test_policy_matches_executable_types_and_required_flags(self) -> None:
        policy_path = Path(__file__).parents[1] / "manifests" / "artifact_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(
            frozenset(policy["allowed_artifact_classes"]),
            privacy.ALLOWED_PUBLIC_ARTIFACT_TYPES,
        )
        self.assertEqual(
            frozenset(policy["required_false_flags"]),
            privacy.REQUIRED_FALSE_FLAGS,
        )
        self.assertEqual(
            frozenset(policy["aggregate_metric_fields"]),
            privacy.ALLOWED_AGGREGATE_METRIC_FIELDS,
        )


if __name__ == "__main__":
    unittest.main()
