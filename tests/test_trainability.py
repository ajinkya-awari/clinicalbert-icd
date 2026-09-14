"""Synthetic parameter-graph tests for LoRA and full-mode guards."""

from __future__ import annotations

import unittest

from clinicalbert_icd.models.trainability import (
    ParameterRecord,
    audit_full_trainability,
    audit_lora_trainability,
)


def lora_parameters() -> tuple[ParameterRecord, ...]:
    return (
        ParameterRecord("encoder.layer.0.attention.query.weight", False, 64),
        ParameterRecord("encoder.layer.0.attention.query.lora_A.weight", True, 8),
        ParameterRecord("encoder.layer.0.attention.query.lora_B.weight", True, 8),
        ParameterRecord("encoder.layer.0.attention.value.weight", False, 64),
        ParameterRecord("encoder.layer.0.attention.value.lora_A.weight", True, 8),
        ParameterRecord("encoder.layer.0.attention.value.lora_B.weight", True, 8),
        ParameterRecord("encoder.layer.0.output.dense.weight", False, 64),
        ParameterRecord("classifier.weight", True, 24),
        ParameterRecord("classifier.bias", True, 3),
    )


class LoraTrainabilityTests(unittest.TestCase):
    def test_valid_lora_graph_has_targets_trainable_head_and_bounded_scope(self) -> None:
        audit = audit_lora_trainability(
            lora_parameters(),
            target_suffixes=("query", "value"),
            classifier_path="classifier",
        )

        self.assertTrue(audit.allowed)
        self.assertEqual(audit.reasons, ())
        self.assertEqual(audit.adapter_parameter_count, 32)
        self.assertEqual(audit.classifier_parameter_count, 27)
        self.assertEqual(
            audit.matched_target_modules,
            (
                "encoder.layer.0.attention.query",
                "encoder.layer.0.attention.value",
            ),
        )
        self.assertLess(audit.trainable_parameter_count, audit.total_parameter_count)

    def test_zero_or_partially_matched_targets_fail(self) -> None:
        no_adapters = tuple(
            item for item in lora_parameters() if ".lora_" not in item.name
        )

        zero = audit_lora_trainability(
            no_adapters,
            target_suffixes=("query", "value"),
            classifier_path="classifier",
        )
        partial = audit_lora_trainability(
            lora_parameters(),
            target_suffixes=("query", "key", "value"),
            classifier_path="classifier",
        )

        self.assertFalse(zero.allowed)
        self.assertIn("target_adapters_missing", zero.reasons)
        self.assertFalse(partial.allowed)
        self.assertIn("target_suffix_unmatched", partial.reasons)

    def test_classifier_must_exist_and_be_fully_trainable(self) -> None:
        frozen = tuple(
            ParameterRecord(item.name, False, item.count)
            if item.name == "classifier.bias"
            else item
            for item in lora_parameters()
        )

        audit = audit_lora_trainability(
            frozen,
            target_suffixes=("query", "value"),
            classifier_path="classifier",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("classifier_not_fully_trainable", audit.reasons)

    def test_every_observed_target_adapter_parameter_must_be_trainable(self) -> None:
        partially_frozen = tuple(
            ParameterRecord(item.name, False, item.count)
            if item.name.endswith("query.lora_B.weight")
            else item
            for item in lora_parameters()
        )

        audit = audit_lora_trainability(
            partially_frozen,
            target_suffixes=("query", "value"),
            classifier_path="classifier",
        )

        self.assertFalse(audit.allowed)
        self.assertIn("adapter_not_fully_trainable", audit.reasons)

    def test_unexpected_trainable_base_or_unbounded_scope_fails(self) -> None:
        unexpected = tuple(
            ParameterRecord(item.name, True, item.count)
            if item.name == "encoder.layer.0.output.dense.weight"
            else item
            for item in lora_parameters()
        )
        all_trainable = tuple(
            ParameterRecord(item.name, True, item.count) for item in lora_parameters()
        )

        unexpected_audit = audit_lora_trainability(
            unexpected,
            target_suffixes=("query", "value"),
            classifier_path="classifier",
        )
        unbounded_audit = audit_lora_trainability(
            all_trainable,
            target_suffixes=("query", "value"),
            classifier_path="classifier",
        )

        self.assertFalse(unexpected_audit.allowed)
        self.assertIn("unexpected_trainable_base", unexpected_audit.reasons)
        self.assertEqual(
            unexpected_audit.unexpected_trainable_parameters,
            ("encoder.layer.0.output.dense.weight",),
        )
        self.assertFalse(unbounded_audit.allowed)
        self.assertIn("lora_scope_not_bounded", unbounded_audit.reasons)


class FullTrainabilityTests(unittest.TestCase):
    def test_full_mode_requires_every_parameter_and_classifier_trainable(self) -> None:
        parameters = tuple(
            ParameterRecord(item.name, True, item.count) for item in lora_parameters()
        )

        audit = audit_full_trainability(parameters, classifier_path="classifier")

        self.assertTrue(audit.allowed)
        self.assertEqual(audit.trainable_parameter_count, audit.total_parameter_count)
        self.assertEqual(audit.classifier_parameter_count, 27)

    def test_full_mode_rejects_any_frozen_parameter(self) -> None:
        audit = audit_full_trainability(
            lora_parameters(), classifier_path="classifier"
        )

        self.assertFalse(audit.allowed)
        self.assertIn("full_parameters_not_trainable", audit.reasons)

    def test_invalid_or_duplicate_parameter_records_fail_closed(self) -> None:
        duplicate = lora_parameters() + (lora_parameters()[0],)
        invalid = lora_parameters() + (ParameterRecord("synthetic.invalid", True, 0),)

        duplicate_audit = audit_full_trainability(
            duplicate, classifier_path="classifier"
        )
        invalid_audit = audit_full_trainability(
            invalid, classifier_path="classifier"
        )

        self.assertFalse(duplicate_audit.allowed)
        self.assertIn("parameter_records_invalid", duplicate_audit.reasons)
        self.assertFalse(invalid_audit.allowed)
        self.assertIn("parameter_records_invalid", invalid_audit.reasons)

    def test_unhashable_parameter_names_and_target_suffixes_fail_closed(self) -> None:
        malformed_record = ParameterRecord(["synthetic.invalid"], True, 1)

        record_audit = audit_full_trainability(
            (malformed_record,), classifier_path="classifier"
        )
        suffix_audit = audit_lora_trainability(
            lora_parameters(),
            target_suffixes=(["query"],),
            classifier_path="classifier",
        )

        self.assertFalse(record_audit.allowed)
        self.assertIn("parameter_records_invalid", record_audit.reasons)
        self.assertFalse(suffix_audit.allowed)
        self.assertIn("trainability_config_invalid", suffix_audit.reasons)


if __name__ == "__main__":
    unittest.main()
