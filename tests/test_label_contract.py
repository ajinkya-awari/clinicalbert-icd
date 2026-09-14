from __future__ import annotations

import re
import unittest
from dataclasses import FrozenInstanceError, asdict

from clinicalbert_icd.data.contracts import (
    AdmissionExample,
    EncodedLabels,
    LabelConfig,
    LabelContract,
    LabelFitAudit,
    LabelTransformAudit,
)
from clinicalbert_icd.data.labels import (
    fit_training_label_contract,
    transform_labels,
)


def _example(
    admission: str, labels: tuple[str, ...], partition: str | None = "train"
) -> AdmissionExample:
    return AdmissionExample(
        subject_id=f"SYNTH-SUBJECT-{admission}",
        hadm_id=f"SYNTH-ADMISSION-{admission}",
        text=f"synthetic text {admission}",
        labels=labels,
        partition=partition,
    )


def _config(minimum_support: int = 1) -> LabelConfig:
    return LabelConfig(
        selection="all_training_labels", minimum_support=minimum_support
    )


class TrainingLabelContractTests(unittest.TestCase):
    def test_vocabulary_is_derived_only_from_training_examples(self) -> None:
        training = (
            _example("A", ("4019", "250.00")),
            _example("B", ("4019",)),
        )

        contract = fit_training_label_contract(training, _config())

        self.assertEqual(contract.vocabulary, ("250.00", "4019"))
        self.assertEqual(
            contract.vocabulary_hash,
            "4781b019696fb1dc513d1860ff646cb49ad9368e042ba097b445973c98b0ac67",
        )
        self.assertNotIn("V5869", contract.vocabulary)

    def test_fit_rejects_validation_test_or_unassigned_examples(self) -> None:
        for partition in ("validation", "test", None):
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(ValueError, "training"):
                    fit_training_label_contract(
                        (_example("X", ("V5869",), partition),), _config()
                    )

    def test_vocabulary_and_hash_are_stable_under_input_and_label_order(self) -> None:
        forward = (
            _example("A", ("4019", "250.00")),
            _example("B", ("4019",)),
        )
        reverse = (
            _example("B", ("4019",)),
            _example("A", ("250.00", "4019")),
        )

        first = fit_training_label_contract(forward, _config())
        second = fit_training_label_contract(reverse, _config())

        self.assertEqual(first, second)
        self.assertRegex(first.vocabulary_hash, re.compile(r"^[0-9a-f]{64}$"))

    def test_minimum_support_counts_distinct_admissions(self) -> None:
        training = (
            _example("A", ("250.00", "4019")),
            _example("B", ("4019",)),
        )

        contract = fit_training_label_contract(training, _config(minimum_support=2))

        self.assertEqual(contract.vocabulary, ("4019",))

    def test_fit_rejects_identical_duplicate_admission_before_support_counting(self) -> None:
        duplicate = _example("DUPLICATE", ("4019",))

        with self.assertRaisesRegex(ValueError, "^duplicate_training_admission$"):
            fit_training_label_contract(
                (duplicate, duplicate),
                _config(minimum_support=2),
            )

    def test_fit_rejects_conflicting_duplicate_admission_before_support_counting(self) -> None:
        training = (
            _example("DUPLICATE", ("4019",)),
            _example("DUPLICATE", ("250.00",)),
        )

        with self.assertRaisesRegex(ValueError, "^duplicate_training_admission$"):
            fit_training_label_contract(training, _config())

    def test_unknown_split_labels_are_excluded_and_counted_without_mutation(self) -> None:
        contract = fit_training_label_contract(
            (_example("A", ("250.00", "4019")),), _config()
        )
        before = contract.vocabulary_hash
        split_examples = (
            _example("V1", ("4019", "V5869"), "validation"),
            _example("T1", ("E8798",), "test"),
        )

        encoded = transform_labels(split_examples, contract)

        self.assertEqual(encoded.rows, ((0, 1), (0, 0)))
        self.assertEqual(encoded.audit.unknown_label_count, 2)
        self.assertEqual(encoded.audit.examples_with_unknown_labels, 2)
        self.assertEqual(contract.vocabulary, ("250.00", "4019"))
        self.assertEqual(contract.vocabulary_hash, before)

    def test_label_configs_contracts_and_audits_are_immutable_and_aggregate(self) -> None:
        config = _config()
        contract = fit_training_label_contract((_example("A", ("4019",)),), config)
        encoded = transform_labels(
            (_example("V1", ("V5869",), "validation"),), contract
        )

        with self.assertRaises(FrozenInstanceError):
            config.minimum_support = 2  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            contract.vocabulary = ()  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            encoded.audit.unknown_label_count = 0  # type: ignore[misc]

        self.assertEqual(
            set(asdict(encoded.audit)),
            {"example_count", "unknown_label_count", "examples_with_unknown_labels"},
        )
        self.assertNotIn("SYNTH-SUBJECT", repr(encoded.audit))
        self.assertNotIn("synthetic text", repr(encoded.audit))

    def test_invalid_label_config_or_empty_selected_vocabulary_fails(self) -> None:
        with self.assertRaises(ValueError):
            LabelConfig(selection="all_training_labels", minimum_support=0)
        with self.assertRaises(ValueError):
            LabelConfig(selection="all_rows", minimum_support=1)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            fit_training_label_contract(
                (_example("A", ("4019",)),), _config(minimum_support=2)
            )

    def test_direct_examples_reject_malformed_or_duplicate_label_sets(self) -> None:
        for labels in (
            (),
            ["4019"],
            (" 4019",),
            ("4019", "4019"),
            ("A41.9",),
        ):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    _example("BAD", labels)  # type: ignore[arg-type]

    def test_direct_label_contract_rejects_bad_vocabulary_hash_or_audit(self) -> None:
        valid_hash = "1a799f91e80489f19f12e5f18a5b3fec090c14d22b5ff66251cf34f781101b7d"
        valid_audit = LabelFitAudit(training_example_count=1, vocabulary_size=1)
        valid = LabelContract(
            selection="all_training_labels",
            minimum_support=1,
            vocabulary=("4019",),
            vocabulary_hash=valid_hash,
            audit=valid_audit,
        )
        self.assertEqual(valid.vocabulary, ("4019",))

        cases = (
            {"vocabulary": (), "vocabulary_hash": valid_hash, "audit": valid_audit},
            {
                "vocabulary": ("4019", "4019"),
                "vocabulary_hash": valid_hash,
                "audit": valid_audit,
            },
            {"vocabulary": ("4019",), "vocabulary_hash": "not-a-hash", "audit": valid_audit},
            {
                "vocabulary": ("4019",),
                "vocabulary_hash": "0" * 64,
                "audit": valid_audit,
            },
            {
                "vocabulary": ("4019",),
                "vocabulary_hash": valid_hash,
                "audit": LabelFitAudit(training_example_count=1, vocabulary_size=2),
            },
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    LabelContract(
                        selection="all_training_labels",
                        minimum_support=1,
                        **case,
                    )

    def test_direct_encoded_labels_reject_bad_rows_hash_or_audit(self) -> None:
        valid_hash = "1a799f91e80489f19f12e5f18a5b3fec090c14d22b5ff66251cf34f781101b7d"
        valid_audit = LabelTransformAudit(
            example_count=1,
            unknown_label_count=0,
            examples_with_unknown_labels=0,
        )
        cases = (
            (((1, 2),), valid_hash, valid_audit),
            (((1,), (0, 1)), valid_hash, LabelTransformAudit(2, 0, 0)),
            (((1,),), "not-a-hash", valid_audit),
            (((1,),), valid_hash, LabelTransformAudit(2, 0, 0)),
        )
        for rows, vocabulary_hash, audit in cases:
            with self.subTest(rows=rows, vocabulary_hash=vocabulary_hash, audit=audit):
                with self.assertRaises(ValueError):
                    EncodedLabels(
                        rows=rows,
                        vocabulary_hash=vocabulary_hash,
                        audit=audit,
                    )


if __name__ == "__main__":
    unittest.main()
