from __future__ import annotations

import re
import unittest
from dataclasses import FrozenInstanceError, asdict, replace

from clinicalbert_icd.data.contracts import (
    AdmissionExample,
    SplitAssignment,
    SplitAudit,
    SplitConfig,
)
from clinicalbert_icd.data.split import (
    assert_subject_disjoint,
    partition_manifest_digest,
    split_by_subject,
)


def _synthetic_examples() -> tuple[AdmissionExample, ...]:
    return (
        AdmissionExample(
            subject_id="SYNTH-SUBJECT-A",
            hadm_id="SYNTH-ADMISSION-A1",
            text="synthetic alpha one",
            labels=("250.00",),
        ),
        AdmissionExample(
            subject_id="SYNTH-SUBJECT-A",
            hadm_id="SYNTH-ADMISSION-A2",
            text="synthetic alpha two",
            labels=("4019",),
        ),
        AdmissionExample(
            subject_id="SYNTH-SUBJECT-B",
            hadm_id="SYNTH-ADMISSION-B1",
            text="synthetic beta",
            labels=("E8798",),
        ),
        AdmissionExample(
            subject_id="SYNTH-SUBJECT-C",
            hadm_id="SYNTH-ADMISSION-C1",
            text="synthetic gamma",
            labels=("V5869",),
        ),
    )


def _config(seed: int = 8408) -> SplitConfig:
    return SplitConfig(
        algorithm="sha256_subject",
        seed=seed,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )


def _membership(assignment: SplitAssignment) -> dict[str, str]:
    return {
        example.hadm_id: partition
        for partition, examples in (
            ("train", assignment.train),
            ("validation", assignment.validation),
            ("test", assignment.test),
        )
        for example in examples
    }


class SubjectSplitTests(unittest.TestCase):
    def test_sha256_subject_split_has_known_seeded_membership(self) -> None:
        assignment = split_by_subject(_synthetic_examples(), _config())
        membership = _membership(assignment)

        self.assertEqual(membership["SYNTH-ADMISSION-A1"], "train")
        self.assertEqual(membership["SYNTH-ADMISSION-A2"], "train")
        self.assertEqual(membership["SYNTH-ADMISSION-B1"], "test")
        self.assertEqual(membership["SYNTH-ADMISSION-C1"], "validation")

    def test_all_admissions_for_one_subject_share_a_partition(self) -> None:
        assignment = split_by_subject(_synthetic_examples(), _config())
        subject_partitions: dict[str, set[str]] = {}
        for partition, examples in (
            ("train", assignment.train),
            ("validation", assignment.validation),
            ("test", assignment.test),
        ):
            for example in examples:
                subject_partitions.setdefault(example.subject_id, set()).add(partition)

        self.assertTrue(all(len(partitions) == 1 for partitions in subject_partitions.values()))
        assert_subject_disjoint(assignment)

    def test_input_row_order_does_not_change_membership_or_public_digest(self) -> None:
        examples = _synthetic_examples()
        forward = split_by_subject(examples, _config())
        reverse = split_by_subject(tuple(reversed(examples)), _config())

        self.assertEqual(_membership(forward), _membership(reverse))
        self.assertEqual(
            partition_manifest_digest(forward), partition_manifest_digest(reverse)
        )

    def test_explicit_seed_changes_the_seeded_assignment(self) -> None:
        first = split_by_subject(_synthetic_examples(), _config(seed=8408))
        second = split_by_subject(_synthetic_examples(), _config(seed=1))

        self.assertNotEqual(_membership(first), _membership(second))

    def test_pairwise_overlap_is_rejected(self) -> None:
        example = _synthetic_examples()[0]
        audit = SplitAudit(
            algorithm="sha256_subject",
            seed=8408,
            ratios=(0.6, 0.2, 0.2),
            subject_counts=(("train", 1), ("validation", 0), ("test", 1)),
            admission_counts=(("train", 1), ("validation", 0), ("test", 1)),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            SplitAssignment(
                train=(replace(example, partition="train"),),
                validation=(),
                test=(
                    replace(
                        example,
                        hadm_id="SYNTH-ADMISSION-A-OVERLAP",
                        partition="test",
                    ),
                ),
                audit=audit,
            )

    def test_duplicate_admission_is_rejected_even_for_the_same_subject(self) -> None:
        example = _synthetic_examples()[0]
        duplicates = (
            (example, example),
            (example, replace(example, text="synthetic duplicate changed")),
        )

        for values in duplicates:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "duplicate admission"):
                    split_by_subject(values, _config())

    def test_empty_split_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            split_by_subject((), _config())

    def test_direct_assignment_rejects_partition_or_audit_mismatch(self) -> None:
        assignment = split_by_subject(_synthetic_examples(), _config())
        misplaced = replace(assignment.train[0], partition="validation")
        bad_counts = replace(
            assignment.audit,
            admission_counts=(("train", 999), ("validation", 0), ("test", 0)),
        )

        with self.assertRaisesRegex(ValueError, "partition"):
            SplitAssignment(
                train=(misplaced,),
                validation=assignment.validation,
                test=assignment.test,
                audit=assignment.audit,
            )
        with self.assertRaisesRegex(ValueError, "audit"):
            SplitAssignment(
                train=assignment.train,
                validation=assignment.validation,
                test=assignment.test,
                audit=bad_counts,
            )

    def test_direct_split_audit_rejects_invalid_algorithm_or_counts(self) -> None:
        with self.assertRaises(ValueError):
            SplitAudit(
                algorithm="row_random",
                seed=8408,
                ratios=(0.6, 0.2, 0.2),
                subject_counts=(("train", 1), ("validation", 1), ("test", 1)),
                admission_counts=(("train", 1), ("validation", 1), ("test", 1)),
            )
        with self.assertRaises(ValueError):
            SplitAudit(
                algorithm="sha256_subject",
                seed=8408,
                ratios=(0.6, 0.2, 0.2),
                subject_counts=(("train", -1), ("validation", 1), ("test", 1)),
                admission_counts=(("train", 1), ("validation", 1), ("test", 1)),
            )

    def test_public_audit_and_digest_contain_aggregates_not_records(self) -> None:
        assignment = split_by_subject(_synthetic_examples(), _config())
        digest = partition_manifest_digest(assignment)

        self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
        audit_fields = set(asdict(assignment.audit))
        self.assertEqual(
            audit_fields,
            {"algorithm", "seed", "ratios", "subject_counts", "admission_counts"},
        )
        self.assertNotIn("SYNTH-SUBJECT", repr(assignment.audit))
        self.assertNotIn("synthetic alpha", repr(assignment.audit))

    def test_split_configs_and_audits_are_immutable(self) -> None:
        config = _config()
        audit = split_by_subject(_synthetic_examples(), config).audit

        with self.assertRaises(FrozenInstanceError):
            config.seed = 9  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            audit.seed = 9  # type: ignore[misc]

    def test_invalid_ratios_and_unknown_algorithm_fail(self) -> None:
        with self.assertRaises(ValueError):
            SplitConfig(
                algorithm="sha256_subject",
                seed=8408,
                train_ratio=0.7,
                validation_ratio=0.2,
                test_ratio=0.2,
            )
        with self.assertRaises(ValueError):
            SplitConfig(
                algorithm="row_random",
                seed=8408,
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.2,
            )


if __name__ == "__main__":
    unittest.main()
