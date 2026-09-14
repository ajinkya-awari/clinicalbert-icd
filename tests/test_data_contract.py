from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, asdict

from clinicalbert_icd.data.contracts import (
    AdmissionExample,
    DedupAudit,
    DedupPolicy,
    DedupResult,
    NoteRecord,
)
from clinicalbert_icd.data.prepare import (
    build_admission_examples,
    deduplicate_admissions,
    parse_native_icd9,
)
from tests.fixtures.synthetic_records import SYNTHETIC_DIAGNOSES, SYNTHETIC_NOTES


class AdmissionDeduplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DedupPolicy(
            category_field="CATEGORY",
            category_value="Discharge summary",
            order_field="note_order",
            keep="latest",
            tie_break_fields=("note_row_id",),
        )

    def test_latest_policy_selects_one_note_per_admission(self) -> None:
        result = deduplicate_admissions(SYNTHETIC_NOTES, self.policy)
        selected = {record.hadm_id: record.text for record in result.records}

        self.assertEqual(
            selected["SYNTH-ADMISSION-A1"], "synthetic final alpha note"
        )
        self.assertEqual(len(selected), len(result.records))
        self.assertEqual(result.audit.dropped_record_count, 1)

    def test_tie_break_is_deterministic_under_row_reordering(self) -> None:
        tied = (
            {
                "SUBJECT_ID": "SYNTH-SUBJECT-T",
                "HADM_ID": "SYNTH-ADMISSION-T1",
                "TEXT": "synthetic tie first",
                "CATEGORY": "Discharge summary",
                "note_order": 7,
                "note_row_id": "T-01",
            },
            {
                "SUBJECT_ID": "SYNTH-SUBJECT-T",
                "HADM_ID": "SYNTH-ADMISSION-T1",
                "TEXT": "synthetic tie second",
                "CATEGORY": "Discharge summary",
                "note_order": 7,
                "note_row_id": "T-02",
            },
        )

        forward = deduplicate_admissions(tied, self.policy)
        reverse = deduplicate_admissions(tuple(reversed(tied)), self.policy)

        self.assertEqual(forward.records, reverse.records)
        self.assertEqual(forward.records[0].text, "synthetic tie second")

    def test_unresolved_tie_fails_instead_of_using_row_order(self) -> None:
        ambiguous = (
            {
                "SUBJECT_ID": "SYNTH-SUBJECT-T",
                "HADM_ID": "SYNTH-ADMISSION-T1",
                "TEXT": "synthetic ambiguous one",
                "CATEGORY": "Discharge summary",
                "note_order": 7,
                "note_row_id": "T-01",
            },
            {
                "SUBJECT_ID": "SYNTH-SUBJECT-T",
                "HADM_ID": "SYNTH-ADMISSION-T1",
                "TEXT": "synthetic ambiguous two",
                "CATEGORY": "Discharge summary",
                "note_order": 7,
                "note_row_id": "T-01",
            },
        )

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            deduplicate_admissions(ambiguous, self.policy)

    def test_order_and_tie_break_types_must_be_consistent_per_admission(self) -> None:
        base = {
            "SUBJECT_ID": "SYNTH-SUBJECT-T",
            "HADM_ID": "SYNTH-ADMISSION-T1",
            "CATEGORY": "Discharge summary",
        }
        cases = (
            (
                {**base, "TEXT": "one", "note_order": 1, "note_row_id": 1},
                {**base, "TEXT": "two", "note_order": "2", "note_row_id": 2},
            ),
            (
                {**base, "TEXT": "one", "note_order": 1, "note_row_id": 1},
                {**base, "TEXT": "two", "note_order": 2, "note_row_id": "2"},
            ),
        )

        for records in cases:
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, "type"):
                    deduplicate_admissions(records, self.policy)

    def test_numeric_ordering_preserves_adjacent_integers_above_float_precision(self) -> None:
        base = {
            "SUBJECT_ID": "SYNTH-SUBJECT-T",
            "HADM_ID": "SYNTH-ADMISSION-T1",
            "CATEGORY": "Discharge summary",
            "note_row_id": "T-01",
        }
        records = (
            {**base, "TEXT": "synthetic lower integer", "note_order": 2**53},
            {**base, "TEXT": "synthetic higher integer", "note_order": 2**53 + 1},
        )

        result = deduplicate_admissions(records, self.policy)

        self.assertEqual(result.records[0].text, "synthetic higher integer")

    def test_numeric_ordering_accepts_arbitrary_size_integers_without_overflow(self) -> None:
        lower = 10**10000
        base = {
            "SUBJECT_ID": "SYNTH-SUBJECT-T",
            "HADM_ID": "SYNTH-ADMISSION-T1",
            "CATEGORY": "Discharge summary",
            "note_row_id": "T-01",
        }
        records = (
            {**base, "TEXT": "synthetic lower huge integer", "note_order": lower},
            {
                **base,
                "TEXT": "synthetic higher huge integer",
                "note_order": lower + 1,
            },
        )

        result = deduplicate_admissions(records, self.policy)

        self.assertEqual(result.records[0].text, "synthetic higher huge integer")

    def test_conflicting_admission_ownership_fails(self) -> None:
        conflicting = (
            SYNTHETIC_NOTES[0],
            {**SYNTHETIC_NOTES[1], "SUBJECT_ID": "SYNTH-SUBJECT-OTHER"},
        )

        with self.assertRaisesRegex(ValueError, "ownership"):
            deduplicate_admissions(conflicting, self.policy)

    def test_missing_required_note_field_fails(self) -> None:
        invalid = ({key: value for key, value in SYNTHETIC_NOTES[0].items() if key != "TEXT"},)

        with self.assertRaisesRegex(ValueError, "TEXT"):
            deduplicate_admissions(invalid, self.policy)

    def test_non_target_note_category_is_not_selected(self) -> None:
        extra = {
            **SYNTHETIC_NOTES[1],
            "TEXT": "synthetic non-discharge note",
            "CATEGORY": "Radiology",
            "note_order": 999,
            "note_row_id": "A1-RAD",
        }

        result = deduplicate_admissions((*SYNTHETIC_NOTES, extra), self.policy)
        selected = {record.hadm_id: record.text for record in result.records}

        self.assertEqual(
            selected["SYNTH-ADMISSION-A1"], "synthetic final alpha note"
        )

    def test_contract_and_audit_are_immutable_and_audit_is_aggregate_only(self) -> None:
        result = deduplicate_admissions(SYNTHETIC_NOTES, self.policy)

        with self.assertRaises(FrozenInstanceError):
            self.policy.keep = "earliest"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.audit.input_record_count = 999  # type: ignore[misc]

        audit_fields = set(asdict(result.audit))
        self.assertFalse(audit_fields & {"subject_id", "hadm_id", "text", "records"})
        self.assertNotIn("SYNTH-SUBJECT", repr(result.audit))
        self.assertNotIn("synthetic final", repr(result.audit))

    def test_prepare_identifiers_accept_only_canonicalizable_strings_or_integers(self) -> None:
        valid = {
            **SYNTHETIC_NOTES[2],
            "SUBJECT_ID": "  SYNTH-SUBJECT-A  ",
            "HADM_ID": 7001,
        }

        result = deduplicate_admissions((valid,), self.policy)

        self.assertEqual(result.records[0].subject_id, "SYNTH-SUBJECT-A")
        self.assertEqual(result.records[0].hadm_id, "7001")

        for invalid in (True, 1.5, float("nan"), ["X"], {"X": 1}, "   "):
            with self.subTest(invalid=invalid):
                record = {**SYNTHETIC_NOTES[2], "SUBJECT_ID": invalid}
                with self.assertRaisesRegex(ValueError, "SUBJECT_ID"):
                    deduplicate_admissions((record,), self.policy)

    def test_deduplication_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            deduplicate_admissions((), self.policy)

    def test_policy_requires_an_exact_tuple_of_tie_break_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "tuple"):
            DedupPolicy(
                category_field="CATEGORY",
                category_value="Discharge summary",
                order_field="note_order",
                keep="latest",
                tie_break_fields=["note_row_id"],  # type: ignore[arg-type]
            )


class DirectDedupContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.note = NoteRecord(
            subject_id="SYNTH-SUBJECT-DIRECT",
            hadm_id="SYNTH-ADMISSION-DIRECT",
            text="synthetic direct note",
            category="Discharge summary",
            order_value=1,
            tie_break_values=("DIRECT-01",),
        )

    def test_note_tie_break_members_must_be_finite_deterministic_scalars(self) -> None:
        for invalid in (
            True,
            float("nan"),
            float("inf"),
            float("-inf"),
            ["unhashable"],
            {"unhashable": 1},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "tie_break"):
                    NoteRecord(
                        subject_id="SYNTH-SUBJECT-DIRECT",
                        hadm_id="SYNTH-ADMISSION-DIRECT",
                        text="synthetic direct note",
                        category="Discharge summary",
                        order_value=1,
                        tie_break_values=(invalid,),
                    )

    def test_dedup_audit_rejects_negative_or_inconsistent_counts(self) -> None:
        invalid_counts = (
            (-1, 0, 0, 0, 0),
            (1, 2, 1, 1, 0),
            (2, 1, 2, 0, 0),
            (2, 2, 1, 0, 1),
            (2, 2, 1, 1, 2),
        )
        for counts in invalid_counts:
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    DedupAudit(*counts)

    def test_dedup_audit_requires_an_output_when_eligible_records_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "eligible"):
            DedupAudit(1, 1, 0, 1, 0)

    def test_dedup_audit_requires_a_duplicate_when_records_are_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            DedupAudit(2, 2, 1, 1, 0)

    def test_dedup_result_rejects_a_contradictory_duplicate_audit(self) -> None:
        with self.assertRaises(ValueError):
            DedupResult(
                records=(self.note,),
                audit=DedupAudit(2, 2, 1, 1, 0),
            )

    def test_dedup_result_rejects_empty_duplicate_or_count_mismatched_records(self) -> None:
        empty_audit = DedupAudit(0, 0, 0, 0, 0)
        one_record_audit = DedupAudit(1, 1, 1, 0, 0)
        two_record_audit = DedupAudit(2, 2, 2, 0, 0)
        cases = (
            ((), empty_audit),
            ([self.note], one_record_audit),
            ((self.note,), two_record_audit),
            ((self.note, self.note), two_record_audit),
        )
        for records, audit in cases:
            with self.subTest(records=records, audit=audit):
                with self.assertRaises(ValueError):
                    DedupResult(records=records, audit=audit)  # type: ignore[arg-type]


class DirectAdmissionContractTests(unittest.TestCase):
    def test_direct_identifiers_normalize_integers_and_reject_noncanonical_values(self) -> None:
        example = AdmissionExample(
            subject_id=7,
            hadm_id=11,
            text="synthetic direct identifiers",
            labels=("4019",),
        )

        self.assertEqual(example.subject_id, "7")
        self.assertEqual(example.hadm_id, "11")

        for invalid in (" X ", True, 1.5, float("nan"), ["X"], {"X": 1}, ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    AdmissionExample(
                        subject_id=invalid,
                        hadm_id="SYNTH-ADMISSION-DIRECT",
                        text="synthetic direct identifiers",
                        labels=("4019",),
                    )


class NativeICD9Tests(unittest.TestCase):
    def test_valid_native_lexemes_are_stripped_but_otherwise_preserved(self) -> None:
        cases = {
            " 250.00 ": "250.00",
            "25000": "25000",
            "4019": "4019",
            "V5869": "V5869",
            "E8798": "E8798",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_native_icd9(raw), expected)

    def test_invalid_or_non_native_lexemes_fail_without_conversion(self) -> None:
        for value in ("", "A41.9", "250-00", "v5869", "123456", "E12"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_native_icd9(value)


class AdmissionAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        policy = DedupPolicy(
            category_field="CATEGORY",
            category_value="Discharge summary",
            order_field="note_order",
            keep="latest",
            tie_break_fields=("note_row_id",),
        )
        self.notes = deduplicate_admissions(SYNTHETIC_NOTES, policy).records

    def test_build_examples_aggregates_unique_sorted_native_labels(self) -> None:
        examples = build_admission_examples(self.notes, SYNTHETIC_DIAGNOSES)
        by_admission = {example.hadm_id: example for example in examples}

        self.assertEqual(
            by_admission["SYNTH-ADMISSION-A1"].labels, ("250.00", "4019")
        )
        self.assertEqual(
            by_admission["SYNTH-ADMISSION-C1"].labels, ("25000",)
        )

    def test_build_examples_is_independent_of_note_and_diagnosis_row_order(self) -> None:
        forward = build_admission_examples(self.notes, SYNTHETIC_DIAGNOSES)
        reverse = build_admission_examples(
            tuple(reversed(self.notes)), tuple(reversed(SYNTHETIC_DIAGNOSES))
        )

        self.assertEqual(forward, reverse)

    def test_diagnosis_with_conflicting_subject_ownership_fails(self) -> None:
        conflicting = (
            {
                **SYNTHETIC_DIAGNOSES[0],
                "SUBJECT_ID": "SYNTH-SUBJECT-OTHER",
            },
        )

        with self.assertRaisesRegex(ValueError, "ownership"):
            build_admission_examples(self.notes, conflicting)

    def test_duplicate_note_admission_must_be_deduplicated_first(self) -> None:
        with self.assertRaisesRegex(ValueError, "deduplicate"):
            build_admission_examples(SYNTHETIC_NOTES[:2], SYNTHETIC_DIAGNOSES[:1])

    def test_build_examples_rejects_empty_note_or_diagnosis_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "notes.*empty"):
            build_admission_examples((), SYNTHETIC_DIAGNOSES[:1])
        with self.assertRaisesRegex(ValueError, "diagnosis.*empty"):
            build_admission_examples(self.notes[:1], ())

    def test_admission_note_with_no_matching_diagnosis_rows_raises_naming_the_admission(self) -> None:
        # self.notes contains A1, A2, B1, and C1 after deduplication.
        # Supply all diagnosis rows except C1's; C1 gets no labels → clear error.
        # Sorted iteration visits subjects A→B→C, so C1 is the first to lack labels.
        diagnoses_without_c1 = tuple(
            row for row in SYNTHETIC_DIAGNOSES
            if row["HADM_ID"] != "SYNTH-ADMISSION-C1"
        )
        self.assertGreater(len(diagnoses_without_c1), 0)

        with self.assertRaisesRegex(ValueError, "SYNTH-ADMISSION-C1"):
            build_admission_examples(self.notes, diagnoses_without_c1)


if __name__ == "__main__":
    unittest.main()
