from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ── identifier helpers ─────────────────────────────────────────────────────────

def _canonical_id(value: Any, name: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool not accepted")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if not value or value != value.strip():
            raise ValueError(f"{name}: must be non-empty with no surrounding whitespace")
        return value
    raise ValueError(f"{name}: must be str or int, got {type(value).__name__}")


# ── ICD-9 format check (native codes only, no crosswalk) ──────────────────────

_ICD9_RE = re.compile(
    r"^(?:V\d{2,4}(?:\.\d{1,2})?|E\d{3,4}(?:\.\d{1,2})?|\d{3,5}(?:\.\d{1,2})?)$"
)


def _validate_icd9(code: str) -> None:
    if not _ICD9_RE.match(code):
        raise ValueError(f"invalid native ICD-9 code: {code!r}")


# ── deduplication ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class NoteRecord:
    subject_id: str
    hadm_id: str
    text: str
    category: str
    order_value: Any
    tie_break_values: tuple

    def __post_init__(self) -> None:
        for v in self.tie_break_values:
            if isinstance(v, bool):
                raise ValueError("tie_break_values: bool not accepted")
            if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                raise ValueError("tie_break_values: nan/inf not accepted")
            if isinstance(v, (list, dict)):
                raise ValueError("tie_break_values: mutable collections not accepted")
            try:
                hash(v)
            except TypeError as exc:
                raise ValueError(f"tie_break_values: unhashable {type(v).__name__}") from exc


@dataclass(frozen=True, slots=True)
class DedupPolicy:
    category_field: str
    category_value: str
    order_field: str
    keep: str
    tie_break_fields: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.tie_break_fields, tuple):
            raise ValueError("tie_break_fields must be a tuple")
        if self.keep not in ("latest", "earliest"):
            raise ValueError(f"keep must be 'latest' or 'earliest', got {self.keep!r}")


@dataclass(frozen=True, slots=True)
class DedupAudit:
    input_record_count: int
    eligible_record_count: int
    output_record_count: int
    dropped_record_count: int
    duplicate_record_count: int

    def __post_init__(self) -> None:
        fields = (
            self.input_record_count,
            self.eligible_record_count,
            self.output_record_count,
            self.dropped_record_count,
            self.duplicate_record_count,
        )
        if any(c < 0 for c in fields):
            raise ValueError("all counts must be non-negative")
        if self.eligible_record_count > self.input_record_count:
            raise ValueError("eligible_record_count > input_record_count")
        if self.output_record_count > self.eligible_record_count:
            raise ValueError("output_record_count > eligible_record_count")
        if self.dropped_record_count + self.output_record_count != self.eligible_record_count:
            raise ValueError("dropped + output != eligible")
        if self.eligible_record_count > 0 and self.output_record_count == 0:
            raise ValueError("eligible records present but output_record_count is zero")
        if self.dropped_record_count > 0 and self.duplicate_record_count == 0:
            raise ValueError("records were dropped but duplicate_record_count is zero")
        if self.dropped_record_count == 0 and self.duplicate_record_count > 0:
            raise ValueError("duplicate_record_count > 0 but dropped_record_count is zero")
        if self.duplicate_record_count > self.dropped_record_count:
            raise ValueError("duplicate_record_count > dropped_record_count")


@dataclass(frozen=True, slots=True)
class DedupResult:
    records: tuple
    audit: DedupAudit

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise ValueError("records must be a tuple")
        if len(self.records) == 0:
            raise ValueError("records must not be empty")
        if len(self.records) != self.audit.output_record_count:
            raise ValueError(
                f"len(records)={len(self.records)} != audit.output_record_count="
                f"{self.audit.output_record_count}"
            )
        seen: set[str] = set()
        for r in self.records:
            if r.hadm_id in seen:
                raise ValueError(f"duplicate hadm_id {r.hadm_id!r} in records")
            seen.add(r.hadm_id)


# ── admission examples ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AdmissionExample:
    subject_id: str
    hadm_id: str
    text: str
    labels: tuple
    partition: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _canonical_id(self.subject_id, "subject_id"))
        object.__setattr__(self, "hadm_id", _canonical_id(self.hadm_id, "hadm_id"))
        if not isinstance(self.labels, tuple):
            raise ValueError("labels must be a tuple")
        if len(self.labels) == 0:
            raise ValueError("labels must not be empty")
        seen: set[str] = set()
        for lbl in self.labels:
            if not isinstance(lbl, str) or lbl != lbl.strip() or not lbl:
                raise ValueError(f"label {lbl!r}: must be a non-empty stripped string")
            _validate_icd9(lbl)
            if lbl in seen:
                raise ValueError(f"duplicate label {lbl!r}")
            seen.add(lbl)


# ── label contracts ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LabelConfig:
    selection: str
    minimum_support: int

    def __post_init__(self) -> None:
        if self.selection != "all_training_labels":
            raise ValueError(
                f"selection must be 'all_training_labels', got {self.selection!r}"
            )
        if self.minimum_support < 1:
            raise ValueError(f"minimum_support must be >= 1, got {self.minimum_support}")


@dataclass(frozen=True, slots=True)
class LabelFitAudit:
    training_example_count: int
    vocabulary_size: int

    def __post_init__(self) -> None:
        if self.training_example_count < 0:
            raise ValueError("training_example_count must be non-negative")
        if self.vocabulary_size < 0:
            raise ValueError("vocabulary_size must be non-negative")


@dataclass(frozen=True, slots=True)
class LabelContract:
    selection: str
    minimum_support: int
    vocabulary: tuple
    vocabulary_hash: str
    audit: LabelFitAudit

    def __post_init__(self) -> None:
        if len(self.vocabulary) == 0:
            raise ValueError("vocabulary must not be empty")
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("vocabulary contains duplicates")
        if not re.match(r"^[0-9a-f]{64}$", self.vocabulary_hash):
            raise ValueError("vocabulary_hash must be 64 lowercase hex characters")
        if self.vocabulary_hash == "0" * 64:
            raise ValueError("vocabulary_hash must not be the zero hash")
        if self.audit.vocabulary_size != len(self.vocabulary):
            raise ValueError(
                f"audit.vocabulary_size={self.audit.vocabulary_size} != "
                f"len(vocabulary)={len(self.vocabulary)}"
            )


@dataclass(frozen=True, slots=True)
class LabelTransformAudit:
    example_count: int
    unknown_label_count: int
    examples_with_unknown_labels: int

    def __post_init__(self) -> None:
        if self.example_count < 0:
            raise ValueError("example_count must be non-negative")
        if self.unknown_label_count < 0:
            raise ValueError("unknown_label_count must be non-negative")
        if self.examples_with_unknown_labels < 0:
            raise ValueError("examples_with_unknown_labels must be non-negative")


@dataclass(frozen=True, slots=True)
class EncodedLabels:
    rows: tuple
    vocabulary_hash: str
    audit: LabelTransformAudit

    def __post_init__(self) -> None:
        if not re.match(r"^[0-9a-f]{64}$", self.vocabulary_hash):
            raise ValueError("vocabulary_hash must be 64 lowercase hex characters")
        if self.audit.example_count != len(self.rows):
            raise ValueError(
                f"audit.example_count={self.audit.example_count} != "
                f"len(rows)={len(self.rows)}"
            )
        if len(self.rows) == 0:
            return
        row_len = len(self.rows[0])
        for row in self.rows:
            if len(row) != row_len:
                raise ValueError("all rows must have the same length")
            for v in row:
                if v not in (0, 1):
                    raise ValueError(f"row values must be 0 or 1, got {v!r}")


# ── split contracts ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SplitConfig:
    algorithm: str
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def __post_init__(self) -> None:
        if self.algorithm != "sha256_subject":
            raise ValueError(
                f"algorithm must be 'sha256_subject', got {self.algorithm!r}"
            )
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ratios must sum to 1.0, got {total}")


@dataclass(frozen=True, slots=True)
class SplitAudit:
    algorithm: str
    seed: int
    ratios: tuple
    subject_counts: tuple
    admission_counts: tuple

    def __post_init__(self) -> None:
        if self.algorithm != "sha256_subject":
            raise ValueError(
                f"algorithm must be 'sha256_subject', got {self.algorithm!r}"
            )
        for _, count in self.subject_counts:
            if count < 0:
                raise ValueError("subject counts must be non-negative")
        for _, count in self.admission_counts:
            if count < 0:
                raise ValueError("admission counts must be non-negative")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    train: tuple
    validation: tuple
    test: tuple
    audit: SplitAudit

    def __post_init__(self) -> None:
        partition_map = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        for part_name, examples in partition_map.items():
            for ex in examples:
                if ex.partition != part_name:
                    raise ValueError(
                        f"example {ex.hadm_id!r} has partition={ex.partition!r} "
                        f"but is in {part_name!r} tuple"
                    )
        train_subjects = {ex.subject_id for ex in self.train}
        val_subjects = {ex.subject_id for ex in self.validation}
        test_subjects = {ex.subject_id for ex in self.test}
        overlap_tv = train_subjects & val_subjects
        overlap_tt = train_subjects & test_subjects
        overlap_vt = val_subjects & test_subjects
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"subject overlap detected: train∩val={overlap_tv}, "
                f"train∩test={overlap_tt}, val∩test={overlap_vt}"
            )
        expected_admission_counts = {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }
        actual_audit_counts = dict(self.audit.admission_counts)
        for part_name, expected in expected_admission_counts.items():
            actual = actual_audit_counts.get(part_name, 0)
            if actual != expected:
                raise ValueError(
                    f"audit.admission_counts[{part_name!r}]={actual} != "
                    f"len({part_name})={expected}"
                )


# ── tokenization contracts ─────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    identifier: str
    revision: str
    maximum_length: int
    truncation_side: str
    padding: str

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("revision must not be empty")
        is_synthetic = self.identifier.startswith("synthetic-")
        if is_synthetic:
            if not (self.revision.startswith("test-") or self.revision.startswith("test:")):
                raise ValueError(
                    f"synthetic identifier requires revision starting with 'test-' or 'test:', "
                    f"got {self.revision!r}"
                )
        else:
            commit_ok = bool(re.match(r"^commit:[0-9a-f]{40}$", self.revision))
            sha256_ok = bool(re.match(r"^sha256:[0-9a-f]{64}$", self.revision))
            if not (commit_ok or sha256_ok):
                raise ValueError(
                    f"non-synthetic identifier requires revision matching "
                    f"'commit:<40hex>' or 'sha256:<64hex>', got {self.revision!r}"
                )
        if self.truncation_side not in ("right", "left"):
            raise ValueError(
                f"truncation_side must be 'right' or 'left', got {self.truncation_side!r}"
            )
        if self.padding not in ("max_length", "do_not_pad"):
            raise ValueError(
                f"padding must be 'max_length' or 'do_not_pad', got {self.padding!r}"
            )


@dataclass(frozen=True, slots=True)
class TokenizationAudit:
    tokenizer_identifier: str
    tokenizer_revision: str
    maximum_length: int
    truncation_side: str
    padding: str
    example_count: int
    truncated_example_count: int
    total_content_tokens_before: int
    total_content_tokens_after: int
    special_tokens_per_example: int


@dataclass(frozen=True, slots=True)
class TokenizedBatch:
    input_ids: tuple
    attention_mask: tuple
    audit: TokenizationAudit

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError(
                f"input_ids has {len(self.input_ids)} rows but "
                f"attention_mask has {len(self.attention_mask)} rows"
            )
        for i, (ids_row, mask_row) in enumerate(zip(self.input_ids, self.attention_mask)):
            if len(ids_row) != len(mask_row):
                raise ValueError(
                    f"row {i}: input_ids length {len(ids_row)} != "
                    f"attention_mask length {len(mask_row)}"
                )
        if self.audit.example_count != len(self.input_ids):
            raise ValueError(
                f"audit.example_count={self.audit.example_count} != "
                f"len(input_ids)={len(self.input_ids)}"
            )
