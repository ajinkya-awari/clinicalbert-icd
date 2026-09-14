"""Immutable contracts for the offline synthetic data pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from math import isclose


_PARTITIONS = frozenset({"train", "validation", "test"})
_PARTITION_ORDER = ("train", "validation", "test")
_ICD9_PATTERN = re.compile(
    r"^(?:[0-9]{3,5}|[0-9]{3}\.[0-9]{1,2}|V[0-9]{2,4}|"
    r"V[0-9]{2}\.[0-9]{1,2}|E[0-9]{3,5}|E[0-9]{3}\.[0-9]{1,2})$"
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_REVISION = re.compile(r"commit:[0-9a-f]{40,64}\Z")
_SHA256_REVISION = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TEST_REVISION = re.compile(r"test(?::|-)[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _canonical_identifier(
    name: str, value: object, *, reject_surrounding_whitespace: bool
) -> str:
    if type(value) is int:
        return str(value)
    if type(value) is not str:
        raise ValueError(f"{name} must be a string or integer identifier")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier")
    if reject_surrounding_whitespace and normalized != value:
        raise ValueError(f"{name} must already be canonical without surrounding whitespace")
    return normalized


def _canonical_icd9(value: object, *, reject_surrounding_whitespace: bool) -> str:
    if type(value) is not str:
        raise ValueError("ICD-9 value must be a string")
    normalized = value.strip()
    if reject_surrounding_whitespace and normalized != value:
        raise ValueError("ICD-9 value must already be canonical")
    if not _ICD9_PATTERN.fullmatch(normalized):
        raise ValueError("invalid native ICD-9 lexical value")
    return normalized


def _label_vocabulary_digest(vocabulary: tuple[str, ...]) -> str:
    payload = json.dumps(
        vocabulary, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_revision_valid(identifier: str, revision: str) -> bool:
    if _COMMIT_REVISION.fullmatch(revision) or _SHA256_REVISION.fullmatch(revision):
        return True
    return identifier.startswith("synthetic-") and _TEST_REVISION.fullmatch(revision) is not None


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _deterministic_scalar(name: str, value: object) -> None:
    if type(value) not in (str, int, float):
        raise ValueError(f"{name} must contain deterministic scalar values")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must contain finite scalar values")


@dataclass(frozen=True, slots=True)
class DedupPolicy:
    category_field: str
    category_value: str
    order_field: str
    keep: str
    tie_break_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string("category_field", self.category_field)
        _require_nonempty_string("category_value", self.category_value)
        _require_nonempty_string("order_field", self.order_field)
        if self.keep not in {"earliest", "latest"}:
            raise ValueError("keep must be 'earliest' or 'latest'")
        if not isinstance(self.tie_break_fields, tuple) or not self.tie_break_fields:
            raise ValueError("tie_break_fields must be an explicit non-empty tuple")
        if any(not isinstance(field, str) or not field for field in self.tie_break_fields):
            raise ValueError("tie_break_fields must contain non-empty strings")
        if len(set(self.tie_break_fields)) != len(self.tie_break_fields):
            raise ValueError("tie_break_fields must be unique")


@dataclass(frozen=True, slots=True)
class NoteRecord:
    subject_id: str
    hadm_id: str
    text: str
    category: str
    order_value: str | int | float
    tie_break_values: tuple[str | int | float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject_id",
            _canonical_identifier(
                "subject_id", self.subject_id, reject_surrounding_whitespace=True
            ),
        )
        object.__setattr__(
            self,
            "hadm_id",
            _canonical_identifier(
                "hadm_id", self.hadm_id, reject_surrounding_whitespace=True
            ),
        )
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        _require_nonempty_string("category", self.category)
        _deterministic_scalar("order_value", self.order_value)
        if not isinstance(self.tie_break_values, tuple):
            raise ValueError("tie_break_values must be a tuple")
        for value in self.tie_break_values:
            _deterministic_scalar("tie_break_values", value)


@dataclass(frozen=True, slots=True)
class DedupAudit:
    input_record_count: int
    eligible_record_count: int
    output_record_count: int
    dropped_record_count: int
    duplicate_admission_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("input_record_count", self.input_record_count),
            ("eligible_record_count", self.eligible_record_count),
            ("output_record_count", self.output_record_count),
            ("dropped_record_count", self.dropped_record_count),
            ("duplicate_admission_count", self.duplicate_admission_count),
        ):
            _nonnegative_int(name, value)
        if self.eligible_record_count > self.input_record_count:
            raise ValueError("eligible record count exceeds input record count")
        if self.output_record_count > self.eligible_record_count:
            raise ValueError("output record count exceeds eligible record count")
        if self.dropped_record_count != (
            self.eligible_record_count - self.output_record_count
        ):
            raise ValueError("dropped record count is inconsistent")
        if self.eligible_record_count > 0 and self.output_record_count == 0:
            raise ValueError("eligible records require at least one output record")
        if self.dropped_record_count > 0 and self.duplicate_admission_count == 0:
            raise ValueError("dropped records require at least one duplicate admission")
        if self.duplicate_admission_count > min(
            self.output_record_count, self.dropped_record_count
        ):
            raise ValueError("duplicate admission count is inconsistent")


@dataclass(frozen=True, slots=True)
class DedupResult:
    records: tuple[NoteRecord, ...]
    audit: DedupAudit

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("dedup result records must be a non-empty tuple")
        if any(not isinstance(record, NoteRecord) for record in self.records):
            raise ValueError("dedup result contains invalid records")
        if not isinstance(self.audit, DedupAudit):
            raise ValueError("dedup result audit is invalid")
        if self.audit.output_record_count != len(self.records):
            raise ValueError("dedup result audit does not match records")
        admission_ids = tuple(record.hadm_id for record in self.records)
        if len(admission_ids) != len(set(admission_ids)):
            raise ValueError("dedup result contains duplicate admissions")


@dataclass(frozen=True, slots=True)
class AdmissionExample:
    subject_id: str
    hadm_id: str
    text: str
    labels: tuple[str, ...]
    partition: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject_id",
            _canonical_identifier(
                "subject_id", self.subject_id, reject_surrounding_whitespace=True
            ),
        )
        object.__setattr__(
            self,
            "hadm_id",
            _canonical_identifier(
                "hadm_id", self.hadm_id, reject_surrounding_whitespace=True
            ),
        )
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.labels, tuple) or not self.labels:
            raise ValueError("labels must be a non-empty tuple")
        canonical_labels = tuple(
            _canonical_icd9(label, reject_surrounding_whitespace=True)
            for label in self.labels
        )
        if len(canonical_labels) != len(set(canonical_labels)):
            raise ValueError("labels must be unique within an admission")
        if self.partition is not None and self.partition not in _PARTITIONS:
            raise ValueError("partition must be train, validation, test, or None")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    algorithm: str
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def __post_init__(self) -> None:
        if self.algorithm != "sha256_subject":
            raise ValueError("algorithm must be 'sha256_subject'")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an explicit integer")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ratios):
            raise ValueError("split ratios must be numeric")
        if any(value <= 0 or value >= 1 for value in ratios):
            raise ValueError("split ratios must each be between zero and one")
        if not isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("split ratios must sum to one")

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.train_ratio, self.validation_ratio, self.test_ratio)


@dataclass(frozen=True, slots=True)
class SplitAudit:
    algorithm: str
    seed: int
    ratios: tuple[float, float, float]
    subject_counts: tuple[tuple[str, int], ...]
    admission_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ratios, tuple) or len(self.ratios) != 3:
            raise ValueError("split audit ratios are invalid")
        SplitConfig(self.algorithm, self.seed, *self.ratios)
        for name, counts in (
            ("subject_counts", self.subject_counts),
            ("admission_counts", self.admission_counts),
        ):
            if (
                not isinstance(counts, tuple)
                or tuple(item[0] for item in counts if isinstance(item, tuple) and len(item) == 2)
                != _PARTITION_ORDER
                or len(counts) != len(_PARTITION_ORDER)
                or any(
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or type(item[1]) is not int
                    or item[1] < 0
                    for item in counts
                )
            ):
                raise ValueError(f"split audit {name} are invalid")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    train: tuple[AdmissionExample, ...]
    validation: tuple[AdmissionExample, ...]
    test: tuple[AdmissionExample, ...]
    audit: SplitAudit

    def __post_init__(self) -> None:
        partitions = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        if not isinstance(self.audit, SplitAudit):
            raise ValueError("split audit is invalid")
        if any(not isinstance(values, tuple) for values in partitions.values()):
            raise ValueError("split partitions must be tuples")
        all_examples = tuple(
            example for values in partitions.values() for example in values
        )
        if not all_examples:
            raise ValueError("split assignment cannot be empty")
        if any(not isinstance(example, AdmissionExample) for example in all_examples):
            raise ValueError("split assignment examples are invalid")
        if any(
            example.partition != partition
            for partition, values in partitions.items()
            for example in values
        ):
            raise ValueError("split partition tag mismatch")
        admission_ids = [example.hadm_id for example in all_examples]
        if len(admission_ids) != len(set(admission_ids)):
            raise ValueError("duplicate admission in split assignment")
        subject_sets = {
            partition: {example.subject_id for example in values}
            for partition, values in partitions.items()
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if subject_sets[left] & subject_sets[right]:
                raise ValueError(f"subject overlap between {left} and {right}")
        subject_counts = tuple(
            (partition, len(subject_sets[partition])) for partition in _PARTITION_ORDER
        )
        admission_counts = tuple(
            (partition, len(partitions[partition])) for partition in _PARTITION_ORDER
        )
        if (
            self.audit.subject_counts != subject_counts
            or self.audit.admission_counts != admission_counts
        ):
            raise ValueError("split audit counts do not match assignment")


@dataclass(frozen=True, slots=True)
class LabelConfig:
    selection: str
    minimum_support: int

    def __post_init__(self) -> None:
        if self.selection != "all_training_labels":
            raise ValueError("selection must be 'all_training_labels'")
        if (
            not isinstance(self.minimum_support, int)
            or isinstance(self.minimum_support, bool)
            or self.minimum_support < 1
        ):
            raise ValueError("minimum_support must be a positive integer")


@dataclass(frozen=True, slots=True)
class LabelFitAudit:
    training_example_count: int
    vocabulary_size: int

    def __post_init__(self) -> None:
        if type(self.training_example_count) is not int or self.training_example_count < 1:
            raise ValueError("training_example_count must be positive")
        if type(self.vocabulary_size) is not int or self.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")


@dataclass(frozen=True, slots=True)
class LabelContract:
    selection: str
    minimum_support: int
    vocabulary: tuple[str, ...]
    vocabulary_hash: str
    audit: LabelFitAudit

    def __post_init__(self) -> None:
        LabelConfig(self.selection, self.minimum_support)
        if not isinstance(self.vocabulary, tuple) or not self.vocabulary:
            raise ValueError("vocabulary must be a non-empty tuple")
        canonical = tuple(
            _canonical_icd9(label, reject_surrounding_whitespace=True)
            for label in self.vocabulary
        )
        if canonical != tuple(sorted(set(canonical))):
            raise ValueError("vocabulary must be sorted and unique")
        if not isinstance(self.vocabulary_hash, str) or _HEX64.fullmatch(
            self.vocabulary_hash
        ) is None:
            raise ValueError("vocabulary_hash must be lowercase SHA-256 hex")
        if self.vocabulary_hash != _label_vocabulary_digest(canonical):
            raise ValueError("vocabulary_hash does not match vocabulary")
        if not isinstance(self.audit, LabelFitAudit) or self.audit.vocabulary_size != len(
            canonical
        ):
            raise ValueError("label fit audit does not match vocabulary")


@dataclass(frozen=True, slots=True)
class LabelTransformAudit:
    example_count: int
    unknown_label_count: int
    examples_with_unknown_labels: int

    def __post_init__(self) -> None:
        for name, value in (
            ("example_count", self.example_count),
            ("unknown_label_count", self.unknown_label_count),
            ("examples_with_unknown_labels", self.examples_with_unknown_labels),
        ):
            _nonnegative_int(name, value)
        if self.example_count < 1:
            raise ValueError("example_count must be positive")
        if self.examples_with_unknown_labels > self.example_count:
            raise ValueError("unknown-label example count exceeds example count")
        if (self.unknown_label_count == 0) != (
            self.examples_with_unknown_labels == 0
        ):
            raise ValueError("unknown-label audit counts are inconsistent")


@dataclass(frozen=True, slots=True)
class EncodedLabels:
    rows: tuple[tuple[int, ...], ...]
    vocabulary_hash: str
    audit: LabelTransformAudit

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("encoded label rows must be a non-empty tuple")
        if not isinstance(self.audit, LabelTransformAudit) or self.audit.example_count != len(
            self.rows
        ):
            raise ValueError("encoded label audit does not match rows")
        if not isinstance(self.vocabulary_hash, str) or _HEX64.fullmatch(
            self.vocabulary_hash
        ) is None:
            raise ValueError("vocabulary_hash must be lowercase SHA-256 hex")
        if any(not isinstance(row, tuple) or not row for row in self.rows):
            raise ValueError("encoded label rows must be non-empty tuples")
        width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("encoded label rows must have a stable width")
        if any(type(item) is not int or item not in (0, 1) for row in self.rows for item in row):
            raise ValueError("encoded label rows must contain binary integers")


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    identifier: str
    revision: str
    maximum_length: int
    truncation_side: str
    padding: str

    def __post_init__(self) -> None:
        _require_nonempty_string("identifier", self.identifier)
        _require_nonempty_string("revision", self.revision)
        if self.identifier != self.identifier.strip():
            raise ValueError("identifier must be canonical")
        if self.revision != self.revision.strip() or not _tokenizer_revision_valid(
            self.identifier, self.revision
        ):
            raise ValueError("revision must be an immutable content address or synthetic fixture revision")
        if (
            not isinstance(self.maximum_length, int)
            or isinstance(self.maximum_length, bool)
            or self.maximum_length < 1
        ):
            raise ValueError("maximum_length must be a positive integer")
        if self.truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side must be 'left' or 'right'")
        if self.padding not in {"max_length", "do_not_pad"}:
            raise ValueError("padding must be 'max_length' or 'do_not_pad'")


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

    def __post_init__(self) -> None:
        TokenizerConfig(
            identifier=self.tokenizer_identifier,
            revision=self.tokenizer_revision,
            maximum_length=self.maximum_length,
            truncation_side=self.truncation_side,
            padding=self.padding,
        )
        for name, value in (
            ("example_count", self.example_count),
            ("truncated_example_count", self.truncated_example_count),
            ("total_content_tokens_before", self.total_content_tokens_before),
            ("total_content_tokens_after", self.total_content_tokens_after),
            ("special_tokens_per_example", self.special_tokens_per_example),
        ):
            _nonnegative_int(name, value)
        if self.example_count < 1:
            raise ValueError("example_count must be positive")
        if self.truncated_example_count > self.example_count:
            raise ValueError("truncated_example_count exceeds example_count")
        if self.total_content_tokens_after > self.total_content_tokens_before:
            raise ValueError("content token audit is inconsistent")
        if self.special_tokens_per_example > self.maximum_length:
            raise ValueError("special token count exceeds maximum_length")
        if self.truncated_example_count == 0 and (
            self.total_content_tokens_before != self.total_content_tokens_after
        ):
            raise ValueError("content token audit is inconsistent")


@dataclass(frozen=True, slots=True)
class TokenizedBatch:
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    audit: TokenizationAudit

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, tuple) or not self.input_ids:
            raise ValueError("token batch cannot be empty")
        if not isinstance(self.attention_mask, tuple) or len(self.attention_mask) != len(
            self.input_ids
        ):
            raise ValueError("attention mask batch does not match input IDs")
        if not isinstance(self.audit, TokenizationAudit) or self.audit.example_count != len(
            self.input_ids
        ):
            raise ValueError("tokenization audit does not match batch")
        for token_ids, mask in zip(self.input_ids, self.attention_mask, strict=True):
            if (
                not isinstance(token_ids, tuple)
                or not token_ids
                or not isinstance(mask, tuple)
                or len(mask) != len(token_ids)
            ):
                raise ValueError("attention mask row does not match token IDs")
            if any(type(item) is not int or item < 0 for item in token_ids):
                raise ValueError("token IDs must be non-negative integers")
            if any(type(item) is not int or item not in (0, 1) for item in mask):
                raise ValueError("attention masks must contain binary integers")
            if len(token_ids) > self.audit.maximum_length:
                raise ValueError("token row exceeds maximum_length")
            if self.audit.padding == "max_length" and len(token_ids) != self.audit.maximum_length:
                raise ValueError("max_length padding produced an invalid row length")
        observed_content_tokens = sum(
            sum(mask) - self.audit.special_tokens_per_example
            for mask in self.attention_mask
        )
        if observed_content_tokens != self.audit.total_content_tokens_after:
            raise ValueError("tokenization audit does not match attention masks")
