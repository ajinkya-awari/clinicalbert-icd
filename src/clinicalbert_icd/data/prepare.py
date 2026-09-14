"""Pure admission preparation over injected records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .contracts import (
    AdmissionExample,
    DedupAudit,
    DedupPolicy,
    DedupResult,
    NoteRecord,
    _canonical_icd9,
    _canonical_identifier,
)


_SCALAR_TYPES = (str, int, float)


def _required(record: Mapping[str, object], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError("missing required field(s): " + ", ".join(sorted(missing)))


def _identifier(name: str, value: object) -> str:
    return _canonical_identifier(
        name, value, reject_surrounding_whitespace=False
    )


def _decision_scalar(name: str, value: object) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, _SCALAR_TYPES):
        raise ValueError(f"{name} must be a string or finite numeric scalar")
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        raise ValueError(f"{name} must be finite")
    return value


def _sort_scalar(value: str | int | float) -> tuple[int, str | int | float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))


def _coerce_note(record: Mapping[str, object], policy: DedupPolicy) -> NoteRecord:
    required_fields = (
        "SUBJECT_ID",
        "HADM_ID",
        "TEXT",
        policy.category_field,
        policy.order_field,
        *policy.tie_break_fields,
    )
    _required(record, required_fields)
    text = record["TEXT"]
    category = record[policy.category_field]
    if not isinstance(text, str):
        raise ValueError("TEXT must be a string")
    if not isinstance(category, str):
        raise ValueError(f"{policy.category_field} must be a string")
    return NoteRecord(
        subject_id=_identifier("SUBJECT_ID", record["SUBJECT_ID"]),
        hadm_id=_identifier("HADM_ID", record["HADM_ID"]),
        text=text,
        category=category,
        order_value=_decision_scalar(policy.order_field, record[policy.order_field]),
        tie_break_values=tuple(
            _decision_scalar(field, record[field]) for field in policy.tie_break_fields
        ),
    )


def _selection_key(record: NoteRecord) -> tuple[tuple[int, str | int | float], ...]:
    return (
        _sort_scalar(record.order_value),
        *(_sort_scalar(value) for value in record.tie_break_values),
    )


def _decision_kind(value: str | int | float) -> str:
    return "numeric" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string"


def _validate_decision_types(admission_notes: list[NoteRecord]) -> None:
    positions = (
        tuple(note.order_value for note in admission_notes),
        *(
            tuple(note.tie_break_values[index] for note in admission_notes)
            for index in range(len(admission_notes[0].tie_break_values))
        ),
    )
    if any(len({_decision_kind(value) for value in values}) != 1 for values in positions):
        raise ValueError("dedup decision field types are inconsistent within an admission")


def deduplicate_admissions(
    records: Iterable[Mapping[str, object]], policy: DedupPolicy
) -> DedupResult:
    """Select one target-category note per admission under an explicit policy."""

    raw_records = tuple(records)
    if not raw_records:
        raise ValueError("deduplication input cannot be empty")
    eligible = tuple(
        note
        for note in (_coerce_note(record, policy) for record in raw_records)
        if note.category == policy.category_value
    )
    grouped: dict[str, list[NoteRecord]] = defaultdict(list)
    for note in eligible:
        grouped[note.hadm_id].append(note)

    selected: list[NoteRecord] = []
    duplicate_admissions = 0
    for hadm_id, admission_notes in grouped.items():
        owners = {note.subject_id for note in admission_notes}
        if len(owners) != 1:
            raise ValueError(f"conflicting admission ownership for {hadm_id}")
        if len(admission_notes) > 1:
            duplicate_admissions += 1
        _validate_decision_types(admission_notes)

        keys = [_selection_key(note) for note in admission_notes]
        winning_key = min(keys) if policy.keep == "earliest" else max(keys)
        winners = [
            note for note in admission_notes if _selection_key(note) == winning_key
        ]
        if len(set(winners)) > 1:
            raise ValueError(f"ambiguous deterministic tie for admission {hadm_id}")
        selected.append(winners[0])

    selected_records = tuple(sorted(selected, key=lambda note: (note.subject_id, note.hadm_id)))
    audit = DedupAudit(
        input_record_count=len(raw_records),
        eligible_record_count=len(eligible),
        output_record_count=len(selected_records),
        dropped_record_count=len(eligible) - len(selected_records),
        duplicate_admission_count=duplicate_admissions,
    )
    return DedupResult(records=selected_records, audit=audit)


def parse_native_icd9(value: str) -> str:
    """Strip and validate a native ICD-9 lexeme without converting it."""

    return _canonical_icd9(value, reject_surrounding_whitespace=False)


def _example_note(record: NoteRecord | Mapping[str, object]) -> tuple[str, str, str]:
    if isinstance(record, NoteRecord):
        return record.subject_id, record.hadm_id, record.text
    _required(record, ("SUBJECT_ID", "HADM_ID", "TEXT"))
    text = record["TEXT"]
    if not isinstance(text, str):
        raise ValueError("TEXT must be a string")
    return (
        _identifier("SUBJECT_ID", record["SUBJECT_ID"]),
        _identifier("HADM_ID", record["HADM_ID"]),
        text,
    )


def build_admission_examples(
    notes: Iterable[NoteRecord | Mapping[str, object]],
    diagnosis_rows: Iterable[Mapping[str, object]],
) -> tuple[AdmissionExample, ...]:
    """Aggregate native labels onto already-deduplicated admission notes."""

    note_values = tuple(notes)
    diagnosis_values = tuple(diagnosis_rows)
    if not note_values:
        raise ValueError("admission notes input cannot be empty")
    if not diagnosis_values:
        raise ValueError("diagnosis rows input cannot be empty")

    notes_by_admission: dict[str, tuple[str, str]] = {}
    for record in note_values:
        subject_id, hadm_id, text = _example_note(record)
        if hadm_id in notes_by_admission:
            raise ValueError("notes must be deduplicated before admission aggregation")
        notes_by_admission[hadm_id] = (subject_id, text)

    labels_by_admission: dict[str, set[str]] = defaultdict(set)
    for row in diagnosis_values:
        _required(row, ("SUBJECT_ID", "HADM_ID", "ICD9_CODE"))
        subject_id = _identifier("SUBJECT_ID", row["SUBJECT_ID"])
        hadm_id = _identifier("HADM_ID", row["HADM_ID"])
        if hadm_id not in notes_by_admission:
            raise ValueError(f"diagnosis admission {hadm_id} has no deduplicated note")
        note_subject_id, _ = notes_by_admission[hadm_id]
        if subject_id != note_subject_id:
            raise ValueError(f"conflicting admission ownership for {hadm_id}")
        labels_by_admission[hadm_id].add(parse_native_icd9(row["ICD9_CODE"]))

    examples = []
    for hadm_id, (subject_id, text) in sorted(
        notes_by_admission.items(), key=lambda item: (item[1][0], item[0])
    ):
        labels = tuple(sorted(labels_by_admission.get(hadm_id, set())))
        if not labels:
            raise ValueError(
                f"admission {hadm_id} has no diagnosis rows in the supplied labels"
            )
        examples.append(
            AdmissionExample(
                subject_id=subject_id,
                hadm_id=hadm_id,
                text=text,
                labels=labels,
            )
        )
    return tuple(examples)
