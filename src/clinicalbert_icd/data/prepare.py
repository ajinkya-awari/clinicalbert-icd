from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from clinicalbert_icd.data.contracts import (
    AdmissionExample,
    DedupAudit,
    DedupPolicy,
    DedupResult,
    NoteRecord,
    _validate_icd9,
    _canonical_id,
)


def parse_native_icd9(raw: str) -> str:
    stripped = raw.strip()
    _validate_icd9(stripped)
    return stripped


def _require_canonical_id(row: dict, field: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"{field}: field is required but missing from record")
    if isinstance(value, str):
        value = value.strip()
    return _canonical_id(value, field)


def deduplicate_admissions(
    records: Sequence[Any],
    policy: DedupPolicy,
) -> DedupResult:
    if not records:
        raise ValueError("records must not be empty")

    cat_field = policy.category_field
    order_field = policy.order_field
    tbf = policy.tie_break_fields

    for row in records:
        if row.get(cat_field) != policy.category_value:
            continue
        if order_field not in row:
            raise ValueError(f"record missing required field: {order_field!r}")
        for tf in tbf:
            if tf not in row:
                raise ValueError(f"record missing required tie-break field: {tf!r}")
        if "TEXT" not in row:
            raise ValueError("record missing required field: 'TEXT'")
        _require_canonical_id(row, "SUBJECT_ID")
        _require_canonical_id(row, "HADM_ID")

    eligible: list[dict] = [
        row for row in records if row.get(cat_field) == policy.category_value
    ]

    by_admission: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        hadm = _canonical_id(row["HADM_ID"], "HADM_ID")
        by_admission[hadm].append(row)

    for hadm, rows in by_admission.items():
        subj_ids = {_require_canonical_id(r, "SUBJECT_ID") for r in rows}
        if len(subj_ids) > 1:
            raise ValueError(
                f"ownership conflict: admission {hadm!r} maps to multiple subjects {subj_ids}"
            )
        order_types = {type(r[order_field]) for r in rows}
        if len(order_types) > 1:
            raise ValueError(
                f"type inconsistency for order_field {order_field!r} in admission {hadm!r}"
            )
        for tf in tbf:
            tb_types = {type(r[tf]) for r in rows}
            if len(tb_types) > 1:
                raise ValueError(
                    f"type inconsistency for tie_break_field {tf!r} in admission {hadm!r}"
                )

    selected: list[NoteRecord] = []
    reverse = policy.keep == "latest"

    for hadm in sorted(by_admission):
        rows = by_admission[hadm]
        if len(rows) == 1:
            row = rows[0]
        else:
            def sort_key(r: dict) -> tuple:
                primary = r[order_field]
                tb = tuple(r[tf] for tf in tbf)
                return (primary, *tb)

            ordered = sorted(rows, key=sort_key, reverse=reverse)
            best_key = sort_key(ordered[0])
            tied = [r for r in ordered if sort_key(r) == best_key]
            if len(tied) > 1:
                raise ValueError(
                    f"ambiguous tie in admission {hadm!r}: {len(tied)} rows share "
                    f"the same order and tie-break values {best_key}"
                )
            row = ordered[0]

        tb_vals = tuple(row[tf] for tf in tbf)
        nr = NoteRecord(
            subject_id=_require_canonical_id(row, "SUBJECT_ID"),
            hadm_id=hadm,
            text=str(row["TEXT"]),
            category=str(row[cat_field]),
            order_value=row[order_field],
            tie_break_values=tb_vals,
        )
        selected.append(nr)

    input_count = len(records)
    eligible_count = len(eligible)
    output_count = len(selected)
    dropped_count = eligible_count - output_count
    dup_admissions = sum(1 for rows in by_admission.values() if len(rows) > 1)

    audit = DedupAudit(
        input_record_count=input_count,
        eligible_record_count=eligible_count,
        output_record_count=output_count,
        dropped_record_count=dropped_count,
        duplicate_record_count=dup_admissions,
    )
    return DedupResult(records=tuple(selected), audit=audit)


def _note_hadm_id(note: Any) -> str:
    if hasattr(note, "hadm_id"):
        return note.hadm_id
    return str(note["HADM_ID"]).strip()


def build_admission_examples(
    notes: Sequence[Any],
    diagnoses: Sequence[Any],
) -> tuple[AdmissionExample, ...]:
    if not notes:
        raise ValueError("notes must not be empty")
    if not diagnoses:
        raise ValueError("diagnosis list must not be empty")

    hadm_ids = [_note_hadm_id(n) for n in notes]
    if len(hadm_ids) != len(set(hadm_ids)):
        raise ValueError(
            "duplicate admission found in notes — call deduplicate_admissions first"
        )

    note_subject: dict[str, str] = {}
    for n in notes:
        hadm = _note_hadm_id(n)
        subj = n.subject_id if hasattr(n, "subject_id") else str(n.get("SUBJECT_ID", "")).strip()
        note_subject[hadm] = subj

    diag_by_hadm: dict[str, set[str]] = defaultdict(set)
    for row in diagnoses:
        hadm = str(row["HADM_ID"]).strip()
        if "SUBJECT_ID" in row:
            diag_subj = str(row["SUBJECT_ID"]).strip()
            expected_subj = note_subject.get(hadm)
            if expected_subj is not None and diag_subj != expected_subj:
                raise ValueError(
                    f"ownership conflict: diagnosis for admission {hadm!r} has "
                    f"SUBJECT_ID={diag_subj!r} but note has SUBJECT_ID={expected_subj!r}"
                )
        code = str(row["ICD9_CODE"]).strip()
        try:
            code = parse_native_icd9(code)
        except ValueError:
            continue
        diag_by_hadm[hadm].add(code)

    examples: list[AdmissionExample] = []
    for note in notes:
        hadm = _note_hadm_id(note)
        if hadm not in diag_by_hadm:
            raise ValueError(
                f"no diagnosis rows found for admission {hadm!r}; "
                "ensure diagnoses contains at least one valid ICD-9 code for this admission"
            )
        labels = tuple(sorted(diag_by_hadm[hadm]))
        subj = note.subject_id if hasattr(note, "subject_id") else str(note.get("SUBJECT_ID", "")).strip()
        ex = AdmissionExample(
            subject_id=subj,
            hadm_id=hadm,
            text=note.text if hasattr(note, "text") else str(note.get("TEXT", "")),
            labels=labels,
        )
        examples.append(ex)

    return tuple(sorted(examples, key=lambda e: e.hadm_id))
