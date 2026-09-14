"""Clearly synthetic records for offline data-contract tests."""

from __future__ import annotations


SYNTHETIC_NOTES = (
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A1",
        "TEXT": "synthetic older alpha note",
        "CATEGORY": "Discharge summary",
        "note_order": 1,
        "note_row_id": "A1-01",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A1",
        "TEXT": "synthetic final alpha note",
        "CATEGORY": "Discharge summary",
        "note_order": 2,
        "note_row_id": "A1-02",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A2",
        "TEXT": "synthetic second alpha admission",
        "CATEGORY": "Discharge summary",
        "note_order": 1,
        "note_row_id": "A2-01",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-B",
        "HADM_ID": "SYNTH-ADMISSION-B1",
        "TEXT": "synthetic beta note",
        "CATEGORY": "Discharge summary",
        "note_order": 1,
        "note_row_id": "B1-01",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-C",
        "HADM_ID": "SYNTH-ADMISSION-C1",
        "TEXT": "synthetic gamma note",
        "CATEGORY": "Discharge summary",
        "note_order": 1,
        "note_row_id": "C1-01",
    },
)


SYNTHETIC_DIAGNOSES = (
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A1",
        "ICD9_CODE": "250.00",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A1",
        "ICD9_CODE": "4019",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A1",
        "ICD9_CODE": "4019",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-A",
        "HADM_ID": "SYNTH-ADMISSION-A2",
        "ICD9_CODE": "V5869",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-B",
        "HADM_ID": "SYNTH-ADMISSION-B1",
        "ICD9_CODE": "E8798",
    },
    {
        "SUBJECT_ID": "SYNTH-SUBJECT-C",
        "HADM_ID": "SYNTH-ADMISSION-C1",
        "ICD9_CODE": "25000",
    },
)
