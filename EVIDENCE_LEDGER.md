# Evidence Ledger — Project 08 ClinicalBERT-ICD

Records all authoritative evidence for this project. Each entry is either confirmed (with date and verifiable detail) or pending. Do not fabricate or pre-fill unverified values.

---

## E-001 — Historical synthetic execution (2026-08-19)

| Field | Value |
|---|---|
| Date | 2026-08-19 |
| Environment | Local (laptop) |
| Test runner | pytest |
| Tests passed | 168 |
| Exit code | 0 |
| Duration | 54.828 s |
| Source | HANDOVER.md historical record |
| Status | **HISTORICAL ONLY** — 3 tests were added since (171 total static); current tree not re-executed |

---

## E-002 — Kaggle provider-free synthetic validation (2026-09-18)

| Field | Value |
|---|---|
| Kernel ID | `ajinkya1225/f3-clinicalbert-icd-synthetic-validation` |
| Dataset ID | `ajinkya1225/clinicalbert-icd-source` v10 |
| Kernel version | v12 |
| UTC timestamp | `20260918T161857Z` |
| Python version | 3.12 (Kaggle default) |
| Test runner | `python -m unittest discover -s tests -v` |
| Test count | 171 |
| Exit code | 0 |
| Device | CPU (GPU disabled) |
| Provider access | disabled |
| Model inference | not_run |
| Restricted-artifact count | 0 |
| Evidence path | `evidence/synthetic/synthetic-validation-20260918T161857Z.json` |
| Status | **PASSED** — 171 tests OK, 5 skipped (powershell-only), 0 failures, 0 errors |

---

## E-003 — Public export scan (2026-09-14)

| Field | Value |
|---|---|
| Date | 2026-09-14 |
| Export path | `E:\application\MS CS\clinicalbert-icd-public\` |
| File count | 44 |
| PHI / clinical text | None found |
| Secrets / credentials | None found (scan hits were regex detection patterns, not actual secrets) |
| Model weights / checkpoints | None found |
| AI residue | None found |
| Internal control files | Excluded |
| Status | **PASSED** |

---

## E-004 — Git and GitHub release (2026-09-14)

| Field | Value |
|---|---|
| Date | 2026-09-14 |
| Local commit hash | `20f63d1` |
| Commit message | "Publish ClinicalBERT ICD synthetic evaluation scaffold" |
| Files | 44 (6007 insertions) |
| GitHub URL | https://github.com/ajinkya-awari/clinicalbert-icd |
| Push status | **CONFIRMED** — `git push -u origin main` succeeded 2026-09-14 |
| Status | **COMPLETE** |

---

## Pending gates (not yet evidenced)

- MIMIC-III PhysioNet DUA and credentialing
- CITI human-subjects training completion
- Approved secure processing location
- Retention/deletion and artifact-restriction approval
- Exact dependency lock file
- Immutable tokenizer/model revision pins
- Implemented live preflight command
- Real data loader, trainer, benchmark command
- Real training run
- Real validation and untouched-test evaluation
- Memorization/privacy review
- GPU evidence
- Clinical/regulatory review
