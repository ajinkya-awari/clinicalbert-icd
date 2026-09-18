# Release Checklist — Project 08 ClinicalBERT-ICD

Tracks every gate from synthetic scaffold through clinical release. Checked items have dated evidence in `EVIDENCE_LEDGER.md`.

---

## Synthetic scaffold (local contracts)

- [x] Governance fail-closed contracts implemented and tested (2026-08-19)
- [x] Privacy aggregate-only scan implemented and tested (2026-08-19)
- [x] Patient-grouped split by `SUBJECT_ID` implemented and tested (2026-08-19)
- [x] Train-only label fitting implemented and tested (2026-08-19)
- [x] Validation-only threshold selection implemented and tested (2026-08-19)
- [x] Single-use `UntouchedTestSession` implemented and tested (2026-08-19)
- [x] Five-system registry validated (2026-08-19)
- [x] Command and pre-commit hooks pass adversarial tests (2026-08-19)
- [x] 171 synthetic unit tests written (2026-08-29 static inventory)
- [x] L-015 fix: `build_admission_examples` named ValueError (2026-08-27)
- [x] L-016 fix: governance `location is not None` guard (2026-08-27)
- [x] LICENSE (MIT) added (2026-09-14)
- [x] `pyproject.toml` added (2026-09-14)
- [x] Public export created and scanned clean (2026-09-14)
- [x] Git commit `20f63d1` in public export (2026-09-14)
- [x] GitHub repo created: https://github.com/ajinkya-awari/clinicalbert-icd (2026-09-14)

## Synthetic Kaggle execution

- [x] Kaggle kernel output downloaded (`E-002` in EVIDENCE_LEDGER.md) (2026-09-18)
- [x] `exit_code == 0` confirmed from evidence JSON (2026-09-18)
- [x] Test count ≥ 171 confirmed — 171 tests, 5 skipped (powershell-only) (2026-09-18)
- [x] `restricted_artifact_count == 0` confirmed (2026-09-18)
- [x] `provider_api_usage == "none"` confirmed (2026-09-18)
- [x] Evidence JSON saved to `evidence/synthetic/` in project (2026-09-18)

## Public release

- [x] `git push -u origin main` confirmed; https://github.com/ajinkya-awari/clinicalbert-icd live (2026-09-14)
- [ ] README.md updated with GitHub URL and Kaggle evidence reference (pending E-002)
- [x] EVIDENCE_LEDGER.md E-002 filled with real Kaggle values (2026-09-18)

## Rights and environment (APPROVAL REQUIRED)

- [ ] MIMIC-III PhysioNet DUA verified
- [ ] CITI training completed
- [ ] Secure processing location approved
- [ ] Retention/deletion terms approved
- [ ] Artifact restrictions documented

## Benchmark (APPROVAL REQUIRED)

- [ ] Exact dependency lock file (`requirements.txt`) created and approved
- [ ] Immutable tokenizer/model revision pins documented
- [ ] Live preflight command implemented
- [ ] Real MIMIC data loaded in approved environment
- [ ] Real training completed (5 systems)
- [ ] Thresholds selected on validation only
- [ ] Untouched test set evaluated once
- [ ] Memorization/privacy review completed
- [ ] GPU evidence recorded

## Clinical/regulatory (APPROVAL REQUIRED)

- [ ] No unsupported diagnostic or clinical efficacy claims
- [ ] External clinical/privacy review completed
- [ ] Publication or deployment approved by evidence owner
