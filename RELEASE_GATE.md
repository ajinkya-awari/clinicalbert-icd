# Release Gate — Project 08 ClinicalBERT-ICD

This file is the single authoritative gate for any public artifact release, data access, training, evaluation, memorization review, or deployment. No gate may be bypassed automatically.

---

## Current gate status

| Gate | Status | Evidence |
|---|---|---|
| Synthetic scaffold | COMPLETE | E-001 (historical), CHECKLIST.md synthetic section |
| Kaggle synthetic execution | **COMPLETE** | E-002: kernel v12, exit 0, 171 tests, 0 restricted artifacts (2026-09-18) |
| Public export pushed to GitHub | **COMPLETE** | Commit `20f63d1`; https://github.com/ajinkya-awari/clinicalbert-icd live 2026-09-14 |
| MIMIC-III DUA | BLOCKED | No evidence; requires PhysioNet approval |
| Secure processing location | BLOCKED | No approval |
| Retention/artifact rules | BLOCKED | No approval |
| Dependency lock approved | BLOCKED | No `requirements.txt` exists |
| Model/tokenizer revisions pinned | BLOCKED | No revision document |
| Real training | BLOCKED | All of the above must be approved first |
| Real evaluation (untouched test) | BLOCKED | Training must complete first |
| Memorization review | BLOCKED | Training must complete first |
| Clinical/regulatory review | BLOCKED | Evaluation must complete first |
| Full public release | BLOCKED | All of the above required |

---

## Synthetic-scope completion criteria

The synthetic scope is complete (not the full project) when ALL of:
1. Kaggle output downloaded and `E-002.exit_code == 0`
2. `E-002.test_count >= 171` (or gap explained)
3. `E-002.restricted_artifact_count == 0`
4. `E-002.provider_api_usage == "none"`
5. `git push` confirmed; GitHub URL live
6. EVIDENCE_LEDGER.md E-002 and E-004 filled with verified values

Current synthetic-scope status: **COMPLETE** — all criteria met 2026-09-18 (E-004 not required for synthetic scope; git push confirmed 2026-09-14)

---

## Approval required before any of the following

- Processing MIMIC-III or any restricted clinical dataset
- Moving data to Kaggle, Colab, or any external environment
- Installing exact model or tokenizer (from HuggingFace or elsewhere)
- Running training, threshold selection, or untouched-test evaluation
- Publishing model weights, intermediate checkpoints, or clinical outputs
- Deploying a demo or application
- Sending cold emails citing clinical metrics
- Any W&B, Hugging Face Hub, or external-API integration

---

## Release not authorized

No release of clinical results, model checkpoints, diagnostic claims, or patient-safety claims is authorized. `manifests/artifact_policy.json` carries `release_authorized: false`.
