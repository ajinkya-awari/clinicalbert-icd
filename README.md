# F3 ClinicalBERT-ICD

Leakage-safe comparison of LoRA and full fine-tuning for ICD-9 multi-label classification on clinical discharge summaries.

---

## Status â€” 2026-09-14

**Pre-results. Local contracts and synthetic test scaffolding are present. No MIMIC-III training, model loading, or clinical evaluation has run.**

**Repository:** https://github.com/ajinkya-awari/clinicalbert-icd Â· commit `20f63d1`

| Layer | State |
|---|---|
| Governance and privacy contracts | Implemented; historical synthetic evidence only |
| Admission deduplication (SUBJECT_ID split) | Implemented; historical synthetic evidence only |
| Train-only ICD-9 label vocabulary | Implemented; historical synthetic evidence only |
| Five-system model registry (rule, ClinicalBERT x {LoRA,full}, BERT x {LoRA,full}) | Implemented; historical synthetic evidence only |
| LoRA / full fine-tuning trainability audits | Implemented; historical synthetic evidence only |
| Validation-only threshold selection | Implemented; historical synthetic evidence only |
| Untouched-test evaluation session | Implemented; historical synthetic evidence only |
| Command and commit guards (blocks accidental data/API leakage) | Implemented; Kaggle CPU synthetic suite run (evidence pending download) |
| MIMIC-III data access | **Blocked â€” PhysioNet DUA approval required** |
| Dependency installation / model download | **Blocked â€” Kaggle approval required** |
| Training, evaluation metrics, results | **Not yet run** |
| Public repository | **Live** â€” https://github.com/ajinkya-awari/clinicalbert-icd |

Results (micro-F1, macro-F1, P@K, AUC) will be added here after the approved Kaggle training run.

---

## Research question

Does parameter-efficient fine-tuning (LoRA) match or exceed full fine-tuning for ICD-9 multi-label classification on ClinicalBERT and general BERT, when training data is limited to a single admission per patient and labels are sourced from MIMIC-III DIAGNOSES_ICD only (no GEM crosswalks)?

---

## Five-system design

| Logical system | Model family | Adaptation |
|---|---|---|
| `clinicalbert_lora` | ClinicalBERT | LoRA (frozen base, trainable adapters + classifier head) |
| `clinicalbert_full` | ClinicalBERT | Full fine-tuning |
| `bert_lora` | General BERT | LoRA |
| `bert_full` | General BERT | Full fine-tuning |
| `rule` | Frequency-weighted rule baseline | â€” |

All five systems share a single label vocabulary (fit on training split only) and are evaluated on one untouched test partition.

---

## Data requirements

**MIMIC-III access is required.** This project uses:
- `NOTEEVENTS.csv` â€” discharge summary text
- `DIAGNOSES_ICD.csv` â€” native ICD-9 codes (not ICD-10, not GEM-converted)

Access requires a PhysioNet credentialed account and completion of the CITI human-subjects training course. Apply at [physionet.org](https://physionet.org/register/). Approval typically takes one to two weeks.

Raw MIMIC files must remain in an approved secure local environment. They must not be uploaded to GitHub, HuggingFace, Kaggle, or any external API.

---

## Offline reproducibility (no MIMIC required)

The synthetic offline suite tests all data contracts, privacy gates, model registry validation, and command guards using only Python standard library. No third-party packages, no model downloads, no data access.

```bash
# Set PYTHONPATH to the src directory (Windows PowerShell)
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'

# Run the synthetic suite
python -m unittest discover -s tests -v
```

Historical evidence in `HANDOVER.md` records `Ran 168 tests in 54.828s`, `OK` on 2026-08-19. A 2026-08-29 static inventory found 171 test functions, but the current tree has not been executed.

All tests use clearly labeled synthetic fixtures in `tests/fixtures/`. No patient data, no real model weights.

## Audit and Reconciliation - 2026-08-29

Status label: **PARTIAL**. Portfolio readiness: **43%** by the project rubric: implementation 70%, tests/validation 45%, runtime/execution 15%, reproducibility/provenance 55%, release readiness 10%.

This audit inspected all Markdown, source, tests, notebook, config, manifest, hook, generated-cache, and Git-availability evidence without running Python, Kaggle, providers, model downloads, MIMIC access, training, deployment, commit, or push. The exact next task is to run the provider-free synthetic suite in Kaggle and preserve sanitized evidence under `/kaggle/working/f3-clinicalbert-icd/evidence/synthetic/`.

---

## Privacy and governance controls

- **Data governance gate**: `src/clinicalbert_icd/governance.py` â€” fail-closed; any access attempt without a signed DUA manifest is denied.
- **Privacy audit**: `src/clinicalbert_icd/privacy.py` â€” scans public artifact dicts for note text, patient identifiers, model secrets, and checkpoint paths before release.
- **Pre-commit hook**: `.claude/hooks/pre-commit.sh` â€” blocks staged MIMIC paths, `.env` files, PEM keys, HF tokens, `sk-` style API keys, and AWS AKIA keys.
- **Command guard**: `.claude/hooks/pre-tool-use.sh` â€” blocks Python training calls, `from_pretrained`, `kaggle`, `curl`/`wget`, W&B, HuggingFace CLI upload, and `git push` in the local session.

---

## Leakage controls

- Admissions are deduplicated (one note per `HADM_ID`) before any split.
- Splits are deterministic by `SUBJECT_ID` (SHA-256 bucketing, seed 8408): a patient's admissions cannot appear in both training and test.
- ICD-9 label vocabulary and `MultiLabelBinarizer` equivalent are fit on training split only.
- Threshold selection uses validation-partition predictions only.
- The test partition is evaluated through a single-use `UntouchedTestSession`; a second call raises `RuntimeError`.

---

## Project structure

```
08-f3-clinicalbert-icd/
â”œâ”€â”€ src/clinicalbert_icd/
â”‚   â”œâ”€â”€ governance.py          # Fail-closed DUA and environment contracts
â”‚   â”œâ”€â”€ privacy.py             # Public-artifact PHI and secret scan
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ contracts.py       # Immutable data contracts (dedup, split, labels, tokenizer)
â”‚   â”‚   â”œâ”€â”€ prepare.py         # Admission deduplication and example aggregation
â”‚   â”‚   â”œâ”€â”€ split.py           # SUBJECT_ID-grouped SHA-256 deterministic split
â”‚   â”‚   â”œâ”€â”€ labels.py          # Train-only label fitting and transform
â”‚   â”‚   â””â”€â”€ tokenize.py        # Tokenizer protocol and truncation audit
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ protocol.py        # ModelSpec frozen dataclass
â”‚   â”‚   â”œâ”€â”€ registry.py        # Five-system registry validator
â”‚   â”‚   â””â”€â”€ trainability.py    # LoRA adapter and classifier-head trainability audits
â”‚   â”œâ”€â”€ evaluation/
â”‚   â”‚   â”œâ”€â”€ metrics.py         # Micro/macro F1, P@K, AUC
â”‚   â”‚   â”œâ”€â”€ thresholds.py      # Validation-only global threshold selection
â”‚   â”‚   â”œâ”€â”€ test_session.py    # Single-use untouched-test partition gate
â”‚   â”‚   â””â”€â”€ provenance.py      # Deterministic run provenance and command whitelist
â”‚   â””â”€â”€ training/
â”‚       â””â”€â”€ gates.py           # Training authorization gate (blocks live-data training locally)
â”œâ”€â”€ tests/                     # 171 synthetic standard-library unit tests
â”œâ”€â”€ notebooks/                 # Gated Kaggle training notebook (cells require explicit approval)
â”œâ”€â”€ manifests/                 # Artifact policy and rights environment schema
â”œâ”€â”€ config/synthetic.json      # Synthetic run configuration
â”œâ”€â”€ DESIGN.md                  # Full pipeline specification
â””â”€â”€ FINAL_VULNERABILITY_SCAN.md # Critical controls and high-risk areas
```

---

## Honest limitations

- No training results exist yet. The results table above will be populated after approved Kaggle execution.
- This project is a research comparison, not a clinical decision-support tool. No clinical deployment is intended or claimed.
- ICD-9 labels from MIMIC-III have known noise (missing codes, coding-date effects). Results reflect retrospective coding patterns, not prospective diagnostic accuracy.
- MIMIC-III is a single-centre ICU dataset (BIDMC). Generalisation to other institutions or patient populations is not established.
- LoRA target modules (query/value attention layers) are chosen by convention; the optimal configuration may differ for ClinicalBERT's specific architecture version.

---

## Portfolio context

This is Project 08 in a 21-project pre-UCL portfolio. It targets AI/healthcare engineering roles at UK companies including Kheiron Medical, Medtronic, and NHS AI Lab. The companion projects include:
- Project 07 (F2 HistoGNN) â€” graph-based histopathology classification
- Project 09 (NHSCopilot-Eval) â€” evaluation harness for NHS-facing LLM tools
- Project 13 (ClinVision) â€” clinical image classification pipeline
