<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=ClinicalBERT-ICD&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Leakage-safe%20LoRA%20vs%20full%20fine-tuning%20for%20ICD-9%20multi-label%20classification&descAlignY=58&descAlign=50&descSize=16"/>

<div align="center">

[![Tests](https://img.shields.io/badge/tests-171%20passing-brightgreen?style=flat-square)](https://github.com/ajinkya-awari/clinicalbert-icd)
[![Kaggle](https://img.shields.io/badge/Kaggle-kernel%20v12%20✓-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/)
[![Status](https://img.shields.io/badge/status-synthetic%20complete-blue?style=flat-square)](https://github.com/ajinkya-awari/clinicalbert-icd)
[![MIMIC](https://img.shields.io/badge/MIMIC--III-approval%20pending-orange?style=flat-square)](https://physionet.org/content/mimiciii/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Readiness](https://img.shields.io/badge/readiness-67%25-yellow?style=flat-square)](https://github.com/ajinkya-awari/clinicalbert-icd)

</div>

---

## Overview

A privacy-first, leakage-safe experimental scaffold comparing **LoRA** (parameter-efficient) and **full fine-tuning** strategies for ICD-9 multi-label classification over MIMIC-III discharge summaries. Built for reproducibility before data access — every contract, split, threshold, and evaluation boundary is locked and tested on synthetic data, ready for real training once PhysioNet DUA is approved.

**What this is:**
- A rigorous five-system ML pipeline with patient-grouped splits, train-only label fitting, and single-use test evaluation
- 171 synthetic unit tests covering governance contracts, privacy scans, and leakage guards — all passing (Kaggle kernel v12, exit 0)
- A fair apples-to-apples comparison: same tokenizer, same label set, same evaluation protocol for both fine-tuning strategies

**What this is not:**
- Clinically validated or production-ready
- A claim about LoRA vs full fine-tuning outcomes (no real MIMIC training run yet)
- Anything other than a scaffold awaiting PhysioNet data-use agreement approval

---

## Architecture

Five subsystems with a shared preprocessing and evaluation contract:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ClinicalBERT-ICD Pipeline                   │
├──────────────┬──────────────┬────────────────┬──────────────────┤
│  Governance  │  Ingestion   │  Model Layer   │  Evaluation      │
│  (contracts) │  (split +    │  (LoRA / full  │  (val threshold  │
│              │  binarize)   │  fine-tune)    │  → test once)    │
├──────────────┴──────────────┴────────────────┴──────────────────┤
│                      Privacy Scan Layer                         │
│    (aggregate-only outputs · no note text exposed · PHI guard)  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Split strategy | By `SUBJECT_ID` | Patient disjointness — no leakage across splits |
| Label fitting | Train split only | `MultiLabelBinarizer` never sees val/test labels |
| Threshold selection | Validation only | Prevents optimistic test-set inflation |
| Test evaluation | Single-use `UntouchedTestSession` | Once evaluated, test set is permanently sealed |
| ICD coding | Native ICD-9 | No GEM crosswalk approximations |

---

## Synthetic Execution Evidence

**Kaggle kernel v12 — 2026-09-18 — exit code 0**

| Gate | Result |
|---|---|
| Total tests | 171 passed, 5 skipped (PowerShell-only hooks) |
| Exit code | 0 |
| Restricted artifacts | 0 |
| Provider API usage | none |
| MIMIC data accessed | no (synthetic fixtures only) |

Evidence path: `evidence/synthetic/kaggle_v12_summary.json`

> **All benchmark metrics (F1, AUC, per-label scores) are pending PhysioNet DUA approval and real MIMIC training runs. No model comparison results exist yet.**

---

## Non-Negotiable Rules

1. **Split by `SUBJECT_ID`** — never split rows randomly
2. **Fit label vocabulary on training data only** — no val/test label leakage
3. **Select thresholds on validation only** — test set stays untouched
4. **Single-use test evaluation** — `UntouchedTestSession` seals results after one call
5. **Native ICD-9 labels** — no GEM crosswalk conversions
6. **No raw MIMIC notes outside approved secure environment**
7. **No note text to external APIs** — OpenAI, Anthropic, HuggingFace inference, Groq
8. **No training, download, or GPU without explicit approval**
9. **Deduplicate admissions before splitting**
10. **Privacy scan must pass before any evaluation claim**

---

## Repository Structure

<details>
<summary>Full tree</summary>

```
clinicalbert-icd/
├── src/
│   ├── clinicalbert_icd/
│   │   ├── contracts.py          # Governance contracts + fail-closed guards
│   │   ├── ingestion.py          # MIMIC loading + SUBJECT_ID split
│   │   ├── prepare.py            # Label binarizer (train-only fit)
│   │   ├── model.py              # ClinicalBERT + LoRA config
│   │   ├── train.py              # Training loop (gated — requires MIMIC DUA)
│   │   ├── evaluate.py           # Threshold selection + UntouchedTestSession
│   │   └── privacy_scan.py       # Aggregate-only output verification
├── tests/
│   ├── test_contracts.py         # Governance fail-closed contracts (synthetic)
│   ├── test_ingestion.py         # Split + dedup contracts
│   ├── test_prepare.py           # Train-only label fitting
│   ├── test_evaluate.py          # Threshold + test isolation
│   └── test_privacy.py           # Privacy scan contracts
├── evidence/
│   └── synthetic/
│       └── kaggle_v12_summary.json   # Execution evidence (2026-09-18)
├── pyproject.toml
├── LICENSE
└── README.md
```

</details>

---

## Reproducibility

<details>
<summary>Run synthetic tests locally (no MIMIC required)</summary>

```bash
# Clone
git clone https://github.com/ajinkya-awari/clinicalbert-icd
cd clinicalbert-icd

# Install (Python 3.10+)
pip install -e ".[dev]"

# Run synthetic suite
pytest tests/ -v --tb=short
# Expected: 171 passed, 5 skipped
```

**All 171 tests use synthetic fixtures only. No MIMIC data needed.**

</details>

<details>
<summary>Real training (MIMIC-III — APPROVAL REQUIRED)</summary>

Real training requires:
1. PhysioNet DUA approval for MIMIC-III (apply at physionet.org)
2. CITI training completion
3. Approved secure processing environment

Once approved:
```bash
# Place MIMIC-III files in data/ (never commit to git)
python -m clinicalbert_icd.train --config configs/lora.yaml
python -m clinicalbert_icd.train --config configs/full.yaml
```

</details>

---

## Status

| Dimension | Weight | Score | Notes |
|---|---|---|---|
| Implementation | 30% | 22/30 | All source files clean; governance contracts complete |
| Tests | 20% | 18/20 | 171/171 synthetic tests passing |
| Runtime/execution | 20% | 12/20 | Kaggle synthetic run verified; real training pending DUA |
| Reproducibility | 15% | 8/15 | Pinned revisions documented; real data pipeline pending |
| Release readiness | 15% | 7/15 | GitHub live; HF Space + arXiv pending real results |

**Overall: 67%** — synthetic scope complete, real training blocked on MIMIC PhysioNet DUA approval

---

## Limitations

- No real benchmark results exist — F1/AUC numbers will be added after approved MIMIC training runs
- HuggingFace Space not yet deployed (pending real model weights)
- Comparison between LoRA and full fine-tuning is the research question, not a confirmed outcome
- Not clinically validated; not a medical device; not intended for deployment

---

## Citation

```bibtex
@misc{awari2026clinicalbert,
  author       = {Awari, Ajinkya},
  title        = {ClinicalBERT-ICD: Leakage-Safe LoRA vs Full Fine-tuning for ICD-9 Classification},
  year         = {2026},
  howpublished = {\url{https://github.com/ajinkya-awari/clinicalbert-icd}},
  note         = {Synthetic scope complete; real training pending MIMIC-III PhysioNet DUA}
}
```

---

## Portfolio Context

Part of a 21-project AI/ML portfolio targeting UK healthcare AI and research positions.

| Project | Focus |
|---|---|
| [SolomonoffBench](https://github.com/ajinkya-awari/solomonoff-bench) | Information-theoretic LLM evaluation |
| [XAI Medical Imaging](https://github.com/ajinkya-awari/xai-medical-imaging-project-02) | Grad-CAM / SHAP / IG benchmarked on NIH ChestX-ray14 |
| [AlphaFold Binding Benchmark](https://github.com/ajinkya-awari/afbind) | AF2 structure substitution benchmark |
| [NHSCopilot-Eval](https://github.com/ajinkya-awari/-nhscopilot-eval) | Clinical assistant safety evaluation framework |
| [MedLLM Safety](https://github.com/ajinkya-awari/medllm-safety) | LLM refusal, abstention, PHI-redaction contracts |

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>
