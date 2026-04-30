# Learning Conjunction Dynamics
### A Self-Supervised Approach to Satellite Collision Risk Assessment

**SDC2026 — KAU Aerospace Engineering Team**  
Submitted to the **DebriSolver Competition** at the [Space Debris Conference 2026](https://ssa.gov.sa)  
Riyadh, Saudi Arabia · January 26–27, 2026  
Organized by the **Saudi Space Agency (SSA)**, supported by **UNOOSA** and **ITU**, data provided by **ALDORIA**

---

## The Problem

Low Earth orbit is becoming critically congested. As of 2024, operators receive thousands of conjunction alerts per year — each flagging a potential collision between two objects. These alerts come from diverse tracking systems (radar, optical, TLE-based) with varying accuracy and latency, making it nearly impossible to distinguish genuine threats from noise.

Every alert demands operator attention. Every unnecessary maneuver burns fuel, shortens satellite lifespan, and disrupts service. Every ignored real threat risks mission failure. The cost of getting this wrong — in either direction — is enormous.

The challenge is compounded by a fundamental data problem: **actual collisions are too rare to build supervised classifiers.** The Iridium-Cosmos event (2009) and the FengYun-1C debris cloud remain the most studied cases. No reliable collision labels exist at scale.

---

## Our Proposed Solution

We developed a **self-supervised Bidirectional GRU** that learns conjunction risk dynamics directly from CDM sequence patterns — without requiring any collision labels.

**The core insight**: a conjunction event generates a series of CDMs over time as tracking updates. The trajectory of these CDMs — how Pc evolves, how uncertainty changes, how miss distance shifts — encodes whether the event is converging toward a real threat or dissolving. A model that can predict the next CDM from a historical sequence must learn these dynamics implicitly.

We then extract two outputs from this trained model:
- **Threat score** [0–100]: based on predicted Pc trajectory and time-to-TCA urgency, calibrated to ESA/NASA maneuver thresholds
- **Confidence** [0–1]: derived from MC Dropout uncertainty across 50 forward passes, weighted by data quantity and covariance quality

This produces an operator-ready four-quadrant risk classification:

| Quadrant | Threat | Confidence | Operator Action |
|----------|--------|------------|-----------------|
| **ACT NOW** | High | High | Immediate maneuver evaluation |
| **WATCH CLOSELY** | High | Low | Request more tracking data |
| **SAFELY IGNORE** | Low | High | Deprioritize with confidence |
| **NOT PRIORITY** | Low | Low | Routine monitoring |

---

## Model Architecture

```
CDM Sequence (max 20 timesteps × 10 features)
    │
    ▼
Masking Layer  (sentinel = −999.0, handles variable-length sequences)
    │
    ▼
Bidirectional GRU  (128 units, L2 regularization)
LayerNormalization + Dropout (0.3)
    │
    ▼
Bidirectional GRU  (64 units, L2 regularization)
LayerNormalization + Dropout (0.3)
    │
    ▼
Dense (64 units, ReLU) → Dropout (0.3)
Dense (32 units, ReLU) → Dropout (0.3)
    │
    ▼
Output: Predicted next CDM (10 features, linear)
```

**Key design decisions**:

- **Bidirectional GRU over LSTM**: Captures both forward and backward temporal dependencies with fewer parameters. GRU converges faster than LSTM on our sequence lengths (2–20 CDMs).
- **LayerNormalization over BatchNormalization**: BatchNorm's batch statistics corrupt MC Dropout uncertainty estimates at inference. LayerNorm operates per-sample and is stable regardless of batch size.
- **MC Dropout for uncertainty**: Dropout remains active during inference (`training=True`). 50 forward passes produce a distribution over predictions. The standard deviation of this distribution is our uncertainty signal — a Bayesian approximation without explicit priors.
- **Self-supervised task**: Input = CDM₁ … CDMₙ₋₁, Target = CDMₙ. No collision labels needed. The model learns trajectory dynamics as a side effect of this prediction task.
- **Weighted MSE loss**: `log10_pc` (×2.0) and `MISS_DISTANCE` (×1.5) are upweighted. These features carry the most operational significance.
- **Gradient clipping** (`clipnorm=1.0`): Covariance features span orders of magnitude. Without clipping, a single outlier CDM can produce catastrophic weight updates.

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| GRU units (layer 1 / layer 2) | 128 / 64 |
| Dense units | 64, 32 |
| Dropout rate | 0.3 |
| L2 regularization | 0.001 |
| Learning rate | 0.001 (Adam, decaying) |
| Batch size | 64 |
| Max sequence length | 20 timesteps |
| MC Dropout samples | 50 |
| Total parameters | 244,171 |

---

## Dataset

**Source**: ALDORIA — proprietary CDM dataset provided for SDC2026  
**Format**: KVN (Keyhole Notation) Conjunction Data Messages  
**Coverage**: Full calendar year, global conjunction screenings

> ⚠ The raw KVN data files are **not included** in this repository due to licensing restrictions from ALDORIA.

**Statistics after parsing**:

| Metric | Value |
|--------|-------|
| Total CDMs | 185,511 |
| Unique conjunction events | 2,003 |
| CDMs per event | 2–48 (median: ~8) |
| Training events | 1,602 (80%) |
| Validation events | 200 (10%) |
| Test events | 201 (10%) |

**Split methodology**: Events are split at the event level — all CDMs from a single conjunction event are assigned to exactly one split. This prevents data leakage: no model can memorize the future of an event it has seen training CDMs from.

**Input features** (10 per CDM, after engineering):

| Feature | Description | Transform |
|---------|-------------|-----------|
| `log10_pc` | log₁₀(Collision Probability) | log-scale |
| `MISS_DISTANCE` | Closest approach distance (m) | raw |
| `time_to_tca_hours` | Hours until Time of Closest Approach | raw |
| `RELATIVE_SPEED` | Relative velocity at TCA (m/s) | raw |
| `RELATIVE_POSITION_R` | Radial separation (m) | raw |
| `RELATIVE_POSITION_T` | Transverse separation (m) | raw |
| `RELATIVE_POSITION_N` | Normal separation (m) | raw |
| `combined_cr_r` | Combined radial covariance (m²) | log1p |
| `combined_ct_t` | Combined transverse covariance (m²) | log1p |
| `combined_cn_n` | Combined normal covariance (m²) | log1p |

Covariance features are log1p-transformed before scaling. Raw values span 0–20 billion m²; StandardScaler cannot normalize such a range. Log1p compresses this to [0, ~22], enabling stable training.

---

## Results

Trained for 150 epochs on CPU (~2.7 hours). ReduceLROnPlateau decayed the learning rate 7 times (0.001 → 7.8×10⁻⁶). Early stopping patience = 20.

| Metric | Value |
|--------|-------|
| Best validation loss | 0.628 (epoch 147 of 150) |
| Test loss | 0.620 |
| Test overall MAE | 0.506 |
| Test Pc MAE (`log₁₀` scale) | 0.352 |
| Training / val loss gap | < 8% (well-generalized) |
| Validation loss before engineering fixes | 84.9 |
| Validation loss after fixes (150 epochs) | **0.628** (~99.3% reduction) |

**Risk classification on test set** (201 conjunction events):

| Quadrant | Count | Share |
|----------|-------|-------|
| ACT NOW | 1,259 | 62.9% |
| WATCH CLOSELY | 44 | 2.2% |
| **SAFELY IGNORE** | **638** | **31.9%** |
| NOT PRIORITY | 62 | 3.1% |

35% of all conjunction events are classified as low-priority with high model confidence — these can be safely deprioritized without manual review, directly reducing operator workload.

---

## Repository Structure

```
SDC2026_KAU_AE_TEAM/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
│
└── Scripts/
    │
    ├── config.yaml              ← All hyperparameters (single source of truth)
    ├── model_builder.py         ← BiGRU architecture (used by step3, step3b, step4)
    ├── scoring.py               ← Threat/confidence scoring (used by step3b, step4)
    ├── run_pipeline.py          ← Automated end-to-end pipeline runner
    │
    ├── step1_parse_kvn.py       ← KVN files → parsed_cdm_data.csv
    ├── step2_prepare_sequences.py  ← CSV → numpy sequences (train/val/test)
    ├── step3_train_model.py     ← BiGRU training
    ├── step3b_evaluate_proxy_confidence.py  ← Pre-production evaluation gate
    ├── step4_inference_dashboard.py  ← Production inference → threat scores
    ├── step5_visualize.py       ← Publication figures
    ├── step5b_detailed_reports.py   ← Per-event detailed reports
    │
    ├── tools/                   ← Developer tools (manual use, not part of pipeline)
    │   ├── README.md
    │   ├── inspect_data.py      ← Data audit and distribution diagnostics
    │   ├── calculate_R2.py      ← R² evaluation on test predictions
    │   ├── train_val_test_graph.py  ← Standalone training curve figure
    │   └── visualize_model_architecture.py
    │
    └── tests/                   ← Pytest unit test suite
        ├── README.md
        ├── test_scoring.py      ← Threat/confidence logic (13 tests)
        ├── test_sequences.py    ← Padding and split logic
        ├── test_model_io.py     ← Architecture and weight I/O
        ├── test_parser.py       ← KVN parser logic
        └── smoke_tests.py       ← End-to-end artifact checks
```

---

## Team

**SDC2026 KAU AE Team**  
King Abdulaziz University — Aerospace Engineering Department  
Jeddah, Saudi Arabia

| Name | Role | LinkedIn |
|------|------|----------|
| Ahmad Alharbi | Team Lead & Lead Developer | [linkedin.com/in/ahmed-alharbi-973b63246](https://www.linkedin.com/in/ahmed-alharbi-973b63246/) |
| Abdulelah Mojelad | Research & Development | [linkedin.com/in/abdulellah-mojalled](https://www.linkedin.com/in/abdulellah-mojalled/) |
| Hamzah Alharbi | Research & Development | [linkedin.com/in/hamzah-alharbi-00b18133a](https://www.linkedin.com/in/hamzah-alharbi-00b18133a/) |
| Khalid Alsadoon | Research & Development | [linkedin.com/in/khalid-alsadoon-a95802242](https://www.linkedin.com/in/khalid-alsadoon-a95802242/) |
| Mohamedhakim Hassan | Research & Development | [linkedin.com/in/mohamed-hassan-aero](https://www.linkedin.com/in/mohamed-hassan-aero/) |

---

## Acknowledgments

- **Saudi Space Agency (SSA)** — for organizing the Space Debris Conference 2026 and the DebriSolver Competition
- **ALDORIA** — for providing the CDM dataset that made this research possible
- **King Abdulaziz University** — for institutional support

---

## Citation

```bibtex
@inproceedings{alharbi2026conjunction,
  title        = {Learning Conjunction Dynamics: A Self-Supervised Approach
                  to Satellite Collision Risk Assessment},
  author       = {Alharbi, Ahmad and Mojelad, Abdulelah and Alharbi, Hamzah
                  and Alsadoon, Khalid and Hassan, Mohamedhakim},
  booktitle    = {Space Debris Conference 2026 — DebriSolver Competition},
  address      = {Riyadh, Saudi Arabia},
  year         = {2026},
  organization = {Saudi Space Agency (SSA)}
}
```

---

*King Abdulaziz University Aerospace Engineering Team — built for safer space operations.*
