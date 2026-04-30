# Test Suite

Unit tests for the DebriSolver pipeline. These tests are fast, fully offline, and require no pipeline artifacts (no CSV files, no model weights, no `.npy` arrays).

The goal is to catch logic regressions early — before retraining an expensive model or running a 2-hour pipeline.

---

## Running Tests

```powershell
# From the project root — run all tests
python -m pytest Scripts/tests/ -v

# Run a specific file
python -m pytest Scripts/tests/test_scoring.py -v

# Run with short output (just pass/fail)
python -m pytest Scripts/tests/ -q
```

Expected output: all tests pass in under 10 seconds.

---

## Test Files

### `test_scoring.py` — Threat & Confidence Logic

**What it covers**: The physics-based threat scoring and confidence calculation in `scoring.py`.

**Why it matters**: `scoring.py` is the single source of truth that both step3b and step4 import. A silent bug here would corrupt all threat scores without any obvious error — the pipeline would still "run" and produce wrong outputs.

Key scenarios tested:
- Pc above the ESA/NASA maneuver threshold (1e-4) gives threat > 50
- Pc well below threshold gives threat < 15
- Rising Pc trend gives higher threat than flat Pc
- Imminent TCA (< 24h) adds urgency modifier
- Threat is always clipped to [0, 100]
- Confidence is always in [0.1, 1.0]
- Higher MC Dropout std → lower confidence
- More valid CDM timesteps → higher confidence
- Covariance expm1 correction: log1p-scale covariance after scaler inverse_transform is correctly converted back to m² before confidence scoring

---

### `test_sequences.py` — Sequence Preparation Logic

**What it covers**: The padding logic and event-level split logic from `step2_prepare_sequences.py`.

**Why it matters**: Sequence shape errors or padding bugs would cause silent failures — a wrongly-padded sequence would either crash the model or silently feed garbage inputs. The event-level split is critical for data integrity (CDM leakage between train/val/test would inflate metrics).

Key scenarios tested:
- Short sequences are padded to exactly `max_len` timesteps
- Padding value is **exactly -999.0** — never 0.0 (which conflicts with StandardScaler outputs)
- Real data is preserved unchanged after padding
- Sequences longer than `max_len` are truncated from the left (keep most recent CDMs)
- Event-level split: no event appears in both train and test
- All events covered exactly once across splits

---

### `test_model_io.py` — Model Architecture & Weight Loading

**What it covers**: The BiGRU architecture built by `model_builder.py` and the weight save/load cycle.

**Why it matters**: The architecture must stay consistent between training (step3) and inference (step4). A mismatch would cause silent wrong predictions — the model would load but produce garbage outputs. The MC Dropout behavior is safety-critical: uncertainty estimates depend on dropout being active during inference.

Key scenarios tested:
- Model builds without error from config
- Output shape is `(batch, n_features)` — not any other shape
- Model uses `LayerNormalization` — NOT `BatchNormalization` (BatchNorm corrupts MC Dropout uncertainty)
- Masking layer uses `mask_value=-999.0` — not 0.0
- At least 2 Dropout layers exist
- `training=True` produces different predictions each pass (MC Dropout active)
- `training=False` produces identical predictions each pass (dropout off)
- Weights saved and reloaded produce bit-identical predictions

---

### `test_parser.py` — KVN Parser Logic

**What it covers**: The CDM parsing logic in `step1_parse_kvn.py`.

**Why it matters**: The parser is the first step. If it silently drops fields or misparses dates, every downstream step is affected. These tests use synthetic KVN strings — no real data files needed.

---

### `smoke_tests.py` — Artifact Sanity Checks

**What it covers**: End-to-end artifact validation after a full pipeline run.

**Why it matters**: Unit tests verify logic in isolation. Smoke tests verify that a full pipeline run actually produced the expected files with the expected structure. They catch integration issues that unit tests miss.

```powershell
# Run after a complete pipeline run
python Scripts/tests/smoke_tests.py --stage full

# Or by stage
python Scripts/tests/smoke_tests.py --stage data    # after step2
python Scripts/tests/smoke_tests.py --stage model   # after step3
python Scripts/tests/smoke_tests.py --stage inference  # after step4
```

Note: Unlike pytest tests, smoke tests **do** require pipeline artifacts to exist.

---

## Design Principles

- **No external artifacts**: All pytest tests use synthetic data and mock scalers. Running `pytest` never reads a CSV, loads a model, or touches `processed_sequences/`.
- **Fast**: Full suite runs in < 10 seconds.
- **Isolated**: Each test class is independent. Failures in one class don't affect others.
- **Explicit failures**: Every assertion has a message explaining what went wrong and why it matters.
