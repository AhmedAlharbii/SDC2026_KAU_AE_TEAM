## 4. Our Proposed Solution - The Core Idea

### 4.1 The Central Insight: CDM Sequences Tell a Story

The foundational insight of DebriSolver is simple but powerful: **a sequence of CDMs for a single conjunction event is not a collection of independent snapshots - it is a story with a trajectory, and that trajectory is rich with information about how dangerous the event really is.**

Consider two events, both currently showing Pc = 5×10⁻⁵ (below the 1×10⁻⁴ maneuver threshold):

**Event A:** 12 CDMs over 8 days. Pc started at 1×10⁻⁸ four days ago, rose steadily to 1×10⁻⁶, then jumped to 5×10⁻⁵ in the last 24 hours. Covariance is tightening with each CDM - new tracking data is arriving regularly. TCA is 18 hours away.

**Event B:** 2 CDMs. Pc was 6×10⁻⁵ yesterday, is 5×10⁻⁵ today - slightly declining. Covariance is enormous (tracking is poor). TCA is 11 days away.

Both sit below the 1×10⁻⁴ threshold. A threshold-based system treats them identically. But any experienced conjunction analyst would immediately recognize:
- Event A is **alarming** - rising Pc, imminent TCA, good data quality means the estimate is trustworthy, and the trend is still upward
- Event B is **low priority** - Pc is probably an artifact of poor tracking, TCA is far away, and one more radar track may eliminate the event entirely

This distinction is exactly what our model learns to capture. By training on thousands of CDM sequences, the BiGRU learns what "alarming trajectory" looks like vs. what "noise that will resolve" looks like - without ever being told which events were actually dangerous.

---

### 4.2 The Self-Supervised Formulation

The self-supervised learning task is framed as **next-CDM prediction**:

> Given the first k CDMs of a conjunction event (CDM₁, CDM₂, ..., CDMₖ), predict the feature values of CDMₖ₊₁.

Formally, for a conjunction event with N CDMs:
- We generate N−1 training samples: (input: CDMs 1..k, target: CDM k+1) for k = 1, 2, ..., N−1
- **Input X:** A padded sequence of shape (max_len=20, n_features=11), where CDMs before the first real measurement are filled with the sentinel value −999.0
- **Target Y:** A 1D vector of shape (n_features=11) representing the next CDM's feature values

All features in both X and Y are standardized (StandardScaler, fitted on training data only). Covariance features are log1p-transformed before standardization to handle their extreme scale.

This formulation has elegant properties:
1. **No labels required** - the "label" for each sample is simply the next row in the same CSV. The data supervises itself.
2. **Every CDM in a sequence contributes** - a 10-CDM event generates 9 training samples, each giving the model a different temporal vantage point
3. **Temporal order is preserved** - CDMs are always presented in chronological order (by CREATION_DATE), so the model learns causal dynamics
4. **Variable-length sequences are handled naturally** - via left-padding and the Masking layer, so 2-CDM and 20-CDM events coexist in the same batch

At training time, the model minimizes a weighted MSE loss over all predicted features, with higher weights on Pc (×2.0) and miss distance (×1.5) - reflecting their greater operational importance.

---

### 4.3 What "Learning Conjunction Dynamics" Means

When we say the model "learns conjunction dynamics," we mean it implicitly encodes in its weights several physical and observational relationships:

**Pc evolution near TCA:** As time to TCA decreases, conjunction geometry becomes more constrained. For well-tracked events, Pc often peaks in the last 24-48 hours before TCA, then either spikes (genuine threat) or collapses (tracking data resolution). The model learns this temporal pattern.

**Covariance decay with tracking:** As more radar observations arrive, orbit determination improves and covariance shrinks. A sequence showing steadily decreasing covariance is physically distinct from one showing constant or growing covariance. The model learns this as a signature of data quality.

**Miss distance refinement:** Early CDMs often have large uncertainty in miss distance. As TCA approaches and covariance tightens, miss distance estimates converge. The model learns the typical trajectory of this convergence.

**Relative position evolution in RTN frame:** The radial (R), transverse (T), and normal (N) components of relative position evolve predictably as TCA approaches. The model learns these geometric relationships.

Crucially, the model does **not** learn "event X was a collision." It learns "event X's sequence evolution pattern." When presented with a new event, it predicts what the next CDM should look like. If the actual next CDM is very different from the prediction, the event is behaving anomalously - which is a risk signal.

---

### 4.4 From Prediction to Threat Score

At inference time, for each test event, the model makes a prediction of the next CDM's features. The **threat score** (0-100) is derived from these predictions using physics-based rules in `scoring.py`:

**Base threat from predicted Pc level:**
The predicted log₁₀(Pc) is converted back to physical Pc via inverse-transform. This predicted Pc is mapped to a base threat score using operational thresholds:
- Predicted Pc > 1×10⁻³ → base threat ≈ 80-100 (extreme concern)
- Predicted Pc in [1×10⁻⁴, 1×10⁻³] → base threat ≈ 50-80 (maneuver evaluation zone)
- Predicted Pc in [1×10⁻⁶, 1×10⁻⁴] → base threat ≈ 20-50 (monitoring zone)
- Predicted Pc < 1×10⁻⁶ → base threat ≈ 0-20 (low concern)

**Trend modifier (where is Pc going?):**
The current Pc (from the last real CDM in the sequence) is compared to the predicted next Pc. If predicted Pc > current Pc (rising trend), threat is boosted. If predicted Pc < current Pc (falling trend), threat is penalized. This is the core novel contribution - the model's prediction encodes trajectory direction.

**TCA urgency bonus:**
Events with predicted time-to-TCA < 24 hours receive an urgency bonus. Events with TCA > 7 days receive a suppression factor. Imminent events demand more aggressive triage regardless of absolute Pc level.

The threat score is clipped to [0, 100] and represents a continuous ordering of event urgency.

---

### 4.5 From Uncertainty to Confidence

The **confidence level** (0.0-1.0) answers a different question than threat: *how much should the operator trust this threat assessment?*

```uncertaintyplot
```

Confidence is computed from three independent components, each reflecting a distinct source of epistemic information:

**Component 1 - MC Dropout Uncertainty (weight: 40%)**
The BiGRU is run 50 times with Dropout active (`training=True`). Each pass uses a different dropout mask, producing a different prediction. The standard deviation across 50 predictions quantifies how uncertain the model is about the next CDM. Low std → model is confident in its prediction → higher confidence score. High std → model is confused by this event's trajectory → lower confidence.

**Component 2 - Data Quantity (weight: 35%)**
More CDMs in the input sequence means more information. An event with 15 CDMs gives the model a rich temporal picture; an event with 2 CDMs barely constrains the model. Confidence scales with the number of valid (non-padding) timesteps in the input.

**Component 3 - Covariance Quality (weight: 25%)**
The combined position covariance (CR_R + CT_T + CN_N, in raw m²) of the two objects determines how well-tracked they are. Large covariance means large tracking uncertainty, which means the Pc estimate itself is unreliable. Confidence is penalized as a function of covariance size, with a threshold calibrated to typical LEO tracking quality.

The three components are combined into a final confidence value clipped to [0.10, 1.00] (never zero - some minimum confidence always exists even with 1 CDM).

---

### 4.6 The Four-Quadrant Risk Classification

Threat score and confidence together define a **two-dimensional risk space**. Every conjunction event is placed in this space and assigned to one of four operational quadrants:

**ACT NOW** (High Threat, High Confidence): The model is predicting a dangerous trajectory *and* is confident in that prediction. Data is plentiful, tracking is good, uncertainty is low. These events demand immediate human review and potential maneuver planning.  
**WATCH CLOSELY** (High Threat, Low Confidence): The model flags a potential threat but uncertainty is high - perhaps because there are only 2 CDMs, or because covariance is enormous. More tracking data is needed before a maneuver decision. Human oversight required; request additional observations.  
**SAFELY IGNORE** (Low Threat, High Confidence): The model predicts Pc will remain low *and* is confident about it. The event has good tracking data and a benign trajectory. These can be deprioritized with confidence - operators can focus elsewhere.  
**NOT PRIORITY** (Low Threat, Low Confidence): Low predicted threat but also low confidence. Monitor passively. May warrant a second look if more CDMs arrive.

The quadrant boundaries in our implementation:
- Threat threshold: 50/100 (separates High from Low threat)
- Confidence threshold: 0.5 (separates High from Low confidence)

```quadrantchart
```

---

### 4.7 Why This Is Deployment-Safe (No Label Leakage)

A critical property of our system: **it cannot leak future information, and its outputs are fully explainable without collision ground truth.**

**No label leakage:** The training process never uses any information about whether a conjunction ultimately resulted in a collision. The pretext task (predict next CDM) uses only information available at the time of prediction. There is no "outcome label" anywhere in the pipeline.

**No future data leakage:** Each prediction is made from CDMs available at a specific moment in time (CDMs 1 through k, predicting CDM k+1). The model never sees CDMs beyond the prediction horizon. The Masking layer ensures that padding positions contribute zero gradient.

**Reproducibility:** All random operations (data split, model initialization, dropout masks at inference) are seeded with SEED=42 and TensorFlow deterministic mode is enabled, producing identical results across runs.

**Interpretability:** The threat score and confidence are computed from physical quantities (predicted Pc, predicted time-to-TCA, covariance) - all directly traceable to CDM fields. An operator can always ask "why is this event ACT NOW?" and receive an answer grounded in physics: "predicted Pc = 2.3×10⁻⁴ (above 1×10⁻⁴ threshold), rising from current 4.1×10⁻⁵, TCA in 11 hours, model uncertainty low (std=0.08), 14 CDMs available."

---

### 4.8 Comparison to Our Initial Approach (What We Changed)

The system described above is the result of significant iteration. Our initial design had several critical differences:

**Initially: Raw COLLISION_PROBABILITY as a direct feature**
We included the raw linear Pc value (0 to 1) as a training feature. This caused catastrophic validation loss (~84.9) because StandardScaler cannot handle a distribution that spans 10 orders of magnitude - a few high-Pc events produced scaled values of 100+ standard deviations. **Fix:** Replaced with log₁₀(Pc), compressing the same range into a manageable [-∞ to 0] interval (capped at -10 for near-zero Pc).

**Initially: Raw covariance values without log1p**
Covariance features span from near-zero to 20 billion m². Even with StandardScaler, the resulting scaled values for extreme-covariance events were wildly outside the [-3, 3] range. **Fix:** Applied log1p before StandardScaler, compressing the range from [0, 2×10¹⁰] to [0, ~22].

**Initially: Zero as the padding sentinel**
After StandardScaler, real CDM features are mean-centered - meaning real values can legitimately be zero. The Masking layer using mask_value=0 was masking real data, not just padding. **Fix:** Changed padding sentinel to -999.0, a value that is physically impossible after standardization.

**Initially: BatchNormalization in the model**
The original architecture used BatchNorm for stability. This broke MC Dropout - batch statistics were recomputed each forward pass in a way unrelated to weight uncertainty. **Fix:** Replaced all BatchNorm with LayerNorm, which is sample-local and immune to the training=True/False distinction.

**Initially: Confidence weights heavily biased toward MC Dropout uncertainty**
The original confidence formula weighted MC Dropout std at 60%, leaving data quantity and covariance quality at 20% each. Most events have only 2-5 CDMs, so data_confidence was always low, dragging the total below 0.5. Every event landed in the "WATCH CLOSELY" or "SAFELY IGNORE" zone - the system had no discrimination. **Fix:** Rebalanced to 40/35/25, allowing events with enough CDMs and low uncertainty to reach ACT NOW.

Each of these changes is documented in detail in Section 16 with the exact diagnostic process used to discover and fix each problem.

## 5. Data: From Raw KVN to Structured CSV

```preprocessingdiagram
```

### 5.1 Anatomy of a KVN CDM File

Each KVN file is a plain-text document containing one Conjunction Data Message. The file structure is strictly linear: every key-value pair appears on its own line, and the ordering within the file defines which `OBJECT` block each field belongs to.

A complete CDM file has three logical sections:

**1. Global header** - fields that describe the conjunction as a whole, before any `OBJECT = OBJECT1` delimiter appears:
```
CCSDS_CDM_VERS          = 1.0
CREATION_DATE           = 2025-11-01T12:00:00.000
ORIGINATOR              = ALDORIA
MESSAGE_ID              = CDM_25544_48274_003
TCA                     = 2025-11-03T10:00:00.000
MISS_DISTANCE           = 150.5 [m]
RELATIVE_SPEED          = 12500.0 [m/s]
RELATIVE_POSITION_R     = -42.3 [m]
RELATIVE_POSITION_T     = 143.1 [m]
RELATIVE_POSITION_N     = -11.7 [m]
COLLISION_PROBABILITY   = 1.5E-05
COLLISION_PROBABILITY_METHOD = FOSTER-1992
```

**2. OBJECT1 block** - begins at the `OBJECT = OBJECT1` line:
```
OBJECT                  = OBJECT1
OBJECT_DESIGNATOR       = 25544
OBJECT_NAME             = ISS (ZARYA)
OBJECT_TYPE             = PAYLOAD
MANEUVERABLE            = YES
X                       =  6500.123 [km]
Y                       =  1200.456 [km]
Z                       =   500.789 [km]
X_DOT                   =     1.523 [km/s]
Y_DOT                   =    -7.201 [km/s]
Z_DOT                   =     0.312 [km/s]
CR_R                    =    25.000 [m**2]
CT_T                    =   100.000 [m**2]
CN_N                    =    50.000 [m**2]
```

**3. OBJECT2 block** - begins at `OBJECT = OBJECT2`:
```
OBJECT                  = OBJECT2
OBJECT_DESIGNATOR       = 48274
OBJECT_NAME             = COSMOS DEB
OBJECT_TYPE             = DEBRIS
MANEUVERABLE            = NO
CR_R                    =  8500.000 [m**2]
CT_T                    = 430000.000 [m**2]
CN_N                    = 12000.000 [m**2]
```

Key parsing challenges: field order is not guaranteed, units appear in brackets on the same line as values, some fields appear only in one object block, and the OBJECT1/OBJECT2 distinction must be tracked by the parser's state machine.

---

### 5.2 The KVN Parser (step1_parse_kvn.py) - Why It Was Needed and How It Works

Due to severe timestamp corruption in the provided CSV database (with approximately 60% of the timestamps being unreadable), a custom Python parser (`step1_parse_kvn.py`) was developed to process the original KVN files directly. By avoiding the broken CSV and attacking the raw KVN, the custom parser achieved 99.95% timestamp recovery-retaining 185,415 valid CDMs out of the original 185,511.

`step1_parse_kvn.py` implements a stateful line-by-line parser. The core function `parse_kvn_file(filepath)` returns a flat Python dictionary containing all extracted fields for one CDM, or `None` if parsing fails.

The parser is **fail-soft**: if a field is missing (not all CDMs include all optional fields), it simply isn't added to the record. Missing fields are later handled by the `SimpleImputer` in step 2.

---

### 5.3 Field Extraction & Unit Stripping

KVN values often include physical units in square brackets: `150.5 [m]`, `100.0 [m**2]`, `6500.0 [km]`. These must be stripped before numeric conversion.

The `strip_units()` function uses a regex to remove everything from the first `[` to the end of the string.
After stripping, values are stored as strings in the record dictionary. Numeric conversion to `float` happens downstream in step 2 when building the pandas DataFrame, where `pd.to_numeric(errors='coerce')` handles any remaining non-numeric values by converting them to `NaN` for imputation.

Special cases handled:
- **Scientific notation:** `1.5E-05` is valid Python float notation and converts cleanly
- **Negative values:** `-42.3 [m]` strips correctly to `-42.3`
- **Empty values:** `COLLISION_PROBABILITY =` (blank after `=`) - stored as empty string, later becomes NaN
- **COMMENT lines:** Explicitly skipped; some KVN files embed commentary inline

---

### 5.4 Object-Specific Field Namespacing

CDM files contain two objects, and many fields (especially covariance terms) appear for both. Without namespacing, the second object's `CR_R` would overwrite the first's.

The parser tracks the current object context using the state variable `current_object`:
- Before any `OBJECT = OBJECT1` line: fields are global (e.g., `TCA`, `MISS_DISTANCE`)
- After `OBJECT = OBJECT1`: fields are prefixed `object1_` (e.g., `object1_CR_R`)
- After `OBJECT = OBJECT2`: fields are prefixed `object2_` (e.g., `object2_CR_R`)

This flat, namespaced structure makes the downstream pandas processing straightforward - every CDM row in the CSV has the same column schema.

---

### 5.5 Event ID Construction from Filenames

The event identifier - which groups all CDMs for the same conjunction together - is derived from the KVN **filename**, not from its contents. The ALDORIA naming convention is:

```
CDM_<NORAD_ID_1>_<NORAD_ID_2>_<MESSAGE_NUMBER>.kvn
```

For example: `CDM_25544_48274_003.kvn`
- `25544` = NORAD ID of Object 1
- `48274` = NORAD ID of Object 2
- `003` = this is the 3rd CDM for this conjunction event

This is a robust approach because:
- The filename is always present (parsing doesn't depend on file contents for identification)
- The NORAD IDs in the filename are guaranteed to be consistent across all CDMs for the same event
- Different message numbers for the same event naturally share the same `event_id`

---

### 5.6 Derived Feature Engineering at Parse Time

Beyond extracting raw CDM fields, the parser computes several derived features that are more useful for modeling than the raw fields:

**`time_to_tca_hours`** - Time remaining until closest approach, in hours. This is arguably the most operationally important derived feature - it tells the model exactly where in the conjunction timeline this CDM was issued.

**`log10_pc`** - Log-10 of collision probability.
Converts the 10-order-of-magnitude range of Pc values into a compact [-10, 0] scale suitable for neural network training.

**`combined_cr_r`, `combined_ct_t`, `combined_cn_n`** - Sum of both objects' covariance diagonal elements.
The combined covariance represents the total positional uncertainty of the conjunction. A single combined value is more useful to the model than four separate (per-object × per-axis) values.

---

### 5.7 Schema Validation & Fail-Fast Design

After parsing, each record is validated before being accepted into the output CSV. A CDM is rejected (`parse_kvn_file` returns `None`) if:

- `TCA` is missing or cannot be parsed as a datetime
- `CREATION_DATE` is missing or cannot be parsed
- `MISS_DISTANCE` is missing (required for basic conjunction geometry)
- `COLLISION_PROBABILITY` is missing (required for threat scoring)
- `time_to_tca_hours` is negative (CDM was created after TCA - stale, operationally useless)

Records that pass validation but have some optional fields missing (e.g., covariance terms for one object) are accepted with those fields as `NaN`, to be handled by the `SimpleImputer` in step 2.

---

### 5.8 Output: parsed_cdm_data.csv - Structure & Statistics

The parser collects all valid CDM records into a list of dictionaries, then writes them to `parsed_cdm_data.csv`.

**File statistics:**
- **Total rows:** 185,511 CDMs
- **Total columns:** ~45 (including all raw + derived + object metadata)
- **File size:** 213 MB
- **Unique event_ids:** 20,506 raw events; 2,003 events after filtering for ≥ 2 CDMs per event

---

### 5.9 Data Quality Observations

Running the inspection tool (`Scripts/tools/inspect_data.py`) on the parsed and processed sequences revealed several important data quality characteristics:

```preprocessingdiagram
```

**Padding dominance:** 69.3% of all values in X_train are the sentinel value −999.0. This is expected - most events have 2-8 CDMs, leaving 12-18 of the 20 timestep slots padded. The Masking layer is therefore critically important.

**Covariance extreme outliers:** Even after log1p transform, the `combined_ct_t` feature showed values up to 32× the training standard deviation in the validation set.

**Distribution skew in COLLISION_PROBABILITY:** The raw linear Pc distribution is extremely right-skewed. After log₁₀ transform, the distribution is approximately Gaussian.

**Cross-split consistency:** The train/val/test distributions for most features are well-matched.

**Missing value rates:** Approximately 3-8% of CDMs are missing at least one covariance term, resolved by imputation.

---

## 6. Feature Engineering & Sequence Preparation

### 6.1 Feature Selection - The 11 Model Features

The final feature set used for training contains **11 features** per CDM timestep. These were chosen to capture the three most important dimensions of conjunction information: **risk level** (Pc and miss distance), **temporal urgency** (time to TCA), and **data quality** (covariance in RTN frame and relative position).

| # | Feature | Source | Why Included |
|---|---------|--------|-------------|
| 0 | `COLLISION_PROBABILITY` | CDM global | Raw Pc - retained alongside log10_pc for model redundancy |
| 1 | `log10_pc` | Derived | Compresses 10-order-of-magnitude Pc range into [-10, 0] |
| 2 | `MISS_DISTANCE` | CDM global | Direct conjunction geometry - key risk indicator |
| 3 | `time_to_tca_hours` | Derived | Temporal urgency - where are we in the event timeline? |
| 4 | `combined_cr_r` | Derived (log1p) | Radial covariance quality after log1p compression |
| 5 | `combined_ct_t` | Derived (log1p) | Transverse covariance quality - most variable axis |
| 6 | `combined_cn_n` | Derived (log1p) | Normal covariance quality |
| 7 | `RELATIVE_SPEED` | CDM global | Hypervelocity indicator - affects collision energy |
| 8 | `RELATIVE_POSITION_R` | CDM global | Radial component of separation vector |
| 9 | `RELATIVE_POSITION_T` | CDM global | Transverse component of separation vector |
| 10 | `RELATIVE_POSITION_N` | CDM global | Normal component of separation vector |

Features intentionally excluded:
- **State vectors (X, Y, Z, X_DOT, Y_DOT, Z_DOT):** Absolute position in ECI frame; not meaningful for conjunction geometry without both objects simultaneously
- **Per-object covariance** (before combining): Replaced by the combined form, which is what actually determines the Pc uncertainty
- **OBJECT_NAME, OBJECT_TYPE, MANEUVERABLE:** Categorical metadata - not appropriate for direct numeric input to a recurrent network

---

### 6.2 Why Raw COLLISION_PROBABILITY Was Excluded as the Primary Feature

This was the single most impactful engineering decision in the project. In the initial design, `COLLISION_PROBABILITY` (raw linear scale) was the primary Pc feature. The result was a validation loss of **~84.9** that did not converge.

The fix: use `log10_pc` as the primary feature, retaining raw `COLLISION_PROBABILITY` as an additional feature (now scaled to a manageable range). After the fix, scaled target values for all features stayed within roughly ±5 standard deviations, and validation loss dropped from 84.9 to 0.628.

---

### 6.3 The Covariance Problem: Spans 0 to 20 Billion m²

The three combined covariance features (`combined_cr_r`, `combined_ct_t`, `combined_cn_n`) posed the second major data engineering challenge.
Physical covariance values in the ALDORIA dataset:
- **Minimum:** Near-zero (objects tracked with high-precision sensors, uncertainty < 1 m²)
- **Maximum combined_ct_t:** ~20 billion m² (objects with only TLE-quality tracking, uncertainty spanning ±141 km in the transverse direction)
- **Typical range:** 10,000 m² to 500,000,000 m² (a 5-order-of-magnitude spread for "typical" events)

Log transform solves this by giving equal weight to each order of magnitude.

---

### 6.4 The log1p Transform Solution

The log1p transform compresses extreme ranges while preserving the ordering and relative differences between values. It is applied *before* StandardScaler in step 2.

*x_transformed = ln(1 + x)*

Using +1 ensures that x=0 maps to ln(1+0) = 0 rather than -∞, which is important because some objects have zero reported covariance in certain axes.

---

### 6.5 The log10_pc Feature

The `log10_pc` feature transforms raw collision probability into a physically interpretable logarithmic scale. The floor at 1 × 10⁻¹⁰ prevents -∞ for events with Pc = 0.

*x_{log10_pc} = max(log₁₀(Pc), -10)*

After this transform, the feature spans roughly [-10, 0]:
- log10_pc = -10: Pc ≤ 1×10⁻¹⁰ (effectively zero risk)
- log10_pc = -6: Pc = 1×10⁻⁶ (routine monitoring)
- log10_pc = -4: Pc = 1×10⁻⁴ (maneuver evaluation threshold)
- log10_pc = -1: Pc = 0.1 (extreme - collision likely if estimate is accurate)

---

### 6.6 Event-Level Train/Val/Test Split (80/10/10)

The dataset is split **by event**, not by CDM. All CDMs from a given conjunction event (e.g., all 12 CDMs for event `25544_48274`) are assigned exclusively to one of the three splits.

| Split | Events | Self-Supervised Samples |
|-------|--------|------------------------|
| Training | 1,602 (80.0%) | 77,989 |
| Validation | 200 (10.0%) | 9,598 |
| Test | 201 (10.0%) | 9,637 |

---

### 6.7 Why Event-Level Split Prevents Data Leakage

An event-level split is the **only correct** split strategy for self-supervised CDM modeling. The alternative - splitting CDMs randomly regardless of which event they belong to - creates a severe data leakage problem.

With event-level split:
- All 12 CDMs of an event are in the test set (or all in training - never split across sets)
- The model has never seen any CDM from this event during training
- The test evaluation is a genuine measure of generalization to unseen conjunction events

---

### 6.8 The Padding Sentinel: Why -999.0, Not Zero

Sequences of different lengths must be padded to a uniform length (max_sequence_length = 20) for batched training. The padding value must satisfy one constraint: **it must be distinguishable from any legitimate data value after normalization.**

**Why not 0.0?** After `StandardScaler`, real CDM features are mean-centered. A feature with mean 0 and std 1 regularly produces values of 0.0 from real data. A Masking layer with `mask_value=0` would incorrectly mask real timesteps where all features happen to be at their mean simultaneously.

**Why -999.0?** After `StandardScaler`, the minimum physically realistic feature value is approximately -3.7. A value of -999.0 is 999 standard deviations below the mean - completely impossible from real data. It is therefore an unambiguous sentinel.

Left-padding (prepending padding before the real data) ensures that the **most recent CDMs are always at the end of the sequence** - aligned with the temporal direction that matters most for prediction.

---

### 6.9 Self-Supervised Sample Generation

For each conjunction event with N CDMs (N ≥ 2), step 2 generates N−1 training samples. For event k with CDMs sorted by CREATION_DATE.

This gives the model exposure to every temporal vantage point within each event: it must learn to predict from both sparse early-event contexts (few CDMs, lots of padding) and information-rich late-event contexts (many CDMs, no padding). This variety makes the model robust across the full range of event maturities seen at inference time.

---

### 6.10 StandardScaler: Fitted on Training Data Only

A `StandardScaler` is fitted **only on training data** and then applied to transform validation and test data. 

z = (x - μ) / σ

Where μ and σ are the mean and standard deviation computed exclusively from the training set. This is standard machine learning hygiene: fitting the scaler on validation or test data would contaminate the training statistics with unseen-data information, producing overly optimistic performance estimates.

---

### 6.11 SimpleImputer: Median Strategy for Missing Values

Before scaling, missing values (NaN) must be filled. A `SimpleImputer` with `strategy='median'` is used.
**Why median over mean?** The covariance distributions are heavily right-skewed. The median is far more robust than the mean for skewed distributions with outliers.

---

### 6.12 Final Sequence Shapes & Storage

After imputation, log1p transform, scaling, event-level split, sample generation, and padding, the final arrays are saved as NumPy binary files.
`;
