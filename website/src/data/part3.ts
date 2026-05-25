export const part3 = `
## 7. Model Architecture — The BiGRU

### 7.1 Architecture Overview (Layer-by-Layer)

The DebriSolver model is a **Bidirectional Gated Recurrent Unit (BiGRU)** encoder with a Dense regression decoder. It takes a variable-length CDM sequence as input and produces a single-step prediction of the next CDM's features.

### 7.2 The Masking Layer — Handling Variable-Length Sequences

When the Masking layer encounters a timestep where **all features equal -999.0**, it generates a boolean mask of \`False\` for that position. Downstream recurrent layers honor this mask.
This means:
- A sequence with 3 real CDMs and 17 padding positions trains exactly as if only 3 timesteps were present
- The model never "learns from" padding — it learns only from real data

---

### 7.3 Bidirectional GRU Layer 1 (128 units per direction)

This layer processes the sequence in both directions and concatenates the results:
- **Forward GRU (128 units):** Processes CDM₁ → CDM₂ → ... → CDMₙ, building a hidden state that accumulates past context
- **Backward GRU (128 units):** Processes CDMₙ → CDMₙ₋₁ → ... → CDM₁, building a hidden state that incorporates future context

---

### 7.4 Bidirectional GRU Layer 2 (64 units per direction)

The second BiGRU layer refines the temporal representation. With \`return_sequences=False\`, it outputs only the final hidden state — a single vector of size 128 (64 forward + 64 backward) that summarizes the entire sequence.

---

### 7.5 Dense Decoder Layers

After the BiGRU encoder produces a 128-dimensional sequence summary, two Dense layers decode it into a feature prediction.
These layers learn the non-linear mapping from the sequence embedding to the predicted next CDM values. ReLU activations introduce non-linearity while remaining computationally efficient.

---

### 7.6 Output Layer (11 features, linear activation)

The output layer produces 11 real-valued predictions — one for each feature in the scaled feature space. The linear (no) activation is critical: we are performing **regression**, not classification.

---

### 7.7 Why GRU Over LSTM?

\`\`\`grucelldiagram
\`\`\`

GRU was chosen over LSTM for three concrete reasons:
**1. Fewer parameters.** 
**2. Comparable performance on short sequences.** 
**3. Faster convergence.** 

---

### 7.8 Why Bidirectional?

Standard (unidirectional) GRUs process sequences from left to right — earlier CDMs inform later ones, but not vice versa. For a sequence prediction task, this is appropriate at inference time (we can't look into the future). However, during **training**, we have access to the full event sequence, and the backward direction provides valuable context.

---

### 7.9 Why LayerNorm Over BatchNorm? (Critical for MC Dropout)

This is the most architecturally critical decision in the model, and it directly enables MC Dropout uncertainty quantification.

**BatchNormalization** normalizes using statistics computed across the batch dimension. With a different dropout mask each pass, the inputs to each layer change — so the batch statistics change — producing prediction variance that reflects both dropout randomness *and* batch statistic variation.

**LayerNormalization** normalizes across the feature dimension for each sample independently. LayerNorm has no batch-level statistics. Result: with LayerNorm, the only source of variation between MC Dropout forward passes is the dropout mask itself.

---

### 7.10 L2 Regularization on GRU Kernels

L2 regularization (weight decay, λ=0.001) is applied to GRU input-to-hidden kernels, GRU recurrent kernels, and Dense layer kernels.

---

### 7.11 Total Parameters: 244,171

244,171 parameters is lean for this problem — small enough to train on CPU in a reasonable time (~2.7 hours for 150 epochs) while still having sufficient capacity to capture the temporal dynamics of CDM sequences.

---

### 7.12 model_builder.py: The Shared Architecture Module

Rather than defining the model architecture inside \`step3_train_model.py\`, the architecture is encapsulated in \`Scripts/model_builder.py\`.

---

### 7.13 config.yaml: Single Source of Truth

All model hyperparameters are defined in \`Scripts/config.yaml\`. Any hyperparameter change is made in config.yaml and automatically propagated to all pipeline steps that read it.

## 8. Training Strategy & Optimization

### 8.1 The Self-Supervised Training Objective

The model is trained to minimize the difference between its predicted next CDM and the actual next CDM — a **regression task** with no classification labels. The objective function is a weighted Mean Squared Error (wMSE) computed over all 11 features. 

The batch loss function (Equation 1) is given by:

$$ L = \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{11} w_j \left(\hat{y}_{ij} - y_{ij}\right)^2 $$

The total Weighted Mean Squared Error (Equation 2) over the full set is calculated as:

$$ \text{wMSE} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{11} w_j \left(\hat{y}_{ij} - y_{ij}\right)^2 $$

Where:
- $\hat{y}_{ij}$ = predicted value for feature $j$
- $y_{ij}$ = actual next CDM value for feature $j$
- $w_j$ = feature weight

Beyond wMSE, model performance on the held-out test set is quantified using MAE and $R^2$.

Per-feature and overall MAE (Equation 3):

$$ \text{MAE}_{overall} = \frac{1}{N \times 11} \sum_{i=1}^{N} \sum_{j=1}^{11} \left| y_{ij} - \hat{y}_{ij} \right| $$

MAE for collision probability (Equation 4):

$$ \text{MAE}_{P_c} = \frac{1}{N} \sum_{i=1}^{N} \left| P_{c_i}^{pred} - P_{c_i}^{true} \right| $$

Overall $R^2$ is represented as (Equation 6):

$$ R^2 = \frac{1}{11} \sum_{j=1}^{11} \left( 1 - \frac{\sum_{i=1}^{N} \left(y_{ij} - \hat{y}_{ij}\right)^2}{\sum_{i=1}^{N} \left(y_{ij} - \bar{y}_j\right)^2} \right) $$

---

### 8.2 Weighted MSE Loss — Feature Importance Weighting

A standard MSE loss treats all 11 features equally. However, from an operational standpoint, predicting Pc and miss distance accurately is far more important than predicting relative position components. A weighted MSE was implemented to reflect this.

---

### 8.3 Why Gradient Clipping Was Essential (clipnorm=1.0)

Even after log1p transform and StandardScaler, the training data contains rare CDMs with extreme feature values. Gradient clipping with \`clipnorm=1.0\` resolves this by rescaling the **entire gradient vector** if its L2 norm exceeds 1.0.

---

### 8.4 The Adam Optimizer

Adam (Adaptive Moment Estimation) is used over SGD for its adaptive per-parameter learning rates.

---

### 8.5 Learning Rate Scheduling: ReduceLROnPlateau

A \`ReduceLROnPlateau\` callback monitors the validation loss and reduces the learning rate when improvement stalls.

---

### 8.6 EarlyStopping: Patience=20

\`EarlyStopping\` terminates training if the validation loss doesn't improve for 20 consecutive epochs, and restores the best weights seen during training.

---

### 8.7 ModelCheckpoint: Saving Best Weights

\`ModelCheckpoint\` saves the model weights at the epoch with the lowest validation loss.

---

### 8.8 The Full Training Run: 150 Epochs on CPU

The final training run statistics:
- **Hardware:** CPU (no GPU available during development)
- **Duration:** ~2.7 hours total for 150 epochs
- **Batch size:** 256
- **Total samples per epoch:** 77,989 training samples → 305 batches/epoch
- **Best epoch:** 127 (val_loss = 0.628)
- **Stopped at:** epoch 147 (EarlyStopping triggered)

---

### 8.9 Reproducibility: Seed=42, TF Deterministic Ops

Reproducibility was a design requirement. The same training run, on the same hardware, must produce identical results.

---

### 8.10 Training History: Loss & MAE Curves

Training history is saved to \`model_artifacts/training_history.json\`, which records val_loss, train_loss, val_mae, and train_mae for every epoch.

\`\`\`losschart
\`\`\`

---

### 8.11 The gate_passed.flag Invalidation Mechanism

A critical safety rule within the DebriSolver architecture: **every new training run invalidates the previous evaluation gate.** Because production inference (Step 4) requires the exact weighting structure verified during the proxy confidence evaluation (Step 3B), any modification to the model weights—even from the same data and hyperparameters—necessitates re-evaluating the gate. This mechanism guarantees that uncalibrated model artifacts never reach the operational dashboard.

## 9. Uncertainty Quantification — MC Dropout

### 9.1 What Is Monte Carlo Dropout?

Monte Carlo Dropout (MC Dropout) is a technique for obtaining uncertainty estimates from a neural network without requiring any architectural changes beyond the dropout layers already present for regularization. 

---

### 9.2 Dropout as Bayesian Approximation

Formally, Gal & Ghahramani demonstrated that a standard neural network with dropout applied prior to every weight layer is mathematically equivalent to a Bayesian approximation of a deep Gaussian Process. Instead of maintaining enormous distributions over weights, MC Dropout provides computationally feasible, sampling-based Monte Carlo iterations over deterministic weights modulated by Bernoulli distributions. This profound insight allows deterministic sequence models to output calibrated predictive uncertainty.

---

### 9.3 Implementation: training=True at Inference

In Keras, enabling MC Dropout at inference time requires a single change: pass \`training=True\` to every forward pass call.

---

### 9.4 50 Forward Passes — Why This Number?

The number of MC Dropout passes (\`n_passes=50\`) was chosen empirically by monitoring the stability of the mean and std estimates as a function of pass count. 50 passes provide a stable uncertainty estimate with acceptable inference latency.

---

### 9.5 Computing Prediction Mean & Standard Deviation

For each event's input sequence, 50 stochastic forward passes produce 50 predictions (each a vector of 11 scaled feature values) with Dropout active ($p=0.3$). The mean and standard deviation are computed across passes to formulate prediction uncertainty.

Mean prediction (Equation 7):

$$ \hat{y} = \frac{1}{T} \sum_{i=1}^{50} f(\mathbf{x};\, \mathbf{w},\, \mathbf{d}_i) $$

Prediction standard deviation (Equation 8):

$$ \sigma_{pred} = \sqrt{\frac{1}{T-1} \sum_{i=1}^{50} \left(f(\mathbf{x};\, \mathbf{w},\, \mathbf{d}_i) - \hat{y}\right)^2} $$

Where $\mathbf{w}$ represents deterministic weights, and $\mathbf{d}_i$ is the Bernoulli mask applied at the $i$-th inference pass.

---

### 9.6 Why BatchNorm Would Break MC Dropout

BatchNormalization maintains running mean/std statistics of the activations. At test time (training=False), these running statistics are used. When MC Dropout forces training=True, BatchNorm recomputes statistics from each batch, causing the statistics to change.

---

### 9.8 Interpreting Uncertainty Values

Uncertainty and confidence are operationally meaningful.

## 10. Scoring System — Threat & Confidence

### 10.1 The scoring.py Module — Single Source of Truth

\`Scripts/scoring.py\` contains the single authoritative implementation of \`compute_threat_and_confidence()\`. By centralizing this logic, we ensure that the exact same scoring mechanics are used identically across all pipeline stages—from the Step 3B evaluation gate to the Step 4 real-time inference dashboard, and finally in the Step 5 detailed reports. This prevents logic drift and guarantees that a "High Risk" classification means the same thing everywhere.

---

### 10.3 Threat Score: Base Threat from Pc Level

The first component of the threat score is derived from the **predicted Pc** (the model's prediction of the *next* CDM's collision probability), $P_c$ trends, inverse miss distance, and time urgency. Because collision probabilities span many orders of magnitude (from $10^{-10}$ up to $10^{-2}$ or higher), we evaluate the base threat using a smooth scaled continuous function mapping the typical $P_c$ range to a 0–100 scale.

Threat score equation (Equation 9):

$$ \text{Threat} = 50 \cdot \left(1 + \tanh(\alpha \cdot \log_{10} P_c^{pred} + \beta)\right) $$

Where parameters $\alpha$ and $\beta$ define the logarithmic backbone, ensuring that order-of-magnitude jumps in physical risk correspond to intuitive, linear increases in the final 0-100 threat score presented to the operator. This corresponds roughly to:
- **Critical (Pc > 1e-3):** Base score ~80-100.
- **High (Pc 1e-4 to 1e-3):** Base score ~60-80.
- **Elevated (Pc 1e-5 to 1e-4):** Base score ~40-60.
- **Nominal (Pc < 1e-5):** Base score < 40.

---

### 10.4 Threat Score: Trend Modifier (Rising vs Falling Pc)

The model's core novel contribution is using the **predicted next Pc** relative to the **current Pc** to assess trajectory direction. A static snapshot of risk is insufficient because a high Pc that is rapidly falling (due to improving orbital geometry) is operationally less threatening than a moderate Pc that is consistently rising toward TCA.

The trend modifier calculates the $\Delta Pc = Pc_{predicted} - Pc_{latest}$. 
- If $\Delta Pc > 0$ (Risk is predicted to rise): A positive multiplier is applied to the base threat score, aggressively flagging the event for operator intervention.
- If $\Delta Pc < 0$ (Risk is expected to decay naturally): A dampening modifier is applied, lowering the threat score and preventing "false alarm" alert fatigue for self-resolving conjunctions.

---

### 10.10 Quadrant Classification

After computing threat_score (0–100) and confidence (0.10–1.00), each event is categorized using binary operational thresholds of Threat = 50 and Confidence = 0.5, resulting in a four-quadrant classification scheme:

| Quadrant | Threat | Confidence | Description |
|---|---|---|---|
| **ACT NOW** | $\ge 50$ | $\ge 0.5$ | Immediate operational attention required |
| **WATCH CLOSELY** | $\ge 50$ | $< 0.5$ | High threat but low certainty — monitor further |
| **SAFELY IGNORE** | $< 50$ | $\ge 0.5$ | Routine conjunction — safely deprioritize |
| **NOT PRIORITY** | $< 50$ | $< 0.5$ | Low risk, insufficient data for confident assessment |

## 11. The Evaluation Gate — Step 3B

### 11.1 Why an Evaluation Gate Exists

Step 3B (\`step3b_evaluate_proxy_confidence.py\`) is a **mandatory quality gate** between training (step 3) and production inference (step 4).

## 12. Production Inference — The Dashboard

### 12.6 Event Dashboard: Structure & Fields

The primary output is \`inference_outputs/event_dashboard.csv\` — one row per event.

## 13. Visualization & Reporting

### 13.1 Figure 1: Risk Assessment Quadrant Dashboard

A 2D scatter plot of all 2,003 events plotted in the threat-score vs. confidence space. The two quadrant boundaries (threat=50, confidence=0.5) are drawn as dashed lines, dividing the plot into four labeled regions. Points are color-coded by quadrant.

\`\`\`quadrantchart
\`\`\`
`;
