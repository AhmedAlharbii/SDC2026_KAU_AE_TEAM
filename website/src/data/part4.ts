export const part4 = `
## 14. Software Architecture & Pipeline Design

### 14.1 Pipeline Philosophy: Linear, Artifact-Driven

The DebriSolver pipeline follows a strictly **linear, artifact-driven** design. Each step produces one or more output artifacts (files), and each step consumes only the artifacts produced by previous steps. There is no shared in-memory state between steps.

---

### 14.2 The 7-Step Pipeline (1 → 2 → 3 → 3B → 4 → 5 → 5B)

\`\`\`pipelinediagram
\`\`\`

The gate (step 3B) is the only non-linear element: it must complete successfully before step 4 is permitted.

---

### 14.3 run_pipeline.py: The Orchestrator

\`run_pipeline.py\` is the top-level orchestrator that executes all steps in sequence.

---

### 14.4 Step Resumption & Partial Runs

The \`--from-step\` and \`--to-step\` flags allow partial pipeline runs.

---

### 14.5 config.yaml: Single Source of Truth

All tunable parameters are centralized in \`Scripts/config.yaml\`. This file is read at the start of every pipeline step that needs parameters.

---

## 15. Testing Strategy & Test Suite

### 15.1 Testing Philosophy: No Artifacts Required

The DebriSolver test suite is designed around one principle: **unit tests must not require any pipeline artifacts to run.** This means tests should not need \`X_train.npy\`, \`best_model.keras\`, or \`parsed_cdm_data.csv\` to exist. Tests that require artifacts are integration tests, handled separately by \`smoke_tests.py\`.

---

## 16. Problems Encountered & How We Solved Them

### 16.1 Problem 1: Validation Loss Explosion (84.9 → fixed)
- **What happened:** val_loss stuck at ~85 while train_loss converged to ~1.0
- **Root cause:** COLLISION_PROBABILITY (raw linear scale) had catastrophic outliers (max=117 std devs). StandardScaler couldn't normalize it.
- **Fix:** Excluded raw COLLISION_PROBABILITY entirely. Used log10_pc instead. Val loss dropped from 84.9 → 0.628 (~99.3% reduction).

### 16.2 Problem 2: Covariance Features Causing Training Instability
- **What happened:** Covariance values span 0 to ~20 billion m². StandardScaler's mean/std were astronomically large.
- **Fix:** Applied log1p transform to all covariance features before scaling. Compressed range from [0, 2×10¹⁰] to [0, ~22]. Scaler then worked correctly.

### 16.3 Problem 3: Padding Value of Zero Conflicting with Scaled Data
- **What happened:** Initial implementation used 0.0 as the padding sentinel. After StandardScaler, real CDM values can be zero (mean-centered). Masking layer couldn't distinguish padding from real data.
- **Fix:** Changed padding sentinel to -999.0, a value physically impossible after StandardScaler normalization.

### 16.4 Problem 4: BatchNorm Breaking MC Dropout (KD-12)
- **What happened:** Using BatchNormalization with training=True for MC Dropout caused batch statistics to be recomputed each forward pass.
- **Fix:** Replaced all BatchNormalization with LayerNormalization.

## 17. Results & Performance

### 17.1 Model Training Performance
This section presents the performance evaluation and operational analysis of the bidirectional GRU model. The test set consisted of 9,637 samples from 6,411 unique conjunction events, enabling a robust assessment of the model's generalization to previously unseen satellite encounters. 

The bidirectional GRU model underwent training for 48 epochs with early stopping, selecting epoch 28 as optimal based on validation loss. Training was completed in 24.8 minutes using standard CPU hardware. The training convergence curves demonstrated smooth learning with no evidence of instability or divergence. The weighted mean squared error (MSE) loss decreased from approximately 88 to 73 on the validation set, while the overall mean absolute error (MAE) improved consistently, indicating minimal overfitting. Notably, the collision probability prediction error decreased from 0.54 to 0.43 on the training set and from 0.44 to 0.41 on the validation set, demonstrating effective learning of the most operationally significant feature.

| Metric | Value |
|--------|-------|
| Overall MAE (all normalized features) | 0.464 |
| Collision probability MAE | **0.403** |
| log₁₀($P_c$) $R^2$ | **0.649** |
| Validation $P_c$ MAE | 0.41 |
| Test $P_c$ MAE | 0.404 |

### 17.2 Self-Supervised Learning Validation
A primary challenge in this study was the absence of ground-truth collision outcomes for supervised training, given the rarity of actual satellite collisions. The framework required validating that the self-supervised approach identified meaningful risk patterns rather than arbitrary mappings. 

Analysis illustrated a strong positive correlation between threat scores and actual final Pc values (with a linear regression slope of 2.86), confirming that the model independently discovered risk relationships from CDM sequence patterns without explicit labeling. The self-supervised framework was particularly effective at managing the extensive dynamic range of collision probabilities, spanning eight orders of magnitude (from roughly 10⁻¹⁰ to 10⁻²). 

### 17.3 Uncertainty Quantification and Confidence Calibration
Monte Carlo Dropout, implemented through 50 stochastic forward passes during inference with dropout layers active, provided built-in uncertainty quantification. Analysis of the test set revealed a mean uncertainty of 0.127 (ranging from 0.037 to 9.81), indicating that the model appropriately expresses varying degrees of confidence across different conjunction scenarios.

Confidence levels showed a clear positive relationship with CDM count. Events observed through 15-40 CDMs achieved a mean confidence exceeding 0.7, while sparsely observed events (2-5 CDMs) typically registered confidence below 0.4. This emergent behavior demonstrates that the model naturally recognizes when it lacks sufficient information. 

### 17.4 Model Architecture Selection and Advantages
The bidirectional GRU architecture was chosen for its ability to capture the full temporal context in both directions while maintaining computational efficiency. GRU cells were preferred over LSTM units due to their simplified gating mechanism, reducing parameter count and training time while preserving sequence modeling capabilities. The model, with just 244,171 parameters, balances complexity and speed—allowing calculation within milliseconds for real-time traffic screening.

### 17.5 Operational Risk Classification
The four-quadrant risk classification framework, which combined threat scores with confidence levels, produced operationally realistic event distributions across the test set of 2,003 unique events, allowing a 96% reduction in urgent alerts requiring immediate analyst review:

| Quadrant | Events | Percentage | Description |
|---|---|---|---|
| **SAFELY IGNORE** | 1,132 | 56.5% | Routine conjunctions deprioritized without analyst review |
| **NOT PRIORITY** | 641 | 32.0% | Low-risk with insufficient observational data |
| **WATCH CLOSELY** | 149 | 7.4% | High-threat cases requiring further observation |
| **ACT NOW** | 81 | 4.0% | Immediate operational attention required |

The **ACT NOW** quadrant prominently exhibited high confidence (>0.5) across the entire threat spectrum, proving the model correctly identifies and distinguishes confident and safe trajectories from those with operational threat.

### 17.6 Top 20 High-Priority Events Table

| Event ID | Threat | Confidence | Final $P_c$ | TCA | CDMs | Object 1 | Object 2 |
|---|---|---|---|---|---|---|---|
| 49402_13771 | 69.9 | 0.52 | 2.08E-06 | 4/30/2024 14:08 | 5 | KOSEN-1 | SL-3 R/B |
| 32162_43643 | 57.3 | 0.51 | 1.99E-06 | 5/27/2024 14:40 | 2 | FENGYUN 1C DEB | YAOGAN-32 B |
| 25865_43924 | 57.2 | 0.53 | 1.46E-08 | 2/12/2024 0:19 | 2 | SL-16 DEB | IRIDIUM 168 |
| 58598_58597 | 55.8 | 0.52 | 2.40E-06 | 1/4/2024 10:55 | 2 | STARLINK-31079 | STARLINK-31087 |
| 54073_53841 | 55.8 | 0.82 | 2.04E-07 | 6/10/2024 4:09 | 13 | STARLINK-5146 | STARLINK-4750 |
| 53918_51891 | 55.4 | 0.85 | 1.57E-06 | 2/3/2024 5:19 | 21 | STARLINK-5012 | STARLINK-3549 |
| 36861_40399 | 55.4 | 0.55 | 3.49E-07 | 4/2/2024 8:07 | 2 | METEOR 2-8 DEB | DMSP 5D-2 F13 DEB |
| 36878_36270 | 55.2 | 0.51 | 7.24E-06 | 4/27/2024 21:13 | 4 | METEOR 2-7 DEB | FENGYUN 1C DEB |
| 17642_41508 | 54.3 | 0.50 | 4.41E-05 | 4/28/2024 22:56 | 4 | COSMOS 1275 DEB | NOAA 16 DEB |
| 42339_39657 | 54.3 | 0.50 | 3.83E-05 | 4/1/2024 9:18 | 3 | NOAA 16 DEB | COSMOS 1867 COOLANT |
| 52000_53544 | 54.2 | 0.85 | 6.79E-06 | 3/6/2024 1:27 | 16 | STARLINK-3653 | STARLINK-4303 |
| 45550_48105 | 54.1 | 0.61 | 4.05E-07 | 5/22/2024 6:06 | 5 | STARLINK-1390 | STARLINK-2440 |
| 52691_53998 | 53.9 | 0.66 | 4.76E-07 | 2/3/2024 22:42 | 6 | STARLINK-3945 | STARLINK-5128 |
| 30197_21420 | 53.8 | 0.52 | 1.00E-10 | 4/13/2024 15:41 | 2 | FENGYUN 1C DEB | SL-8 DEB |
| 50176_53844 | 53.7 | 0.83 | 2.10E-05 | 3/4/2024 11:24 | 35 | STARLINK-3293 | STARLINK-4745 |
| 54186_52850 | 53.5 | 0.79 | 3.88E-06 | 1/15/2024 1:15 | 9 | STARLINK-5194 | STARLINK-4187 |
| 53912_51883 | 53.5 | 0.78 | 8.68E-06 | 2/16/2024 16:12 | 9 | STARLINK-5059 | STARLINK-3576 |
| 46753_45228 | 53.5 | 0.80 | 5.71E-06 | 2/11/2024 20:45 | 23 | STARLINK-1922 | STARLINK-1222 |
| 47398_45409 | 53.4 | 0.52 | 4.45E-07 | 2/11/2024 20:31 | 2 | STARLINK-2120 | STARLINK-1282 |
| 46129_46791 | 53.4 | 0.75 | 2.72E-06 | 3/4/2024 5:11 | 19 | STARLINK-1623 | STARLINK-1933 |

## 18. Libraries, Frameworks & Tools
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn
- PyYAML, pytest
- Matplotlib, Chart.js

## 19. Lessons Learned & Future Work

### 19.1 What We Would Do Differently
**Start with data profiling before modeling.** The most expensive mistakes in this project (val_loss = 84.9, covariance instability) were all data-related. Exploring and handling anomalies earlier saves tremendous architectural debugging time.

### 19.2 Limitations and Future Directions
Several limitations of the current approach are acknowledged:
1. **Absence of Confirmed Collision Labels:** The extreme rarity of satellite collisions prevents direct validation of threat classifications against ground truth events beyond prediction metrics. 
2. **Object Characteristics Excluded:** The model treats all spaces objects equally rather than factoring in active maneuverability, mass, cross-sectional area, or spacecraft type.
3. **External Influences:** The current framework implicitly relies on observed sequence features to capture perturbations (atmospheric density variations, solar activity, gravitional effects) without active ingestion of external operational datasets.

**Future Strategies:**
- Integrate object metadata (maneuverability, cross-sectional area) directly into the feature space.
- Couple trajectory mapping with physical environmental space weather indices and atmospheric density forecasts.
- Construct ensemble architectures combining RNN variants and Transformers.

## 20. Team & Acknowledgments

### 20.1 Team Members & Roles

The DebriSolver project was developed by the **SDC2026 KAU AE Team** from King Abdulaziz University, Aerospace Engineering Department, competing in the Saudi Space Data Challenge 2026 (Space Debris Conference, Riyadh, January 26–27, 2026).

| Name | Role | LinkedIn |
|------|------|----------|
| **Ahmad Alharbi** | Team Lead & Lead Developer | [ahmed-alharbi-973b63246](https://www.linkedin.com/in/ahmed-alharbi-973b63246/) |
| **Abdulelah Mojelad** | AI Research & Development | [abdulellah-mojalled](https://www.linkedin.com/in/abdulellah-mojalled/) |
| **Hamzah Alharbi** | Research & Development | [hamzah-alharbi-00b18133a](https://www.linkedin.com/in/hamzah-alharbi-00b18133a/) |
| **Khalid Alsadoon** | Research & Development | [khalid-alsadoon-a95802242](https://www.linkedin.com/in/khalid-alsadoon-a95802242/) |
| **Mohamedhakim Hassan** | Research & Development | [mohamed-hassan-aero](https://www.linkedin.com/in/mohamed-hassan-aero/) |

## 21. References & Citation

### 21.1 Citing This Work

The correct citation for the research project:

\`\`\`bibtex
@inproceedings{alharbi2026conjunction,
  title        = {Learning Conjunction Dynamics: A Self-Supervised Approach
                  to Satellite Collision Risk Assessment},
  author       = {Alharbi, Ahmad and Mojelad, Abdulelah and Alharbi, Hamzah
                  and Alsadoon, Khalid and Hassan, Mohamedhakim},
  booktitle    = {Space Debris Conference 2026 -- DebriSolver Competition},
  address      = {Riyadh, Saudi Arabia},
  year         = {2026},
  organization = {Saudi Space Agency (SSA)}
}
\`\`\`

### 21.2 Key References

The following are the **exact 10 references cited in the submitted paper**, as they appear in the final submission PDF:

---

**[1]** "ESA Space Environment Report 2024," ESA.
Available: https://www.esa.int/Space_Safety/Space_Debris/ESA_Space_Environment_Report_2024
*(Accessed: Nov. 10, 2025)*
> *Cited in §1 (Introduction) — motivation for LEO congestion and mega-constellation risks.*

---

**[2]** L. Sanchez, M. Vasile, E. Minisci (2020). "On the Use of Machine Learning and Evidence Theory to Improve Collision Risk Management." Paper presented at the *2nd IAA International Conference in Space Situational Awareness*, Washington, D.C., USA, 14–16 January 2020.
*(Accessed: Oct. 25, 2025)*
> *Cited in §1 & §1.1 — limitations of fixed-threshold approaches and false-positive cost.*

---

**[3]** A. K. Mashiku, L. K. Newman, and D. E. Highsmith, "NASA Conjunction Assessment Risk Analysis (CARA) Compendium for Artificial Intelligence and Machine Learning for Satellite Collision Avoidance," in *Proc. 26th AMOS Advanced Maui Optical and Space Surveillance Technologies Conference*, Wailea, HI, USA, 2025.
Available: https://ntrs.nasa.gov/api/citations/20250008251/downloads/AMOS_2025_AIML_Paper_UpdatedContractorAddress.pdf
*(Accessed: Dec. 12, 2025)*
> *Cited in §2.3 (Methodology) — self-supervised sequence learning task definition.*

---

**[4]** Y. Qiao, H.-M. Xu, W.-J. Zhou, B. Peng, B. Hu, and X. Guo, "A BiGRU joint optimized attention network for recognition of drilling conditions," *Petroleum Science*, vol. 20, no. 6, pp. 3624–3637, 2023.
doi: 10.1016/j.petsci.2023.05.021
*(Accessed: Nov. 3, 2025)*
> *Cited for BiGRU architecture justification.*

---

**[5]** D. Xu et al., "A survey on multi-output learning," *arXiv.org*, arXiv:1901.00248.
Available: https://arxiv.org/abs/1901.00248
*(Accessed: Dec. 1, 2025)*
> *Cited for the multi-output regression formulation (predicting all 11 CDM features simultaneously).*

---

**[6]** J. Terven, D.-M. Córdova-Esparza, J.-A. Romero-González, A. Ramírez-Pedraza, and E. A. Chávez-Urbiola, "A comprehensive survey of loss functions and metrics in Deep Learning," *Artificial Intelligence Review*, SpringerLink, 2025.
Available: https://link.springer.com/article/10.1007/s10462-025-11198-7
*(Accessed: Nov. 5, 2025)*
> *Cited for weighted MSE loss design and metric selection.*

---

**[7]** Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in Deep Learning," *arXiv.org*, arXiv:1506.02142.
Available: https://arxiv.org/abs/1506.02142
*(Accessed: Nov. 29, 2025)*
> *Primary theoretical foundation for Monte Carlo Dropout (Section 9 of this document).*

---

**[8]** M. Hasan, A. Khosravi, I. Hossain, A. Rahman, and S. Nahavandi, "Controlled dropout for uncertainty estimation," *arXiv.org*, arXiv:2205.03109.
Available: https://arxiv.org/abs/2205.03109
*(Accessed: Nov. 25, 2025)*
> *Cited for controlled dropout uncertainty estimation methodology.*

---

**[9]** A. Pim and T. Pryer, "Surrogate modelling of proton dose with Monte Carlo dropout uncertainty quantification," *arXiv.org*, arXiv:2509.18155.
Available: https://arxiv.org/abs/2509.18155
*(Accessed: Dec. 2, 2025)*
> *Cited as applied MC Dropout case study in a safety-critical domain.*

---

**[10]** X. Cao, Y. Xu, and X. Yang, "Customer lifetime value prediction with uncertainty estimation using Monte Carlo Dropout," *arXiv.org*, arXiv:2411.15944.
Available: https://arxiv.org/abs/2411.15944
*(Accessed: Nov. 28, 2025)*
> *Cited as applied MC Dropout uncertainty estimation example.*

---

### 21.3 Technical Standards (CCSDS CDM Standard)

- **CCSDS 508.0-B-1** (2013): *Conjunction Data Message, Recommended Standard.* Consultative Committee for Space Data Systems (CCSDS), Blue Book. Washington, D.C.: CCSDS Secretariat.
  - Defines the official KVN (Key-Value Notation) and XML formats for Conjunction Data Messages
  - Specifies all mandatory and optional fields, their units, and their physical meanings
  - The ALDORIA dataset follows this standard; our KVN parser implements the field definitions from this document

- **CCSDS 502.0-B-2** (2019): *Orbit Data Messages, Recommended Standard.* CCSDS, Blue Book.
  - Defines the OPM, OEM, and OMM orbit data formats referenced by CDM object blocks
  - Relevant for understanding the coordinate frames (RTN, ECI) used in CDM relative position fields
`;
