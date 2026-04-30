"""Shared threat/confidence scoring logic for offline evaluation and production inference.

This is the SINGLE SOURCE OF TRUTH for scoring. Both step3b and step4 import from here.
"""

import numpy as np


def compute_threat_and_confidence(pred_mean, pred_std, X_input, feature_names, scaler):
    """
    Compute threat score and confidence from model predictions.

    Predictions are inverse-transformed to physical units before scoring.
    Threat thresholds are based on conjunction assessment practice:
      - Pc > 1e-4  (log10 > -4):  maneuver threshold (ESA/NASA)
      - Pc > 1e-6  (log10 > -6):  elevated risk
      - Pc > 1e-8  (log10 > -8):  low but notable

    Confidence (0-1):
    - Based on model's prediction uncertainty (MC Dropout std)
    - Also considers data quantity and covariance quality
    - Does NOT use ground-truth future labels (deployment-safe)
    """

    n_samples = pred_mean.shape[0]
    threat_scores = np.zeros(n_samples)
    confidence_levels = np.zeros(n_samples)

    # Inverse-transform predictions and std to physical units
    pred_physical = scaler.inverse_transform(pred_mean)
    # For std: scale back using scaler's scale_ (std of training features)
    pred_std_physical = pred_std * scaler.scale_

    # Undo log1p for covariance features (KD-14: they were log1p-transformed
    # before scaling in step2). scaler.inverse_transform returns log1p(cov);
    # expm1 converts back to raw m² so physical thresholds apply correctly.
    _LOG_FEATS = {'combined_cr_r', 'combined_ct_t', 'combined_cn_n'}
    for _fi, _fn in enumerate(feature_names):
        if _fn in _LOG_FEATS:
            pred_physical[:, _fi] = np.expm1(np.clip(pred_physical[:, _fi], 0, None))

    # Find feature indices
    pc_idx = None
    log10_pc_idx = None
    miss_dist_idx = None
    time_to_tca_idx = None
    cr_r_idx = None

    for i, feat in enumerate(feature_names):
        if feat == 'COLLISION_PROBABILITY':
            pc_idx = i
        elif feat == 'log10_pc':
            log10_pc_idx = i
        elif feat == 'MISS_DISTANCE':
            miss_dist_idx = i
        elif feat == 'time_to_tca_hours':
            time_to_tca_idx = i
        elif feat == 'combined_cr_r':
            cr_r_idx = i

    # Use log10_pc if available, otherwise COLLISION_PROBABILITY
    main_pc_idx = log10_pc_idx if log10_pc_idx is not None else pc_idx

    # Also inverse-transform the input sequences for current Pc comparison
    # Reshape for scaler: (samples * timesteps, features) then back
    X_flat = X_input.reshape(-1, X_input.shape[2])
    X_physical_flat = scaler.inverse_transform(X_flat)
    X_physical = X_physical_flat.reshape(X_input.shape)

    for i in range(n_samples):
        # =====================================================================
        # THREAT SCORE (using physical units)
        # =====================================================================

        # 1. Get current Pc from input sequence (last non-padding timestep)
        input_seq = X_input[i]  # scaled, for padding detection
        input_phys = X_physical[i]  # physical units, for values
        non_padding_mask = ~np.all(input_seq == -999.0, axis=1)

        if np.any(non_padding_mask):
            last_valid_idx = np.where(non_padding_mask)[0][-1]
            current_pc = input_phys[last_valid_idx, main_pc_idx]
        else:
            current_pc = -10.0 if log10_pc_idx is not None else 0.0

        # 2. Get predicted Pc in physical units
        predicted_pc = pred_physical[i, main_pc_idx]

        # 3. Base threat from absolute Pc level (physics-based thresholds)
        if log10_pc_idx is not None:
            # predicted_pc is log10(Pc) in real units
            if predicted_pc > -4:        # Pc > 1e-4: maneuver threshold
                base_threat = 70 + (predicted_pc + 4) * 10  # 70-80+
            elif predicted_pc > -6:      # Pc > 1e-6: elevated risk
                base_threat = 40 + (predicted_pc + 6) * 15  # 40-70
            elif predicted_pc > -8:      # Pc > 1e-8: low but notable
                base_threat = 15 + (predicted_pc + 8) * 12.5  # 15-40
            else:                        # Pc < 1e-8: negligible
                base_threat = max(0, 15 + (predicted_pc + 8) * 2)
            base_threat = np.clip(base_threat, 0, 70)
        else:
            # Raw COLLISION_PROBABILITY (0 to 1 range)
            base_threat = np.clip(-np.log10(max(predicted_pc, 1e-15)) * (-7), 0, 70)

        # 4. Trend modifier: is Pc increasing or decreasing?
        pc_change = predicted_pc - current_pc
        if log10_pc_idx is not None:
            # 1 order of magnitude change in log10 space is significant
            trend_modifier = np.clip(pc_change * 10, -15, 15)
        else:
            trend_modifier = np.clip(pc_change * 30, -15, 15)

        # 5. Time urgency (physical hours to TCA)
        if time_to_tca_idx is not None:
            predicted_hours = pred_physical[i, time_to_tca_idx]
            # Less than 24h = very urgent, less than 72h = urgent
            if predicted_hours < 24:
                urgency = 15
            elif predicted_hours < 72:
                urgency = 10
            elif predicted_hours < 168:  # 1 week
                urgency = 5
            else:
                urgency = 0
        else:
            urgency = 0

        # Combine threat components
        threat = base_threat + trend_modifier + urgency
        threat_scores[i] = np.clip(threat, 0, 100)

        # =====================================================================
        # CONFIDENCE (using physical units for covariance; scaled for uncertainty)
        # =====================================================================

        # 1. Model uncertainty (from MC Dropout std in SCALED space)
        # ---------------------------------------------------------------
        # IMPORTANT: Do NOT compute relative uncertainty in physical space.
        # After expm1(), covariance features are millions of m², which overwhelms
        # avg_pred_physical → relative_uncertainty ≈ 0 → confidence ≈ 1.0 always.
        # pred_std is already in StandardScaler units (zero-mean, unit-variance).
        # avg std of 0.05 = tight agreement = high confidence.
        # avg std of 0.60 = wide spread     = low confidence.
        avg_std_scaled = np.mean(pred_std[i])   # scaled units
        uncertainty_confidence = 1 / (1 + avg_std_scaled * 5)

        # 2. Data quantity confidence
        # More input timesteps = more confident (max at 10 CDMs)
        n_valid_timesteps = np.sum(non_padding_mask)
        data_confidence = min(1.0, n_valid_timesteps / 10)

        # 3. Covariance-based confidence (physical m² units after expm1)
        # < 1,000 m² = well-tracked (high confidence)
        # > 100,000 m² = poorly tracked (low confidence)
        if cr_r_idx is not None:
            predicted_cov_physical = pred_physical[i, cr_r_idx]
            cov_confidence = 1 / (1 + max(0, predicted_cov_physical) / 1000)
        else:
            cov_confidence = 0.5

        # Combine confidence components
        # Weights rebalanced so that low-data / high-uncertainty events can fall
        # below 0.5 → enables all 4 quadrants (WATCH CLOSELY, NOT PRIORITY).
        confidence = (
            uncertainty_confidence * 0.40 +  # MC Dropout spread (scaled space)
            data_confidence        * 0.35 +  # Data quantity (key real-world signal)
            cov_confidence         * 0.25    # Covariance quality (tracking accuracy)
        )
        confidence_levels[i] = np.clip(confidence, 0.05, 1.0)

    return threat_scores, confidence_levels
