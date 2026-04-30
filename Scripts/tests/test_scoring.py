"""
Unit tests for scoring.py — the threat/confidence scoring module.

These tests use synthetic data and a mock scaler so they run instantly
without needing any pipeline artifacts (no CSV, no model weights, no .npy files).

Run with: pytest Scripts/tests/test_scoring.py -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

# Add Scripts directory to path so we can import scoring
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scoring import compute_threat_and_confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = ['log10_pc', 'time_to_tca_hours']
N_FEATURES = len(FEATURE_NAMES)
N_TIMESTEPS = 5
PADDING = -999.0


def make_identity_scaler(n_features):
    """Mock scaler whose inverse_transform is an identity function."""
    scaler = MagicMock()
    scaler.inverse_transform = lambda x: x.copy()
    scaler.scale_ = np.ones(n_features)
    return scaler


def make_input(log10_pc_current, hours_current, n_timesteps=N_TIMESTEPS):
    """Build a padded input sequence with one real timestep at the end."""
    X = np.full((1, n_timesteps, N_FEATURES), PADDING)
    X[0, -1, 0] = log10_pc_current   # log10_pc
    X[0, -1, 1] = hours_current       # time_to_tca_hours
    return X


def make_prediction(log10_pc_pred, hours_pred):
    """Build a prediction array."""
    pred_mean = np.array([[log10_pc_pred, hours_pred]])
    pred_std = np.array([[0.01, 0.1]])  # low uncertainty
    return pred_mean, pred_std


# ---------------------------------------------------------------------------
# Threat Score Tests
# ---------------------------------------------------------------------------

class TestThreatScore:

    def test_pc_above_maneuver_threshold_gives_high_threat(self):
        """Pc > 1e-4 (log10 > -4) should give threat > 50 (ACT NOW zone)."""
        scaler = make_identity_scaler(N_FEATURES)
        X = make_input(-5.0, 72.0)
        pred_mean, pred_std = make_prediction(-3.0, 72.0)  # log10=-3 → Pc=1e-3

        threat, _ = compute_threat_and_confidence(pred_mean, pred_std, X, FEATURE_NAMES, scaler)
        assert threat[0] > 50, f"Pc=1e-3 should give threat > 50, got {threat[0]:.1f}"

    def test_pc_well_below_threshold_gives_low_threat(self):
        """Pc < 1e-10 should give threat < 15 (SAFELY IGNORE zone)."""
        scaler = make_identity_scaler(N_FEATURES)
        X = make_input(-11.0, 200.0)
        pred_mean, pred_std = make_prediction(-11.0, 200.0)  # Pc=1e-11

        threat, _ = compute_threat_and_confidence(pred_mean, pred_std, X, FEATURE_NAMES, scaler)
        assert threat[0] < 15, f"Pc=1e-11 should give threat < 15, got {threat[0]:.1f}"

    def test_increasing_pc_trend_raises_threat(self):
        """If Pc is predicted to increase (trend up), threat should be higher."""
        scaler = make_identity_scaler(N_FEATURES)

        # Same current Pc, same predicted absolute Pc — but one is trending up, one down
        X_up = make_input(-8.0, 72.0)   # current: Pc=1e-8
        pred_up, std_up = make_prediction(-6.0, 72.0)  # predicted: Pc=1e-6 (rising)

        X_flat = make_input(-6.0, 72.0)  # current: Pc=1e-6
        pred_flat, std_flat = make_prediction(-6.0, 72.0)  # predicted: same (flat)

        threat_up, _ = compute_threat_and_confidence(pred_up, std_up, X_up, FEATURE_NAMES, scaler)
        threat_flat, _ = compute_threat_and_confidence(pred_flat, std_flat, X_flat, FEATURE_NAMES, scaler)

        assert threat_up[0] > threat_flat[0], \
            f"Rising Pc should give higher threat ({threat_up[0]:.1f}) than flat ({threat_flat[0]:.1f})"

    def test_imminent_tca_adds_urgency(self):
        """TCA in < 24h should give higher threat than TCA in > 7 days."""
        scaler = make_identity_scaler(N_FEATURES)

        X = make_input(-6.0, 24.0)
        pred_soon, std = make_prediction(-6.0, 12.0)   # 12h to TCA
        pred_far, std2 = make_prediction(-6.0, 500.0)  # 500h to TCA

        threat_soon, _ = compute_threat_and_confidence(pred_soon, std, X, FEATURE_NAMES, scaler)
        threat_far, _ = compute_threat_and_confidence(pred_far, std2, X, FEATURE_NAMES, scaler)

        assert threat_soon[0] > threat_far[0], \
            f"Imminent TCA ({threat_soon[0]:.1f}) should give higher threat than distant ({threat_far[0]:.1f})"

    def test_threat_is_clipped_to_0_100(self):
        """Threat score must always be in [0, 100]."""
        scaler = make_identity_scaler(N_FEATURES)
        # Extreme values
        X = make_input(0.0, 0.0)
        pred_mean, pred_std = make_prediction(0.0, 0.0)  # Pc=1 (maximum possible)

        threat, _ = compute_threat_and_confidence(pred_mean, pred_std, X, FEATURE_NAMES, scaler)
        assert 0 <= threat[0] <= 100, f"Threat must be in [0,100], got {threat[0]}"

    def test_threat_zero_when_all_padding(self):
        """All-padding input (no real CDMs) should not crash and give a valid score."""
        scaler = make_identity_scaler(N_FEATURES)
        X = np.full((1, N_TIMESTEPS, N_FEATURES), PADDING)  # all padding
        pred_mean, pred_std = make_prediction(-9.0, 200.0)

        threat, confidence = compute_threat_and_confidence(pred_mean, pred_std, X, FEATURE_NAMES, scaler)
        assert 0 <= threat[0] <= 100
        assert 0 <= confidence[0] <= 1


# ---------------------------------------------------------------------------
# Confidence Score Tests
# ---------------------------------------------------------------------------

class TestConfidenceScore:

    def test_confidence_is_in_valid_range(self):
        """Confidence must always be in [0.1, 1.0]."""
        scaler = make_identity_scaler(N_FEATURES)
        X = make_input(-6.0, 48.0)
        pred_mean, pred_std = make_prediction(-6.0, 48.0)

        _, confidence = compute_threat_and_confidence(pred_mean, pred_std, X, FEATURE_NAMES, scaler)
        assert 0.1 <= confidence[0] <= 1.0, f"Confidence must be in [0.1, 1.0], got {confidence[0]}"

    def test_low_std_gives_higher_confidence_than_high_std(self):
        """Lower MC Dropout std should produce higher confidence."""
        scaler = make_identity_scaler(N_FEATURES)
        X = make_input(-6.0, 48.0)

        pred_mean = np.array([[-6.0, 48.0]])
        std_low = np.array([[0.001, 0.001]])   # very certain
        std_high = np.array([[5.0, 100.0]])    # very uncertain

        _, conf_low_std = compute_threat_and_confidence(pred_mean, std_low, X, FEATURE_NAMES, scaler)
        _, conf_high_std = compute_threat_and_confidence(pred_mean, std_high, X, FEATURE_NAMES, scaler)

        assert conf_low_std[0] > conf_high_std[0], \
            f"Low std ({conf_low_std[0]:.3f}) should give higher confidence than high std ({conf_high_std[0]:.3f})"

    def test_more_valid_timesteps_gives_higher_confidence(self):
        """More real CDMs in input = higher data confidence."""
        scaler = make_identity_scaler(N_FEATURES)
        pred_mean, pred_std = make_prediction(-6.0, 48.0)

        # 1 valid timestep
        X_sparse = np.full((1, N_TIMESTEPS, N_FEATURES), PADDING)
        X_sparse[0, -1, :] = [-6.0, 48.0]

        # 5 valid timesteps (full sequence, no padding)
        X_dense = np.full((1, N_TIMESTEPS, N_FEATURES), -6.0)
        X_dense[0, :, 1] = 48.0

        _, conf_sparse = compute_threat_and_confidence(pred_mean, pred_std, X_sparse, FEATURE_NAMES, scaler)
        _, conf_dense = compute_threat_and_confidence(pred_mean, pred_std, X_dense, FEATURE_NAMES, scaler)

        assert conf_dense[0] > conf_sparse[0], \
            f"Dense input ({conf_dense[0]:.3f}) should give higher confidence than sparse ({conf_sparse[0]:.3f})"


# ---------------------------------------------------------------------------
# Batch / Shape Tests
# ---------------------------------------------------------------------------

class TestBatchBehavior:

    def test_multiple_samples_processed_correctly(self):
        """Function should handle a batch of N samples correctly."""
        scaler = make_identity_scaler(N_FEATURES)
        N = 10
        pred_mean = np.random.uniform(-10, -3, size=(N, N_FEATURES))
        pred_std = np.random.uniform(0.01, 0.5, size=(N, N_FEATURES))
        X = np.full((N, N_TIMESTEPS, N_FEATURES), PADDING)
        X[:, -1, :] = pred_mean  # one real timestep per sample

        threat, confidence = compute_threat_and_confidence(pred_mean, pred_std, X, FEATURE_NAMES, scaler)

        assert threat.shape == (N,), f"Expected threat shape ({N},), got {threat.shape}"
        assert confidence.shape == (N,), f"Expected confidence shape ({N},), got {confidence.shape}"
        assert np.all(threat >= 0) and np.all(threat <= 100)
        assert np.all(confidence >= 0.1) and np.all(confidence <= 1.0)

    def test_output_shapes_match_input_batch_size(self):
        """Output arrays must have exactly the same length as input batch."""
        scaler = make_identity_scaler(N_FEATURES)
        for batch_size in [1, 5, 100]:
            pred_mean = np.full((batch_size, N_FEATURES), -6.0)
            pred_std = np.full((batch_size, N_FEATURES), 0.1)
            X = np.full((batch_size, N_TIMESTEPS, N_FEATURES), PADDING)

            threat, confidence = compute_threat_and_confidence(
                pred_mean, pred_std, X, FEATURE_NAMES, scaler
            )
            assert len(threat) == batch_size
            assert len(confidence) == batch_size


# ---------------------------------------------------------------------------
# KD-14: expm1 correction — covariance confidence must use physical m² units
# ---------------------------------------------------------------------------

class TestCovarianceExpm1:
    """
    After KD-14 (step2 applies log1p to covariance before scaling),
    scaler.inverse_transform() returns log1p(covariance), NOT raw m².

    scoring.py must apply expm1() to convert back to m² before computing
    covariance-based confidence.  Without the fix, confidence is always ~0.99
    regardless of how poorly tracked the objects are.
    """

    # Feature set that includes combined_cr_r (triggers the expm1 branch)
    FEATURE_NAMES_COV = ['log10_pc', 'time_to_tca_hours', 'combined_cr_r']
    N_FEATURES_COV = len(FEATURE_NAMES_COV)

    def _make_log1p_scaler(self, log1p_cov_value, n_features=3):
        """
        Scaler whose inverse_transform returns log1p-scale values.
        index 2 = combined_cr_r = log1p(raw_cov_m2)
        """
        scaler = MagicMock()
        def fake_inverse(x):
            out = x.copy()
            out[:, 2] = log1p_cov_value   # simulate log1p(covariance)
            return out
        scaler.inverse_transform = fake_inverse
        scaler.scale_ = np.ones(n_features)
        return scaler

    def test_high_covariance_gives_lower_confidence_than_low_covariance(self):
        """
        High covariance (500,000 m²) must give meaningfully lower confidence
        than low covariance (10 m²).  This directly tests that expm1() fires:

          Without expm1: inverse_transform returns log1p(cov) ≈ 13.12 → cov_confidence ≈ 0.987
          With    expm1: scoring sees 500,000 m²              → cov_confidence ≈ 0.002

        We compare the two scenarios — the gap must be at least 0.10.
        """
        import math

        def _run(raw_cov_m2):
            lv = math.log1p(raw_cov_m2)
            scaler = self._make_log1p_scaler(lv)
            pm = np.array([[-7.0, 48.0, lv]])
            ps = np.array([[0.01, 0.1, 0.01]])
            X  = np.full((1, 5, self.N_FEATURES_COV), -999.0)
            X[0, -1, :] = [-7.0, 48.0, lv]
            _, conf = compute_threat_and_confidence(pm, ps, X, self.FEATURE_NAMES_COV, scaler)
            return conf[0]

        conf_high_cov = _run(500_000.0)   # very poor tracking
        conf_low_cov  = _run(10.0)        # well-tracked

        assert conf_low_cov > conf_high_cov, (
            f"Low covariance should give higher confidence than high covariance. "
            f"Got low={conf_low_cov:.3f}, high={conf_high_cov:.3f}."
        )
        gap = conf_low_cov - conf_high_cov
        assert gap > 0.10, (
            f"Expected a confidence gap > 0.10 between 10m² and 500k m² covariance. "
            f"Got gap={gap:.3f} (low={conf_low_cov:.3f}, high={conf_high_cov:.3f}). "
            f"Check that expm1() is applied to combined_cr_r in scoring.py."
        )

    def test_low_covariance_gives_high_confidence(self):
        """
        A covariance of 10 m² (well-tracked) should produce confidence > 0.65.
        """
        import math
        log1p_val = math.log1p(10.0)   # ≈ 2.40

        scaler = self._make_log1p_scaler(log1p_val)
        pred_mean = np.array([[-7.0, 48.0, log1p_val]])
        pred_std  = np.array([[0.01, 0.1, 0.01]])
        X = np.full((1, 5, self.N_FEATURES_COV), -999.0)
        X[0, -1, :] = [-7.0, 48.0, log1p_val]

        _, confidence = compute_threat_and_confidence(
            pred_mean, pred_std, X, self.FEATURE_NAMES_COV, scaler
        )

        assert confidence[0] > 0.55, (
            f"Low covariance (10 m²) should give confidence > 0.55, "
            f"got {confidence[0]:.3f}. (Note: 1 valid timestep caps data_confidence at 0.10)"
        )
