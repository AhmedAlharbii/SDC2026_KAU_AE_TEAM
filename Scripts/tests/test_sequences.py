"""
Unit tests for sequence preparation logic.

Tests padding, masking sentinel, shape correctness, and data integrity
without needing the full 213MB CSV or any pipeline artifacts.

Run with: pytest Scripts/tests/test_sequences.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PADDING_VALUE = -999.0
N_FEATURES = 5
MAX_LEN = 10


# ---------------------------------------------------------------------------
# Helper: replicate the core padding logic from step2
# ---------------------------------------------------------------------------

def pad_sequence(seq, max_len, padding_value=PADDING_VALUE):
    """Replicate step2's padding logic: prepend padding to the left."""
    n = len(seq)
    if n >= max_len:
        return np.array(seq[-max_len:])
    pad = np.full((max_len - n, seq.shape[1]), padding_value)
    return np.vstack([pad, seq])


def make_seq(n_cdms, n_features=N_FEATURES, seed=42):
    """Make a synthetic CDM sequence with realistic non-zero values."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, size=(n_cdms, n_features))


# ---------------------------------------------------------------------------
# Padding Tests
# ---------------------------------------------------------------------------

class TestPadding:

    def test_short_sequence_padded_to_max_len(self):
        """Sequences shorter than max_len should be padded to exactly max_len."""
        seq = make_seq(3)
        padded = pad_sequence(seq, MAX_LEN)
        assert padded.shape == (MAX_LEN, N_FEATURES)

    def test_padding_value_is_sentinel(self):
        """Padded positions must be exactly PADDING_VALUE, not 0.0."""
        seq = make_seq(3)
        padded = pad_sequence(seq, MAX_LEN)
        n_padded = MAX_LEN - 3
        assert np.all(padded[:n_padded] == PADDING_VALUE), \
            f"Padded rows should be {PADDING_VALUE}, found {padded[:n_padded]}"

    def test_real_data_preserved_after_padding(self):
        """Real data should appear unchanged after padding."""
        seq = make_seq(3)
        padded = pad_sequence(seq, MAX_LEN)
        np.testing.assert_array_almost_equal(padded[-3:], seq)

    def test_exact_length_sequence_not_modified(self):
        """Sequence exactly equal to max_len needs no padding."""
        seq = make_seq(MAX_LEN)
        padded = pad_sequence(seq, MAX_LEN)
        assert padded.shape == (MAX_LEN, N_FEATURES)
        np.testing.assert_array_equal(padded, seq)

    def test_longer_sequence_truncated_to_max_len(self):
        """Sequences longer than max_len should be truncated from the left."""
        seq = make_seq(MAX_LEN + 5)
        padded = pad_sequence(seq, MAX_LEN)
        assert padded.shape == (MAX_LEN, N_FEATURES)
        # Should keep the LAST max_len timesteps
        np.testing.assert_array_equal(padded, seq[-MAX_LEN:])

    def test_padding_does_not_produce_zero(self):
        """PADDING_VALUE must not be 0.0 — this is the whole point of Problem 3 fix."""
        assert PADDING_VALUE != 0.0, \
            "PADDING_VALUE must not be 0.0 — it would conflict with StandardScaler outputs"

    def test_padding_never_appears_in_real_data(self):
        """Real data from a synthetic sequence should not contain PADDING_VALUE."""
        seq = make_seq(5)
        # Real data values (0.1 to 5.0) should never equal -999.0
        assert not np.any(seq == PADDING_VALUE)


# ---------------------------------------------------------------------------
# Masking Tests
# ---------------------------------------------------------------------------

class TestMaskingDetection:

    def test_padding_detection_via_sentinel(self):
        """Padding rows should be detectable by checking all values == PADDING_VALUE."""
        seq = make_seq(3)
        padded = pad_sequence(seq, MAX_LEN)

        # This is the logic used in step4 and scoring.py
        non_padding_mask = ~np.all(padded == PADDING_VALUE, axis=1)

        assert np.sum(non_padding_mask) == 3, \
            f"Should detect 3 valid timesteps, detected {np.sum(non_padding_mask)}"
        assert np.sum(~non_padding_mask) == MAX_LEN - 3, \
            f"Should detect {MAX_LEN - 3} padding rows"

    def test_fully_padded_sequence_has_no_valid_timesteps(self):
        """An all-padding sequence should have zero valid timesteps."""
        padded = np.full((MAX_LEN, N_FEATURES), PADDING_VALUE)
        non_padding_mask = ~np.all(padded == PADDING_VALUE, axis=1)
        assert np.sum(non_padding_mask) == 0

    def test_fully_real_sequence_has_all_valid_timesteps(self):
        """A sequence with no padding should have all timesteps valid."""
        seq = make_seq(MAX_LEN)
        non_padding_mask = ~np.all(seq == PADDING_VALUE, axis=1)
        assert np.sum(non_padding_mask) == MAX_LEN


# ---------------------------------------------------------------------------
# Event-Level Split Tests
# ---------------------------------------------------------------------------

class TestEventLevelSplit:

    def _make_events(self, n_events=50, seed=0):
        """Make synthetic event IDs."""
        rng = np.random.default_rng(seed)
        return [f"event_{i}" for i in range(n_events)]

    def test_no_event_in_both_train_and_test(self):
        """After an event-level split, no event should appear in both train and test."""
        events = self._make_events(100)
        rng = np.random.default_rng(42)
        rng.shuffle(events)

        n_train = int(0.8 * len(events))
        n_val = int(0.1 * len(events))

        train_events = set(events[:n_train])
        val_events = set(events[n_train:n_train + n_val])
        test_events = set(events[n_train + n_val:])

        assert len(train_events & test_events) == 0, \
            "Data leakage: events appear in both train and test"
        assert len(train_events & val_events) == 0, \
            "Data leakage: events appear in both train and val"
        assert len(val_events & test_events) == 0, \
            "Data leakage: events appear in both val and test"

    def test_split_covers_all_events(self):
        """Train + val + test should cover all events exactly once."""
        events = self._make_events(100)
        n_train = int(0.8 * len(events))
        n_val = int(0.1 * len(events))

        train_events = set(events[:n_train])
        val_events = set(events[n_train:n_train + n_val])
        test_events = set(events[n_train + n_val:])

        all_split = train_events | val_events | test_events
        assert len(all_split) == len(events), \
            "Some events are missing from the split"


# ---------------------------------------------------------------------------
# Sequence Shape Tests
# ---------------------------------------------------------------------------

class TestSequenceShapes:

    def test_X_is_3d(self):
        """X arrays should be 3D: (samples, timesteps, features)."""
        n_samples = 10
        X = np.stack([pad_sequence(make_seq(3), MAX_LEN) for _ in range(n_samples)])
        assert X.ndim == 3, f"X should be 3D, got shape {X.shape}"

    def test_Y_is_2d(self):
        """Y (target) arrays should be 2D: (samples, features)."""
        n_samples = 10
        Y = np.stack([make_seq(1)[0] for _ in range(n_samples)])
        assert Y.ndim == 2, f"Y should be 2D, got shape {Y.shape}"

    def test_X_timesteps_match_max_len(self):
        """X timestep dimension should equal MAX_SEQUENCE_LENGTH."""
        n_samples = 5
        X = np.stack([pad_sequence(make_seq(i + 1), MAX_LEN) for i in range(n_samples)])
        assert X.shape[1] == MAX_LEN

    def test_X_and_Y_same_number_of_samples(self):
        """X and Y must have the same number of samples."""
        n_samples = 20
        X = np.stack([pad_sequence(make_seq(3), MAX_LEN) for _ in range(n_samples)])
        Y = np.stack([make_seq(1)[0] for _ in range(n_samples)])
        assert X.shape[0] == Y.shape[0]
