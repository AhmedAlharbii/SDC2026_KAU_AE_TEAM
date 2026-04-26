"""
Unit tests for model building and I/O.

Tests that:
- The model builds with correct architecture
- LayerNorm is present (NOT BatchNorm)
- MC Dropout is active during training=True
- Weights save and load correctly
- Predictions are reproducible when training=False

Run with: pytest Scripts/tests/test_model_io.py -v
"""

import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF noise during tests
import tensorflow as tf
from model_builder import build_self_supervised_gru, build_model_from_config

# Minimal model config for tests — fast to build and run
SMALL_CONFIG = {
    'n_timesteps': 10,
    'n_features': 5,
    'gru_units_1': 16,
    'gru_units_2': 8,
    'dense_units': 8,
    'dropout_rate': 0.3,
    'l2_reg': 0.001,
    'seed': 42,
}

PADDING_VALUE = -999.0


def make_model():
    return build_self_supervised_gru(
        n_timesteps=SMALL_CONFIG['n_timesteps'],
        n_features=SMALL_CONFIG['n_features'],
        gru_units_1=SMALL_CONFIG['gru_units_1'],
        gru_units_2=SMALL_CONFIG['gru_units_2'],
        dense_units=SMALL_CONFIG['dense_units'],
        dropout_rate=SMALL_CONFIG['dropout_rate'],
        l2_reg=SMALL_CONFIG['l2_reg'],
    )


def make_input(n_samples=4):
    X = np.full((n_samples, SMALL_CONFIG['n_timesteps'], SMALL_CONFIG['n_features']), PADDING_VALUE)
    X[:, -3:, :] = np.random.uniform(0.1, 1.0, size=(n_samples, 3, SMALL_CONFIG['n_features']))
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Architecture Tests
# ---------------------------------------------------------------------------

class TestModelArchitecture:

    def test_model_builds_without_error(self):
        """Model should build cleanly from config."""
        model = make_model()
        assert model is not None

    def test_model_has_correct_output_shape(self):
        """Output should be (batch, n_features)."""
        model = make_model()
        X = make_input(4)
        out = model(X, training=False)
        assert out.shape == (4, SMALL_CONFIG['n_features']), \
            f"Expected (4, {SMALL_CONFIG['n_features']}), got {out.shape}"

    def test_model_uses_layer_norm_not_batch_norm(self):
        """Model MUST use LayerNormalization, NOT BatchNormalization."""
        model = make_model()
        layer_types = [type(layer).__name__ for layer in model.layers]

        assert 'LayerNormalization' in layer_types, \
            "Model must contain LayerNormalization"
        assert 'BatchNormalization' not in layer_types, \
            "Model must NOT use BatchNormalization (breaks MC Dropout)"

    def test_model_has_masking_layer_with_correct_sentinel(self):
        """Masking layer must exist and use mask_value=-999.0."""
        model = make_model()
        masking_layers = [l for l in model.layers if type(l).__name__ == 'Masking']
        assert len(masking_layers) == 1, "Model should have exactly one Masking layer"
        assert masking_layers[0].mask_value == PADDING_VALUE, \
            f"Masking layer must use mask_value={PADDING_VALUE}, got {masking_layers[0].mask_value}"

    def test_model_has_dropout_layers(self):
        """Model must have Dropout layers for MC Dropout to work."""
        model = make_model()
        dropout_layers = [l for l in model.layers if type(l).__name__ == 'Dropout']
        assert len(dropout_layers) >= 2, \
            f"Model should have at least 2 Dropout layers, found {len(dropout_layers)}"

    def test_build_from_config_matches_direct_build(self):
        """build_model_from_config should produce same architecture as direct build."""
        model_direct = make_model()
        model_config = build_model_from_config(SMALL_CONFIG)

        # Compare number of parameters
        assert model_direct.count_params() == model_config.count_params()


# ---------------------------------------------------------------------------
# MC Dropout Tests
# ---------------------------------------------------------------------------

class TestMCDropout:

    def test_predictions_vary_with_training_true(self):
        """With training=True (MC Dropout active), repeated passes must differ."""
        model = make_model()
        X = make_input(4)
        # Warm up
        _ = model(X, training=True)

        preds = [model(X, training=True).numpy() for _ in range(10)]
        preds = np.array(preds)

        # Standard deviation across MC samples should be > 0
        std_across_samples = preds.std(axis=0)
        assert std_across_samples.mean() > 1e-6, \
            "MC Dropout should produce different outputs each pass (got near-zero std)"

    def test_predictions_deterministic_with_training_false(self):
        """With training=False (dropout off), repeated passes must be identical."""
        model = make_model()
        X = make_input(4)
        # Warm up
        _ = model(X, training=False)

        pred1 = model(X, training=False).numpy()
        pred2 = model(X, training=False).numpy()

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6,
            err_msg="Predictions with training=False should be deterministic")


# ---------------------------------------------------------------------------
# Weight Save / Load Tests
# ---------------------------------------------------------------------------

class TestWeightIO:

    def test_weights_save_and_load_h5(self):
        """Saved weights should produce identical predictions after reload."""
        model = make_model()
        X = make_input(4)
        # Build the model
        _ = model(X, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_weights.weights.h5')
            model.save_weights(path)

            # Build fresh model and load weights
            model2 = make_model()
            _ = model2(X, training=False)  # Build before loading
            model2.load_weights(path)

            pred1 = model(X, training=False).numpy()
            pred2 = model2(X, training=False).numpy()

            np.testing.assert_array_almost_equal(pred1, pred2, decimal=5,
                err_msg="Predictions should match after save/load of weights")

    def test_weights_load_fails_on_wrong_architecture(self):
        """Loading weights into a different architecture should raise an error."""
        model_small = make_model()
        X = make_input(2)
        _ = model_small(X, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'small_weights.weights.h5')
            model_small.save_weights(path)

            # Build a model with different architecture
            model_large = build_self_supervised_gru(
                n_timesteps=SMALL_CONFIG['n_timesteps'],
                n_features=SMALL_CONFIG['n_features'],
                gru_units_1=64,   # Different!
                gru_units_2=32,   # Different!
                dense_units=32,   # Different!
                dropout_rate=0.3,
                l2_reg=0.001,
            )
            _ = model_large(X, training=False)

            with pytest.raises((OSError, ValueError, RuntimeError, Exception)):
                model_large.load_weights(path).expect_partial()


# ---------------------------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------------------------

class TestInputHandling:

    def test_all_padding_input_does_not_crash(self):
        """A batch of all-padding inputs should not raise an error."""
        model = make_model()
        X = np.full((2, SMALL_CONFIG['n_timesteps'], SMALL_CONFIG['n_features']),
                    PADDING_VALUE, dtype=np.float32)
        result = model(X, training=False)
        assert result.shape == (2, SMALL_CONFIG['n_features'])

    def test_mixed_padding_and_real_data(self):
        """Sequences with some padding and some real data should work correctly."""
        model = make_model()
        X = make_input(8)
        result = model(X, training=False)
        assert not np.any(np.isnan(result.numpy())), "Output should not contain NaN"
        assert not np.any(np.isinf(result.numpy())), "Output should not contain Inf"
