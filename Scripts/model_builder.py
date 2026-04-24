"""Shared model construction utilities for training and inference scripts."""

from tensorflow.keras import Model, layers, regularizers


def build_self_supervised_gru(
    n_timesteps,
    n_features,
    gru_units_1=128,
    gru_units_2=64,
    dense_units=64,
    dropout_rate=0.3,
    l2_reg=0.001,
):
    """Build the shared BiGRU model used across step3/step3b/step4."""

    inputs = layers.Input(shape=(n_timesteps, n_features), name="cdm_sequence")
    x = layers.Masking(mask_value=-999.0)(inputs)

    x = layers.Bidirectional(
        layers.GRU(
            gru_units_1,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
        ),
        name="bigru_1",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(
        layers.GRU(
            gru_units_2,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
        ),
        name="bigru_2",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        dense_units // 2,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_features, name="predicted_cdm")(x)
    return Model(inputs=inputs, outputs=outputs, name="SelfSupervised_CDM_GRU")


def build_model_from_config(config):
    """Build model from persisted config with safe defaults for old artifacts."""

    return build_self_supervised_gru(
        n_timesteps=config["n_timesteps"],
        n_features=config["n_features"],
        gru_units_1=config.get("gru_units_1", 128),
        gru_units_2=config.get("gru_units_2", 64),
        dense_units=config.get("dense_units", 64),
        dropout_rate=config.get("dropout_rate", 0.3),
        l2_reg=config.get("l2_reg", 0.001),
    )
