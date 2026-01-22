import numpy as np
from typing import Union

from live_trader.ml_model.evaluations import brier
from live_trader.ml_model.layers import *

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Attention, LayerNormalization, Add, GlobalAveragePooling1D, 
    Conv1D, MultiHeadAttention, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


# --------------------  Basic LSTM  --------------------

def build_lstm(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds and compiles an LSTM model for sequence data.

    Args:
        X_train_seq (Union[np.ndarray, list]): Input feature sequences used to determine input shape.
            Expected shape is (num_samples, time_steps, num_features).

    Returns:
        Model: Compiled Keras LSTM model ready for training.
    """
    # Determine number of features dynamically
    n_features = X_train_seq.shape[2]

    # Use functional API with Input layer (instead of input_shape in LSTM)
    inputs = Input(shape=(None, n_features))  # variable time steps allowed¨
 
    x = LSTM(50, return_sequences = False)(inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs, name = "basic_lstm")

    model.compile(
        loss = 'binary_crossentropy', 
        optimizer = Adam(learning_rate=1e-3), 
        metrics = [
            AUC(name = "roc_auc"),
            brier
            ]
    )
    
    return model



# --------------------  Attention BiLSTM  --------------------



def build_attention_bilstm(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds and compiles an Attention-enhanced BiLSTM model for binary stock movement prediction.

    Args:
        X_train_seq (Union[np.ndarray, list]): Input feature sequences used to determine input shape.
            Expected shape is (num_samples, time_steps, num_features).

    Returns:
        Model: Compiled Keras BiLSTM-Attention model.
    """

    n_features = X_train_seq.shape[2]

    # Input Layer
    inputs = Input(shape = (None, n_features))

    # Bidirectional LSTM (returns full sequence for attention)
    x = Bidirectional(LSTM(64, return_sequences = True))(inputs)
    x = Dropout(0.3)(x)

    # Attention mechanism
    attn = Attention()([x, x])               # Self-attention over time steps
    x = Add()([x, attn])                     # Residual connection
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Binary output (up/down)
    outputs = Dense(1, activation='sigmoid')(x)

    # Build and compile
    model = Model(inputs, outputs, name = "attention_bilstm")

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = Adam(learning_rate = 1e-3),
        metrics = [
            AUC(name = "roc_auc"),
            brier
            ]
    )

    return model



# --------------------  TCN-lite  --------------------



def build_tcn_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds a lightweight Temporal Convolutional Network (TCN-style)
    for noisy financial time series classification.

    Designed to be robust to non-stationarity and overfitting.

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    x = Conv1D(
        filters=32,
        kernel_size=3,
        padding="causal",
        activation="relu"
    )(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(
        filters=16,
        kernel_size=3,
        padding="causal",
        activation="relu"
    )(x)
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="tcn_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            brier
        ]
    )

    return model



# --------------------  PatchTST-lite --------------------



def build_patchtst_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds a lightweight PatchTST-style Transformer model for
    noisy financial time series classification.

    The model splits the time dimension into patches, embeds them,
    and applies a Transformer encoder for temporal modeling.

    Designed for robustness to non-stationarity and overfitting.

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    patch_len: int = 16
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.3

    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    # Patch embedding
    x = Patchify(patch_len=patch_len, name="patchify")(inputs)

    x = Dense(d_model, activation="linear")(x)
    x = LayerNormalization()(x)

    # Transformer Encoder Block
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)

    x = LayerNormalization()(x + attn_out)

    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dropout(dropout)(ff_out)
    ff_out = Dense(d_model)(ff_out)

    x = LayerNormalization()(x + ff_out)

    # Pooling & Head
    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="patchtst_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            brier
        ]
    )

    return model



# --------------------  GNN-lite --------------------



def build_gnn_lite(X_train_seq: Union[np.ndarray, list],) -> Model:
    """
    Builds a lightweight Graph Neural Network (GNN-style) model
    for noisy financial time series classification.

    Nodes represent features (indicators).
    Edges are learned implicitly via feature interactions.

    Designed for robustness to:
    - Non-stationarity
    - Variable-length sequences
    - Small batch sizes

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    hidden_dim: int = 32
    gnn_layers: int = 2
    dropout: float = 0.3

    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    # Temporal aggregation
    # (B, T, F) → (B, F)
    x = GlobalAveragePooling1D(name="temporal_pool")(inputs)

    # Treat features as nodes
    # (B, F) → (B, F, 1)
    x = ExpandDims(axis=-1, name="expand_dims")(x)

    # GNN layers
    for i in range(gnn_layers):
        x = GraphMessagePassing(
            hidden_dim=hidden_dim,
            dropout=dropout,
            name=f"gnn_layer_{i}"
        )(x)

    # Graph pooling
    x = GlobalAveragePooling1D(name="graph_pool")(x)

    # Head
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="gnn_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[AUC(name="auc"), brier]
    )

    return model



# --------------------  Neural Anomaly Detector (NAD-lite)  --------------------



def build_autoencoder_classifier_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds an Autoencoder + Classifier model for
    neural anomaly detection in time series.

    Fully compatible with Keras 3 and existing pipelines.

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    latent_dim: int = 16
    hidden_dim: int = 64
    dropout: float = 0.3
    recon_weight: float = 0.3

    n_features = X_train_seq.shape[2]

    model = AutoencoderClassifierLite(
        n_features=n_features,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        recon_weight=recon_weight,
        name="autoencoder_classifier_lite"
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[AUC(name="auc"), brier]
    )

    return model



# --------------------  CNN-GRU-lite --------------------



def build_cnn_gru_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds a lightweight CNN-GRU model for
    noisy financial time series classification.

    Combines shallow temporal convolutions for
    local pattern extraction with a compact GRU
    layer for sequence modeling.

    Designed to be robust to non-stationarity
    and overfitting.

    Args:
        X_train_seq (array-like):
            Training sequences of shape
            (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """
    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    # CNN block
    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)
    x = LayerNormalization()(x)

    # GRU block
    x = GRU(
        units=32,
        dropout=0.3
    )(x)

    # Head
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="cnn_gru_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            brier
        ]
    )

    return model